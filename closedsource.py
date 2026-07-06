# closedsource.py
"""
Closed-source runner with alias mapping.

Function:
    run_closedsource(model_name, benchmark_json, data_dir, output_file, batch_size=360)

- Reads API keys from environment:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
- Recognizes short aliases or full vendor model ids and routes calls appropriately.
"""

import os
import base64
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import gc
from typing import Optional

import torch
from utils import BenchmarkTester
from utils import resolve_image_path

# Attempt SDK imports; raise error if missing
# NOTE: modern vendor SDKs (used throughout this file):
#   - OpenAI:    openai>=1.0 client-based API (openai.OpenAI(...).chat.completions.create)
#   - Anthropic: anthropic>=0.25 Messages API (anthropic.Anthropic(...).messages.create)
#   - Gemini:    the new unified `google-genai` SDK (`from google import genai`),
#                NOT the legacy `google-generativeai` package. The old package exposes
#                `genai.configure()` / `genai.GenerativeModel()` and has no `genai.Client`,
#                so mixing the two raises AttributeError at call time.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

# Defaults if env var not provided
DEFAULT_MODELS = {
    "gpt": "gpt-4o-mini-2024-07-18",
    "claude": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-2.5-flash"
}

# OpenAI "reasoning" models (o1, o3, o4, ...) use a different call signature:
# no `temperature` argument (only the default of 1 is supported) and
# `max_completion_tokens` instead of `max_tokens`.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model_id: str) -> bool:
    """True for OpenAI o-series reasoning models (o1, o3, o4, ...)."""
    return (model_id or "").lower().startswith(_REASONING_MODEL_PREFIXES)


def _guess_image_media_type(image_path) -> str:
    """Best-effort MIME type for an image file, used by the Anthropic vision payload."""
    suffix = Path(str(image_path)).suffix.lower()
    return {
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

def _looks_like_model_id(s: str) -> bool:
    """Heuristic: model ids typically contain '-' or '/' (repo-like or vendor model names)."""
    if not s:
        return False
    return ("-" in s) or ("/" in s)


def _is_exact_openai_model_id(s: str) -> bool:
    """Recognize OpenAI model ids that lack dashes (e.g. o3, o1)."""
    lower = s.lower()
    return lower.startswith(("o1", "o3", "gpt"))

def resolve_closed_model(user_input: str):
    """
    Resolve user_input (alias or full id) into (provider, model_id)
    provider in {'gpt','claude','gemini'}.
    Returns: (provider_str, model_id)
    """
    if not user_input:
        raise ValueError("model_name is required")

    ui = user_input.strip()
    ui_lower = ui.lower()

    # Case 0: bare provider aliases (no dash, no repo-like id) always resolve via
    # the alias branch below, even though "gpt" would otherwise pass
    # _is_exact_openai_model_id's startswith("gpt") check.
    _BARE_ALIASES = {"gpt", "claude", "gemini", "openai", "anthropic", "google"}
    if ui_lower in _BARE_ALIASES:
        pass  # fall through to Case 2

    # Case 1: user passed a full model id (use exact string)
    elif _looks_like_model_id(ui) or _is_exact_openai_model_id(ui):
        # infer provider by substring
        if "gpt" in ui_lower or ui_lower.startswith(("o1", "o3")):
            return "gpt", ui
        if "claude" in ui_lower:
            return "claude", ui
        if "gemini" in ui_lower or "google" in ui_lower:
            return "gemini", ui
        # fallback: try to choose GPT by default if ambiguous
        return "gpt", ui

    # Case 2: user passed a short alias (or something containing provider keyword)
    if "gpt" in ui_lower:
        model_id = os.getenv("OPENAI_MODEL", DEFAULT_MODELS["gpt"])
        return "gpt", model_id
    if "claude" in ui_lower or "anthropic" in ui_lower:
        model_id = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODELS["claude"])
        return "claude", model_id
    if "gemini" in ui_lower or "google" in ui_lower:
        model_id = os.getenv("GOOGLE_MODEL", DEFAULT_MODELS["gemini"])
        return "gemini", model_id

    # Last resort: default to GPT branch with default model id
    return "gpt", os.getenv("OPENAI_MODEL", DEFAULT_MODELS["gpt"])


def _image_to_base64_str(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_closedsource(model_name: str, benchmark_json: str, data_dir: str, output_file: str, start_idx: int = 0, batch_size: int = 360):
    """
    Run closed-source vendor multimodal models on the benchmark.
    - model_name: alias or full vendor model id (see resolve_closed_model)
    - keys are read from env variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
    """
    provider, resolved_model_id = resolve_closed_model(model_name)
    provider = provider.lower()
    tester = BenchmarkTester(benchmark_json, data_dir)

    # read keys from env
    openai_key = os.getenv("OPENAI_API_KEY")
    anth_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # configure SDKs / clients if needed
    openai_client = None
    anthropic_client = None
    genai_client = None

    if provider == "gpt":
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed or too old. pip install --upgrade openai")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI requests")
        openai_client = OpenAI(api_key=openai_key)

    if provider == "claude":
        if anthropic is None:
            raise RuntimeError("Anthropic SDK not installed. pip install anthropic")
        if not anth_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is required for Anthropic requests")
        anthropic_client = anthropic.Anthropic(api_key=anth_key)

    if provider == "gemini":
        if genai is None:
            raise RuntimeError(
                "Google Gen AI SDK not installed. pip install google-genai "
                "(note: this is the `google-genai` package, not `google-generativeai`)"
            )
        if not google_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required for Gemini requests")
        genai_client = genai.Client(api_key=google_key)

    results = []
    images = tester.benchmark['benchmark']['images'][start_idx:start_idx + batch_size]
    total_images = len(images)

    for idx, image_data in enumerate(tqdm(images, desc="Processing images")):
        image_path = resolve_image_path(image_data['path'], data_dir, benchmark_json)
        if not image_path.exists():
            print(f"Warning: missing image {image_path}")
            continue

        image_results = []
        for question in image_data['questions']:
            prompt = f"{question['question']} Your response MUST be in the following format and nothing else:\n <NUMBER> [<OBJECT1>, <OBJECT2>, <OBJECT3>, ...]"
            try:
                raw_answer = ""

                if provider == "gpt":
                    # OpenAI multimodal call via the modern client SDK; pass base64 image as data URI.
                    b64 = _image_to_base64_str(str(image_path))
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    create_kwargs = dict(model=resolved_model_id, messages=messages)
                    if _is_reasoning_model(resolved_model_id):
                        # o-series reasoning models: no temperature, different token-limit kwarg
                        create_kwargs["max_completion_tokens"] = 2000
                    else:
                        create_kwargs["max_tokens"] = 2000
                        create_kwargs["temperature"] = 0.0
                    resp = openai_client.chat.completions.create(**create_kwargs)
                    raw_answer = (resp.choices[0].message.content or "").strip()

                elif provider == "claude":
                    # Anthropic Messages API: send the image as a real vision content block.
                    with open(image_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    media_type = _guess_image_media_type(image_path)
                    resp = anthropic_client.messages.create(
                        model=resolved_model_id,
                        max_tokens=200,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": img_b64,
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    )
                    raw_answer = "".join(
                        block.text for block in resp.content if getattr(block, "type", None) == "text"
                    ).strip()

                elif provider == "gemini":
                    # Google Gen AI SDK: pass the PIL image directly, no upload round-trip needed.
                    pil_image = Image.open(image_path).convert("RGB")
                    response = genai_client.models.generate_content(
                        model=resolved_model_id,
                        contents=[pil_image, prompt],
                        config=genai_types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=2000,
                        ),
                    )
                    raw_answer = (getattr(response, "text", None) or "").strip()

                else:
                    raise RuntimeError(f"Unhandled provider: {provider}")

                # Parse and append result using your BenchmarkTester cleaning logic
                cleaned = tester.clean_answer(raw_answer)
                image_results.append({
                    "image_id": image_data["image_id"],
                    "image_type": image_data.get("image_type", "unknown"),
                    "question_id": question["id"],
                    "question": question["question"],
                    "ground_truth": question.get("answer"),
                    "model_answer": cleaned["count"],
                    "model_reasoning": cleaned["reasoning"],
                    "raw_answer": raw_answer,
                    "property_category": question.get("property_category")
                })

            except Exception as e:
                print(f"Error for image {image_data['image_id']} q {question['id']}: {e}")
                continue

        results.extend(image_results)

    # Save results to output_file
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Closed-source run complete: results saved to {output_file}")
    return results
