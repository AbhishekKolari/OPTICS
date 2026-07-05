#!/usr/bin/env python3
"""
Unified runner for the merged-image comparison benchmark.

Supports open-source Hugging Face models and closed-source API models (GPT,
Claude, Gemini) in a single script. Model-specific loading and inference follow
the same patterns as opensource.py and closedsource.py for the main ORBIT
benchmark.
"""

import argparse
import base64
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from closedsource import resolve_closed_model
from utils import load_image_internvl

try:
    import openai
except Exception:
    openai = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

MERGED_ANSWER_SUFFIX = (
    "\n\nAnswer ONLY in this exact format. Do NOT provide reasoning, "
    "explanation, or any other text:\n <A_OR_B>"
)


def format_merged_prompt(question: str) -> str:
    return f"{question.strip()}{MERGED_ANSWER_SUFFIX}"


def extract_final_label(raw_output: str) -> Optional[str]:
    """Extract the final A/B label from model output."""
    text = (raw_output or "").strip()
    if not text:
        return None

    upper_text = text.upper()

    match = re.search(r"<([AB])>", upper_text)
    if match:
        return match.group(1).upper()

    if re.fullmatch(r"\s*[AB]\s*", upper_text):
        return upper_text.strip()

    match = re.search(r"(?:answer|final)\s*[:=]?\s*([AB])\b", upper_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    assistant_blocks = re.findall(r"assistant\s*[\n: ]+\s*([AB])\b", upper_text, re.IGNORECASE)
    if assistant_blocks:
        return assistant_blocks[-1].upper()

    match = re.search(r"([AB])\s*$", upper_text)
    if match:
        return match.group(1).upper()

    return None


def resolve_question_image_path(image_path_str: str, images_dir: str) -> str:
    """Resolve an image path from the questions JSON against a user-provided directory."""
    path = Path(image_path_str)
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path.resolve())
    candidate = Path(images_dir) / path.name
    if candidate.exists():
        return str(candidate)
    candidate = Path(images_dir) / path
    return str(candidate)


def detect_backend(model_name: str) -> str:
    """
    Return 'opensource' or 'closedsource'.
    HF repo ids and local paths route to opensource; API aliases route to closedsource.
    """
    ui = model_name.lower().strip()

    if os.path.isdir(model_name) or (os.path.isfile(model_name) and model_name.endswith((".bin", ".safetensors"))):
        return "opensource"

    if "/" in model_name:
        if ui.startswith("gpt-"):
            return "closedsource"
        return "opensource"

    if ui in {"gpt", "claude", "gemini", "openai", "anthropic"}:
        return "closedsource"

    if any(token in ui for token in ("gpt-", "claude", "anthropic", "gemini")):
        return "closedsource"

    return "opensource"


def load_opensource_model(model_name: str, processor_path: Optional[str] = None):
    """Load an open-source VLM and its processor/tokenizer."""
    resolved_lower = model_name.lower()
    processor_path = processor_path or model_name
    dtype_cuda = torch.float16 if torch.cuda.is_available() else torch.float32

    if "qwen2.5-vl" in resolved_lower or "qwen2_5_vl" in resolved_lower:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype_cuda,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    elif "blip2" in resolved_lower:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype_cuda,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        processor = Blip2Processor.from_pretrained(processor_path, trust_remote_code=True)

    elif "internvl" in resolved_lower:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype_cuda,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        processor = AutoTokenizer.from_pretrained(
            processor_path, trust_remote_code=True, use_fast=False
        )

    elif "gemma" in resolved_lower:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        token = os.getenv("HF_TOKEN")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            processor_path, token=token, trust_remote_code=True
        )

    else:
        from transformers import AutoModelForCausalLM, AutoProcessor

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_cuda,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    if hasattr(model.config, "use_memory_efficient_attention"):
        model.config.use_memory_efficient_attention = True

    model.eval()
    return model, processor


def infer_opensource(
    model_name: str,
    model,
    processor,
    image_path: str,
    question: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Run one merged-benchmark inference for an open-source model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompt = format_merged_prompt(question)
    model_key = model_name.lower()

    if "internvl" in model_key:
        pixel_values = load_image_internvl(image_path, input_size=448, max_num=12)
        pixel_values = pixel_values.to(
            device=device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=False,
        )
        with torch.no_grad():
            response = model.chat(
                processor,
                pixel_values,
                f"<image>\n {prompt}",
                generation_config=generation_config,
            )
        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response

    pil_image = Image.open(image_path).convert("RGB")

    if "gemma" in model_key:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            input_len = inputs["input_ids"].shape[-1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                early_stopping=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            generated = outputs[0][input_len:]
            answer = processor.decode(generated, skip_special_tokens=True)
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer

    if "blip2" in model_key:
        blip_prompt = f"Question: {prompt} Answer:"
        inputs = processor(images=pil_image, text=blip_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                early_stopping=False,
            )
            answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=pil_image, padding=True, return_tensors="pt").to(device)
    else:
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=False,
            pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        )
        try:
            answer = processor.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        except Exception:
            answer = processor.decode(outputs[0], skip_special_tokens=True)

    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answer


def _image_to_base64_str(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def setup_closedsource_client(provider: str):
    """Configure and return a closed-source API client."""
    provider = provider.lower()

    if provider == "gpt":
        if openai is None:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")
        openai.api_key = api_key
        return provider, openai

    if provider == "claude":
        if anthropic is None:
            raise RuntimeError("Anthropic SDK not installed. pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is required")
        try:
            client = anthropic.Client(api_key=api_key)
        except Exception:
            client = anthropic.Anthropic(api_key=api_key)
        return provider, client

    if provider == "gemini":
        if genai is None:
            raise RuntimeError("Google Generative AI SDK not installed. pip install google-generativeai")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required")
        genai.configure(api_key=api_key)
        return provider, genai.Client(api_key=api_key)

    raise RuntimeError(f"Unhandled provider: {provider}")


def infer_closedsource(
    provider: str,
    client,
    model_id: str,
    image_path: str,
    question: str,
) -> str:
    """Run one merged-benchmark inference for a closed-source API model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    prompt = format_merged_prompt(question)

    if provider == "gpt":
        b64 = _image_to_base64_str(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        resp = openai.ChatCompletion.create(
            model=model_id,
            messages=messages,
            max_tokens=2000,
            temperature=0.0,
        )
        try:
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return resp.choices[0].message.content.strip()

    if provider == "claude":
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        combined = f"IMAGE_BASE64:{img_b64}\n\n{prompt}"
        resp = client.completions.create(
            model=model_id,
            prompt=combined,
            max_tokens_to_sample=200,
        )
        return getattr(resp, "completion", getattr(resp, "text", str(resp))).strip()

    if provider == "gemini":
        uploaded = client.files.upload(file=image_path)
        response = client.models.generate_content(model=model_id, contents=[uploaded, prompt])
        return getattr(response, "text", getattr(response, "content", str(response))).strip()

    raise RuntimeError(f"Unhandled provider: {provider}")


def build_result_record(
    sample_index: int,
    question_data: Dict[str, Any],
    image_path: str,
    model_answer: Optional[str],
    raw_answer: str,
    include_metadata: bool,
    skipped: bool = False,
    skip_reason: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    ground_truth = question_data.get("answer_2", "")
    question_text = question_data.get("merged_question_2", "")

    record: Dict[str, Any] = {
        "sample_index": sample_index,
        "image_path": image_path,
        "question": question_text,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "raw_answer": raw_answer,
        "correct": (
            model_answer == ground_truth.upper()
            if model_answer and ground_truth
            else None
        ),
        "skipped": skipped,
    }
    if skip_reason:
        record["skip_reason"] = skip_reason
    if error:
        record["error"] = error
    if include_metadata:
        record["metadata"] = {
            "question_left": question_data.get("question1"),
            "question_right": question_data.get("question2"),
            "bucket": question_data.get("bucket"),
            "similarity": question_data.get("similarity"),
            "count_diff": question_data.get("count_diff"),
            "count_left": question_data.get("count1"),
            "count_right": question_data.get("count2"),
        }
    return record


def process_merged_benchmark(
    model_name: str,
    questions_json: str,
    output_file: str,
    images_dir: str,
    processor_path: Optional[str] = None,
    backend: str = "auto",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_new_tokens: int = 512,
    include_metadata: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate a model on the merged-image comparison benchmark."""
    resolved_backend = detect_backend(model_name) if backend == "auto" else backend
    print(f"Backend: {resolved_backend}")

    with open(questions_json, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    total_questions = len(all_questions)
    if end_idx is None:
        end_idx = total_questions
    questions = all_questions[start_idx:end_idx]
    print(f"Processing samples {start_idx} to {end_idx - 1} ({len(questions)} total)")

    model = None
    processor = None
    closed_provider = None
    closed_client = None
    closed_model_id = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if resolved_backend == "opensource":
        print(f"Loading open-source model from {model_name}...")
        model, processor = load_opensource_model(model_name, processor_path)
        print("Model loaded.")
    else:
        closed_provider, closed_model_id = resolve_closed_model(model_name)
        closed_provider, closed_client = setup_closedsource_client(closed_provider)
        print(f"Using closed-source model: {closed_model_id} ({closed_provider})")

    results: List[Dict[str, Any]] = []
    skipped_zero_diff = 0
    accuracy_by_diff: Dict[int, Dict[str, int]] = {}

    try:
        for offset, question_data in enumerate(tqdm(questions, desc="Evaluating")):
            sample_index = start_idx + offset
            image_rel = question_data.get("merged_image_path", "")
            image_path = resolve_question_image_path(image_rel, images_dir)

            ground_truth_count_diff = question_data.get("count_diff")
            if ground_truth_count_diff == 0:
                skipped_zero_diff += 1
                results.append(
                    build_result_record(
                        sample_index=sample_index,
                        question_data=question_data,
                        image_path=image_rel,
                        model_answer=None,
                        raw_answer="",
                        include_metadata=include_metadata,
                        skipped=True,
                        skip_reason="count_diff_zero",
                    )
                )
                continue

            try:
                if resolved_backend == "opensource":
                    raw_answer = infer_opensource(
                        model_name=model_name,
                        model=model,
                        processor=processor,
                        image_path=image_path,
                        question=question_data.get("merged_question_2", ""),
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    raw_answer = infer_closedsource(
                        provider=closed_provider,
                        client=closed_client,
                        model_id=closed_model_id,
                        image_path=image_path,
                        question=question_data.get("merged_question_2", ""),
                    )

                model_answer = extract_final_label(raw_answer)
                record = build_result_record(
                    sample_index=sample_index,
                    question_data=question_data,
                    image_path=image_rel,
                    model_answer=model_answer,
                    raw_answer=raw_answer,
                    include_metadata=include_metadata,
                )
                results.append(record)

                if ground_truth_count_diff is not None and record["correct"] is not None:
                    diff_key = int(ground_truth_count_diff)
                    stats = accuracy_by_diff.setdefault(diff_key, {"total": 0, "correct": 0})
                    stats["total"] += 1
                    if record["correct"]:
                        stats["correct"] += 1

            except Exception as exc:
                print(f"\nError on sample {sample_index}: {exc}")
                results.append(
                    build_result_record(
                        sample_index=sample_index,
                        question_data=question_data,
                        image_path=image_rel,
                        model_answer=None,
                        raw_answer="",
                        include_metadata=include_metadata,
                        error=str(exc),
                    )
                )

            if (offset + 1) % 25 == 0 or offset == len(questions) - 1:
                checkpoint = f"{output_file}.checkpoint.json"
                with open(checkpoint, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    finally:
        if model is not None:
            del model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    checkpoint_path = Path(f"{output_file}.checkpoint.json")
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    evaluated = sum(1 for r in results if not r.get("skipped"))
    successful = sum(1 for r in results if not r.get("skipped") and "error" not in r)
    correct = sum(1 for r in results if not r.get("skipped") and r.get("correct") is True)
    with_gt = sum(
        1 for r in results if not r.get("skipped") and r.get("ground_truth")
    )

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"  Samples in batch: {len(questions)}")
    print(f"  Skipped (count_diff = 0): {skipped_zero_diff}")
    print(f"  Evaluated: {evaluated}")
    print(f"  Successful: {successful}")
    print(f"  Errors: {evaluated - successful}")
    if with_gt > 0:
        print(f"  Correct: {correct}/{with_gt} ({100 * correct / with_gt:.2f}%)")
    if accuracy_by_diff:
        print("  Accuracy by count diff:")
        for diff_value in sorted(accuracy_by_diff):
            stats = accuracy_by_diff[diff_value]
            if stats["total"] > 0:
                acc = 100 * stats["correct"] / stats["total"]
                print(f"    diff={diff_value}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 60}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run any open- or closed-source VLM on the merged-image comparison benchmark"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HF repo id / local path (open-source) or API alias / model id (closed-source)",
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Processor path for open-source models (defaults to --model_name)",
    )
    parser.add_argument(
        "--questions_json",
        type=str,
        default="benchmark_merged.json",
        help="Path to merged questions JSON",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./merged_images",
        help="Directory containing comparison images",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "opensource", "closedsource"],
        default="auto",
        help="Force open- or closed-source backend (default: auto-detect from model_name)",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in questions list")
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index in questions list (exclusive; default: all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate (open-source models)",
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Include optional dataset metadata fields in each result record",
    )
    args = parser.parse_args()

    process_merged_benchmark(
        model_name=args.model_name,
        questions_json=args.questions_json,
        output_file=args.output_file,
        images_dir=args.images_dir,
        processor_path=args.processor_path,
        backend=args.backend,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_new_tokens=args.max_new_tokens,
        include_metadata=args.include_metadata,
    )


if __name__ == "__main__":
    main()
