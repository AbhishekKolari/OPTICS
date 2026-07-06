"""
Uncertainty estimation utilities for OPTICS benchmarks.

Supports:
  - Open-source VLMs: MSP, perplexity, and mean token entropy from generation scores.
  - Closed-source APIs (OpenAI): lm-polygraph BlackboxModel estimators when installed.

Open-source metrics follow lm-polygraph conventions (higher = more uncertain).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Open-source uncertainty from generation scores
# ---------------------------------------------------------------------------

UncertaintyScores = Dict[str, float]


@torch.no_grad()
def extract_step_logprobs_and_entropies(gen_out, prompt_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract per-step log-probabilities and entropies from model.generate output.

    Args:
        gen_out: output of model.generate(..., return_dict_in_generate=True, output_scores=True)
        prompt_len: number of tokens in the prompt (input_ids length)

    Returns:
        token_logprobs: Tensor[T] log p of the generated token at each step
        token_entropies: Tensor[T] entropy of next-token distribution at each step
    """
    seq = gen_out.sequences
    scores = gen_out.scores

    gen_token_ids = seq[:, prompt_len:]
    num_steps = gen_token_ids.size(1)
    if num_steps != len(scores):
        raise ValueError(f"Mismatch: gen tokens {num_steps} vs scores {len(scores)}")

    token_logprobs = []
    token_entropies = []

    for t in range(num_steps):
        logits_t = scores[t]
        logp_t = F.log_softmax(logits_t, dim=-1)

        tok_id_t = gen_token_ids[:, t].unsqueeze(-1)
        tok_logp_t = logp_t.gather(dim=-1, index=tok_id_t).squeeze(-1)

        p_t = logp_t.exp()
        ent_t = -(p_t * logp_t).sum(dim=-1)

        token_logprobs.append(tok_logp_t)
        token_entropies.append(ent_t)

    token_logprobs = torch.stack(token_logprobs, dim=0)
    token_entropies = torch.stack(token_entropies, dim=0)

    return token_logprobs[:, 0], token_entropies[:, 0]


def uncertainty_msp(token_logprobs: torch.Tensor) -> float:
    """Maximum Sequence Probability (lm-polygraph style): negative total log-likelihood."""
    return float((-token_logprobs.sum()).item())


def uncertainty_perplexity(token_logprobs: torch.Tensor) -> float:
    """Perplexity estimator: negative mean log-likelihood (no exp)."""
    if token_logprobs.numel() == 0:
        return float("nan")
    return float((-(token_logprobs.mean())).item())


def uncertainty_mean_token_entropy(token_entropies: torch.Tensor) -> float:
    """Mean token entropy across generated steps."""
    if token_entropies.numel() == 0:
        return float("nan")
    return float(token_entropies.mean().item())


def compute_opensource_uncertainty_scores(
    gen_out,
    prompt_len: int,
) -> UncertaintyScores:
    """Compute all open-source uncertainty metrics from a generate() output."""
    token_logprobs, token_entropies = extract_step_logprobs_and_entropies(gen_out, prompt_len)
    return {
        "msp": uncertainty_msp(token_logprobs),
        "perplexity": uncertainty_perplexity(token_logprobs),
        "mean_token_entropy": uncertainty_mean_token_entropy(token_entropies),
        "gen_len": int(token_logprobs.numel()),
    }


def generate_with_uncertainty(
    model,
    inputs: dict,
    prompt_len: int,
    max_new_tokens: int,
    pad_token_id=None,
    eos_token_id=None,
) -> Tuple[Any, UncertaintyScores]:
    """
    Run greedy generation with score output and return (sequences, uncertainty_scores).

    Returns sequences suitable for processor.decode / batch_decode.
    """
    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        early_stopping=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    scores = compute_opensource_uncertainty_scores(gen_out, prompt_len)
    return gen_out.sequences, scores


# ---------------------------------------------------------------------------
# Closed-source uncertainty via lm-polygraph (optional dependency)
# ---------------------------------------------------------------------------

CLOSED_UNCERTAINTY_METHODS = {
    "msp": "MaximumSequenceProbability",
    "maximum_sequence_probability": "MaximumSequenceProbability",
    "perplexity": "Perplexity",
    "mean_token_entropy": "MeanTokenEntropy",
    "kernel_language_entropy": "KernelLanguageEntropy",
}


def resolve_closed_uncertainty_method(method: str):
    """Resolve a user-facing method name to an lm-polygraph estimator class."""
    try:
        from lm_polygraph.estimators import (
            KernelLanguageEntropy,
            MaximumSequenceProbability,
            MeanTokenEntropy,
            Perplexity,
        )
    except ImportError as exc:
        raise ImportError(
            "Closed-source uncertainty requires lm-polygraph. "
            "Install with: pip install lm-polygraph"
        ) from exc

    registry = {
        "msp": MaximumSequenceProbability,
        "maximum_sequence_probability": MaximumSequenceProbability,
        "perplexity": Perplexity,
        "mean_token_entropy": MeanTokenEntropy,
        "kernel_language_entropy": KernelLanguageEntropy,
    }
    key = (method or "msp").strip().lower()
    if key not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unknown uncertainty method '{method}'. Supported: {supported}")
    return registry[key]()


def build_openai_multimodal_messages(image_b64: str, prompt: str) -> List[dict]:
    """Build OpenAI chat messages for a single image + text prompt."""
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def estimate_closedsource_uncertainty(
    image_b64: str,
    prompt: str,
    model_id: str,
    uncertainty_method: str = "msp",
    openai_api_key: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Estimate uncertainty for an OpenAI multimodal model via lm-polygraph.

    Returns:
        (uncertainty_score, generation_text)
    """
    try:
        from lm_polygraph import BlackboxModel
        from lm_polygraph.utils import estimate_uncertainty
    except ImportError as exc:
        raise ImportError(
            "Closed-source uncertainty requires lm-polygraph. "
            "Install with: pip install lm-polygraph"
        ) from exc

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for closed-source uncertainty estimation")

    estimator = resolve_closed_uncertainty_method(uncertainty_method)
    messages = build_openai_multimodal_messages(image_b64, prompt)

    model = BlackboxModel.from_openai(
        openai_api_key=api_key,
        model_path=model_id,
        supports_logprobs=True,
        temperature=0.0,
    )
    outs = estimate_uncertainty(model, estimator, input_text=messages)
    return float(outs.uncertainty), outs.generation_text.strip()


def supports_opensource_uncertainty(model_name: str) -> bool:
    """
    Whether open-source uncertainty (via output_scores) is supported for this model.

    InternVL (.chat API) and BLIP-2 do not expose step-wise scores through generate().
    """
    key = model_name.lower()
    if "internvl" in key or "blip2" in key:
        return False
    return True
