# opensource.py
"""
Open-source runner with alias mapping.

Function:
    run_opensource(model_name, processor_path, benchmark_json, data_dir, output_file, batch_size=360)

- Supports hardcoded tested models (Qwen2.5-VL, BLIP2, InternVL3, Gemma3) using the special loading
  & config required for reproducibility.
- Falls back to a generic Hugging Face loader for any other HF repo id or local path.
"""

import gc
import torch
import os
from pathlib import Path
from utils import BenchmarkTester

# Hardcoded aliases -> representative repo ids (lowercase)
MODEL_ALIAS_MAP = {
    "qwen2.5-vl": ["qwen/qwen2.5-vl-7b-instruct", "qwen/qwen2.5-vl-32b-instruct"],
    "qwen3.5": ["qwen/qwen3.5-9b"],
    "blip2": ["salesforce/blip2-flan-t5-xxl", "salesforce/blip2-opt-2.7b", "salesforce/blip2-opt-6.7b"],
    "internvl3": ["opengvlab/internvl3-8b", "opengvlab/internvl3-14b"],
    "gemma-3": ["google/gemma-3-27b-it"]
}

def resolve_model_alias(user_input):
    user_input_lower = user_input.lower()
    for alias in MODEL_ALIAS_MAP.keys():
        if alias in user_input_lower:  # detect model variants
            return user_input
    return user_input


def run_opensource(model_name: str, processor_path: str, benchmark_json: str, data_dir: str, output_file: str, start_idx: int = 0, batch_size: int = 360):
    """
    Load either a hardcoded tested model (alias) or any HF model by repo id/local path.
    Calls BenchmarkTester.evaluate_model(...) which contains per-model generation/preproc branches.
    """
    resolved_name = resolve_model_alias(model_name)
    resolved_lower = resolved_name.lower()
    tester = BenchmarkTester(benchmark_json, data_dir)

    # Hardcoded model branches for reproducibility
    if "qwen2.5-vl" in resolved_lower:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"Loading hardcoded model Qwen2.5-VL from '{model_name}' (alias resolved to qwen2.5-vl)")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        processor = AutoProcessor.from_pretrained(processor_path or model_name, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels)

    elif "qwen3.5" in resolved_lower or "qwen3_5" in resolved_lower or "qwen3" in resolved_lower:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print(f"Loading hardcoded model Qwen3.5 from '{model_name}' (alias resolved to qwen3.5)")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(processor_path or model_name, trust_remote_code=True)

    elif "blip2" in resolved_lower:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        print(f"Loading hardcoded model BLIP-2 from '{model_name}' (alias resolved to blip2)")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        processor = Blip2Processor.from_pretrained(processor_path or model_name, trust_remote_code=True)

    elif "internvl3" in resolved_lower:
        from transformers import AutoModel, AutoTokenizer
        print(f"Loading hardcoded model InternVL3 from '{model_name}' (alias resolved to internvl3)")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        processor = AutoTokenizer.from_pretrained(processor_path or model_name, trust_remote_code=True, use_fast=False)

    elif "gemma-3" in resolved_lower:
        from transformers import Gemma3ForConditionalGeneration, AutoProcessor
        """
        HF_TOKEN is set in the environment variables, e.g. HF_TOKEN="hf_..."
        """
        print(f"Loading hardcoded model Gemma3 from '{model_name}' (alias resolved to gemma3)")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            token=os.getenv("HF_TOKEN"),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(processor_path or model_name, token=os.getenv("HF_TOKEN"), trust_remote_code=True)

    else:
        # Generic HF fallback: attempt to load processor + AutoModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"No hardcoded alias resolved for '{model_name}'. Loading generically from HF or local path.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        # processor_path default fallback to model_name (repo id or local path)
        processor = AutoProcessor.from_pretrained(processor_path or model_name, trust_remote_code=True)

    # optional small tweak if model supports memory efficient attention
    if hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = True

    try:
        tester.evaluate_model(model_name, model, processor, output_file, start_idx=start_idx, batch_size=batch_size)
    finally:
        # cleanup
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
