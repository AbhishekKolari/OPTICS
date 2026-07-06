# utils.py
import json
import time
import gc
from pathlib import Path
from collections import defaultdict
from typing import List, Any, Dict
from typing import Optional
import pandas as pd

from PIL import Image
from tqdm import tqdm
import torch

# torchvision transforms for InternVL preprocessing
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# ----------------------------
# Answer cleaning
# ----------------------------
import re
def clean_answer_improved(answer: str):
    """Robust cleaning: extract digits in canonical '<N> [a,b,..]' or fallbacks (words, roman)."""
    if not answer:
        return {"count": "0", "reasoning": []}

    answer = str(answer).strip()
    answer_lower = answer.lower()

    pattern = r'(\d+)\s*\[(.*?)\]'
    match = re.search(pattern, answer)
    if match:
        number = match.group(1)
        objects = [obj.strip() for obj in match.group(2).split(',') if obj.strip()]
        return {"count": number, "reasoning": objects}

    numbers = re.findall(r'\d+', answer)
    if numbers:
        return {"count": numbers[0], "reasoning": []}

    roman_to_num = {
        'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7,
        'viii': 8, 'ix': 9, 'x': 10, 'xi': 11, 'xii': 12, 'xiii': 13,
        'xiv': 14, 'xv': 15, 'xvi': 16, 'xvii': 17, 'xviii': 18, 'xix': 19,
        'xx': 20
    }

    for roman in sorted(roman_to_num.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(roman) + r'\b', answer_lower):
            return {"count": str(roman_to_num[roman]), "reasoning": []}

    units = {
        'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
        'six':6, 'seven':7, 'eight':8, 'nine':9
    }
    teens = {
        'ten':10, 'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15,
        'sixteen':16, 'seventeen':17, 'eighteen':18, 'nineteen':19
    }
    tens = {
        'twenty':20, 'thirty':30, 'forty':40, 'fifty':50,
        'sixty':60, 'seventy':70, 'eighty':80, 'ninety':90
    }

    for w, n in {**teens, **units}.items():
        if re.search(r'\b' + re.escape(w) + r'\b', answer_lower):
            return {"count": str(n), "reasoning": []}

    tens_pattern = r'\b(' + '|'.join(re.escape(k) for k in tens.keys()) + r')(?:[\s-]+(' + '|'.join(re.escape(k) for k in units.keys()) + r'))?\b'
    tens_match = re.search(tens_pattern, answer_lower)
    if tens_match:
        tens_word = tens_match.group(1)
        unit_word = tens_match.group(2)
        value = tens.get(tens_word, 0)
        if unit_word:
            value += units.get(unit_word, 0)
        return {"count": str(value), "reasoning": []}

    tokenized = re.findall(r"[a-zA-Z]+", answer_lower)
    for tok in tokenized:
        if tok in units:
            return {"count": str(units[tok]), "reasoning": []}
        if tok in teens:
            return {"count": str(teens[tok]), "reasoning": []}
        if tok in tens:
            return {"count": str(tens[tok]), "reasoning": []}

    # fallback -> zero
    return {"count": "0", "reasoning": []}


# ----------------------------
# InternVL preprocessing helpers
# ----------------------------
IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1,1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = int(image_size * target_aspect_ratio[0])
    target_height = int(image_size * target_aspect_ratio[1])
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    cols = max(1, target_width // image_size)
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_internvl(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ------------------ robust path resolution ------------------
def resolve_image_path(image_path_str: str, data_dir: Optional[str], benchmark_json_path: Optional[str]) -> Path:
    """
    Resolve an image path from benchmark.json robustly (absolute / data_dir / benchmark parent).
    """
    img_p = Path(image_path_str)
    if img_p.is_absolute():
        return img_p
    if data_dir:
        return Path(data_dir) / image_path_str
    if benchmark_json_path:
        return Path(benchmark_json_path).parent / image_path_str
    return Path(image_path_str)

# ----------------------------
# BenchmarkTester (per-model branches)
# ----------------------------
class BenchmarkTester:
    def __init__(self, benchmark_path: str, data_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark_path = benchmark_path
        with open(benchmark_path, 'r') as f:
            self.benchmark = json.load(f)
        self.data_dir = data_dir

    def clean_answer(self, answer: str):
        return clean_answer_improved(answer)

    def _print_evaluation_summary(self, model_name, total_images, successful_images, failed_images,
                                  total_questions_processed, total_questions_failed, total_results):
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY FOR {model_name.upper()}")
        print(f"{'='*60}")
        num_successful = len(successful_images)
        num_failed = len(failed_images)
        print(f"📊 IMAGE PROCESSING SUMMARY:")
        print(f"   Total images attempted: {total_images}")
        print(f"   Successfully processed: {num_successful} ({(num_successful/total_images*100) if total_images else 0:.1f}%)")
        print(f"   Failed images: {num_failed} ({(num_failed/total_images*100) if total_images else 0:.1f}%)")
        questions_succeeded = total_questions_processed - total_questions_failed
        print(f"\n📝 QUESTION PROCESSING SUMMARY:")
        print(f"   Total questions attempted: {total_questions_processed}")
        print(f"   Successfully processed: {questions_succeeded} ({(questions_succeeded/total_questions_processed*100) if total_questions_processed else 0:.1f}%)")
        print(f"   Failed questions: {total_questions_failed} ({(total_questions_failed/total_questions_processed*100) if total_questions_processed else 0:.1f}%)")
        print(f"   Results saved: {total_results}")
        if successful_images:
            print(f"\n✅ SUCCESSFUL IMAGES ({len(successful_images)}):")
            for img in successful_images:
                print(f"   • {img['image_id']} (Type: {img['image_type']}, "
                      f"Questions: {img['questions_succeeded']}/{img['questions_total']}, "
                      f"Time: {img['processing_time']:.1f}s)")
        if failed_images:
            print(f"\n❌ FAILED/PROBLEMATIC IMAGES ({len(failed_images)}):")
            for img in failed_images:
                et = img.get('error_type', 'unknown')
                print(f"   • {img['image_id']} (Type: {img.get('image_type','unknown')}) - {et}: {img.get('error_message', img.get('success_rate',''))}")
        if failed_images:
            print(f"\n📈 FAILURE ANALYSIS BY IMAGE TYPE:")
            failures_by_type = defaultdict(list)
            for img in failed_images:
                failures_by_type[img.get('image_type','unknown')].append(img)
            for img_type, failures in failures_by_type.items():
                print(f"   • {img_type}: {len(failures)} failed images")
        print(f"{'='*60}\n")

    def evaluate_model(self, model_name: str, model, processor, save_path: str, start_idx=0, batch_size=5):
        """
        Supports model-specific branches for:
          - "qwen2.5-vl"  (Qwen2_5_VLForConditionalGeneration)
          - "blip2"       (Blip2ForConditionalGeneration)
          - "internvl3"   (InternVL3-style .chat API)
          - "gemma3"      (HF-style generate)
        For other model_names it runs a generic HF-style flow using apply_chat_template if available,
        otherwise the processor(images, text) -> model.generate path.
        """
        model_key = model_name.lower()
        results = []
        successful_images = []
        failed_images = []
        total_questions_processed = 0
        total_questions_failed = 0

        print(f"\nEvaluating {model_name}...")
        print(f"Using device: {self.device}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        images = self.benchmark['benchmark']['images'][start_idx:start_idx + batch_size]
        total_images = len(images)

        for idx, image_data in enumerate(tqdm(images, desc="Processing images")):
            image_start_time = time.time()
            cur_failed = 0
            cur_total = 0
            try:
                image_path = resolve_image_path(image_data['path'], self.data_dir, self.benchmark_path)
                if not image_path.exists():
                    failed_images.append({
                        'image_id': image_data['image_id'],
                        'image_type': image_data.get('image_type', 'unknown'),
                        'error_type': 'file_not_found',
                        'error_message': f'Image not found at {image_path}'
                    })
                    continue

                pil_image = Image.open(image_path).convert('RGB')
                image_results = []

                for question in image_data['questions']:
                    cur_total += 1
                    total_questions_processed += 1
                    try:
                        # InternVL3 special path
                        if "internvl" in model_key:
                            torch.cuda.empty_cache()
                            pixel_values = load_image_internvl(image_path, input_size=448, max_num=12)
                            pixel_values = pixel_values.to(dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
                            if torch.cuda.is_available():
                                pixel_values = pixel_values.cuda()
                            prompt = f'<image>\n {question["question"]} Your response MUST be in the following format and nothing else:\n <NUMBER> [<OBJECT1>, <OBJECT2>, <OBJECT3>, ...]'
                            generation_config = dict(
                                max_new_tokens=200,
                                do_sample=False,
                                num_beams=1,
                                early_stopping=False
                            )
                            # model.chat is expected for InternVL-style models
                            answer = model.chat(processor, pixel_values, prompt, generation_config)
                            cleaned = self.clean_answer(answer)
                            image_results.append({
                                "image_id": image_data["image_id"],
                                "image_type": image_data.get("image_type", "unknown"),
                                "question_id": question["id"],
                                "question": question["question"],
                                "ground_truth": question.get("answer"),
                                "model_answer": cleaned["count"],
                                "model_reasoning": cleaned["reasoning"],
                                "raw_answer": answer,
                                "property_category": question.get("property_category")
                            })
                            del pixel_values
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        # Gemma3 special path
                        elif "gemma" in model_key:
                            messages = [
                                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                                {"role": "user", "content": [
                                    {"type": "image", "image": pil_image},
                                    {"type": "text", "text": f"{question['question']} Your response MUST be in the following format and nothing else:\n <NUMBER> [<OBJECT1>, <OBJECT2>, <OBJECT3>, ...]"}
                                ]}
                            ]
                            torch.cuda.empty_cache()
                            inputs = processor.apply_chat_template(
                                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                            ).to(model.device)
                            with torch.no_grad():
                                input_len = inputs["input_ids"].shape[-1]
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=200,
                                    do_sample=False,
                                    num_beams=1,
                                    early_stopping=False,
                                    pad_token_id=processor.tokenizer.eos_token_id,
                                    eos_token_id=processor.tokenizer.eos_token_id
                                )
                                generated = outputs[0][input_len:]
                                answer = processor.decode(generated, skip_special_tokens=True)
                            cleaned = self.clean_answer(answer)
                            image_results.append({
                                "image_id": image_data["image_id"],
                                "image_type": image_data.get("image_type", "unknown"),
                                "question_id": question["id"],
                                "question": question["question"],
                                "ground_truth": question.get("answer"),
                                "model_answer": cleaned["count"],
                                "model_reasoning": cleaned["reasoning"],
                                "raw_answer": answer,
                                "property_category": question.get("property_category")
                            })
                            del inputs, outputs
                            torch.cuda.empty_cache()

                        # BLIP-2 special path
                        elif "blip2" in model_key:
                            prompt = f"Question: {question['question']} Answer(total number):"
                            torch.cuda.empty_cache()
                            inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=200,
                                    do_sample=False,
                                    num_beams=1,
                                    early_stopping=False
                                )
                                answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                            cleaned = self.clean_answer(answer)
                            image_results.append({
                                "image_id": image_data["image_id"],
                                "image_type": image_data.get("image_type", "unknown"),
                                "question_id": question["id"],
                                "question": question["question"],
                                "ground_truth": question.get("answer"),
                                "model_answer": cleaned["count"],
                                "model_reasoning": cleaned["reasoning"],
                                "raw_answer": answer,
                                "property_category": question.get("property_category")
                            })
                            del inputs, outputs
                            torch.cuda.empty_cache()

                        # Qwen3.5-style path (thinking-mode chat models):
                        # needs tokenize=True chat templating and enable_thinking=False,
                        # unlike the generic tokenize=False flow used below for Qwen2.5-VL.
                        elif "qwen3" in model_key:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": pil_image},
                                        {"type": "text", "text": f"{question['question']} Your response MUST be in the following format and nothing else:\n <NUMBER> [<OBJECT1>, <OBJECT2>, <OBJECT3>, ...]"}
                                    ]
                                }
                            ]
                            torch.cuda.empty_cache()
                            try:
                                inputs = processor.apply_chat_template(
                                    messages,
                                    add_generation_prompt=True,
                                    tokenize=True,
                                    return_dict=True,
                                    return_tensors="pt",
                                    enable_thinking=False,
                                ).to(model.device)
                            except TypeError:
                                # fallback if this tokenizer/chat template doesn't support enable_thinking
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
                                    max_new_tokens=200,
                                    do_sample=False,
                                    num_beams=1,
                                    pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
                                    eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
                                )
                                generated = outputs[0][input_len:]
                                answer = processor.batch_decode(
                                    generated.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False
                                )[0].strip()
                            cleaned = self.clean_answer(answer)
                            image_results.append({
                                "image_id": image_data["image_id"],
                                "image_type": image_data.get("image_type", "unknown"),
                                "question_id": question["id"],
                                "question": question["question"],
                                "ground_truth": question.get("answer"),
                                "model_answer": cleaned["count"],
                                "model_reasoning": cleaned["reasoning"],
                                "raw_answer": answer,
                                "property_category": question.get("property_category")
                            })
                            del inputs, outputs
                            torch.cuda.empty_cache()

                        # Qwen2.5-VL and generic HF flow (use apply_chat_template if available)
                        else:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": pil_image},
                                        {"type": "text", "text": f"{question['question']} Your response MUST be in the following format and nothing else:\n <NUMBER> [<OBJECT1>, <OBJECT2>, <OBJECT3>, ...]"}
                                    ]
                                }
                            ]
                            torch.cuda.empty_cache()
                            # prefer apply_chat_template if processor has it
                            if hasattr(processor, "apply_chat_template"):
                                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                                inputs = processor(text=text, images=pil_image, padding=True, return_tensors="pt").to(self.device)
                            else:
                                # fallback: direct images+text call
                                text_prompt = messages[0]["content"][1]["text"]
                                inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)

                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=200,
                                    do_sample=False,
                                    num_beams=1,
                                    early_stopping=False,
                                    pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
                                    eos_token_id=getattr(processor.tokenizer, "eos_token_id", None)
                                )
                                try:
                                    answer = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                                except Exception:
                                    try:
                                        answer = processor.decode(outputs[0], skip_special_tokens=True)
                                    except Exception:
                                        answer = ""
                            cleaned = self.clean_answer(answer)
                            image_results.append({
                                "image_id": image_data["image_id"],
                                "image_type": image_data.get("image_type", "unknown"),
                                "question_id": question["id"],
                                "question": question["question"],
                                "ground_truth": question.get("answer"),
                                "model_answer": cleaned["count"],
                                "model_reasoning": cleaned["reasoning"],
                                "raw_answer": answer,
                                "property_category": question.get("property_category")
                            })
                            del inputs, outputs
                            torch.cuda.empty_cache()

                    except Exception as e:
                        cur_failed += 1
                        total_questions_failed += 1
                        # keep running other questions
                        continue

                # append this image results
                results.extend(image_results)
                succeeded = cur_total - cur_failed
                if cur_failed == 0:
                    successful_images.append({
                        'image_id': image_data['image_id'],
                        'image_type': image_data.get('image_type', 'unknown'),
                        'questions_total': cur_total,
                        'questions_succeeded': succeeded,
                        'processing_time': time.time() - image_start_time
                    })
                else:
                    failed_images.append({
                        'image_id': image_data['image_id'],
                        'image_type': image_data.get('image_type', 'unknown'),
                        'error_type': 'partial_failure',
                        'questions_total': cur_total,
                        'questions_failed': cur_failed,
                        'questions_succeeded': succeeded,
                        'success_rate': f"{(succeeded/cur_total*100) if cur_total>0 else 0:.1f}%"
                    })

                if (idx + 1) % 2 == 0 or idx == total_images - 1:
                    checkpoint_path = f"{save_path}_checkpoint.json"
                    with open(checkpoint_path, 'w') as f:
                        json.dump(results, f, indent=4)

            except Exception as e:
                failed_images.append({
                    'image_id': image_data['image_id'],
                    'image_type': image_data.get('image_type', 'unknown'),
                    'error_type': 'complete_failure',
                    'error_message': str(e)
                })
                continue

        # final save
        if results:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)

        # print summary
        self._print_evaluation_summary(model_name, total_images, successful_images, failed_images,
                                       total_questions_processed, total_questions_failed, len(results))
        return results


# ----------------------------
# Metrics utilities
# ----------------------------
def _safe_int(x):
    if x is None:
        return None
    try:
        return int(str(x).strip())
    except Exception:
        try:
            return int(round(float(str(x).strip())))
        except Exception:
            return None

def calculate_accuracy(predictions: List[Any], ground_truths: List[Any]) -> float:
    preds = [_safe_int(p) for p in predictions]
    truths = [_safe_int(g) for g in ground_truths]
    pairs = [(p, t) for p, t in zip(preds, truths) if t is not None]
    if not pairs:
        return 0.0
    correct = sum(1 for p, t in pairs if p == t)
    return correct / len(pairs)

def calculate_rmse(predictions: List[Any], ground_truths: List[Any]) -> float:
    preds = [_safe_int(p) for p in predictions]
    truths = [_safe_int(g) for g in ground_truths]
    pairs = [(p, t) for p, t in zip(preds, truths) if p is not None and t is not None]
    if not pairs:
        return float("nan")
    mse = sum((p - t) ** 2 for p, t in pairs) / len(pairs)
    return mse ** 0.5

def calculate_off_by_n_accuracy(predictions: List[Any], ground_truths: List[Any], n: int = 1) -> Dict[str, float]:
    preds = [_safe_int(p) for p in predictions]
    truths = [_safe_int(g) for g in ground_truths]
    pairs = [(p, t) for p, t in zip(preds, truths) if p is not None and t is not None]
    total = len(pairs)
    if total == 0:
        return {f"off_by_{k}": 0.0 for k in range(n+1)} | {f"exactly_{k}": 0.0 for k in range(n+1)}
    diffs = [abs(p - t) for p, t in pairs]
    results = {}
    for k in range(n+1):
        results[f"off_by_{k}"] = sum(1 for d in diffs if d <= k) / total
    for k in range(n+1):
        results[f"exactly_{k}"] = sum(1 for d in diffs if d == k) / total
    return results

def mean_error(predictions: List[Any], ground_truths: List[Any]) -> float:
    preds = [_safe_int(p) for p in predictions]
    truths = [_safe_int(g) for g in ground_truths]
    pairs = [(p, t) for p, t in zip(preds, truths) if p is not None and t is not None]
    if not pairs:
        return float("nan")
    errors = [(p - t) for p, t in pairs]
    return sum(errors) / len(errors)

def load_results_to_dataframe(results_path: str) -> pd.DataFrame:
    with open(results_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Normalize column names
    if 'model_answer' not in df.columns and 'model_prediction' in df.columns:
        df = df.rename(columns={'model_prediction': 'model_answer'})
    df['pred_count'] = df.get('model_answer').apply(_safe_int)
    df['gt_count'] = df.get('ground_truth').apply(_safe_int)
    return df

def compute_metrics_from_results(results_path: str, off_by_n: int = 1, by_category: bool = True) -> Dict[str, Any]:
    df = load_results_to_dataframe(results_path)
    df_valid = df.dropna(subset=['gt_count'])
    if df_valid.empty:
        return {'overall': None, 'by_category': {}}
    overall_preds = df_valid['pred_count'].tolist()
    overall_gts = df_valid['gt_count'].tolist()
    overall = {
        'accuracy': calculate_accuracy(overall_preds, overall_gts),
        'rmse': calculate_rmse(overall_preds, overall_gts),
        'mean_error': mean_error(overall_preds, overall_gts),
    }
    overall.update(calculate_off_by_n_accuracy(overall_preds, overall_gts, n=off_by_n))
    result = {'overall': overall}
    if by_category and 'property_category' in df_valid.columns:
        by_cat = {}
        grouped = df_valid.groupby('property_category')
        for cat, group in grouped:
            preds = group['pred_count'].tolist()
            gts = group['gt_count'].tolist()
            cat_metrics = {
                'accuracy': calculate_accuracy(preds, gts),
                'rmse': calculate_rmse(preds, gts),
                'mean_error': mean_error(preds, gts)
            }
            cat_metrics.update(calculate_off_by_n_accuracy(preds, gts, n=off_by_n))
            cat_metrics['sample_size'] = len(gts)
            by_cat[cat] = cat_metrics
        result['by_category'] = by_cat
    return result
