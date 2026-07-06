<h1 align="center"> OPTICS: <ins>O</ins>bject <ins>P</ins>roperty Reasoning <ins>T</ins>asks for Evaluating <ins>I</ins>mage-based <ins>C</ins>ommon<ins>S</ins>ense</h1>

OPTICS is a comprehensive benchmark suite for evaluating Vision-Language Models (VLMs) on object property reasoning and image-based common sense. It is divided into two benchmarks:

- **OPTICS-CNT** — single-image counting tasks where models predict object counts from property-based questions (`benchmark.json`).
- **OPTICS-CMP** — side-by-side comparison tasks where models decide which of two merged images has a greater count (`benchmark_merged.json`).

Both benchmarks span four object property dimensions (physical, taxonomic, functional, relational), three reasoning complexity levels (direct recognition, property inference, counterfactual), and three visual domains (photographic/real, animated, AI-generated).

<!-- 📄 [Paper](https://arxiv.org/pdf/2508.10956) |  -->

📚 [arXiv](https://arxiv.org/abs/2508.10956) | 
🤗 [Dataset](https://huggingface.co/datasets/Abk802/ORBIT)

<!-- 💻 [Code](https://github.com/AbhishekKolari/ORBIT.git) | -->

<!-- 📓 [Notebooks](https://colab.research.google.com/github/yourname/yourrepo/blob/main/notebook.ipynb) |  -->

<!-- 🌐 [Website](https://yourprojectwebsite.com) -->

<h1 align="center"><img width="584" height="400" alt="orbit-tax-500-1" src="https://github.com/user-attachments/assets/52bd4e19-ca8f-45ab-aa44-0992726c3897" /></h1>

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AbhishekKolari/OPTICS.git
   cd OPTICS
   ```
2. **Install the Framework**:

   Install Anaconda or Miniconda distribution based on Python3+ from their downloads' site once.

   ```bash
   # create the environment
   conda create -n [env_name] python=3.12

   # activate
   conda activate [env_name]

   # install dependencies
   pip install -r requirements.txt
   ```

## Usage

OPTICS provides scripts to reproduce and evaluate multiple open-source and closed-source models on both benchmarks:

- **`main.py`** — Single CLI entrypoint for both benchmarks. Use `--benchmark cnt` for OPTICS-CNT (counting) or `--benchmark cmp` for OPTICS-CMP (comparison). Dispatches to the appropriate runner based on benchmark and mode.
- **`opensource.py`** — Loads and evaluates open-source HuggingFace (HF) models on **OPTICS-CNT**. Contains hardcoded models (different size variants) used for evaluation as in the paper (BLIP2, Qwen2.5-VL, Qwen3.5, InternVL3, and Gemma 3) and falls back to generic HF loading for arbitrary repo IDs. Gemma 3 is a gated model on HF and requires an HF token. Accepts full HF paths (`opengvlab/internvl3-8b`) or local model directories.
- **`closedsource.py`** — Routes multimodal requests to closed-source providers (OpenAI GPT/o-series, Anthropic Claude, Google Gemini) on **OPTICS-CNT**. Accepts short aliases (`gpt`, `claude`, `gemini`) or full provider model ids (`gpt-4o-mini-2024-07-18`, `claude-3-7-sonnet-20250219`, `gemini-2.5-flash`, `o3`). Reads API keys from env vars.
- **`run_merged_benchmark.py`** — Unified runner for **OPTICS-CMP**. Supports both open-source and closed-source models in a single script (auto-detects backend from `--model_name`, or force with `--backend`). Can also be invoked via `main.py --benchmark cmp`. Optionally computes per-answer uncertainty scores via `--compute_uncertainty` (see `uncertainty.py` below). Results are saved in a model-agnostic JSON format.
- **`uncertainty.py`** — Standalone uncertainty-estimation utilities used by `run_merged_benchmark.py` on **OPTICS-CMP**. For open-source models, computes MSP, perplexity, and mean token entropy directly from generation scores (`generate_with_uncertainty()`); InternVL and BLIP-2 are not supported since they don't expose step-wise scores. For closed-source OpenAI models, estimates uncertainty (MSP, perplexity, mean token entropy, or kernel language entropy) via the optional `lm-polygraph` dependency (`estimate_closedsource_uncertainty()`).
- **`utils.py`** — Contains main logic behind evaluating models in `evaluate_model()` under the class `BenchmarkTester` (OPTICS-CNT), InternVL preprocessing helpers, path resolution in `resolve_image_path()`, and metric helpers (accuracy, off-by-n, RMSE, mean error) in `compute_metrics_from_results()`.
- **`question_generator.py`** — Generate MLLM-based question triples on existing images from `data/` or on a new set of images (organized similarly to `data/`) and write them into a benchmark-style JSON (same top-level structure as OPTICS-CNT's `benchmark.json`). Uses closed-source providers (OpenAI GPT-4o, Anthropic Claude, Google Gemini). Accepts short aliases (`gpt`, `claude`, `gemini`). Supports `--test_mode` for quick checks and `--max_images` (default 3 in test mode).

### Benchmark files

| Benchmark  | JSON file                 | Images             | Task                                                   |
| ---------- | ------------------------- | ------------------ | ------------------------------------------------------ |
| OPTICS-CNT | `benchmark.json`        | `data/`          | Count objects in a single image                        |
| OPTICS-CMP | `benchmark_merged.json` | `merged_images/` | Compare counts across two side-by-side images (A vs B) |

## Environment Variables

Create a `.env` file to store the keys and token in the same manner as given in the template:

```bash
   # API keys
   OPENAI_API_KEY="sk-..."
   ANTHROPIC_API_KEY="claude-..."
   GOOGLE_API_KEY="ya29..."

   # Gemma 3
   HF_TOKEN="hf_..."

   # default model ids (used when --model is an alias)
   OPENAI_MODEL=gpt-4o-mini-2024-07-18
   ANTHROPIC_MODEL=claude-3-7-sonnet-20250219
   GOOGLE_MODEL=gemini-2.5-flash
```

For closed-source models, also include default model versions used as `*_MODEL`.

## Execution

### OPTICS-CNT (counting benchmark)

1. **Open-source models**:

   ```bash
   python main.py \
     --benchmark cnt \
     --mode opensource \
     --model_name qwen/qwen2.5-vl-7b-instruct \
     --processor_path qwen/qwen2.5-vl-7b-instruct \
     --benchmark_json ./benchmark.json \
     --data_dir . \
     --output_file qwen_cnt_results.json \
     --start_idx 0 \
     --batch_size 5
   ```
2. **Closed-source models**:

   ```bash
   python main.py \
     --benchmark cnt \
     --mode closedsource \
     --model_name gpt \
     --benchmark_json ./benchmark.json \
     --data_dir . \
     --output_file gpt_cnt_results.json \
     --start_idx 0 \
     --batch_size 10
   ```

By default, `start_idx` and `batch_size` are set to 0 and 360 (total number of images in OPTICS-CNT) respectively. Use these arguments to evaluate on a smaller subset of images.

### OPTICS-CMP (comparison benchmark)

1. **Open-source models** (via `main.py`):

   ```bash
   python main.py \
     --benchmark cmp \
     --mode opensource \
     --model_name opengvlab/internvl3-8b \
     --questions_json ./benchmark_merged.json \
     --images_dir ./merged_images \
     --output_file internvl3_cmp_results.json \
     --start_idx 0 \
     --end_idx 100
   ```
2. **Closed-source models** (via `main.py`):

   ```bash
   python main.py \
     --benchmark cmp \
     --mode closedsource \
     --model_name gpt \
     --questions_json ./benchmark_merged.json \
     --images_dir ./merged_images \
     --output_file gpt_cmp_results.json
   ```
3. **Standalone runner** (`run_merged_benchmark.py` can also be used directly):

   ```bash
   python run_merged_benchmark.py \
     --model_name qwen/qwen2.5-vl-7b-instruct \
     --questions_json ./benchmark_merged.json \
     --images_dir ./merged_images \
     --output_file qwen_cmp_results.json
   ```
4. **With uncertainty estimation** (OPTICS-CMP only, see `uncertainty.py`):

   ```bash
   python run_merged_benchmark.py \
     --model_name qwen/qwen2.5-vl-7b-instruct \
     --questions_json ./benchmark_merged.json \
     --images_dir ./merged_images \
     --output_file qwen_cmp_results.json \
     --compute_uncertainty \
     --uncertainty_method msp
   ```

   Open-source models always report MSP, perplexity, and mean token entropy together when `--compute_uncertainty` is set (InternVL and BLIP-2 are unsupported). `--uncertainty_method` selects the estimator for closed-source OpenAI models (`msp`, `perplexity`, `mean_token_entropy`, or `kernel_language_entropy`), and requires the optional `lm-polygraph` dependency (see the commented-out line in `requirements.txt`).

For OPTICS-CMP, `--mode` is optional when using `main.py` — the backend is auto-detected from `--model_name`. Use `--end_idx` to limit the number of comparison questions processed. Add `--include_metadata` to include optional dataset fields (bucket, similarity, count diff) in result records.

### Results

**OPTICS-CNT** — compute accuracy, RMSE, mean error, and off-by-n metrics after a run:

```bash
python main.py \
  --benchmark cnt \
  --mode closedsource \
  --model_name gpt \
  --benchmark_json benchmark.json \
  --output_file gpt_cnt_results.json \
  --analyze \
  --off_by_n 2
```

Include `--analyze` and optionally `--off_by_n` (default tolerance 1) to display metrics similar to the tables shown in the paper. Analysis is only supported for OPTICS-CNT.

**OPTICS-CMP** — prints an inline accuracy summary at the end of each run (overall accuracy and breakdown by count difference). Result records use a portable schema:

```json
{
  "sample_index": 0,
  "image_path": "./merged_images/merged_image_0.jpg",
  "question": "...",
  "ground_truth": "B",
  "model_answer": "B",
  "raw_answer": "B",
  "correct": true,
  "skipped": false
}
```

### OPTICS MLLM-based question generation

To generate questions on OPTICS-CNT images or a new set of images, run:

```bash
python question_generator.py \
  --model gpt \
  --data_dir ./data/ANIMATED \
  --image_type ANIMATED \
  --output_json generated.json \
  --test_mode \
  --max_images 2
```

By default, `--image_type` is set to `REAL`. The questions in the output JSON file `generated.json` can be manually refined and then evaluated using the commands above with `--benchmark cnt` and `--mode` set to either `opensource` or `closedsource`.

## Citation (BibTeX)

If you found OPTICS useful, please cite us:

```bash
@misc{kolari2025orbitobjectpropertyreasoning,
      title={OPTICS: Object Property Reasoning Tasks for Evaluating Image-based Common Sense}, 
      author={Abhishek Kolari and Mohammadhossein Khojasteh and Yifan Jiang and Floris den Hengst and Filip Ilievski},
      year={2025},
      eprint={2508.10956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.10956}, 
}
```

<!-- ## Project Structure

- **benchmark.json**: The OPTICS-CNT benchmark dataset containing annotated questions and ground truth answers for real, animated, and AI-generated images.
- **benchmark_merged.json**: The OPTICS-CMP benchmark dataset containing side-by-side comparison questions and ground truth A/B answers.
- **merged_images/**: Side-by-side merged images used by OPTICS-CMP.
- **merged_data/**: Contains subfolders for different image types (`REAL/`, `ANIMATED/`, `AI_GENERATED/`) used in the benchmark.
- **download_models.py**: Script to download and cache all required vision-language models from HuggingFace or other sources. This ensures reproducibility and consistent model versions across experiments.
- **shell_scripts/run_models.sh**: Example shell script to run model inference in batch mode.
- **OPTICS_results/**: Store the output JSON files from model runs. Each file contains the model's answers and reasoning for all benchmark questions.
- **analyse_results.ipynb**: The main analysis notebook. Loads model outputs, computes evaluation metrics (accuracy, off-by-N, MAE, RMSE), and generates plots for thesis figures. This notebook is central to the quantitative and qualitative analysis in the thesis.
- **OPTICS_analysis_plots/** and **OPTICS_model_plots/**: Contain figures generated from the analysis notebook, including accuracy curves, error distributions, and model comparison plots. These are directly used in the thesis to illustrate findings.
- **OPTICS_notebooks/opa-benchmark-<model-names>.ipynb**: Contains wrappers and utility functions for running open-source models on the benchmark.
<!-- - **pdf2bench.py**: Utility for converting PDF-based datasets into the benchmark format. -->

<!-- - **create_notebook.py**: Script to auto-generate Jupyter notebooks for new experiments or model evaluations. -->

<!-- ## How This Supports the Thesis Experiments

1. **Benchmark Construction**: The `benchmark.json` and `merged_data/` directories define the experimental setup, ensuring a diverse and challenging set of counting and reasoning tasks.
2. **Model Evaluation**: `download_models.py` and `opa-benchmark-<model-names>.ipynb` allow for systematic downloading, setup, and inference with a wide range of vision-language models, as required for the thesis comparison.
3. **Result Storage**: All model outputs are saved in a standardized format in `OPTICS_results/`, enabling fair and reproducible evaluation.
4. **Analysis & Visualization**: `analyse_results.ipynb` computes all key metrics reported in the thesis (accuracy, off-by-N, MAE, RMSE, error clustering, etc.) and produces publication-ready plots found in `OPTICS_analysis_plots/` and `OPTICS_model_plots/`.
5. **Reproducibility**: Scripts and notebooks are organized to allow any researcher to reproduce the thesis experiments from model download to final analysis. -->

<!-- ## Getting Started

1. **Setup and Dependencies**:  
   Install Anaconda or Miniconda distribution based on Python3+ from their downloads' site.
   ```bash 
   conda create -n [env_name] python=3.12
   ```
   Activate it and install all necessary libraries:   -->

<!-- ```bash 
   pip install -r requirements.txt
   ```
   Create ipykernel for the use of Jupyter Notebooks:
   ```bash
   python -m ipykernel install --user --name [env_name] --display-name "[any_name]"
   ```

2. **Download models**:
   ```bash
   python download_models.py
   ```

3. **Tweak model parameters and dataset batches** in `opa-benchmark-<model-names>.ipynb`

4. **Run inference via SLURM** (change file paths accordingly):
   ```bash
   sbatch run_models.sh
   ```

5. **Analyze results**:  
   Tweak `analyse_results.ipynb` and run  
   ```bash
   sbatch analyse.sh
   -->
