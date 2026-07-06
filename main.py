#!/usr/bin/env python3
import argparse
import json

from closedsource import run_closedsource
from opensource import run_opensource
from run_merged_benchmark import process_merged_benchmark
from utils import compute_metrics_from_results


def main():
    parser = argparse.ArgumentParser(
        description="OPTICS benchmark runner (OPTICS-CNT and OPTICS-CMP)"
    )
    parser.add_argument(
        "--benchmark",
        choices=["cnt", "cmp"],
        default="cnt",
        help='Benchmark to run: "cnt" (OPTICS-CNT, counting) or "cmp" (OPTICS-CMP, comparison)',
    )
    parser.add_argument(
        "--mode",
        choices=["opensource", "closedsource"],
        default=None,
        help='Required for OPTICS-CNT. Optional for OPTICS-CMP (auto-detects if omitted).',
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or HF repo id (opensource) or API model id (closedsource)",
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Processor path or HF repo id (opensource). If omitted, model_name is used.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for batch processing",
    )

    # OPTICS-CNT (counting benchmark)
    parser.add_argument(
        "--benchmark_json",
        type=str,
        default="benchmark.json",
        help="Path to OPTICS-CNT benchmark JSON (used when --benchmark cnt)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Optional root for relative image paths in OPTICS-CNT",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=360,
        help="Number of images to process for OPTICS-CNT (default=360)",
    )

    # OPTICS-CMP (comparison benchmark)
    parser.add_argument(
        "--questions_json",
        type=str,
        default="benchmark_merged.json",
        help="Path to OPTICS-CMP questions JSON (used when --benchmark cmp)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./merged_images",
        help="Directory containing comparison images for OPTICS-CMP",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for OPTICS-CMP (exclusive; default: all questions)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate for open-source models on OPTICS-CMP",
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Include optional dataset metadata in OPTICS-CMP result records",
    )
    parser.add_argument(
        "--compute_uncertainty",
        action="store_true",
        help="Compute uncertainty scores on OPTICS-CMP (see uncertainty.py)",
    )
    parser.add_argument(
        "--uncertainty_method",
        type=str,
        default="msp",
        choices=["msp", "perplexity", "mean_token_entropy", "kernel_language_entropy"],
        help="Closed-source OpenAI uncertainty estimator for OPTICS-CMP (default: msp)",
    )

    # Analysis (OPTICS-CNT only)
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Compute analysis metrics on saved OPTICS-CNT results after run",
    )
    parser.add_argument(
        "--off_by_n",
        type=int,
        default=1,
        help="Tolerance for off-by-n analysis (OPTICS-CNT only)",
    )

    args = parser.parse_args()

    if args.benchmark == "cnt":
        if args.mode is None:
            parser.error('--mode is required when --benchmark cnt')

        if args.mode == "opensource":
            run_opensource(
                model_name=args.model_name,
                processor_path=(args.processor_path or args.model_name),
                benchmark_json=args.benchmark_json,
                data_dir=args.data_dir,
                output_file=args.output_file,
                start_idx=args.start_idx,
                batch_size=args.batch_size,
            )
        else:
            run_closedsource(
                model_name=args.model_name,
                benchmark_json=args.benchmark_json,
                data_dir=args.data_dir,
                output_file=args.output_file,
                start_idx=args.start_idx,
                batch_size=args.batch_size,
            )

        if args.analyze:
            print("\nRunning analysis on results...")
            metrics = compute_metrics_from_results(
                args.output_file, off_by_n=args.off_by_n, by_category=True
            )
            print(json.dumps(metrics, indent=2))

    else:
        if args.analyze:
            parser.error("--analyze is only supported for OPTICS-CNT (--benchmark cnt)")

        backend = args.mode if args.mode else "auto"
        process_merged_benchmark(
            model_name=args.model_name,
            questions_json=args.questions_json,
            output_file=args.output_file,
            images_dir=args.images_dir,
            processor_path=args.processor_path,
            backend=backend,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            max_new_tokens=args.max_new_tokens,
            include_metadata=args.include_metadata,
            compute_uncertainty=args.compute_uncertainty,
            uncertainty_method=args.uncertainty_method,
        )


if __name__ == "__main__":
    main()
