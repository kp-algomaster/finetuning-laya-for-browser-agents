#!/usr/bin/env python3
"""
Master Orchestration Script for Local Mac Training & Evaluation Pipeline.
End-to-end execution on Apple Silicon Metal (MPS) or CPU.

Usage:
    python3 scripts/run_local_pipeline.py [--epochs 2] [--micro-batch 4]
"""

import os
import sys
import subprocess
import time
import argparse
import platform
import torch

MODEL_CACHE_DIR = "/Users/kp/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"
DATA_DIR = "data"
OUTPUT_DIR = "models/laya-browser-agent"


def run_cmd(cmd, description):
    print(f"\n>>> [{description}] Running: {cmd}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        print(f"Error: Step '{description}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f">>> [{description}] Completed in {time.time() - t0:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Run complete local Mac Laya training and evaluation pipeline")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2 for fast local iteration)")
    parser.add_argument("--micro-batch", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--model-dir", type=str, default=MODEL_CACHE_DIR, help="Base Laya model directory")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directory for training items")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for fine-tuned weights")
    args = parser.parse_args()

    print("=" * 65)
    print("   LAYA BROWSER AGENT: LOCAL MAC TRAINING PIPELINE")
    print(f"   Architecture    : {platform.machine()} ({platform.platform()})")
    print(f"   Apple Silicon   : {'MPS Active' if torch.backends.mps.is_available() else 'CPU Fallback'}")
    print(f"   PyTorch Version : {torch.__version__}")
    print("=" * 65)

    # Stage 1: Data Preparation & Preprocessing
    step1_cmd = f"python3 scripts/generate_mac_training_data.py {args.data_dir}"
    run_cmd(step1_cmd, "Stage 1: Generate & Tokenize Browser Decision Trajectories")

    # Stage 2: Model Training on Mac (MPS)
    train_items = os.path.join(args.data_dir, "browser_train_items.pt")
    calib_items = os.path.join(args.data_dir, "browser_calib_items.pt")
    step2_cmd = (
        f"python3 scripts/train_laya_mac.py "
        f"--model-dir {args.model_dir} "
        f"--train-items {train_items} "
        f"--calib-items {calib_items} "
        f"--output-dir {args.output_dir} "
        f"--epochs {args.epochs} "
        f"--micro-batch {args.micro_batch} "
        f"--grad-accum {args.grad_accum}"
    )
    run_cmd(step2_cmd, "Stage 2: Train Model Locally on Apple Silicon MPS")

    # Stage 3: Benchmark & Evaluation Metrics
    test_file = os.path.join(args.data_dir, "browser_test_cases.jsonl")
    step3_cmd = (
        f"python3 scripts/eval_laya_mac.py "
        f"--model-dir {args.output_dir} "
        f"--test-file {test_file}"
    )
    run_cmd(step3_cmd, "Stage 3: Compute Model Training Evaluation Metrics")

    print("\n" + "=" * 65)
    print("  PIPELINE EXECUTION COMPLETE!")
    print(f"  Fine-Tuned Model Weights : {args.output_dir}/model.safetensors")
    print(f"  Calibrated Config        : {args.output_dir}/rl_agent_config.json")
    print(f"  Evaluation Metrics JSON  : {args.output_dir}/eval_metrics.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
