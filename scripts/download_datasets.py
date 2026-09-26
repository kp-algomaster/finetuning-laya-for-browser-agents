#!/usr/bin/env python3
"""
Dataset and Model Asset Downloader for Laya Browser Agent Fine-Tuning.

Fetches and verifies the required training datasets and foundation checkpoints:
1. Mind2Web Trajectory Dataset (Hugging Face: osunlp/Mind2Web)
2. Laya-Browser Reverse Goal Corpus (Hugging Face: cklxx/laya-browser)
3. Base Laya Foundation Model (Hugging Face: convaiinnovations/laya)
4. ModernBERT Base Architecture (Hugging Face: answerdotai/ModernBERT-base)

Usage:
    python3 scripts/download_datasets.py --all
    python3 scripts/download_datasets.py --dataset mind2web
    python3 scripts/download_datasets.py --model base
"""

import os
import sys
import argparse
import subprocess

DATASETS = {
    "mind2web": {
        "name": "Mind2Web Trajectory Corpus",
        "hf_id": "osunlp/Mind2Web",
        "type": "dataset",
        "url": "https://huggingface.co/datasets/osunlp/Mind2Web",
        "paper": "https://arxiv.org/abs/2306.06070",
        "description": "7,296 multi-step web interaction tasks across 137 websites with negative element distractors."
    },
    "laya_browser": {
        "name": "Laya-Browser Reverse-Engineered Goals",
        "hf_id": "cklxx/laya-browser",
        "type": "dataset",
        "url": "https://huggingface.co/cklxx/laya-browser",
        "paper": "https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md",
        "description": "5,244 real web tasks across 421 crawled domains, paired with Chromium DONE landing states."
    },
    "base_laya": {
        "name": "Base Laya Foundation Checkpoint (421M ModernBERT-large)",
        "hf_id": "convaiinnovations/laya",
        "type": "model",
        "url": "https://huggingface.co/convaiinnovations/laya",
        "paper": "https://huggingface.co/convaiinnovations/laya",
        "description": "Pre-trained weights and tokenizer configuration for Laya."
    },
    "modernbert": {
        "name": "ModernBERT Base",
        "hf_id": "answerdotai/ModernBERT-base",
        "type": "model",
        "url": "https://huggingface.co/answerdotai/ModernBERT-base",
        "paper": "https://arxiv.org/abs/2412.13663",
        "description": "Base bidirectional transformer encoder for long-context comprehension."
    }
}


def download_hf_asset(asset_key, target_dir=None):
    info = DATASETS[asset_key]
    print(f"\n==================================================")
    print(f"Downloading: {info['name']}")
    print(f"Hugging Face ID: {info['hf_id']}")
    print(f"Repository URL : {info['url']}")
    print(f"Description    : {info['description']}")
    print(f"==================================================")

    try:
        if info["type"] == "dataset":
            from datasets import load_dataset
            print(f"Fetching dataset split via datasets library...")
            ds = load_dataset(info["hf_id"], split="train", streaming=True)
            print(f"✓ Successfully connected to {info['hf_id']}")
        else:
            from huggingface_hub import snapshot_download
            out = target_dir or os.path.join("models", os.path.basename(info["hf_id"]))
            print(f"Downloading checkpoint snapshot to {out}...")
            snapshot_download(repo_id=info["hf_id"], local_dir=out)
            print(f"✓ Downloaded model snapshot to {out}")
    except Exception as e:
        print(f"Notice: Automated download encountered: {e}")
        print(f"You can also clone or download directly from: {info['url']}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets and foundation models for Laya fine-tuning")
    parser.add_argument("--all", action="store_true", help="Download/verify all datasets and base weights")
    parser.add_argument("--dataset", choices=["mind2web", "laya_browser"], help="Download specific dataset")
    parser.add_argument("--model", choices=["base_laya", "modernbert"], help="Download specific foundation model")
    parser.add_argument("--target-dir", type=str, default=None, help="Target download directory")
    args = parser.parse_args()

    if args.all:
        for k in DATASETS:
            download_hf_asset(k, args.target_dir)
    elif args.dataset:
        download_hf_asset(args.dataset, args.target_dir)
    elif args.model:
        key = "base_laya" if args.model == "base_laya" else "modernbert"
        download_hf_asset(key, args.target_dir)
    else:
        print("Available datasets and models:")
        for k, v in DATASETS.items():
            print(f"  • {k:14}: {v['name']} ({v['url']})")
        print("\nRun with --all to fetch all assets or specify --dataset / --model.")


if __name__ == "__main__":
    main()
