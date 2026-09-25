#!/usr/bin/env python3
"""
Benchmark Evaluation Script for Laya Browser Agent.
Computes official metrics against held-out browser tasks and generates a head-to-head
comparison against TypeSafe Jev and ModernBERT-base.

Usage:
    python3 scripts/eval_laya_browser.py \
        --model-dir models/laya-browser-agent \
        --test-data data/browser_test.jsonl \
        --device cuda
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import laya
from laya.common import ece_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Laya browser agent")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing fine-tuned model")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test split JSONL")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    args = parser.parse_args()

    print(f"Loading fine-tuned Laya model from {args.model_dir} on {args.device}...")
    agent = laya.Agent(args.model_dir, device=args.device)

    items = []
    with open(args.test_data, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))

    print(f"Loaded {len(items)} test benchmark cases. Running inference...")

    latencies_ms = []
    predictions = []

    for idx, item in enumerate(items):
        state = item["state"] if isinstance(item["state"], dict) else json.loads(item["state"])
        questions = item["questions"] if isinstance(item["questions"], dict) else json.loads(item["questions"])
        gold = item["gold"] if isinstance(item["gold"], dict) else json.loads(item["gold"])

        t0 = time.perf_counter()
        pred = agent.predict(state, questions)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

        predictions.append({"pred": pred, "gold": gold, "questions": questions})

        if (idx + 1) % 50 == 0:
            print(f"  Evaluated {idx + 1}/{len(items)} cases (mean latency: {np.mean(latencies_ms):.1f} ms)...")

    # Metric computation
    accuracies = []
    soft_accuracies = []
    brier_scores = []
    kl_divs = []
    tv_distances = []
    all_confs = []
    all_corrects = []

    for item in predictions:
        pred_answers = item["pred"]
        gold_answers = item["gold"]
        questions = item["questions"]

        for qid, qdef in questions.items():
            if qid not in pred_answers or qid not in gold_answers:
                continue
            p_ans = pred_answers[qid]
            g_ans = gold_answers[qid]
            q_type = qdef["type"]

            if q_type == "choice":
                keys = list(qdef["criteria"].keys())
                pred_choice = str(p_ans.get("choice", ""))
                gold_label = str(g_ans.get("label", g_ans.get("choice", "")))

                is_corr = float(pred_choice == gold_label)
                accuracies.append(is_corr)
                all_corrects.append(is_corr)

                p_probs = np.array([p_ans.get("probabilities", {}).get(k, 1e-6) for k in keys])
                g_probs = np.array([g_ans.get("probabilities", {}).get(k, 1e-6) for k in keys])
                p_probs = p_probs / max(1e-12, p_probs.sum())
                g_probs = g_probs / max(1e-12, g_probs.sum())

                all_confs.append(float(p_probs.max()))
                soft_accuracies.append(float((p_probs * g_probs).sum()))
                brier_scores.append(float(((p_probs - g_probs) ** 2).sum()))
                tv_distances.append(float(0.5 * np.abs(p_probs - g_probs).sum()))
                kl_divs.append(float((g_probs * np.log(np.clip(g_probs / p_probs, 1e-12, 1e4))).sum()))

            elif q_type == "noul":
                p_val = p_ans.get("noul", p_ans.get("probabilities", {}).get("true", 0.5))
                g_val = g_ans.get("noul", g_ans.get("probabilities", {}).get("true", 0.5))
                gold_label = str(g_ans.get("label", "true" if g_val >= 0.5 else "false")).lower()

                pred_label = "true" if p_val >= 0.5 else "false"
                is_corr = float(pred_label == gold_label)
                accuracies.append(is_corr)
                all_corrects.append(is_corr)
                all_confs.append(float(max(p_val, 1.0 - p_val)))

                p_dist = np.array([1.0 - p_val, p_val])
                g_dist = np.array([1.0 - g_val, g_val])

                soft_accuracies.append(float((p_dist * g_dist).sum()))
                brier_scores.append(float(((p_dist - g_dist) ** 2).sum()))
                tv_distances.append(float(0.5 * np.abs(p_dist - g_dist).sum()))
                kl_divs.append(float((g_dist * np.log(np.clip(g_dist / p_dist, 1e-12, 1e4))).sum()))

    laya_acc = float(np.mean(accuracies))
    laya_soft_acc = float(np.mean(soft_accuracies))
    laya_brier = float(np.mean(brier_scores))
    laya_kl = float(np.mean(kl_divs))
    laya_tv = float(np.mean(tv_distances))
    laya_ece = float(ece_score(np.array(all_confs), np.array(all_corrects)))
    laya_latency = float(np.percentile(latencies_ms, 50))

    comparison_data = [
        {
            "Model": "TypeSafe Jev 1.13.0",
            "Kind": "Cloud API (Proprietary)",
            "Accuracy": 0.727,
            "Soft Acc": 0.580,
            "Brier": 0.148,
            "ECE": 0.144,
            "ms/case": 710,
            "Cost/Case": "$0.0004",
        },
        {
            "Model": "Laya Browser (Fine-Tuned)",
            "Kind": "Self-Hosted / Local",
            "Accuracy": round(laya_acc, 3),
            "Soft Acc": round(laya_soft_acc, 3),
            "Brier": round(laya_brier, 3),
            "ECE": round(laya_ece, 3),
            "ms/case": round(laya_latency, 1),
            "Cost/Case": "$0.00",
        },
        {
            "Model": "ModernBERT-base (149M)",
            "Kind": "Specialist Local",
            "Accuracy": 0.646,
            "Soft Acc": 0.542,
            "Brier": 0.119,
            "ECE": 0.179,
            "ms/case": 349,
            "Cost/Case": "$0.00",
        },
    ]

    df_comp = pd.DataFrame(comparison_data)
    print("\n=================== BENCHMARK REPORT ===================")
    print(df_comp.to_markdown(index=False))
    print("========================================================\n")

    report_path = os.path.join(args.model_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "n_cases": len(items),
                "n_decisions": len(accuracies),
                "accuracy": laya_acc,
                "soft_accuracy": laya_soft_acc,
                "brier_score": laya_brier,
                "ece": laya_ece,
                "tv_distance": laya_tv,
                "kl_div": laya_kl,
                "median_latency_ms": laya_latency,
            },
            f,
            indent=2,
        )
    print(f"Saved benchmark report to {report_path}")


if __name__ == "__main__":
    main()
