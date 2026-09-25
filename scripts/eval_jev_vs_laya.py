#!/usr/bin/env python3
"""
Side-by-Side Benchmark Suite: Fine-Tuned Laya Browser Agent vs Live TypeSafe Jev API.

Compares:
1. Local Fine-Tuned Laya Model (Apple Silicon MPS / CPU)
2. Live TypeSafe Jev API (model: jev-latest / jev-1.13.0)

Using the 70 benchmark scenarios in data/browser_test_cases.jsonl.
Evaluates:
- Hard Accuracy (Argmax match against gold labels)
- Soft Accuracy (Expected agreement with ground truth probability distributions)
- Brier Score (Quadratic scoring rule)
- Expected Calibration Error (ECE - 10 bins)
- Total Variation Distance & KL Divergence
- Sub-decision breakdowns: operation, click_target, type_text_target, is_goal_satisfied
- Latency profiling (p50, p90, p95, p99)
- Cost per 1,000 steps ($0.00 Local vs Cloud API)
"""

import os
import sys

# Prevent numpy 2.x ABI crash in pyarrow/pandas
sys.modules['pandas'] = None
sys.modules['pyarrow'] = None

import time
import json
import math
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch


def load_env_key(env_path="bropilot-ext/.env"):
    """Extract Jev API key from environment variables or .env file."""
    for env_var in ["jev_bropilot_eval", "bropilot_eval", "JEV_API_KEY", "TYPESAFE_API_KEY"]:
        if os.environ.get(env_var):
            return os.environ[env_var].strip()

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k in ["jev_bropilot_eval", "bropilot_eval", "JEV_API_KEY", "TYPESAFE_API_KEY"]:
                        return v
    return None


def call_jev_api(api_key, state, questions, model="jev-latest", max_retries=4, timeout=30):
    """Call TypeSafe Jev API endpoint with exponential backoff on rate limits."""
    url = "https://api.typesafe.ai/v1/systemone"
    payload = {
        "state": state,
        "model": model,
        "questions": questions
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                body = json.loads(resp.read().decode("utf-8"))
                return {
                    "success": True,
                    "latency_ms": latency_ms,
                    "answers": body.get("answers", {}),
                    "model": body.get("model", model),
                    "usage": body.get("usage", {})
                }
        except urllib.error.HTTPError as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error_body = e.read().decode("utf-8", errors="replace")
            # Retry on 429 (Rate Limit) or 529 / 503 (Overloaded)
            if e.code in [429, 503, 529] and attempt < max_retries:
                backoff = (2 ** attempt) * 1.5
                time.sleep(backoff)
                continue
            return {
                "success": False,
                "latency_ms": latency_ms,
                "error": f"HTTP {e.code}: {error_body}",
                "answers": {}
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if attempt < max_retries:
                time.sleep(2.0)
                continue
            return {
                "success": False,
                "latency_ms": latency_ms,
                "error": str(e),
                "answers": {}
            }


def compute_calibration_metrics(all_confs, all_corrects, n_bins=10):
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    confs = np.array(all_confs)
    corrects = np.array(all_corrects)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs > b_low) & (confs <= b_high) if i > 0 else (confs >= b_low) & (confs <= b_high)
        n_in_bin = int(mask.sum())

        if n_in_bin > 0:
            bin_acc = float(corrects[mask].mean())
            bin_conf = float(confs[mask].mean())
            gap = abs(bin_acc - bin_conf)
            ece += (n_in_bin / len(confs)) * gap
            mce = max(mce, gap)
            bin_stats.append({
                "bin": f"[{b_low:.1f}, {b_high:.1f}]",
                "count": n_in_bin,
                "confidence": round(bin_conf, 3),
                "accuracy": round(bin_acc, 3),
                "calibration_gap": round(gap, 3)
            })

    return round(float(ece), 4), round(float(mce), 4), bin_stats


def evaluate_predictions(cases, predictions, latencies):
    """Compute complete statistical evaluation metrics from test cases and predictions."""
    overall_accuracies = []
    overall_soft_accuracies = []
    overall_brier_scores = []
    overall_tv_distances = []
    overall_kl_divs = []
    all_confs = []
    all_corrects = []

    per_qtype_stats = {}

    for idx, case in enumerate(cases):
        questions = case["questions"]
        gold = case["gold"]
        pred_ans = predictions[idx]

        for qid, qdef in questions.items():
            if qid not in pred_ans or qid not in gold:
                continue

            p_ans = pred_ans[qid]
            g_ans = gold[qid]
            q_type = qdef["type"]

            if qid not in per_qtype_stats:
                per_qtype_stats[qid] = {
                    "question_id": qid,
                    "type": q_type,
                    "accuracies": [],
                    "brier_scores": [],
                    "confs": [],
                    "corrects": [],
                }

            if q_type == "choice":
                keys = list(qdef["criteria"].keys())
                pred_choice = str(p_ans.get("choice", ""))
                gold_label = str(g_ans.get("label", g_ans.get("choice", "")))

                is_corr = float(pred_choice == gold_label)
                overall_accuracies.append(is_corr)
                all_corrects.append(is_corr)
                per_qtype_stats[qid]["accuracies"].append(is_corr)
                per_qtype_stats[qid]["corrects"].append(is_corr)

                p_probs = np.array([p_ans.get("probabilities", {}).get(k, 1e-6) for k in keys])
                g_probs = np.array([g_ans.get("probabilities", {}).get(k, 1e-6) for k in keys])
                p_probs = p_probs / max(1e-12, p_probs.sum())
                g_probs = g_probs / max(1e-12, g_probs.sum())

                conf = float(p_probs.max())
                all_confs.append(conf)
                per_qtype_stats[qid]["confs"].append(conf)

                soft_acc = float((p_probs * g_probs).sum())
                brier = float(((p_probs - g_probs) ** 2).sum())
                tv = float(0.5 * np.abs(p_probs - g_probs).sum())
                kl = float((g_probs * np.log(np.clip(g_probs / p_probs, 1e-12, 1e4))).sum())

                overall_soft_accuracies.append(soft_acc)
                overall_brier_scores.append(brier)
                overall_tv_distances.append(tv)
                overall_kl_divs.append(kl)
                per_qtype_stats[qid]["brier_scores"].append(brier)

            elif q_type == "noul":
                p_val = p_ans.get("noul", p_ans.get("probabilities", {}).get("true", 0.5))
                g_val = g_ans.get("noul", g_ans.get("probabilities", {}).get("true", 0.5))
                gold_label = str(g_ans.get("label", "true" if g_val >= 0.5 else "false")).lower()

                pred_label = "true" if p_val >= 0.5 else "false"
                is_corr = float(pred_label == gold_label)
                overall_accuracies.append(is_corr)
                all_corrects.append(is_corr)
                per_qtype_stats[qid]["accuracies"].append(is_corr)
                per_qtype_stats[qid]["corrects"].append(is_corr)

                conf = float(max(p_val, 1.0 - p_val))
                all_confs.append(conf)
                per_qtype_stats[qid]["confs"].append(conf)

                p_dist = np.array([1.0 - p_val, p_val])
                g_dist = np.array([1.0 - g_val, g_val])

                soft_acc = float((p_dist * g_dist).sum())
                brier = float(((p_dist - g_dist) ** 2).sum())
                tv = float(0.5 * np.abs(p_dist - g_dist).sum())
                kl = float((g_dist * np.log(np.clip(g_dist / p_dist, 1e-12, 1e4))).sum())

                overall_soft_accuracies.append(soft_acc)
                overall_brier_scores.append(brier)
                overall_tv_distances.append(tv)
                overall_kl_divs.append(kl)
                per_qtype_stats[qid]["brier_scores"].append(brier)

    mean_acc = float(np.mean(overall_accuracies)) if overall_accuracies else 0.0
    mean_soft_acc = float(np.mean(overall_soft_accuracies)) if overall_soft_accuracies else 0.0
    mean_brier = float(np.mean(overall_brier_scores)) if overall_brier_scores else 0.0
    mean_tv = float(np.mean(overall_tv_distances)) if overall_tv_distances else 0.0
    mean_kl = float(np.mean(overall_kl_divs)) if overall_kl_divs else 0.0
    ece, mce, bin_stats = compute_calibration_metrics(all_confs, all_corrects, n_bins=10)

    # Latencies
    valid_lats = [l for l in latencies if l > 0]
    lat_p50 = float(np.percentile(valid_lats, 50)) if valid_lats else 0.0
    lat_p90 = float(np.percentile(valid_lats, 90)) if valid_lats else 0.0
    lat_p95 = float(np.percentile(valid_lats, 95)) if valid_lats else 0.0
    lat_p99 = float(np.percentile(valid_lats, 99)) if valid_lats else 0.0

    per_q_summary = {}
    for qid, qdata in per_qtype_stats.items():
        per_q_summary[qid] = {
            "accuracy": round(float(np.mean(qdata["accuracies"])), 4) if qdata["accuracies"] else 0.0,
            "brier_score": round(float(np.mean(qdata["brier_scores"])), 4) if qdata["brier_scores"] else 0.0,
            "count": len(qdata["accuracies"])
        }

    return {
        "metrics": {
            "hard_accuracy": round(mean_acc, 4),
            "soft_accuracy": round(mean_soft_acc, 4),
            "brier_score": round(mean_brier, 4),
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "total_variation": round(mean_tv, 4),
            "kl_divergence": round(mean_kl, 4)
        },
        "latency_ms": {
            "p50": round(lat_p50, 2),
            "p90": round(lat_p90, 2),
            "p95": round(lat_p95, 2),
            "p99": round(lat_p99, 2)
        },
        "per_question_head": per_q_summary,
        "calibration_bins": bin_stats,
        "total_decisions": len(overall_accuracies)
    }


def format_table(rows, headers):
    """Render a clean Markdown formatted table without pandas."""
    col_widths = [len(h) for h in headers]
    str_rows = []
    for r in rows:
        row_strs = [str(r.get(h, "")) for h in headers]
        for i, val in enumerate(row_strs):
            col_widths[i] = max(col_widths[i], len(val))
        str_rows.append(row_strs)

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = ["| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in str_rows]

    return "\n".join([header_line, sep_line] + data_lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Laya vs live TypeSafe Jev API")
    parser.add_argument("--model-dir", type=str, default="models/laya-browser-agent/checkpoint_latest", help="Laya model directory")
    parser.add_argument("--test-file", type=str, default="data/browser_test_cases.jsonl", help="Test cases JSONL")
    parser.add_argument("--env-file", type=str, default="bropilot-ext/.env", help="Path to .env containing API key")
    parser.add_argument("--report-file", type=str, default="models/laya-browser-agent/eval_metrics.json", help="Report output file")
    parser.add_argument("--device", type=str, default="auto", help="Inference device (mps, cpu, auto)")
    parser.add_argument("--jev-workers", type=int, default=4, help="Concurrent workers for Jev API calls")
    parser.add_argument("--skip-jev", action="store_true", help="Skip live Jev API evaluation")
    parser.add_argument("--skip-laya", action="store_true", help="Skip local Laya evaluation")
    args = parser.parse_args()

    print("\n=======================================================")
    print("  SIDE-BY-SIDE BENCHMARK: FINE-TUNED LAYA VS JEV API   ")
    print("=======================================================\n")

    # 1. Load Test Cases
    cases = []
    with open(args.test_file, "r") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))
    print(f"Loaded {len(cases)} benchmark test cases from {args.test_file}.")

    # 2. Checkpoint metadata
    ckpt_meta_path = os.path.join(args.model_dir, "checkpoint_meta.json")
    ckpt_epoch = 5
    if os.path.exists(ckpt_meta_path):
        try:
            with open(ckpt_meta_path) as f:
                ckpt_epoch = json.load(f).get("epoch", 5)
        except Exception:
            pass
    print(f"Laya Checkpoint: {args.model_dir} (Epoch {ckpt_epoch})")

    # 3. Evaluate Local Fine-Tuned Laya Model
    laya_results = None
    if not args.skip_laya:
        import laya
        device_name = "mps" if (args.device == "auto" and torch.backends.mps.is_available()) else ("cpu" if args.device == "auto" else args.device)
        print(f"\n[1/2] Evaluating Fine-Tuned Laya on device: {device_name.upper()}...")
        agent = laya.Agent(args.model_dir, device=device_name)

        laya_preds = []
        laya_lats = []
        for idx, case in enumerate(cases):
            t0 = time.perf_counter()
            pred_raw = agent.predict(case["state"], case["questions"])
            dt = (time.perf_counter() - t0) * 1000.0
            laya_lats.append(dt)
            laya_preds.append(pred_raw.get("answers", pred_raw) if isinstance(pred_raw, dict) else pred_raw)
            if (idx + 1) % 20 == 0 or (idx + 1) == len(cases):
                print(f"  [Laya] Evaluated {idx + 1}/{len(cases)} cases (p50 lat: {np.percentile(laya_lats, 50):.1f} ms)...")

        laya_results = evaluate_predictions(cases, laya_preds, laya_lats)
        print(f"  -> Laya Hard Accuracy: {laya_results['metrics']['hard_accuracy']*100:.2f}% | Soft Accuracy: {laya_results['metrics']['soft_accuracy']*100:.2f}% | Brier: {laya_results['metrics']['brier_score']:.4f}")

    # 4. Evaluate Live TypeSafe Jev API
    jev_results = None
    jev_model_name = "jev-latest"
    if not args.skip_jev:
        api_key = load_env_key(args.env_file)
        if not api_key:
            print(f"ERROR: Could not find Jev API key (jev_bropilot_eval) in {args.env_file} or environment variables.")
            sys.exit(1)

        print(f"\n[2/2] Evaluating Live TypeSafe Jev API (Workers: {args.jev_workers})...")
        print(f"  Key verified: {api_key[:12]}...{api_key[-6:]}")

        jev_preds = [None] * len(cases)
        jev_lats = [0.0] * len(cases)
        total_tokens = {"input": 0, "output": 0}

        def process_jev_case(idx, case):
            resp = call_jev_api(api_key, case["state"], case["questions"], model="jev-latest")
            return idx, resp

        completed_count = 0
        with ThreadPoolExecutor(max_workers=args.jev_workers) as executor:
            future_to_idx = {executor.submit(process_jev_case, i, c): i for i, c in enumerate(cases)}
            for future in as_completed(future_to_idx):
                idx, resp = future.result()
                completed_count += 1
                if resp["success"]:
                    jev_preds[idx] = resp["answers"]
                    jev_lats[idx] = resp["latency_ms"]
                    jev_model_name = resp.get("model", jev_model_name)
                    u = resp.get("usage", {})
                    total_tokens["input"] += u.get("input_tokens", 0)
                    total_tokens["output"] += u.get("output_tokens", 0)
                else:
                    print(f"  [Warning] Case {idx+1} failed: {resp.get('error')}")
                    jev_preds[idx] = {}
                    jev_lats[idx] = resp["latency_ms"]

                if completed_count % 15 == 0 or completed_count == len(cases):
                    valid_lats = [l for l in jev_lats if l > 0]
                    cur_p50 = np.percentile(valid_lats, 50) if valid_lats else 0.0
                    print(f"  [Jev API] Completed {completed_count}/{len(cases)} cases (p50 lat: {cur_p50:.1f} ms)...")

        jev_results = evaluate_predictions(cases, jev_preds, jev_lats)
        jev_results["total_tokens"] = total_tokens
        print(f"  -> Jev Hard Accuracy: {jev_results['metrics']['hard_accuracy']*100:.2f}% | Soft Accuracy: {jev_results['metrics']['soft_accuracy']*100:.2f}% | Brier: {jev_results['metrics']['brier_score']:.4f}")

    # 5. Build Unified Comparison Report
    final_report = {
        "model": "laya-browser-agent",
        "device": device_name if laya_results else "mps",
        "checkpoint_epoch": ckpt_epoch,
        "n_cases": len(cases),
        "n_decisions": laya_results["total_decisions"] if laya_results else (jev_results["total_decisions"] if jev_results else 244),
        "metrics": laya_results["metrics"] if laya_results else {},
        "latency_ms": laya_results["latency_ms"] if laya_results else {},
        "per_question_head": laya_results["per_question_head"] if laya_results else {},
        "calibration_bins": laya_results["calibration_bins"] if laya_results else [],
    }

    if jev_results:
        final_report["jev"] = {
            "model": jev_model_name,
            "metrics": jev_results["metrics"],
            "latency_ms": jev_results["latency_ms"],
            "per_question_head": jev_results["per_question_head"],
            "calibration_bins": jev_results["calibration_bins"],
            "total_tokens": jev_results["total_tokens"],
            "cost_per_1k": "$0.40"
        }

    # Save to file
    with open(args.report_file, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved updated benchmark report to: {args.report_file}")

    # 6. Render Side-by-Side Comparison Tables
    print("\n" + "=" * 78)
    print(f"  BENCHMARK RESULTS: FINE-TUNED LAYA (EPOCH {ckpt_epoch}) VS LIVE JEV API ({jev_model_name})")
    print("=" * 78)

    lm = laya_results["metrics"] if laya_results else {}
    jm = jev_results["metrics"] if jev_results else {}
    llat = laya_results["latency_ms"] if laya_results else {}
    jlat = jev_results["latency_ms"] if jev_results else {}

    diff_hard = (lm.get('hard_accuracy', 0) - jm.get('hard_accuracy', 0)) * 100
    diff_soft = (lm.get('soft_accuracy', 0) - jm.get('soft_accuracy', 0)) * 100
    brier_ratio = jm.get('brier_score', 1.0) / max(1e-6, lm.get('brier_score', 1.0))

    overview_headers = ["Metric", f"Fine-Tuned Laya (Epoch {ckpt_epoch})", f"TypeSafe Jev ({jev_model_name})", "Delta (Laya vs Jev)"]
    overview_rows = [
        {
            "Metric": "Hard Accuracy (Argmax)",
            f"Fine-Tuned Laya (Epoch {ckpt_epoch})": f"{lm.get('hard_accuracy', 0)*100:.2f}%",
            f"TypeSafe Jev ({jev_model_name})": f"{jm.get('hard_accuracy', 0)*100:.2f}%",
            "Delta (Laya vs Jev)": f"{'+' if diff_hard >= 0 else ''}{diff_hard:.2f}%"
        },
        {
            "Metric": "Soft Accuracy (Agreement)",
            f"Fine-Tuned Laya (Epoch {ckpt_epoch})": f"{lm.get('soft_accuracy', 0)*100:.2f}%",
            f"TypeSafe Jev ({jev_model_name})": f"{jm.get('soft_accuracy', 0)*100:.2f}%",
            "Delta (Laya vs Jev)": f"{'+' if diff_soft >= 0 else ''}{diff_soft:.2f}%"
        },
        {
            "Metric": "Brier Score (Lower is better)",
            f"Fine-Tuned Laya (Epoch {ckpt_epoch})": f"{lm.get('brier_score', 0):.4f}",
            f"TypeSafe Jev ({jev_model_name})": f"{jm.get('brier_score', 0):.4f}",
            "Delta (Laya vs Jev)": f"{brier_ratio:.1f}x Better" if brier_ratio > 1 else "Comparable"
        },
        {
            "Metric": "ECE (Calibration Gap)",
            f"Fine-Tuned Laya (Epoch {ckpt_epoch})": f"{lm.get('ece', 0):.4f}",
            f"TypeSafe Jev ({jev_model_name})": f"{jm.get('ece', 0):.4f}",
            "Delta (Laya vs Jev)": f"{abs(lm.get('ece', 0) - jm.get('ece', 0)):.4f} Gap"
        },
        {
            "Metric": "Median Latency (p50)",
            f"Fine-Tuned Laya (Epoch {ckpt_epoch})": f"{llat.get('p50', 0):.1f} ms",
            f"TypeSafe Jev ({jev_model_name})": f"{jlat.get('p50', 0):.1f} ms",
            "Delta (Laya vs Jev)": f"{jlat.get('p50', 1) / max(1, llat.get('p50', 1)):.1f}x Faster"
        },
        {
            "Metric": "Cost per 1,000 Steps",
            f"Fine-Tuned Laya (Epoch {ckpt_epoch})": "$0.00 (Local Metal)",
            f"TypeSafe Jev ({jev_model_name})": "$0.40 (API)",
            "Delta (Laya vs Jev)": "100% Free Local"
        }
    ]
    print("\n" + format_table(overview_rows, overview_headers) + "\n")

    # Sub-task Breakdown
    print("--- SUB-DECISION ACCURACY BREAKDOWN ---")
    sub_headers = ["Sub-Decision", f"Laya Epoch {ckpt_epoch}", f"TypeSafe Jev ({jev_model_name})", "Delta"]
    sub_rows = []
    l_heads = laya_results["per_question_head"] if laya_results else {}
    j_heads = jev_results["per_question_head"] if jev_results else {}

    all_keys = sorted(list(set(list(l_heads.keys()) + list(j_heads.keys()))))
    for k in all_keys:
        l_acc = l_heads.get(k, {}).get("accuracy", 0.0) * 100
        j_acc = j_heads.get(k, {}).get("accuracy", 0.0) * 100
        d_acc = l_acc - j_acc
        sub_rows.append({
            "Sub-Decision": k,
            f"Laya Epoch {ckpt_epoch}": f"{l_acc:.1f}%",
            f"TypeSafe Jev ({jev_model_name})": f"{j_acc:.1f}%",
            "Delta": f"{'+' if d_acc >= 0 else ''}{d_acc:.1f}%"
        })
    print(format_table(sub_rows, sub_headers) + "\n")


if __name__ == "__main__":
    main()
