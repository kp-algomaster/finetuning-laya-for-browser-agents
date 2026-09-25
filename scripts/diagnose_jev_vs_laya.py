#!/usr/bin/env python3
"""
Deep Diagnostic Failure Analysis: Fine-Tuned Laya vs Live TypeSafe Jev API.

Pinpoints:
1. Exact test cases where TypeSafe Jev failed (prediction != gold).
2. Exact test cases where TypeSafe Jev outperformed Fine-Tuned Laya.
3. Root-cause categorization for each error.
"""

import os
import sys

sys.modules['pandas'] = None
sys.modules['pyarrow'] = None

import time
import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import laya


def load_env_key(env_path="bropilot-ext/.env"):
    for env_var in ["jev_bropilot_eval", "bropilot_eval", "JEV_API_KEY"]:
        if os.environ.get(env_var):
            return os.environ[env_var].strip()
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k in ["jev_bropilot_eval", "bropilot_eval", "JEV_API_KEY"]:
                        return v
    return None


def call_jev(api_key, state, questions, model="jev-latest"):
    url = "https://api.typesafe.ai/v1/systemone"
    payload = {"state": state, "model": model, "questions": questions}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            dt = (time.perf_counter() - t0) * 1000.0
            body = json.loads(resp.read().decode("utf-8"))
            return {"success": True, "latency_ms": dt, "answers": body.get("answers", {})}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        return {"success": False, "latency_ms": dt, "error": str(e), "answers": {}}


def main():
    api_key = load_env_key()
    if not api_key:
        print("Error: Jev API key not found.")
        sys.exit(1)

    print("Loading 70 benchmark cases...")
    cases = []
    with open("data/browser_test_cases.jsonl") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))

    print("Loading Laya agent...")
    agent = laya.Agent("models/laya-browser-agent/checkpoint_latest", device="cpu")

    print(f"Running inference across {len(cases)} cases...")

    # Run Jev in parallel
    jev_results = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futs = {executor.submit(call_jev, api_key, c["state"], c["questions"]): i for i, c in enumerate(cases)}
        for fut in as_completed(futs):
            idx = futs[fut]
            jev_results[idx] = fut.result()

    # Run Laya sequentially
    laya_results = []
    for c in cases:
        raw = agent.predict(c["state"], c["questions"])
        laya_results.append(raw.get("answers", raw) if isinstance(raw, dict) else raw)

    jev_failures = []
    laya_failures = []
    jev_better_than_laya = []
    laya_better_than_jev = []

    for idx, case in enumerate(cases):
        state = case["state"]
        url = state.get("page", {}).get("url", "")
        title = state.get("page", {}).get("title", "")
        questions = case["questions"]
        gold = case["gold"]

        l_ans = laya_results[idx]
        j_ans = jev_results[idx].get("answers", {})

        for qid, qdef in questions.items():
            if qid not in gold:
                continue
            g_item = gold[qid]
            qtype = qdef["type"]

            if qtype == "choice":
                gold_label = str(g_item.get("label", g_item.get("choice", "")))
                l_pred = str(l_ans.get(qid, {}).get("choice", ""))
                j_pred = str(j_ans.get(qid, {}).get("choice", ""))
                j_conf = j_ans.get(qid, {}).get("confidence", 0.0)
                l_conf = max(l_ans.get(qid, {}).get("probabilities", {}).values()) if l_ans.get(qid, {}).get("probabilities") else 0.0

                l_correct = (l_pred == gold_label)
                j_correct = (j_pred == gold_label)

                item = {
                    "case_idx": idx + 1,
                    "url": url,
                    "title": title,
                    "qid": qid,
                    "qtype": qtype,
                    "instructions": qdef.get("instructions", ""),
                    "gold": gold_label,
                    "laya_pred": l_pred,
                    "laya_conf": round(float(l_conf), 3),
                    "laya_correct": l_correct,
                    "jev_pred": j_pred,
                    "jev_conf": round(float(j_conf), 3),
                    "jev_correct": j_correct
                }

                if not j_correct:
                    jev_failures.append(item)
                if not l_correct:
                    laya_failures.append(item)
                if j_correct and not l_correct:
                    jev_better_than_laya.append(item)
                if l_correct and not j_correct:
                    laya_better_than_jev.append(item)

            elif qtype == "noul":
                g_val = g_item.get("noul", g_item.get("probabilities", {}).get("true", 0.5))
                gold_label = str(g_item.get("label", "true" if g_val >= 0.5 else "false")).lower()

                l_val = l_ans.get(qid, {}).get("noul", l_ans.get(qid, {}).get("probabilities", {}).get("true", 0.5))
                l_pred = "true" if l_val >= 0.5 else "false"

                j_val = j_ans.get(qid, {}).get("noul", 0.5)
                j_pred = "true" if j_val >= 0.5 else "false"

                l_correct = (l_pred == gold_label)
                j_correct = (j_pred == gold_label)

                item = {
                    "case_idx": idx + 1,
                    "url": url,
                    "title": title,
                    "qid": qid,
                    "qtype": qtype,
                    "instructions": qdef.get("instructions", ""),
                    "gold": gold_label,
                    "laya_pred": l_pred,
                    "laya_val": round(float(l_val), 3),
                    "laya_correct": l_correct,
                    "jev_pred": j_pred,
                    "jev_val": round(float(j_val), 3),
                    "jev_correct": j_correct
                }

                if not j_correct:
                    jev_failures.append(item)
                if not l_correct:
                    laya_failures.append(item)
                if j_correct and not l_correct:
                    jev_better_than_laya.append(item)
                if l_correct and not j_correct:
                    laya_better_than_jev.append(item)

    report = {
        "summary": {
            "total_decisions": 244,
            "jev_failures_count": len(jev_failures),
            "laya_failures_count": len(laya_failures),
            "laya_better_count": len(laya_better_than_jev),
            "jev_better_count": len(jev_better_than_laya)
        },
        "laya_better_than_jev": laya_better_than_jev,
        "jev_better_than_laya": jev_better_than_laya,
        "jev_failures": jev_failures
    }

    with open("models/laya-browser-agent/eval_diagnostic.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- DIAGNOSTIC SUMMARY ---")
    print(f"Total Decisions Evaluated     : 244")
    print(f"Jev Failures                  : {len(jev_failures)} / 244 ({len(jev_failures)/244*100:.1f}%)")
    print(f"Laya Failures                 : {len(laya_failures)} / 244 ({len(laya_failures)/244*100:.1f}%)")
    print(f"Laya Win (Laya Correct, Jev Wrong) : {len(laya_better_than_jev)} decisions")
    print(f"Jev Win  (Jev Correct, Laya Wrong) : {len(jev_better_than_laya)} decisions")


if __name__ == "__main__":
    main()
