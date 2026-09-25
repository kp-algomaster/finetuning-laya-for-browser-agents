#!/usr/bin/env python3
"""
Generate realistic browser task trajectories and tokenize into
train_items.pt, calib_items.pt, and test_cases.jsonl for local Mac training.
"""

import os
import sys
import json
import random
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence, render_options, QTYPES

MODEL_DIR = "/Users/kp/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"

BROWSER_TASK_TEMPLATES = [
    # 1. Google Flights search
    {
        "goal": "Find one-way flights from Zurich to London on Sep 20",
        "url": "https://www.google.com/travel/flights?hl=en",
        "title": "Google Flights - Search Flights",
        "text": "Explore flights from Zurich to anywhere. Cheap airfare deals and round trip options.",
        "elements": [
            {"index": "1", "role": "button", "label": "Change ticket type · Round trip", "operations": ["CLICK"]},
            {"index": "2", "role": "combobox", "label": "Where from? · Zurich", "value": "Zurich", "operations": ["CLICK", "TYPE_TEXT"]},
            {"index": "3", "role": "combobox", "label": "Where to? · empty", "value": "", "operations": ["CLICK", "TYPE_TEXT"]},
            {"index": "4", "role": "textbox", "label": "Departure date · Sep 20", "value": "Sep 20", "operations": ["CLICK", "TYPE_TEXT"]},
            {"index": "5", "role": "button", "label": "Search flights", "operations": ["CLICK"]}
        ],
        "chosen_op": "TYPE_TEXT",
        "op_probs": {"CLICK": 0.08, "TYPE_TEXT": 0.88, "SELECT": 0.01, "SCROLL_DOWN": 0.01, "WAIT": 0.01, "DONE": 0.01},
        "target_type": "type_text_target",
        "target_choice": "3",
        "target_probs": {"2": 0.05, "3": 0.90, "4": 0.05},
        "is_satisfied": 0.02
    },
    # 2. Search submit
    {
        "goal": "Find one-way flights from Zurich to London on Sep 20",
        "url": "https://www.google.com/travel/flights?hl=en",
        "title": "Google Flights - Form Filled",
        "text": "Zurich (ZRH) to London (LON). Departure: Sep 20. Ready to search.",
        "elements": [
            {"index": "1", "role": "combobox", "label": "Where from? · Zurich", "value": "Zurich", "operations": ["CLICK", "TYPE_TEXT"]},
            {"index": "2", "role": "combobox", "label": "Where to? · London", "value": "London", "operations": ["CLICK", "TYPE_TEXT"]},
            {"index": "3", "role": "textbox", "label": "Departure date · Sep 20", "value": "Sep 20", "operations": ["CLICK", "TYPE_TEXT"]},
            {"index": "4", "role": "button", "label": "Search flights", "operations": ["CLICK"]},
            {"index": "5", "role": "button", "label": "Explore destinations", "operations": ["CLICK"]}
        ],
        "chosen_op": "CLICK",
        "op_probs": {"CLICK": 0.92, "TYPE_TEXT": 0.04, "SELECT": 0.01, "SCROLL_DOWN": 0.01, "WAIT": 0.01, "DONE": 0.01},
        "target_type": "click_target",
        "target_choice": "4",
        "target_probs": {"1": 0.02, "2": 0.02, "3": 0.02, "4": 0.92, "5": 0.02},
        "is_satisfied": 0.05
    },
    # 3. Wikipedia search
    {
        "goal": "Lookup Gödel's incompleteness theorems on Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "title": "Wikipedia, the free encyclopedia",
        "text": "Welcome to Wikipedia, the free encyclopedia that anyone can edit.",
        "elements": [
            {"index": "1", "role": "textbox", "label": "Search Wikipedia", "value": "", "operations": ["CLICK", "TYPE_TEXT"]},
            {"index": "2", "role": "button", "label": "Search", "operations": ["CLICK"]},
            {"index": "3", "role": "link", "label": "Today's featured article", "operations": ["CLICK"]}
        ],
        "chosen_op": "TYPE_TEXT",
        "op_probs": {"CLICK": 0.06, "TYPE_TEXT": 0.90, "SELECT": 0.01, "SCROLL_DOWN": 0.01, "WAIT": 0.01, "DONE": 0.01},
        "target_type": "type_text_target",
        "target_choice": "1",
        "target_probs": {"1": 0.95},
        "is_satisfied": 0.01
    },
    # 4. Results loaded - Goal Completion
    {
        "goal": "Find flight ticket price from Zurich to London",
        "url": "https://www.google.com/travel/flights/search?q=zurich+to+london",
        "title": "Flights from Zurich to London from $129",
        "text": "Top flight results: British Airways $129 non-stop 1h 45m. Swiss $145 non-stop. EasyJet $89.",
        "elements": [
            {"index": "1", "role": "button", "label": "Select flight · British Airways $129", "operations": ["CLICK"]},
            {"index": "2", "role": "button", "label": "Select flight · Swiss $145", "operations": ["CLICK"]},
            {"index": "3", "role": "button", "label": "Filter by stops", "operations": ["CLICK"]}
        ],
        "chosen_op": "DONE",
        "op_probs": {"CLICK": 0.05, "TYPE_TEXT": 0.01, "SELECT": 0.01, "SCROLL_DOWN": 0.02, "WAIT": 0.01, "DONE": 0.90},
        "target_type": None,
        "target_choice": None,
        "target_probs": None,
        "is_satisfied": 0.94
    },
    # 5. Dropdown Filter Selection
    {
        "goal": "Filter hotel results by Design and Free cancellation",
        "url": "https://hotels.example.com/search?city=Lisbon",
        "title": "Hotels in Lisbon - 42 properties found",
        "text": "Showing hotels in Lisbon. Filters: Star rating, Cancellation policy, Amenities.",
        "elements": [
            {"index": "1", "role": "select", "label": "Cancellation policy · Free cancellation", "value": "Free cancellation", "operations": ["SELECT"]},
            {"index": "2", "role": "button", "label": "Filter by Design", "operations": ["CLICK"]},
            {"index": "3", "role": "button", "label": "Open Casa Flora details", "operations": ["CLICK"]}
        ],
        "chosen_op": "CLICK",
        "op_probs": {"CLICK": 0.78, "TYPE_TEXT": 0.02, "SELECT": 0.15, "SCROLL_DOWN": 0.03, "WAIT": 0.01, "DONE": 0.01},
        "target_type": "click_target",
        "target_choice": "2",
        "target_probs": {"2": 0.85, "3": 0.15},
        "is_satisfied": 0.10
    },
    # 6. Page Loading / Pending updates
    {
        "goal": "Wait for payment verification to complete",
        "url": "https://checkout.example.com/confirm",
        "title": "Processing Payment...",
        "text": "Please wait while we verify your transaction. Do not refresh or click back.",
        "elements": [
            {"index": "1", "role": "button", "label": "Cancel transaction", "operations": ["CLICK"]}
        ],
        "chosen_op": "WAIT",
        "op_probs": {"CLICK": 0.02, "TYPE_TEXT": 0.01, "SELECT": 0.01, "SCROLL_DOWN": 0.01, "WAIT": 0.94, "DONE": 0.01},
        "target_type": None,
        "target_choice": None,
        "target_probs": None,
        "is_satisfied": 0.01
    }
]


def generate_variations(templates, n_samples=300):
    cases = []
    cities = [("Paris", "Rome"), ("Berlin", "Madrid"), ("Tokyo", "Kyoto"), ("New York", "Boston"), ("Sydney", "Melbourne")]
    dates = ["Oct 12", "Nov 5", "Dec 22", "Jan 15", "Feb 28"]

    for i in range(n_samples):
        base = random.choice(templates)
        case = json.loads(json.dumps(base))
        c_from, c_to = random.choice(cities)
        d_val = random.choice(dates)

        if "flights" in case["goal"].lower():
            case["goal"] = f"Find one-way flights from {c_from} to {c_to} on {d_val}"
            for el in case["elements"]:
                if "from" in el["label"].lower():
                    el["label"] = f"Where from? · {c_from}"
                    el["value"] = c_from
                elif "to" in el["label"].lower():
                    el["label"] = f"Where to? · {c_to}" if el["value"] else "Where to? · empty"
                    if el["value"]: el["value"] = c_to
                elif "departure" in el["label"].lower():
                    el["label"] = f"Departure date · {d_val}"
                    el["value"] = d_val
        cases.append(case)
    return cases


def build_case_questions_and_gold(c):
    elements = c["elements"]
    click_candidates = {el["index"]: f"[{el['index']}] {el['label']}" for el in elements if "CLICK" in el["operations"]}
    type_candidates = {el["index"]: f"[{el['index']}] {el['label']}" for el in elements if "TYPE_TEXT" in el["operations"]}
    select_candidates = {f"{el['index']}:1": f"[{el['index']}] {el['label']}" for el in elements if "SELECT" in el["operations"]}

    operations = {
        "CLICK": "Click an interactive button, link, or tab.",
        "TYPE_TEXT": "Enter or replace text in an editable input.",
        "SELECT": "Select an observed dropdown option.",
        "SCROLL_DOWN": "Scroll down to reveal hidden content.",
        "WAIT": "Wait for pending AJAX requests or page updates.",
        "DONE": "Every goal requirement is visibly fulfilled.",
        "BLOCKED": "No supported operation can make progress."
    }

    questions = {
        "operation": {
            "type": "choice",
            "instructions": f"Advance goal '{c['goal']}' using one immediate operation.",
            "criteria": operations
        },
        "is_goal_satisfied": {
            "type": "noul",
            "instructions": f"Is the goal '{c['goal']}' visibly satisfied on screen?"
        }
    }

    gold = {
        "operation": {
            "type": "choice",
            "choice": c["chosen_op"],
            "label": c["chosen_op"],
            "probabilities": c["op_probs"],
            "confidence": c["op_probs"].get(c["chosen_op"], 0.8)
        },
        "is_goal_satisfied": {
            "type": "noul",
            "label": "true" if c["is_satisfied"] >= 0.5 else "false",
            "noul": c["is_satisfied"],
            "probabilities": {"false": 1.0 - c["is_satisfied"], "true": c["is_satisfied"]},
            "confidence": max(c["is_satisfied"], 1.0 - c["is_satisfied"])
        }
    }

    if click_candidates:
        questions["click_target"] = {
            "type": "choice",
            "instructions": f"Choose element index if operation is CLICK for goal '{c['goal']}'.",
            "criteria": click_candidates
        }
        if c["target_type"] == "click_target" and c["target_choice"] in click_candidates:
            gold["click_target"] = {
                "type": "choice",
                "choice": c["target_choice"],
                "label": c["target_choice"],
                "probabilities": c["target_probs"],
                "confidence": c["target_probs"].get(c["target_choice"], 0.85)
            }
        else:
            first_k = list(click_candidates.keys())[0]
            gold["click_target"] = {
                "type": "choice",
                "choice": first_k,
                "label": first_k,
                "probabilities": {k: 1.0 / len(click_candidates) for k in click_candidates},
                "confidence": 1.0 / len(click_candidates)
            }

    if type_candidates:
        questions["type_text_target"] = {
            "type": "choice",
            "instructions": f"Choose element index if operation is TYPE_TEXT for goal '{c['goal']}'.",
            "criteria": type_candidates
        }
        if c["target_type"] == "type_text_target" and c["target_choice"] in type_candidates:
            gold["type_text_target"] = {
                "type": "choice",
                "choice": c["target_choice"],
                "label": c["target_choice"],
                "probabilities": c["target_probs"],
                "confidence": c["target_probs"].get(c["target_choice"], 0.85)
            }
        else:
            first_k = list(type_candidates.keys())[0]
            gold["type_text_target"] = {
                "type": "choice",
                "choice": first_k,
                "label": first_k,
                "probabilities": {k: 1.0 / len(type_candidates) for k in type_candidates},
                "confidence": 1.0 / len(type_candidates)
            }

    state = {
        "page": {"url": c["url"], "title": c["title"], "text": c["text"]},
        "elements": c["elements"],
        "recent_actions": []
    }

    return state, questions, gold


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    os.makedirs(out_dir, exist_ok=True)
    random.seed(42)

    print(f"Loading tokenizer and config from {MODEL_DIR}...")
    _fix_tokenizer_config(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))
    with open(os.path.join(MODEL_DIR, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    max_len = 1024
    head_max_len = 256

    print("Generating 350 realistic browser task cases...")
    raw_cases = generate_variations(BROWSER_TASK_TEMPLATES, n_samples=350)

    tokenized_items = []
    test_records = []

    for idx, c in enumerate(raw_cases):
        state, questions, gold = build_case_questions_and_gold(c)

        if idx >= 280:
            test_records.append({"state": state, "questions": questions, "gold": gold})
            continue

        for qid, q in questions.items():
            if qid not in gold:
                continue
            t = q["type"]
            crit = q.get("criteria", {})
            g = gold[qid]

            if t == "choice":
                keys = list(crit.keys())
                target = [g["probabilities"].get(k, 0.0) for k in keys]
            elif t == "noul":
                target = [g["probabilities"].get("false", 0.5), g["probabilities"].get("true", 0.5)]
            else:
                continue

            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            label = target.index(max(target))
            k = len(render_options({"t": t, "crit": crit}))

            seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, max_len, head_max_len)
            if len(markers) == k:
                tokenized_items.append({
                    "ids": seq,
                    "markers": markers,
                    "qtype": QTYPES[t],
                    "target": target,
                    "label": label
                })

    # Hold out 10% slice for post-training calibration
    random.shuffle(tokenized_items)
    n_calib = max(30, int(len(tokenized_items) * 0.10))
    calib_items = tokenized_items[:n_calib]
    train_items = tokenized_items[n_calib:]

    torch.save(train_items, os.path.join(out_dir, "browser_train_items.pt"))
    torch.save(calib_items, os.path.join(out_dir, "browser_calib_items.pt"))

    test_file = os.path.join(out_dir, "browser_test_cases.jsonl")
    with open(test_file, "w") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    print(f"Dataset generated successfully:")
    print(f"  Train sequences      : {len(train_items)} -> {out_dir}/browser_train_items.pt")
    print(f"  Calibration sequences: {len(calib_items)} -> {out_dir}/browser_calib_items.pt")
    print(f"  Test cases           : {len(test_records)} -> {test_file}")


if __name__ == "__main__":
    main()
