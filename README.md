# Fine-Tuning Laya for Autonomous Browser Agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS%20%7C%20CUDA%20DDP-success.svg)](#training-on-apple-silicon-metal-mps)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-cklxx%2Flaya--browser-yellow)](https://huggingface.co/cklxx/laya-browser)

This repository contains the complete codebase, evaluation harness, dataset curation scripts, and benchmark artifacts for fine-tuning **Laya** (a 149M-parameter ModernBERT encoder with multi-task decision heads) for high-frequency, structured browser automation.

In a held-out **70-case / 244-decision benchmark**, Fine-Tuned Laya outperformed the commercial **TypeSafe Jev Cloud API (`jev-1.13.0`)**:
- **Offline Case-Level Pass**: **84.3% (59/70 cases with all graded outputs correct)** vs. Jev's **70.0% (49/70)** (▲ +14.3 points)
- **Decision-Level Accuracy**: **94.3% (230/244 decisions)** vs. Jev's **86.9% (212/244)** (▲ +7.4 points)
- **Median Decision Latency**: **116.8 ms** (Apple Silicon MPS) vs. **841.8 ms** (Cloud Jev API) (▲ 7.2× faster)
- **Calibration (Brier Score)**: **0.0002** vs. Jev's **0.1608** (▲ 804× lower squared error across categorical output distributions)
- **Marginal Cloud Cost**: **$0.00** at zero SaaS API invoices

---

## Benchmark Highlights & Verified Results

| Metric | Base ModernBERT (Zero-Shot) | TypeSafe Jev (`jev-1.13.0` Live API) | Fine-Tuned Laya (Local Mac MPS) | Delta (Laya vs. Jev) |
| :--- | :---: | :---: | :---: | :---: |
| **Offline Case-Level Pass (All Graded Outputs)** | 22 / 70 (31.43%) | 49 / 70 (70.00%) | **59 / 70 (84.29%)** | **▲ +10 cases (+14.29%)** |
| **Decisions Correct** | 158 / 244 | 212 / 244 | **230 / 244** | **▲ +18 decisions** |
| **Decision Accuracy (Argmax)** | 64.60% | 86.89% (~86.9%) | **94.26% (~94.3%)** | **▲ +7.37%** |
| **Soft Accuracy (Agreement)** | 54.20% | 71.03% | **76.66%** | **▲ +5.63%** |
| **Brier Score (Lower is better)** | 0.1190 | 0.1608 | **0.0002** | **▲ 804× lower error** |
| **Expected Calibration Error (ECE)** | 17.90% | 7.66% | **7.55%** | **▲ Comparable (~7.6%)** |
| **Total Variation Distance (TV)** | 0.4580 | 0.2051 | **0.0091** | **▲ 22× closer to gold** |
| **Median Latency ($p_{50}$)** | 349.0 ms | 841.8 ms | **116.8 ms (MPS)** | **▲ 7.2× faster** |
| **Marginal API Cost / 1k Steps** | $0.00 | ~$0.40 (workload est.) | **$0.00 (Zero API fees)** | **100% cloud cost reduction** |

### Reconciled Sub-Decision Breakdown

| Decision Head | Number of Decisions | Base ModernBERT | TypeSafe Jev | Fine-Tuned Laya | Delta (Laya vs Jev) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`operation`** (7-way primitive action) | 70 | 61.4% (43/70) | 81.4% (57/70) | **100.0% (70/70)** | **▲ +13 decisions (+18.6%)** |
| **`is_goal_satisfied`** (Noul verification) | 70 | 88.6% (62/70) | **100.0% (70/70)** | **100.0% (70/70)** | **Tied at parity (70/70)** |
| **`type_text_target`** (Input field index) | 34 | 52.9% (18/34) | 73.5% (25/34) | **82.4% (28/34)** | **▲ +3 decisions (+8.9%)** |
| **`click_target`** (Element index) | 70 | 55.7% (39/70) | 85.7% (60/70) | **88.6% (62/70)** | **▲ +2 decisions (+2.9%)** |
| **Total Decisions Correct** | **244** | **158 / 244 (64.6%)** | **212 / 244 (86.9%)** | **230 / 244 (94.3%)** | **▲ +18 decisions (+7.4%)** |

<p align="center">
  <img src="docs/reports/images/overall_accuracy_brier_chart.png" width="48%" alt="Accuracy and Brier Calibration" />
  <img src="docs/reports/images/subdecisions_ece_chart.png" width="48%" alt="Sub-decisions and ECE Calibration" />
</p>

---

## Dataset Provenance & Download URLs

The training corpus comprises **14,076 curated interaction states** synthesized across five distinct open sources:

| Source Dataset | Public Repository / URL | Volume | Role & Representation |
| :--- | :--- | :---: | :--- |
| **Mind2Web** | [osunlp/Mind2Web on Hugging Face](https://huggingface.co/datasets/osunlp/Mind2Web) | 7,296 steps | Multi-turn human web interaction trajectories across 137 domains, paired with 44 negative distractors. |
| **Laya-Browser Goal Corpus** | [cklxx/laya-browser on Hugging Face](https://huggingface.co/cklxx/laya-browser) | 5,244 goals | Reverse-engineered goals across 421 real crawled domains (Wikipedia, GitHub, arXiv, e-commerce). |
| **DONE Landing States** | Chromium Telemetry Traces | 700 states | Genuine post-action landing states confirming goal satisfaction (`is_satisfied = 1.0`). |
| **Step-2 Negative Controls** | Chromium Telemetry Traces | 659 states | Non-terminal landing states paired with prior action history to prevent premature stopping heuristics. |
| **DAgger Interactive Traces** | Live Rollout Corrections | 177 traces | Expert corrective demonstrations recorded when execution drifted during live rollouts. |

### Download & Verification Script

To download and verify all public dataset splits and foundation checkpoints automatically:

```bash
# Download and verify all datasets and models
python3 scripts/download_datasets.py --all

# Or download specific components
python3 scripts/download_datasets.py --dataset mind2web
python3 scripts/download_datasets.py --dataset laya_browser
python3 scripts/download_datasets.py --model base_laya
```

---

## Training on Apple Silicon Metal (MPS)

### Reinforcement Learning with Calibrated Decisions (RLCD)

Rather than generating tokens autoregressively, Laya treats browser navigation as calibrated probability distributions over discrete action vocabularies and candidate element indices. In each training step, the engine executes a policy gradient update using strictly proper scoring rules (Spherical scoring $w_{\text{sph}}=0.75$ and Ranked Probability Scoring $w_{\text{rps}}=1.0$) paired with soft cross-entropy guidance:

```python
# 1. Sample G noisy logit distributions with zero-mean projection
eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
z = logits.detach().unsqueeze(0) + eps
q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

# 2. Evaluate proper scoring reward (w_sph=0.75, w_rps=1.0)
with torch.no_grad():
    r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
    adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)

# 3. Policy gradient loss + soft cross-entropy guidance
logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
loss_rl = -(adv * logp).mean()
loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM
loss.backward()
```

### The Metal Unified Memory Leak & Fix

During initial training runs on Apple Silicon unified memory, PyTorch RSS memory bloated to **32.6 GB**, forcing **21.5 GB into disk swap** and slowing throughput by 10×. The PyTorch Metal Performance Shaders (MPS) allocator retained intermediate computation graph allocations across gradient accumulation boundaries.

**The Fix**: In `scripts/train_laya_mac.py`, explicit Metal allocator cache purges were placed at gradient accumulation steps:

```python
if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.empty_cache()  # Purges Metal graph buffers
```

System RAM usage immediately stabilized at **~263 MB RSS**, eliminating swap thrashing and accelerating training epochs from ~3,000s down to **~180s per epoch** at 98% Metal GPU compute utilization.

<p align="center">
  <img src="docs/reports/images/training_cockpit.png" width="90%" alt="Real-Time Training Visualizer Cockpit" />
</p>

### Run Local Training on Mac

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run local end-to-end training and evaluation pipeline
python3 scripts/run_local_pipeline.py --epochs 3 --micro-batch 4 --grad-accum 4
```

Or trigger training directly with custom checkpoints:

```bash
python3 scripts/train_laya_mac.py \
    --train-items data/browser_train_items.pt \
    --calib-items data/browser_calib_items.pt \
    --output-dir models/laya-browser-agent \
    --epochs 3 \
    --micro-batch 4 \
    --grad-accum 4
```

---

## Multi-GPU Distributed Training (DDP)

For large clusters (Kaggle 2×T4, A100, H100):

```bash
torchrun --nproc_per_node=2 scripts/train_laya_browser_ddp.py \
    --train-items data/browser_train_items.pt \
    --output-dir models/laya-browser-agent \
    --epochs 5 \
    --batch-size 8
```

Alternatively, open and run the verified Kaggle notebook:
[`notebooks/laya_finetune_browser_tasks_2xT4_kaggle.ipynb`](notebooks/laya_finetune_browser_tasks_2xT4_kaggle.ipynb).

---

## Running the 70-Case Side-by-Side Benchmark

To evaluate your fine-tuned model side-by-side against the live TypeSafe Jev API:

```bash
# Export your TypeSafe API key (or add to .env)
export TYPESAFE_API_KEY="your_typesafe_api_key_here"

# Execute side-by-side benchmark
python3 scripts/eval_jev_vs_laya.py \
    --model-dir models/laya-browser-agent/checkpoint_latest \
    --test-file data/browser_test_cases.jsonl \
    --device mps
```

### Standalone Failure Diagnosis

Inspect the exact failure cases where Jev or Laya struggled:

```bash
python3 scripts/diagnose_jev_vs_laya.py \
    --model-dir models/laya-browser-agent/checkpoint_latest \
    --test-file data/browser_test_cases.jsonl
```

---

## Real-Time Telemetry Cockpit

Monitor memory RSS, swap, GPU compute utilization, loss convergence, and step latency via a live WebSocket dashboard:

```bash
python3 scripts/training_visualizer_server.py --port 8765
# Open http://localhost:8765 in your browser
```

---

## Proposed Edge–Cloud Cascade (Engineering Hypothesis)

Given the complementary strengths of local and cloud models:
1. **Local Primary (MPS)**: Evaluate state with Fine-Tuned Laya (~117 ms).
2. **Confidence Gating**: If output confidence exceeds 0.40 (projected for ~88% of standard navigation actions), execute locally at $0 marginal API fees.
3. **Cloud Fallback (TypeSafe Jev)**: If confidence is ambiguous (~12% of steps), route the state to TypeSafe Jev for cloud reasoning.

*Roadmap*: Closed-loop multi-turn evaluation on live WebArena and Mind2Web environments is ongoing.

---

## Repository Structure

```
.
├── README.md                                  # Comprehensive documentation & benchmark results
├── requirements.txt                           # Minimal Python dependencies
├── .gitignore                                 # Ignore large binary weights, logs, cache
├── LICENSE                                    # Apache 2.0 License
├── data/
│   ├── browser_test_cases.jsonl               # 70 held-out benchmark evaluation scenarios
│   ├── browser_calib_items.pt                 # Calibration dataset tensor
│   └── browser_train_items.pt                 # Tokenized training dataset tensor
├── scripts/
│   ├── train_laya_mac.py                      # Apple Silicon MPS training engine with memory fix
│   ├── train_laya_browser_ddp.py              # Multi-GPU Distributed Data Parallel training engine
│   ├── generate_mac_training_data.py          # Data synthesis & tokenization pipeline
│   ├── preprocess_browser_decisions.py        # Decision preprocessor
│   ├── eval_jev_vs_laya.py                    # Side-by-side benchmark runner vs TypeSafe Jev API
│   ├── eval_laya_browser.py                   # Standalone Laya evaluation runner
│   ├── diagnose_jev_vs_laya.py                # Failure analysis & error taxonomy diagnostics
│   ├── run_local_pipeline.py                  # Master orchestration pipeline
│   ├── training_visualizer_server.py          # Real-time WebSocket visualizer cockpit
│   └── download_datasets.py                   # Automated Hugging Face dataset downloader
├── notebooks/
│   └── laya_finetune_browser_tasks_2xT4_kaggle.ipynb # End-to-end Kaggle 2xT4 DDP training notebook
├── models/
│   └── laya-browser-agent/                    # Checkpoint configs, tokenizer, and eval telemetry
└── docs/
    └── reports/
        ├── laya_vs_jev_whitepaper.md          # Full Technical Whitepaper
        ├── laya_vs_jev_whitepaper.html        # Interactive HTML Report
        └── images/                            # Benchmark and training telemetry charts
```

---

## References & Citations

1. **Mind2Web**: Deng et al., *Mind2Web: Towards a Generalist Agent for the Web*, NeurIPS 2023. [arXiv:2306.06070](https://arxiv.org/abs/2306.06070).
2. **ModernBERT**: Warner et al., *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory-Efficient, and Long-Context Understanding*, Answer.AI & LightOn (2024). [arXiv:2412.13663](https://arxiv.org/abs/2412.13663).
3. **Laya Architecture**: NandhaKishorM, *Fine-Tuning Laya Browser Agent Specification*. [GitHub Guide](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md).
4. **Calibration**: Brier, G. W. (1950), *Verification of forecasts expressed in terms of probability*, Monthly Weather Review. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
5. **TypeSafe AI System One**: [TypeSafe AI Console](https://console.typesafe.ai) & [Platform](https://typesafe.ai).

---

## Author & License

Authored by **Kalpesh Patil** ([@kp-algomaster](https://github.com/kp-algomaster)). Released under the [Apache 2.0 License](LICENSE).
