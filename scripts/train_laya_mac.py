#!/usr/bin/env python3
"""
Native Mac Local Training Engine for Laya on Browser Tasks.
Optimized for Apple Silicon Metal (MPS) and CPU.

Usage:
    python3 scripts/train_laya_mac.py \
        --model-dir /Users/kp/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b \
        --train-items data/browser_train_items.pt \
        --calib-items data/browser_calib_items.pt \
        --output-dir models/laya-browser-agent \
        --epochs 3 \
        --micro-batch 4 \
        --grad-accum 4
"""

import os
import sys

# Prevent numpy 2.x ABI crash in pyarrow/pandas when transformers imports candidate generators
sys.modules['pandas'] = None
sys.modules['pyarrow'] = None

import time
import json
import random
import argparse
import warnings

# Suppress optional library import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, proper_reward, QTYPES
from laya.agent import _fix_tokenizer_config


def collate_train_batch(items, pad_id):
    """Collate variable-length tokenized items into padded PyTorch tensors."""
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)

    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)

    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items], dtype=torch.long),
        "label": torch.tensor([it["label"] for it in items], dtype=torch.long),
    }


def fit_one_temp(sel):
    """Fit softmax calibration temperature via L-BFGS to minimize negative log likelihood."""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def main():
    parser = argparse.ArgumentParser(description="Train Laya locally on Mac (MPS/CPU)")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to base Laya model directory")
    parser.add_argument("--train-items", type=str, default="data/browser_train_items.pt", help="Path to train items")
    parser.add_argument("--calib-items", type=str, default="data/browser_calib_items.pt", help="Path to calib items")
    parser.add_argument("--output-dir", type=str, default="models/laya-browser-agent", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--micro-batch", type=int, default=4, help="Micro batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5, help="Learning rate for encoder")
    parser.add_argument("--lr-head", type=float, default=1.0e-4, help="Learning rate for decision heads")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO baseline exploration samples")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (mps, cpu, or auto)")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from checkpoint_latest if present")
    parser.add_argument("--log-file", type=str, default="models/laya-browser-agent/training.log", help="Path to append training logs")
    args = parser.parse_args()

    def log_msg(msg):
        print(msg, flush=True)
        if args.log_file:
            try:
                os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
                with open(args.log_file, "a") as f_log:
                    f_log.write(msg + "\n")
            except Exception:
                pass

    # Determine hardware device
    if args.device == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    log_msg(f"\n=======================================================")
    log_msg(f"  LAYA BROWSER AGENT LOCAL TRAINING PIPELINE (MAC)     ")
    log_msg(f"  Hardware Device : {device_name.upper()} (Apple Silicon Metal)" if device_name == "mps" else f"  Hardware Device : {device_name.upper()}")
    log_msg(f"  Base Model Dir  : {args.model_dir}")
    log_msg(f"  Output Dir      : {args.output_dir}")
    log_msg(f"=======================================================\n")

    _fix_tokenizer_config(args.model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))

    with open(os.path.join(args.model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    log_msg("Building model architecture and loading weights...")
    model = build_model(cfg, encoder_dir=os.path.join(args.model_dir, "encoder"))

    start_epoch = 0
    ckpt_latest_dir = os.path.join(args.output_dir, "checkpoint_latest")
    ckpt_latest_weights = os.path.join(ckpt_latest_dir, "model.safetensors")
    ckpt_meta_file = os.path.join(ckpt_latest_dir, "checkpoint_meta.json")

    if args.resume and os.path.exists(ckpt_latest_weights):
        log_msg(f"Resuming model weights from checkpoint: {ckpt_latest_weights}...")
        weights = load_file(ckpt_latest_weights)
        model.load_state_dict(weights, strict=True)
        if os.path.exists(ckpt_meta_file):
            with open(ckpt_meta_file) as f:
                meta = json.load(f)
                start_epoch = meta.get("epoch", 0)
                log_msg(f"Resumed from completed Epoch {start_epoch}. Continuing training: Epoch {start_epoch + 1} to {args.epochs}...")
    else:
        weights_path = os.path.join(args.model_dir, "model.safetensors")
        weights = load_file(weights_path)
        model.load_state_dict(weights, strict=True)

    model.to(device)
    model.train()

    log_msg(f"Loading training items from {args.train_items}...")
    train_items = torch.load(args.train_items, weights_only=False)
    calib_items = torch.load(args.calib_items, weights_only=False) if os.path.exists(args.calib_items) else []

    log_msg(f"Loaded {len(train_items)} training sequences | {len(calib_items)} held-out calibration items.")

    EPOCHS = args.epochs
    MICRO_BATCH = args.micro_batch
    GRAD_ACCUM = args.grad_accum
    GROUP_SIZE = args.group_size
    SIGMA_START = 0.4
    SIGMA_END = 0.1

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )

    total_updates = (len(train_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)

    # Fast forward scheduler for completed epochs if resuming
    updates_per_epoch = len(train_items) // (MICRO_BATCH * GRAD_ACCUM)
    for _ in range(start_epoch * updates_per_epoch):
        scheduler.step()

    log_msg(f"Starting local training: Epochs {start_epoch + 1}-{EPOCHS} | Micro-batch: {MICRO_BATCH} | Grad accum: {GRAD_ACCUM}")
    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):
        random.seed(42 + epoch)
        random.shuffle(train_items)
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress
        t_epoch_start = time.time()

        for b_idx in range(0, len(train_items), MICRO_BATCH):
            chunk = train_items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            logits, act = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["marker_pos"].to(device),
                batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Evaluate proper scoring reward (w_sph=0.75 for soft target matching, w_rps=1.0)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient loss + soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            loss.backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(train_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if device_name == "mps":
                    torch.mps.empty_cache()

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if (n_batches % 20) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                log_msg(
                    f"  [Epoch {epoch+1}/{EPOCHS}] Step {n_batches:03d} | "
                    f"Loss: {loss.item()*GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f} | LR: {cur_lr:.2e}"
                )
                if device_name == "mps":
                    torch.mps.empty_cache()

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed_epoch = time.time() - t_epoch_start
        log_msg(f"=== Epoch {epoch+1}/{EPOCHS} Finished in {elapsed_epoch:.1f}s | Avg Loss: {avg_loss:.4f} ===")

        # Save rolling checkpoint
        ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(ckpt_sd, os.path.join(ckpt_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
        tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
        with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
            json.dump({"epoch": epoch + 1, "total_epochs": EPOCHS, "avg_loss": avg_loss}, f, indent=2)

        # Release Metal MPS allocator pool after each epoch
        if device_name == "mps":
            torch.mps.empty_cache()

    # Post-training L-BFGS calibration temperature fitting
    print("\nFitting post-training calibration temperatures on held-out slice...")
    model.eval()
    calib_preds = []

    if calib_items:
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 8):
                c_chunk = calib_items[c_idx : c_idx + 8]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                l_sub, _ = model(
                    cb["input_ids"].to(device),
                    cb["attention_mask"].to(device),
                    cb["marker_pos"].to(device),
                    cb["marker_mask"].to(device),
                    cb["qtype"].to(device),
                )
                l_np = l_sub.float().cpu().numpy()
                for r_idx, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r_idx, :k], it["target"]))

    fitted_temps = [1.2, 1.2, 1.2]
    try:
        for qt in range(3):
            sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
            if sel:
                fitted_temps[qt] = fit_one_temp(sel)
        print("Fitted calibration temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])
    except Exception as e:
        print("Temperature fitting fallback:", e)

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(args.output_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.output_dir, "encoder"))
    tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-browser-agent"
    cfg["temperature"] = fitted_temps
    cfg.pop("temperature_by_options", None)

    with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining pipeline completed successfully in {total_time:.1f}s!")
    print(f"Fine-tuned model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
