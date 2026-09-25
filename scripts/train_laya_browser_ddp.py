#!/usr/bin/env python3
"""
Multi-GPU Distributed RLCD Fine-Tuning Script for Laya on Browser Control Tasks.
Reference:
- https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
- https://github.com/browser-use/jev-ultrafast

Usage:
    torchrun --standalone --nproc_per_node=2 scripts/train_laya_browser_ddp.py \
        convaiinnovations/laya \
        data/train_items.pt \
        models/laya-browser-agent
"""

import os
import sys
import time
import json
import random
import math
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.common import build_model, proper_reward, QTYPES
from laya.agent import _fix_tokenizer_config


def collate_train_batch(items, pad_id):
    """Collate variable-length tokenized items and option markers into padded tensors."""
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
    parser = argparse.ArgumentParser(description="Fine-tune Laya on browser control tasks via DDP")
    parser.add_argument("model_id_or_dir", type=str, help="Hugging Face model ID or path to base Laya model")
    parser.add_argument("train_items_path", type=str, help="Path to preprocessed train_items.pt")
    parser.add_argument("output_dir", type=str, help="Destination directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--micro-batch", type=int, default=8, help="Batch size per forward pass per GPU")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO baseline exploration samples")
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5, help="Learning rate for ModernBERT encoder")
    parser.add_argument("--lr-head", type=float, default=1.0e-4, help="Learning rate for decision heads")
    args = parser.parse_args()

    # Initialize Distributed Process Group
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Download model assets if given a Hugging Face repo ID
    model_dir = args.model_id_or_dir
    if not os.path.isdir(model_dir):
        if rank == 0:
            print(f"Downloading base Laya model from Hugging Face: {model_dir}...")
        model_dir = snapshot_download(model_dir)
        _fix_tokenizer_config(model_dir)
    dist.barrier()

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))

    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    # Enable gradient checkpointing to fit in 16GB T4 VRAM
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Load preprocessed browser items
    all_items = torch.load(args.train_items_path, weights_only=False)

    # Hold out 10% slice for post-training calibration (fixed random seed across all ranks)
    rng = random.Random(42)
    indices = list(range(len(all_items)))
    rng.shuffle(indices)
    calib_count = max(50, int(len(all_items) * 0.10))
    calib_indices = set(indices[:calib_count])
    train_indices = [idx for idx in indices if idx not in calib_indices]

    train_items = [all_items[i] for i in train_indices]
    calib_items = [all_items[i] for i in calib_indices]

    # Shard training items evenly across DDP ranks
    my_items = train_items[rank::world_size]

    EPOCHS = args.epochs
    MICRO_BATCH = args.micro_batch
    GRAD_ACCUM = args.grad_accum
    GROUP_SIZE = args.group_size
    SIGMA_START = 0.4
    SIGMA_END = 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]

    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_encoder}, {"params": head_params, "lr": args.lr_head}],
        weight_decay=0.01,
    )

    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        effective_batch = MICRO_BATCH * world_size * GRAD_ACCUM
        print(f"=== Starting DDP Fine-Tuning on Browser Tasks ===")
        print(f"Total training items: {len(train_items)} | Held-out calibration: {len(calib_items)}")
        print(f"Items per rank: {len(my_items)} | GPUs: {world_size} | Effective batch size: {effective_batch}")
        print(f"Epochs: {EPOCHS} | Encoder LR: {args.lr_encoder} | Head LR: {args.lr_head}")

    t0 = time.time()

    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0

        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(
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

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | "
                    f"Loss: {loss.item()*GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f} | LR: {cur_lr:.2e}"
                )

        if rank == 0:
            avg_loss = epoch_loss / max(1, n_batches)
            print(f"=== Epoch {epoch+1}/{EPOCHS} Completed in {time.time()-t0:.1f}s | Avg Loss: {avg_loss:.4f} ===")

            # Save rolling checkpoint after each epoch to prevent data loss
            ckpt_dir = os.path.join(args.output_dir, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(ckpt_sd, os.path.join(ckpt_dir, "model.safetensors"))
            model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "total_epochs": EPOCHS, "avg_loss": avg_loss}, f, indent=2)
            print(f"  Saved rolling checkpoint to {ckpt_dir}")

        dist.barrier()

    # Post-training temperature calibration on rank 0
    if rank == 0:
        print("\nFitting post-training calibration temperatures on held-out slice...")
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()

        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx : c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                l_np = l_sub.float().cpu().numpy()
                for r, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r, :k], it["target"]))

        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print("Fitted calibration temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])
        except Exception as e:
            print("Temperature fitting fallback:", e)

        os.makedirs(args.output_dir, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(args.output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(args.output_dir, "encoder"))
        tok.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-browser-agent"
        cfg["temperature"] = fitted_temps
        cfg.pop("temperature_by_options", None)

        with open(os.path.join(args.output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"\nModel successfully saved to {args.output_dir}!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
