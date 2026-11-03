"""Train a TinyGPT on a plain-text corpus, character level.

  python -m tiny_lab.train --corpus tiny_lab/data/tinyshakespeare.txt --run_name demo

Writes tiny_lab/runs/<run_name>/ckpt.pt (state_dict, config, vocab) at every
eval and at the end, plus metrics.jsonl. Progress is printed as the same
`@@METRIC {"step","loss","val_loss","lr"}` lines finetune/train.py emits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .model import TinyConfig, TinyGPT

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
VAL_BATCHES = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Plain-text file")
    p.add_argument("--run_name", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10, help="Train-loss metric interval")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def emit(metrics_f, step: int, loss=None, val_loss=None, lr=None) -> None:
    rec = {"step": step, "loss": loss, "val_loss": val_loss, "lr": lr}
    print("@@METRIC " + json.dumps(rec), flush=True)
    metrics_f.write(json.dumps(rec) + "\n")
    metrics_f.flush()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"Corpus: {args.corpus} ({len(text):,} chars, vocab {len(vocab)})")
    print(f"  train {len(train_data):,} / val {len(val_data):,} chars")
    print(f"  chance-level loss ln({len(vocab)}) = {math.log(len(vocab)):.3f}")

    cfg = TinyConfig(vocab_size=len(vocab), block_size=args.block_size,
                     n_layer=args.n_layer, n_head=args.n_head,
                     n_embd=args.n_embd, dropout=args.dropout)
    model = TinyGPT(cfg).to(device)
    print(f"Model: {model.num_params():,} parameters on {device}")
    print(f"  config: {json.dumps(cfg.to_dict())}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def get_batch(split: torch.Tensor):
        ix = torch.randint(len(split) - cfg.block_size - 1, (args.batch,))
        x = torch.stack([split[i:i + cfg.block_size] for i in ix])
        y = torch.stack([split[i + 1:i + 1 + cfg.block_size] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        total = 0.0
        for _ in range(VAL_BATCHES):
            _, loss = model(*get_batch(val_data))
            total += loss.item()
        model.train()
        return total / VAL_BATCHES

    run_dir = os.path.join(RUNS_DIR, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    def save(step: int, last_val: float | None) -> None:
        torch.save({
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": vocab,
            "step": step,
            "val_loss": last_val,
            "train_args": vars(args),
        }, ckpt_path)

    metrics_f = open(os.path.join(run_dir, "metrics.jsonl"), "w")
    t0 = time.time()
    v = val_loss()
    emit(metrics_f, 0, val_loss=v, lr=args.lr)
    print(f"step 0: val {v:.4f}", flush=True)

    model.train()
    running, n_running = 0.0, 0
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item()
        n_running += 1

        train_avg = None
        if step % args.log_every == 0 or step == args.steps:
            train_avg = running / n_running
            running, n_running = 0.0, 0
        v = None
        if step % args.eval_every == 0 or step == args.steps:
            v = val_loss()
        if train_avg is not None or v is not None:
            emit(metrics_f, step, loss=train_avg, val_loss=v, lr=args.lr)
        if v is not None:
            save(step, v)
            print(f"step {step}: train {train_avg if train_avg is not None else float('nan'):.4f}"
                  f" val {v:.4f} ({time.time() - t0:.1f}s)", flush=True)

    metrics_f.close()
    print(f"Saved {ckpt_path}")
    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
