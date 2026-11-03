"""Generate text from a TinyGPT checkpoint.

  python -m tiny_lab.sample --run_name demo --n 300 --temperature 0.8
"""
from __future__ import annotations

import argparse
import os

import torch

from .model import TinyConfig, TinyGPT

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def load(run_name: str, device: str = "cpu") -> tuple[TinyGPT, list[str]]:
    ckpt = torch.load(os.path.join(RUNS_DIR, run_name, "ckpt.pt"),
                      map_location=device, weights_only=False)
    model = TinyGPT(TinyConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["vocab"]


def sample(run_name: str, n: int = 300, temperature: float = 0.8,
           prompt: str = "\n", device: str = "cpu", seed: int | None = None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    model, vocab = load(run_name, device)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    # Characters outside the training vocab are dropped from the prompt.
    ids = [stoi[c] for c in (prompt or "\n") if c in stoi] or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, n, temperature=temperature)[0, len(ids):]
    return "".join(vocab[i] for i in out.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--prompt", default="\n")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    print(sample(a.run_name, a.n, a.temperature, a.prompt, a.device, a.seed))


if __name__ == "__main__":
    main()
