"""tiny_lab — a from-scratch, character-level GPT (~1M params) for LLM Lab.

Standalone: nothing here imports the LoRA pipeline, and nothing it writes lands
in finetune/ or any model directory the registry scans.

  python -m tiny_lab.train  --corpus tiny_lab/data/tinyshakespeare.txt --run_name demo
  python -m tiny_lab.sample --run_name demo --n 300
"""
