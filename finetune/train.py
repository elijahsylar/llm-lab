"""
LoRA fine-tuning script for LLM Lab.

Dataset format (JSONL, one record per line), either:
  {"instruction": "...", "output": "..."}
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]}

Training text is rendered with the tokenizer's chat template (the same template
llama_cpp applies from the exported GGUF at serve time). Tokenizers without a
chat template fall back to the Alpaca "### Instruction / ### Response" format.

Profiles:
  full  (default) QLoRA: 4-bit base, paged 8-bit AdamW, auto-detected targets.
  lite  small bases (SmolLM2-135M, Qwen2.5-0.5B): bf16 base, q/v LoRA,
        200 steps capped, 10% val split with periodic eval.

Settings resolve as: profile defaults < --config JSON < explicit CLI flags.

Usage:
  python finetune/train.py \
    --model  meta-llama/Meta-Llama-3-8B-Instruct \
    --data   finetune/datasets/my_data.jsonl \
    --output finetune/exports/my-adapter \
    [--profile full|lite] [--config cfg.json] [--epochs 3] [--batch 4] ...

The adapter is saved in HuggingFace PEFT format. Run export_gguf.py to merge
and convert to GGUF for use with llama-cpp-python. Progress is printed
as `@@METRIC {json}` lines, which the LLM Lab backend parses for its chart.
"""

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(LAB_ROOT, "finetune", "runs")

PROFILES: dict[str, dict] = {
    "full": {
        "epochs": 3, "batch": 4, "grad_accum": 4, "lr": 2e-4, "max_len": 2048,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "max_steps": -1, "eval_steps": None, "logging_steps": 10,
        "val_split": 0.0, "max_samples": None,
        "quant": "4bit", "optim": "paged_adamw_8bit", "target_modules": None,
        "save_strategy": "epoch",
    },
    "lite": {
        "epochs": 3, "batch": 8, "grad_accum": 2, "lr": 2e-4, "max_len": 512,
        "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
        "max_steps": 200, "eval_steps": 20, "logging_steps": 2,
        "val_split": 0.1, "max_samples": 500,
        "quant": "none", "optim": "adamw_torch", "target_modules": "q_proj,v_proj",
        "save_strategy": "no",
    },
}

# Candidate pad tokens when the tokenizer's pad would equal its end-of-turn
# token (the LM collator masks pad ids out of the loss, so a shared id means
# the model never learns to stop).
PAD_CANDIDATES = ("<|endoftext|>", "<|finetune_right_pad_id|>",
                  "<|reserved_special_token_0|>", "<pad>", "<unk>")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True,  help="HF model ID or local path")
    p.add_argument("--data",    required=True,  help="JSONL dataset file")
    p.add_argument("--output",  required=True,  help="Output dir for LoRA adapter")
    p.add_argument("--profile", choices=sorted(PROFILES), default="full")
    p.add_argument("--config",  default=None, help="JSON file of setting overrides")
    # Tunables default to None so we can tell "not given" from "given".
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--batch",   type=int,   default=None, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--max_len", type=int,   default=None)
    p.add_argument("--lora_r",  type=int,   default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None, help="-1 = run full epochs")
    p.add_argument("--eval_steps", type=int, default=None)
    p.add_argument("--logging_steps", type=int, default=None)
    p.add_argument("--val_split", type=float, default=None, help="0 disables eval")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--quant",   choices=["none", "4bit"], default=None)
    p.add_argument("--optim",   default=None)
    p.add_argument("--target_modules", default=None,
                   help="Comma list; overrides auto-detection")
    return p.parse_args()


def resolve_config(args) -> dict:
    """profile defaults < --config JSON < explicit CLI flags."""
    cfg = dict(PROFILES[args.profile])
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        unknown = set(overrides) - set(cfg)
        if unknown:
            sys.exit(f"Unknown keys in {args.config}: {sorted(unknown)}")
        cfg.update(overrides)
    for key in cfg:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


def to_messages(record: dict) -> list[dict]:
    """Normalize a dataset row to a chat message list."""
    if record.get("messages"):
        return [{"role": m["role"], "content": m["content"]} for m in record["messages"]]
    return [
        {"role": "user", "content": record["instruction"]},
        {"role": "assistant", "content": record["output"]},
    ]


def load_jsonl(path: str) -> Dataset:
    # Normalized to a single `messages` column so rows of either schema mix.
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append({"messages": to_messages(json.loads(line))})
    return Dataset.from_list(records)


def format_prompt(example: dict) -> dict:
    """Alpaca fallback: render messages as Instruction/Response turns."""
    parts = []
    for m in example["messages"]:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append(f"### Instruction:\n{m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"### Response:\n{m['content']}")
    return {"text": "\n\n".join(parts)}


def make_formatter(tokenizer):
    """Return (format_fn, format_name) for this tokenizer."""
    if getattr(tokenizer, "chat_template", None):
        def format_chat(example: dict) -> dict:
            return {"text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False)}
        return format_chat, "chat_template"
    return format_prompt, "alpaca"


def fix_pad_token(tokenizer) -> None:
    """Pick a pad token distinct from EOS so end-of-turn stays in the loss."""
    if tokenizer.pad_token is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        return
    vocab = tokenizer.get_vocab()
    for cand in PAD_CANDIDATES:
        if cand in vocab and vocab[cand] != tokenizer.eos_token_id:
            tokenizer.pad_token = cand
            print(f"  Pad token: {cand!r} (distinct from EOS {tokenizer.eos_token!r})")
            return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  WARNING: pad token == EOS ({tokenizer.eos_token!r}); "
          "end-of-turn tokens are masked from the loss")


def find_lora_targets(model) -> list[str]:
    """Auto-detect linear projection layers for LoRA."""
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            short = name.split(".")[-1]
            # Common attention/MLP projection names
            if any(k in short for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj",
                                         "query", "key", "value", "dense")):
                targets.add(short)
    return list(targets) if targets else ["q_proj", "v_proj"]


class MetricCallback(TrainerCallback):
    """Emit one machine-readable line per log event for the backend to parse:
    @@METRIC {"step", "loss", "val_loss", "lr"}. Events carrying neither a
    train nor a val loss (e.g. the end-of-run summary) are skipped."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("eval_loss")
        if loss is None and val_loss is None:
            return
        print("@@METRIC " + json.dumps({
            "step": state.global_step,
            "loss": loss,
            "val_loss": val_loss,
            "lr": logs.get("learning_rate"),
        }), flush=True)


def main():
    args = parse_args()
    cfg = resolve_config(args)

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("WARNING: CUDA not available — training will be very slow on CPU")
        # bitsandbytes 4-bit and paged/8-bit optimizers are CUDA-only.
        cfg["quant"] = "none"
        if "8bit" in cfg["optim"] or cfg["optim"].startswith("paged"):
            cfg["optim"] = "adamw_torch"
    bf16 = use_cuda and torch.cuda.is_bf16_supported()

    print(f"Profile: {args.profile}")
    print("  Config: " + json.dumps(cfg))

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    fix_pad_token(tokenizer)
    tokenizer.padding_side = "right"

    print(f"Loading dataset: {args.data}")
    formatter, fmt_name = make_formatter(tokenizer)
    dataset = load_jsonl(args.data)
    if cfg["max_samples"] and len(dataset) > cfg["max_samples"]:
        dataset = dataset.shuffle(seed=42).select(range(cfg["max_samples"]))
        print(f"  Capped to {cfg['max_samples']} samples")
    dataset = dataset.map(formatter, remove_columns=dataset.column_names)
    eval_dataset = None
    if cfg["val_split"] and cfg["val_split"] > 0 and len(dataset) >= 2:
        split = dataset.train_test_split(test_size=cfg["val_split"], seed=42)
        dataset, eval_dataset = split["train"], split["test"]
    print(f"  {len(dataset)} train examples"
          + (f", {len(eval_dataset)} val examples" if eval_dataset is not None else ""))
    print(f"  Prompt format: {fmt_name}")
    print("  Example formatted training text:")
    for ln in dataset[0]["text"].splitlines():
        print(f"    | {ln}")

    if cfg["quant"] == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading model (4-bit): {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)
        print(f"Loading model ({str(dtype).replace('torch.', '')}, "
              f"{'GPU' if use_cuda else 'CPU'}): {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )
    model.config.use_cache = False

    if cfg["target_modules"]:
        target_modules = [t.strip() for t in cfg["target_modules"].split(",") if t.strip()]
    else:
        target_modules = find_lora_targets(model)
    print(f"  LoRA targets: {target_modules}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    os.makedirs(args.output, exist_ok=True)
    run_dir = os.path.join(RUNS_DIR, os.path.basename(os.path.normpath(args.output)))
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Checkpoints: {run_dir}")
    # Recorded for export (base model + profile decide the export target).
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump({"model": args.model, "profile": args.profile,
                   "prompt_format": fmt_name, **cfg}, f, indent=2)

    has_eval = eval_dataset is not None and bool(cfg["eval_steps"])
    training_args = SFTConfig(
        output_dir=run_dir,
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"] if cfg["max_steps"] is not None else -1,
        per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        fp16=use_cuda and not bf16,
        bf16=bf16,
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps" if has_eval else "no",
        eval_steps=cfg["eval_steps"] if has_eval else None,
        save_strategy=cfg["save_strategy"],
        optim=cfg["optim"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=cfg["max_len"],
        # A chat template already carries BOS/special tokens; don't add twice.
        dataset_kwargs={"add_special_tokens": fmt_name != "chat_template"},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset if has_eval else None,
        args=training_args,
        callbacks=[MetricCallback()],
    )

    print("Starting training...")
    trainer.train()
    if has_eval and trainer.state.global_step % cfg["eval_steps"] != 0:
        # Final val point so the chart ends on the trained model.
        trainer.evaluate()

    print(f"Saving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
