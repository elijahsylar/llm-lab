"""
Merge a LoRA adapter into its base model, then convert to GGUF.

The output GGUF lands in --out_dir (default $LLM_LAB_SHARED_MODELS_DIR or <repo>/models/shared) and is picked
up by the LLM Lab model registry (backend/model_registry.py) automatically.

Requirements:
  - llama.cpp convert_hf_to_gguf.py  (in the llama.cpp source tree)
  - pip install llama-cpp-python transformers peft accelerate

Usage:
  python finetune/export_gguf.py \
    --adapter  finetune/exports/my-adapter \
    --name     my-model \
    [--base    meta-llama/Meta-Llama-3-8B-Instruct]  (default: from adapter_config.json)
    [--quant   Q4_K_M]   (default: Q4_K_M)
    [--out_dir models/shared]
    [--llama_cpp_dir  /path/to/llama.cpp]

Output:
  <out_dir>/my-model-Q4_K_M.gguf
  (or <out_dir>/my-model-F16.gguf if llama-quantize is missing or fails)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


LLAMA_CPP_DIR = os.path.expanduser(os.environ.get("LLAMA_CPP_DIR", "~/llama.cpp"))
SHARED_MODELS_DIR = os.path.expanduser(os.environ.get(
    "LLM_LAB_SHARED_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "shared")))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",  required=True,  help="Path to saved LoRA adapter dir")
    p.add_argument("--base",     default=None,
                   help="Base model HF ID or local path (default: adapter_config.json's base)")
    p.add_argument("--name",     required=True,  help="Output model name (no extension)")
    p.add_argument("--quant",    default="Q4_K_M", help="GGUF quantization type")
    p.add_argument("--out_dir",  default=SHARED_MODELS_DIR, help="Where the final GGUF lands")
    p.add_argument("--llama_cpp_dir", default=LLAMA_CPP_DIR)
    return p.parse_args()


def run(cmd: list[str], check: bool = True, **kwargs) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"Command failed (exit {result.returncode})", file=sys.stderr, flush=True)
        if check:
            sys.exit(1)
    return result.returncode


def main():
    args = parse_args()

    try:
        import torch
        from peft import AutoPeftModelForCausalLM, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing dependency: {e}. Run: pip install transformers peft accelerate")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="llmlab_merge_") as tmpdir:
        merged_dir = os.path.join(tmpdir, "merged")

        # ── Step 1: merge adapter into base ──────────────────────────────
        print(f"\n[1/4] Loading adapter from {args.adapter}")
        if args.base:
            print(f"      base override: {args.base}")
            base = AutoModelForCausalLM.from_pretrained(
                args.base, device_map="cpu",  # merge on CPU to avoid VRAM limit
            )
            model = PeftModel.from_pretrained(base, args.adapter)
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.adapter,
                device_map="cpu",   # merge on CPU to avoid VRAM limit
            )
        tokenizer = AutoTokenizer.from_pretrained(args.adapter)

        print(f"[2/4] Merging LoRA weights...")
        model = model.merge_and_unload()

        print(f"[3/4] Saving merged model to {merged_dir}")
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        del model

        # ── Step 2: convert to GGUF F16 ──────────────────────────────────
        convert_script = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script):
            convert_script = os.path.join(args.llama_cpp_dir, "convert-hf-to-gguf.py")
        if not os.path.exists(convert_script):
            print(f"convert_hf_to_gguf.py not found in {args.llama_cpp_dir}")
            sys.exit(1)

        f16_path = os.path.join(tmpdir, f"{args.name}-F16.gguf")
        run([
            sys.executable, convert_script,
            merged_dir,
            "--outfile", f16_path,
            "--outtype", "f16",
        ])

        # ── Step 3: quantize ──────────────────────────────────────────────
        quantize_bin = os.path.join(
            args.llama_cpp_dir, "build", "bin", "llama-quantize"
        )
        if not os.path.exists(quantize_bin):
            quantize_bin = shutil.which("llama-quantize") or ""

        os.makedirs(args.out_dir, exist_ok=True)
        out_name = f"{args.name}-{args.quant}.gguf"
        out_path = os.path.join(args.out_dir, out_name)

        quantized = False
        if os.path.exists(quantize_bin):
            # The binary's RUNPATH points at the tree's old location; point the
            # loader at the libllama.so / libggml*.so that sit next to it.
            env = dict(os.environ)
            lib_dir = os.path.dirname(quantize_bin)
            env["LD_LIBRARY_PATH"] = lib_dir + (
                ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
            )
            rc = run([quantize_bin, f16_path, out_path, args.quant], check=False, env=env)
            quantized = rc == 0 and os.path.exists(out_path)
            if not quantized:
                print("WARNING: llama-quantize failed — keeping the F16 gguf instead",
                      flush=True)
        else:
            print("WARNING: llama-quantize not found — keeping the F16 gguf (larger file)",
                  flush=True)

        if not quantized:
            if os.path.exists(out_path):
                os.remove(out_path)  # partial output from a failed quantize
            out_path = os.path.join(args.out_dir, f"{args.name}-F16.gguf")
            shutil.copy2(f16_path, out_path)

    print(f"\n[4/4] Done! GGUF saved to:\n  {out_path}")
    print("The LLM Lab registry picks it up on the next /api/models call.")


if __name__ == "__main__":
    main()
