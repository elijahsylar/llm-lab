"""
model_registry.py — LLM Lab model registry.

A PROMPT_FORMATS table of chat-template structures + stop tokens, and a
MODEL_REGISTRY mapping stable model keys to config objects.

LLM Lab drives inference through llama_cpp's create_chat_completion / Ollama's chat API,
both of which template the conversation from the model's own metadata. So
`prompt_format` here is **metadata** (useful for display and any future
raw-prompt path), not something the chat path has to apply by hand.

The registry is built dynamically at import/refresh time by scanning:
  - Ollama's installed models (GET /api/tags)        → provider "ollama"
  - GGUF files in $LLM_LAB_SHARED_MODELS_DIR and <repo>/models/ → provider "llama_cpp"
  - GGUF files in <repo>/models/lite/                           → llama_cpp, scope "lab"

Known GGUF filenames are enriched from MODEL_META (format / context / blurb,
per-model presets); unknown files fall back to sane defaults.
"""
from __future__ import annotations

import hashlib
import os
import re

import httpx

LAB_ROOT = os.environ.get(
    "LLM_LAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Optional directory of GGUFs shared with other local apps; full fine-tune
# exports land here too. Defaults to <repo>/models/shared.
SHARED_MODELS_DIR = os.path.expanduser(os.environ.get(
    "LLM_LAB_SHARED_MODELS_DIR", os.path.join(LAB_ROOT, "models", "shared")))
LAB_MODELS_DIR = os.path.join(LAB_ROOT, "models")
# Lite LoRA exports: lab-only. Entries get scope "lab" and are hidden from
# /api/models unless ?include_lab=1 (PHT never passes it).
LITE_MODELS_DIR = os.path.join(LAB_MODELS_DIR, "lite")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")

# Files that are vision encoders / non-chat companions — never registered as
# standalone chat models.
GGUF_EXCLUDE_SUBSTRINGS = ("mmproj",)


# =============================================================================
# PROMPT FORMATS  (template structures only)
# =============================================================================
PROMPT_FORMATS: dict[str, dict] = {
    "llama3": {
        "template": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
    },
    "llama2": {
        "template": "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] ",
        "stop_tokens": ["</s>", "[INST]"],
    },
    "mistral": {
        "template": "[INST] {system}\n\n{user} [/INST] ",
        "stop_tokens": ["</s>", "[INST]"],
    },
    "chatml": {
        "template": (
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "stop_tokens": ["<|im_end|>", "<|im_start|>"],
    },
    "gemma": {
        "template": (
            "<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        "stop_tokens": ["<end_of_turn>", "<start_of_turn>"],
    },
    "phi3": {
        "template": "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
        "stop_tokens": ["<|end|>", "<|user|>", "<|assistant|>"],
    },
    "deepseek": {
        "template": "<|begin▁of▁sentence|><|User|>{system}\n\n{user}<|Assistant|>",
        "stop_tokens": ["<|end▁of▁sentence|>", "<|User|>"],
    },
    "deepseek-v2": {
        "template": "<|begin▁of▁sentence|>{system}\n\nUser: {user}\n\nAssistant: ",
        "stop_tokens": ["<|EOT|>", "<|end▁of▁sentence|>"],
    },
    "command-r": {
        "template": (
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}<|END_OF_TURN_TOKEN|>"
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>"
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        ),
        "stop_tokens": ["<|END_OF_TURN_TOKEN|>", "<|START_OF_TURN_TOKEN|>"],
    },
    "raw": {
        "template": "{system}\n\n{user}\n\n",
        "stop_tokens": ["\n\n\n", "Human:", "User:"],
    },
}

DEFAULT_FORMAT = "chatml"
DEFAULT_CONTEXT = 8192
DEFAULT_GPU_LAYERS = -1  # offload all layers (fits ≤9B Q4 on an 8GB 4060)


# =============================================================================
# Curated metadata for known local GGUF files (filename → enrichment).
# Per-model presets. Big models get a conservative gpu_layers
# so an 8GB card does not try to fully offload a 22-24B model and OOM.
# =============================================================================
MODEL_META: dict[str, dict] = {
    "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf":
        {"format": "llama3", "ctx": 8192, "desc": "Llama 3 8B Instruct"},
    "Meta-Llama-3-8B-Instruct-abliterated-v3-Q4_K_M.gguf":
        {"format": "llama3", "ctx": 8192, "desc": "Llama 3 8B (abliterated)"},
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf":
        {"format": "llama3", "ctx": 8192, "desc": "Llama 3.1 8B Instruct"},
    "llama-2-7b-chat.Q4_K_M.gguf":
        {"format": "llama2", "ctx": 4096, "desc": "Llama 2 7B Chat"},
    "mistral-7b-instruct-v0.2.Q4_K_M.gguf":
        {"format": "mistral", "ctx": 8192, "desc": "Mistral 7B Instruct v0.2"},
    "openhermes-2.5-mistral-7b.Q4_K_M.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "OpenHermes 2.5 (Mistral 7B)"},
    "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "Hermes 3 (Llama 3.1 8B)"},
    "qwen2.5-3b-instruct-q4_k_m.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "Qwen 2.5 3B Instruct"},
    "Qwen3-8B-Q4_K_M.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "Qwen3 8B"},
    "Qwen3.5-9B-Q4_K_M.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "Qwen 3.5 9B"},
    "qwen2.5-coder-7b-instruct-q4_k_m.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "Qwen2.5-Coder 7B"},
    "glm-4-9b-chat-Q4_K_M.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "GLM-4 9B Chat"},
    "Phi-3-mini-4k-instruct-Q4_K_M.gguf":
        {"format": "phi3", "ctx": 4096, "desc": "Phi-3 Mini 3.8B"},
    "Phi-4-mini-instruct-Q4_K_M.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "Phi-4 Mini 3.8B"},
    "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf":
        {"format": "deepseek", "ctx": 8192, "desc": "DeepSeek R1 Distill Llama 8B"},
    "DeepSeek-Coder-V2-Lite-Instruct-Q3_K_M.gguf":
        {"format": "deepseek-v2", "ctx": 8192, "desc": "DeepSeek-Coder-V2-Lite 16B MoE"},
    "gemma-2-9b-it-Q4_K_M.gguf":
        {"format": "gemma", "ctx": 8192, "desc": "Gemma 2 9B Instruct"},
    "gemma-3-4b-it-Q4_K_M.gguf":
        {"format": "gemma", "ctx": 8192, "desc": "Gemma 3 4B Instruct"},
    "Yi-1.5-9B-Chat-16K.Q4_K_M.gguf":
        {"format": "chatml", "ctx": 16384, "desc": "Yi 1.5 9B Chat 16K"},
    "internlm2_5-7b-chat-Q4_K_M.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "InternLM 2.5 7B Chat"},
    "c4ai-command-r7b-12-2024-Q4_K_M.gguf":
        {"format": "command-r", "ctx": 8192, "desc": "Command-R 7B (Cohere)"},
    "SmolLM2-1.7B-Instruct-Q8_0.gguf":
        {"format": "chatml", "ctx": 8192, "desc": "SmolLM2 1.7B Instruct"},
    "codellama-7b-instruct.Q4_K_M.gguf":
        {"format": "llama2", "ctx": 8192, "desc": "Code Llama 7B Instruct"},
    # Lite LoRA bases. Lite exports are named "<base>--<run>-Q8_0.gguf" and
    # matched to these by base-name prefix (lab dir only, see _lite_meta).
    "SmolLM2-135M-Instruct-Q8_0.gguf":
        {"format": "chatml", "ctx": 2048, "desc": "SmolLM2 135M Instruct (lite)"},
    "Qwen2.5-0.5B-Instruct-Q8_0.gguf":
        {"format": "chatml", "ctx": 2048, "desc": "Qwen2.5 0.5B Instruct (lite)"},
    # Large models — partial offload so an 8GB card does not OOM.
    "Codestral-22B-v0.1-Q2_K.gguf":
        {"format": "mistral", "ctx": 8192, "desc": "Codestral 22B Q2 (partial offload)",
         "gpu_layers": 12},
    "Mistral-Small-3.1-24B-Instruct-2503-Q2_K.gguf":
        {"format": "mistral", "ctx": 8192, "desc": "Mistral Small 24B Q2 (partial offload)",
         "gpu_layers": 12},
}

# Base-model GGUFs we don't want surfaced as coach options (no instruct tuning).
GGUF_SKIP_FILES = {
    "llama-2-7b-base.Q4_K_M.gguf",
    "mistral-7b-base-Q4_K_M.gguf",
    "mistral-7b-base-f16.gguf",
}


def _slug(stem: str) -> str:
    """Readable stable key from a GGUF filename stem."""
    s = stem
    # Drop common quant / format suffixes for readability.
    s = re.sub(r"[.\-_](Q\d[\w]*|f16|fp16|bf16|gguf)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or stem.lower()


def _ollama_format_guess(family: str) -> str:
    family = (family or "").lower()
    if "llama" in family:
        return "llama3"
    if "gemma" in family:
        return "gemma"
    if "phi" in family:
        return "phi3"
    if "qwen" in family:
        return "chatml"
    return DEFAULT_FORMAT


def _lite_meta(fname: str) -> dict:
    """MODEL_META for a lite export "<base>--<run>-<quant>.gguf", keyed by base."""
    base = fname.split("--", 1)[0]
    for known, meta in MODEL_META.items():
        if "--" in fname and known.startswith(base + "-"):
            desc = f"{meta['desc']} · {fname[len(base) + 2:-len('.gguf')]}"
            return {**meta, "desc": desc}
    return {}


def _scan_gguf_dir(path: str, registry: dict, scope: str | None = None) -> None:
    if not os.path.isdir(path):
        return
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".gguf"):
            continue
        if fname in GGUF_SKIP_FILES:
            continue
        if any(sub in fname.lower() for sub in GGUF_EXCLUDE_SUBSTRINGS):
            continue
        full = os.path.join(path, fname)
        meta = MODEL_META.get(fname) or (_lite_meta(fname) if scope == "lab" else {})
        stem = fname[: -len(".gguf")]
        key = _slug(stem)
        # Avoid clobbering an identically-keyed entry from an earlier dir. The
        # suffix is derived from the path so it is stable across restarts
        # (str hash() is salted per process).
        if key in registry:
            key = f"{key}-{hashlib.sha1(full.encode()).hexdigest()[:6]}"
        try:
            size_bytes = os.path.getsize(full)
        except OSError:
            size_bytes = 0
        registry[key] = {
            "key": key,
            "provider": "llama_cpp",
            "path": full,
            "ollama_name": None,
            "prompt_format": meta.get("format", DEFAULT_FORMAT),
            "context_length": meta.get("ctx", DEFAULT_CONTEXT),
            "gpu_layers": meta.get("gpu_layers", DEFAULT_GPU_LAYERS),
            "default_temperature": 0.7,
            "description": meta.get("desc", stem),
            "size_bytes": size_bytes,
        }
        if scope:
            registry[key]["scope"] = scope


def _scan_ollama(registry: dict) -> None:
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=4.0)
        if r.status_code != 200:
            return
        for m in r.json().get("models", []):
            name = m.get("name")
            if not name:
                continue
            details = m.get("details", {}) or {}
            registry[name] = {
                "key": name,
                "provider": "ollama",
                "path": None,
                "ollama_name": name,
                "prompt_format": _ollama_format_guess(
                    (details.get("families") or [details.get("family", "")])[0]
                    if (details.get("families") or details.get("family")) else ""
                ),
                "context_length": DEFAULT_CONTEXT,
                "gpu_layers": DEFAULT_GPU_LAYERS,  # managed by Ollama itself
                "default_temperature": 0.7,
                "description": f"{name} ({details.get('parameter_size', '?')}, "
                               f"{details.get('quantization_level', '?')})",
                "size_bytes": m.get("size", 0),
            }
    except Exception:
        # Ollama down — local GGUFs still register.
        pass


def build_registry() -> dict:
    """Construct the full model registry by scanning Ollama + GGUF dirs."""
    registry: dict = {}
    _scan_ollama(registry)
    _scan_gguf_dir(SHARED_MODELS_DIR, registry)
    _scan_gguf_dir(LAB_MODELS_DIR, registry)
    _scan_gguf_dir(LITE_MODELS_DIR, registry, scope="lab")
    return registry


def public_entry(entry: dict) -> dict:
    """Strip absolute filesystem paths before sending the registry to clients."""
    return {
        "key": entry["key"],
        "provider": entry["provider"],
        "prompt_format": entry["prompt_format"],
        "context_length": entry["context_length"],
        "default_temperature": entry["default_temperature"],
        "description": entry["description"],
        "size_bytes": entry.get("size_bytes", 0),
        # Only lab entries carry a scope, so PHT's response shape is unchanged.
        **({"scope": entry["scope"]} if entry.get("scope") else {}),
    }
