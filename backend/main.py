import asyncio
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel

# Support both `uvicorn backend.main:app` (package) and direct execution.
try:
    from .model_registry import build_registry, public_entry, OLLAMA_BASE, LITE_MODELS_DIR, SHARED_MODELS_DIR, LAB_ROOT
    from .coach import build_coach_prompt, messages_to_training_pairs
except ImportError:  # pragma: no cover - direct-run fallback
    from model_registry import build_registry, public_entry, OLLAMA_BASE, LITE_MODELS_DIR, SHARED_MODELS_DIR, LAB_ROOT
    from coach import build_coach_prompt, messages_to_training_pairs

app = FastAPI(title="LLM Lab API")

# Origins allowed to call this backend from a browser. Shared by CORSMiddleware
# and the PNA preflight middleware below so they never drift.
# Extra origins (for example a deployed habit-tracker frontend) come from
# LLM_LAB_ALLOWED_ORIGINS, comma-separated.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "null",
] + [o.strip() for o in os.environ.get("LLM_LAB_ALLOWED_ORIGINS", "").split(",") if o.strip()]

# CORS — the PHT frontend (served from a remote host, or Vite dev) calls this
# backend directly from the browser on the local machine. "null" keeps the
# lab's own file:// frontend working; localhost:8000 covers same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Private Network Access (PNA): a page on an https origin (e.g. https://pht.example.com) calling
# http://localhost:8000 is a public→private request. Chrome sends a preflight
# carrying `Access-Control-Request-Private-Network: true` and requires the
# response to be 2xx AND include `Access-Control-Allow-Private-Network: true`.
# Starlette's CORSMiddleware doesn't speak PNA (it returns 400 for that
# preflight), so this middleware — registered AFTER CORSMiddleware and therefore
# OUTERMOST (runs first) — short-circuits PNA preflights from allowed origins.
# Non-PNA OPTIONS and all non-OPTIONS requests pass straight through.
@app.middleware("http")
async def private_network_access(request: Request, call_next):
    if (
        request.method == "OPTIONS"
        and request.headers.get("access-control-request-private-network") == "true"
        and request.headers.get("origin") in ALLOWED_ORIGINS
    ):
        origin = request.headers.get("origin", "")
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Private-Network": "true",
            "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }
        acrh = request.headers.get("access-control-request-headers")
        if acrh:
            headers["Access-Control-Allow-Headers"] = acrh
        return Response(status_code=200, headers=headers)
    return await call_next(request)

DATASETS_DIR    = f"{LAB_ROOT}/finetune/datasets"
EXPORTS_DIR     = f"{LAB_ROOT}/finetune/exports"
RUNS_DIR        = f"{LAB_ROOT}/finetune/runs"
VENV_PYTHON     = f"{LAB_ROOT}/.venv/bin/python"
TRAIN_SCRIPT    = f"{LAB_ROOT}/finetune/train.py"
EXPORT_SCRIPT   = f"{LAB_ROOT}/finetune/export_gguf.py"
TINY_DATA_DIR   = f"{LAB_ROOT}/tiny_lab/data"
TINY_RUNS_DIR   = f"{LAB_ROOT}/tiny_lab/runs"

# ── Model registry + single-slot VRAM state ───────────────────────────────────
# Registry is rebuilt on demand (cheap: one Ollama call + dir listdir) so newly
# pulled Ollama models / new GGUFs show up without a restart.
MODEL_REGISTRY: dict = build_registry()

_llm: Llama | None = None          # currently resident llama_cpp model (or None)
_loaded_key: str | None = None     # key of the currently selected/loaded model
_llm_lock = threading.Lock()       # serializes swaps + generation setup


def _refresh_registry() -> dict:
    global MODEL_REGISTRY
    MODEL_REGISTRY = build_registry()
    return MODEL_REGISTRY


def _unload_llama() -> None:
    """Free any resident llama_cpp model from VRAM (single resident model)."""
    global _llm
    if _llm is not None:
        del _llm
        _llm = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


MODEL_LOAD_TIMEOUT_S = 120  # hard cap on a single GGUF load


def _load_llama_with_timeout(path: str, gpu_layers: int, n_ctx: int) -> Llama:
    """Construct a Llama on a worker thread, abandoning it if it exceeds the
    timeout (a corrupted GGUF can otherwise hang the load forever and wedge the
    swap lock). The orphaned thread writes only to a local holder, never to the
    global `_llm`, so a late-finishing load can't leak in."""
    holder: dict = {}

    def _build():
        try:
            holder["llm"] = Llama(
                model_path=path,
                n_gpu_layers=gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )
        except Exception as e:  # pragma: no cover - load failures
            holder["err"] = e

    t = threading.Thread(target=_build, daemon=True)
    t.start()
    t.join(MODEL_LOAD_TIMEOUT_S)
    if t.is_alive():
        raise TimeoutError(
            f"model load exceeded {MODEL_LOAD_TIMEOUT_S}s (possibly corrupt GGUF)"
        )
    if "err" in holder:
        raise holder["err"]
    return holder["llm"]


def swap_model(model_key: str) -> tuple[bool, str | None, bool]:
    """Make `model_key` the active model. Single model in VRAM at a time.

    llama_cpp: del → gc → cuda.empty_cache → 3s cooldown → construct (≤120s).
    ollama:    evict any resident llama_cpp model (free VRAM), then mark loaded;
               Ollama loads its own weights on first request.
    No-op if the model is already loaded.

    Non-blocking on the swap lock: if another swap is in progress, returns
    immediately with busy=True so the caller can answer 503 instead of queueing.
    Also refuses (busy=True) while a training/export job holds the job slot.
    Returns (ok, error_message, busy).
    """
    global _llm, _loaded_key
    acquired = _llm_lock.acquire(blocking=False)
    if not acquired:
        return False, "Model swap in progress, try again in a few seconds", True
    try:
        if _job_running():
            return False, JOB_BUSY_ERROR, True
        entry = MODEL_REGISTRY.get(model_key)
        if entry is None:
            # Try a refresh in case the registry is stale.
            _refresh_registry()
            entry = MODEL_REGISTRY.get(model_key)
            if entry is None:
                return False, f"Unknown model: {model_key}", False

        already = (
            model_key == _loaded_key
            and (entry["provider"] != "llama_cpp" or _llm is not None)
        )
        if already:
            return True, None, False

        # Free VRAM regardless of target provider (8GB card, one model at a time).
        _unload_llama()
        time.sleep(3)  # cooldown for memory release

        if entry["provider"] == "llama_cpp":
            path = entry.get("path")
            if not path or not os.path.exists(path):
                _loaded_key = None
                return False, f"Model file not found: {path}", False
            try:
                _llm = _load_llama_with_timeout(
                    path,
                    entry.get("gpu_layers", -1),
                    entry.get("context_length", 4096),
                )
            except Exception as e:  # pragma: no cover - load failures / timeout
                _loaded_key = None
                return False, f"Failed to load {model_key}: {e}", False

        _loaded_key = model_key
        return True, None, False
    finally:
        _llm_lock.release()


# ── Job slot ──────────────────────────────────────────────────────────────────
# One global slot shared by every long-running job: LoRA training ("lora"),
# tiny-GPT training ("tiny") and GGUF export ("export"). While it is occupied,
# model loads and chat are refused (503) so nothing competes for the 8GB card.
_job: dict = {
    "running": False,
    "kind": None,          # "lora" | "tiny" | "export"
    "proc": None,
    "run_name": None,
    "log": [],
    "loss_history": [],    # [{"step", "loss", "val_loss", "lr"}]
    "start_time": None,
    "exit_code": None,
}
_job_lock = threading.Lock()

JOB_BUSY_ERROR = "training job running"
METRIC_PREFIX = "@@METRIC "
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # tqdm cursor moves etc.


def _job_running() -> bool:
    with _job_lock:
        return _job["running"]


def _read_job_output(proc: subprocess.Popen):
    """Pump the job's stdout into the log. `@@METRIC {json}` records (emitted
    by the trainers' metric callbacks) go to loss_history only, not the log.
    The prefix is searched for anywhere in the line because stderr progress
    bars (tqdm, no trailing newline) can land in front of it."""
    for raw_line in proc.stdout:
        line = _ANSI_RE.sub("", raw_line).rstrip()
        with _job_lock:
            i = line.find(METRIC_PREFIX)
            if i != -1:
                try:
                    _job["loss_history"].append(json.loads(line[i + len(METRIC_PREFIX):]))
                    line = line[:i].rstrip()
                except json.JSONDecodeError:
                    pass
            if not line:
                continue
            _job["log"].append(line)
            if len(_job["log"]) > 600:
                _job["log"] = _job["log"][-500:]
    proc.wait()
    with _job_lock:
        _job["exit_code"] = proc.returncode
        _job["running"] = False
        _job["proc"] = None


def _start_job(kind: str, run_name: str, cmd: list[str],
               free_vram: bool = True) -> JSONResponse | None:
    """Claim the job slot and spawn `cmd` (cwd = LAB_ROOT). Returns an error
    response if the slot or the model lock is busy, else None.

    The slot is claimed while holding _llm_lock, so a concurrent swap_model()
    either finishes first (and we answer 409) or sees the job and refuses."""
    global _loaded_key
    if not _llm_lock.acquire(blocking=False):
        return JSONResponse({"error": "model is loading, try again"}, status_code=409)
    try:
        with _job_lock:
            if _job["running"]:
                return JSONResponse(
                    {"error": f"a {_job['kind']} job is already running"},
                    status_code=409,
                )
            _job.update(running=True, kind=kind, proc=None, run_name=run_name,
                        log=[f"$ {' '.join(cmd)}", ""], loss_history=[],
                        start_time=time.time(), exit_code=None)
        if free_vram:
            _unload_llama()
            _loaded_key = None
    finally:
        _llm_lock.release()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=LAB_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as e:
        with _job_lock:
            _job.update(running=False, exit_code=-1)
            _job["log"].append(f"failed to launch: {e}")
        return JSONResponse({"error": f"failed to launch: {e}"}, status_code=500)

    with _job_lock:
        _job["proc"] = proc
    threading.Thread(target=_read_job_output, args=(proc,), daemon=True).start()
    return None


def _stop_job(kinds: tuple[str, ...]) -> dict:
    with _job_lock:
        proc = _job.get("proc")
        if proc and _job["running"] and _job["kind"] in kinds:
            proc.terminate()
            return {"status": "stopping", "kind": _job["kind"]}
        return {"status": "not_running"}


def _job_status() -> dict:
    with _job_lock:
        return {
            "running":      _job["running"],
            "kind":         _job["kind"],
            "run_name":     _job["run_name"],
            "log":          list(_job["log"][-150:]),
            "loss_history": list(_job["loss_history"]),
            "start_time":   _job["start_time"],
            "exit_code":    _job["exit_code"],
        }


# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    model: str | None = None          # legacy field (still accepted)
    model_key: str | None = None      # preferred
    messages: list[dict]
    temperature: float | None = None
    system_prompt: str = ""
    protocol_context: dict | None = None


class LoadRequest(BaseModel):
    model_key: str


class TrainingLogRequest(BaseModel):
    protocol_id: str
    messages: list[dict]


class FinetuneRequest(BaseModel):
    base_model: str
    dataset: str
    output_name: str
    # Everything below is optional: unset fields fall through to train.py's
    # profile defaults (full profile == the historical defaults).
    profile: str | None = None            # "full" | "lite"
    epochs: int | None = None
    batch: int | None = None
    grad_accum: int | None = None
    lr: float | None = None
    max_len: int | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    max_steps: int | None = None
    eval_steps: int | None = None
    logging_steps: int | None = None
    val_split: float | None = None
    max_samples: int | None = None
    quant: str | None = None              # "none" | "4bit"
    optim: str | None = None
    target_modules: str | None = None     # comma list


# FinetuneRequest fields forwarded to train.py as `--<name> <value>`.
TRAIN_FLAGS = (
    "profile", "epochs", "batch", "grad_accum", "lr", "max_len",
    "lora_r", "lora_alpha", "lora_dropout", "max_steps", "eval_steps",
    "logging_steps", "val_split", "max_samples", "quant", "optim", "target_modules",
)


class TinyStartRequest(BaseModel):
    corpus: str = "tinyshakespeare.txt"   # file in tiny_lab/data/
    run_name: str
    steps: int = 2000
    batch: int = 32
    lr: float = 3e-4
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    eval_every: int = 100
    device: str = "auto"                  # "auto" | "cuda" | "cpu"


class TinySampleRequest(BaseModel):
    run_name: str
    n_chars: int = 300
    temperature: float = 0.8
    prompt: str = "\n"


class ExportRequest(BaseModel):
    run_name: str
    quant: str | None = None              # default: Q8_0 for lite, Q4_K_M for full


# ── Health / models ───────────────────────────────────────────────────────────
async def _ollama_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


def _visible_loaded_key(include_lab: int) -> str | None:
    """The loaded key, masked to None for callers that must not see lab models."""
    if _loaded_key and not include_lab:
        entry = MODEL_REGISTRY.get(_loaded_key)
        if entry and entry.get("scope") == "lab":
            return None
    return _loaded_key


@app.get("/api/health")
async def health(include_lab: int = 0):
    """Lightweight liveness + which model is currently loaded."""
    return {
        "status": "online",
        "loaded_model": _visible_loaded_key(include_lab),
        "ollama": await _ollama_up(),
    }


@app.get("/api/models")
async def list_models(include_lab: int = 0):
    """Full registry (paths stripped) + which model is loaded.

    Each entry also carries legacy `id`/`source` aliases so the lab's own
    frontend keeps working. Lab-scoped models (lite exports) are hidden unless
    `?include_lab=1` — PHT calls this without the flag and must never see them.
    """
    reg = _refresh_registry()
    models = []
    for entry in reg.values():
        if entry.get("scope") == "lab" and not include_lab:
            continue
        pub = public_entry(entry)
        pub["id"] = pub["key"]  # legacy alias
        pub["source"] = "ollama" if pub["provider"] == "ollama" else "gguf"
        pub["loaded"] = entry["key"] == _loaded_key
        models.append(pub)
    models.sort(key=lambda m: (m["provider"], m["key"]))
    return {"models": models, "loaded_model": _visible_loaded_key(include_lab)}


@app.get("/api/gpu")
async def gpu_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.free,memory.total,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "name": parts[0],
            "vram_used_mb": int(parts[1]),
            "vram_free_mb": int(parts[2]),
            "vram_total_mb": int(parts[3]),
            "temp_c": int(parts[4]),
            "utilization_pct": int(parts[5]),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/models/load")
async def load_model(req: LoadRequest):
    """Eagerly swap to a model so the selector can show a 'Switching…' state.

    (Companion to the in-chat swap; chat also swaps on demand.)
    """
    ok, err, busy = await asyncio.to_thread(swap_model, req.model_key)
    if busy:
        return JSONResponse(
            {"ok": False, "error": err, "loaded_model": _loaded_key},
            status_code=503,
        )
    if not ok:
        return {"ok": False, "error": err, "loaded_model": _loaded_key}
    return {"ok": True, "loaded_model": _loaded_key}


# ── Chat ──────────────────────────────────────────────────────────────────────
async def _stream_ollama(ollama_name, messages, temperature, system_prompt) -> AsyncIterator[str]:
    payload_messages = list(messages)
    if system_prompt:
        payload_messages = [{"role": "system", "content": system_prompt}] + [
            m for m in messages if m.get("role") != "system"
        ]
    payload = {"model": ollama_name, "messages": payload_messages, "stream": True,
               "options": {"temperature": temperature}}
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'Ollama {resp.status_code}'})}\n\n"
                return
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if data.get("done"):
                        yield "data: [DONE]\n\n"
                except json.JSONDecodeError:
                    pass


async def _stream_llama(messages, temperature, system_prompt) -> AsyncIterator[str]:
    """Stream from the resident llama_cpp model (already loaded by swap_model)."""
    if _llm is None:
        yield f"data: {json.dumps({'error': 'No local model loaded'})}\n\n"
        return
    msg_list = []
    if system_prompt:
        msg_list.append({"role": "system", "content": system_prompt})
    msg_list.extend(m for m in messages if m.get("role") != "system")

    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            for chunk in _llm.create_chat_completion(
                messages=msg_list, max_tokens=1024,
                temperature=temperature, stream=True,
            ):
                text = chunk["choices"][0].get("delta", {}).get("content", "")
                if text:
                    asyncio.run_coroutine_threadsafe(q.put(("token", text)), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(("error", str(e))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    threading.Thread(target=_run, daemon=True).start()
    while True:
        item = await q.get()
        if item is None:
            break
        kind, value = item
        if kind == "error":
            yield f"data: {json.dumps({'error': value})}\n\n"
            break
        yield f"data: {json.dumps({'token': value})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    model_key = req.model_key or req.model
    if not model_key:
        async def _err():
            yield f"data: {json.dumps({'error': 'No model specified'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    entry = MODEL_REGISTRY.get(model_key) or _refresh_registry().get(model_key)
    if entry is None:
        async def _err():
            yield f"data: {json.dumps({'error': f'Unknown model: {model_key}'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    # System prompt: explicit string wins; else build from structured context.
    system_prompt = req.system_prompt or ""
    if not system_prompt and req.protocol_context:
        system_prompt = build_coach_prompt(req.protocol_context)

    temperature = (
        req.temperature
        if req.temperature is not None
        else entry.get("default_temperature", 0.7)
    )

    # Ensure the requested model is the loaded one (swap if needed).
    ok, err, busy = await asyncio.to_thread(swap_model, model_key)
    if busy:
        return JSONResponse({"error": err}, status_code=503)
    if not ok:
        # Swap failed (file missing, load timeout, etc.) — don't try to generate.
        return JSONResponse({"error": err}, status_code=503)
    # Guard: a local model must actually be resident before generating.
    if entry["provider"] == "llama_cpp" and _llm is None:
        return JSONResponse({"error": "No model loaded"}, status_code=503)

    async def event_stream():
        if entry["provider"] == "ollama":
            async for chunk in _stream_ollama(
                entry["ollama_name"], req.messages, temperature, system_prompt
            ):
                yield chunk
        else:
            async for chunk in _stream_llama(
                req.messages, temperature, system_prompt
            ):
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Coach training-data accumulation ──────────────────────────────────────────
@app.post("/api/coach/log-training")
async def log_training(req: TrainingLogRequest):
    """Append a chat turn to finetune/datasets/{protocol_id}.jsonl as
    {"instruction","output"} rows (the format train.py consumes)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", req.protocol_id).strip("_")
    if not safe:
        return {"ok": False, "error": "invalid protocol_id"}
    os.makedirs(DATASETS_DIR, exist_ok=True)
    pairs = messages_to_training_pairs(req.messages)
    if not pairs:
        return {"ok": True, "appended": 0, "dataset": f"{safe}.jsonl"}
    path = os.path.join(DATASETS_DIR, f"{safe}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return {"ok": True, "appended": len(pairs), "dataset": f"{safe}.jsonl"}


# ── Fine-tuning endpoints ─────────────────────────────────────────────────────
@app.get("/api/finetune/datasets")
async def list_datasets():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(DATASETS_DIR) if f.endswith(".jsonl"))
    # Include row count for each file
    result = []
    for f in files:
        path = os.path.join(DATASETS_DIR, f)
        try:
            with open(path) as fh:
                count = sum(1 for l in fh if l.strip())
        except Exception:
            count = -1
        result.append({"name": f, "rows": count})
    return {"datasets": result}


@app.post("/api/finetune/start")
async def start_training(req: FinetuneRequest):
    # Both names become paths: plain file/directory names only, as in
    # /api/tiny/start and /api/tiny/sample.
    if not _SAFE_NAME_RE.fullmatch(req.dataset) or req.dataset.startswith("."):
        return JSONResponse({"error": "dataset: letters, digits, _ . - only"}, status_code=400)
    if not _SAFE_NAME_RE.fullmatch(req.output_name) or req.output_name.startswith("."):
        return JSONResponse({"error": "output_name: letters, digits, _ . - only"}, status_code=400)
    dataset_path = os.path.join(DATASETS_DIR, req.dataset)
    if not os.path.isfile(dataset_path):
        return JSONResponse({"error": f"Dataset not found: {req.dataset}"}, status_code=404)
    if _job_running():
        return JSONResponse({"error": JOB_BUSY_ERROR}, status_code=409)

    output_dir = os.path.join(EXPORTS_DIR, req.output_name)
    run_dir = os.path.join(RUNS_DIR, req.output_name)
    created = [d for d in (output_dir, run_dir) if not os.path.exists(d)]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    # Train on a snapshot: PHT's log-training may keep appending to the source.
    snapshot = os.path.join(run_dir, "dataset.jsonl")
    shutil.copyfile(dataset_path, snapshot)

    cmd = [
        VENV_PYTHON, TRAIN_SCRIPT,
        "--model",  req.base_model,
        "--data",   snapshot,
        "--output", output_dir,
    ]
    for name in TRAIN_FLAGS:
        val = getattr(req, name)
        if val is not None:
            cmd += [f"--{name}", str(val)]

    err = _start_job("lora", req.output_name, cmd)
    if err is not None:
        for d in created:  # don't leave empty run dirs behind a refused start
            shutil.rmtree(d, ignore_errors=True)
        return err
    return {"status": "started", "run_name": req.output_name}


def _run_train_config(run_name: str) -> dict:
    try:
        with open(os.path.join(RUNS_DIR, run_name, "train_config.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


@app.post("/api/finetune/export")
async def export_run(req: ExportRequest):
    """Merge + convert a finished run's adapter to GGUF as a job in the shared
    slot. Lite runs go to the lab-only dir (never visible to PHT) as Q8_0;
    full runs go to SHARED_MODELS_DIR as Q4_K_M."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", req.run_name)
    adapter = os.path.join(EXPORTS_DIR, safe)
    if safe != req.run_name or safe.startswith(".") or not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        return JSONResponse({"error": f"No adapter for run: {req.run_name}"}, status_code=404)

    tcfg = _run_train_config(safe)
    lite = tcfg.get("profile") == "lite"
    quant = req.quant or ("Q8_0" if lite else "Q4_K_M")
    if not re.fullmatch(r"[A-Za-z0-9_]+", quant):
        return JSONResponse({"error": f"invalid quant: {quant}"}, status_code=400)
    if lite:
        # "<base>--<run>" lets the registry enrich lite GGUFs by base name.
        base_short = os.path.basename(tcfg.get("model", "").rstrip("/")) or "lite"
        name, out_dir = f"{base_short}--{safe}", LITE_MODELS_DIR
    else:
        name, out_dir = safe, SHARED_MODELS_DIR

    cmd = [VENV_PYTHON, EXPORT_SCRIPT, "--adapter", adapter, "--name", name,
           "--quant", quant, "--out_dir", out_dir]
    # CPU merge + convert: no need to evict the resident model (chat is still
    # refused while the slot is held).
    err = _start_job("export", safe, cmd, free_vram=False)
    if err is not None:
        return err
    return {"status": "started", "run_name": safe, "out_dir": out_dir, "quant": quant}


@app.post("/api/finetune/stop")
async def stop_training():
    return _stop_job(("lora", "export"))


@app.get("/api/finetune/status")
async def training_status():
    return _job_status()


@app.get("/api/finetune/runs")
async def list_runs():
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    runs = []
    for name in sorted(os.listdir(EXPORTS_DIR)):
        path = os.path.join(EXPORTS_DIR, name)
        if not os.path.isdir(path):
            continue
        has_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
        mtime = os.path.getmtime(path)
        tcfg = _run_train_config(name)
        runs.append({
            "name": name,
            "has_adapter": has_adapter,
            "mtime": mtime,
            "profile": tcfg.get("profile"),
            "base_model": tcfg.get("model"),
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {"runs": runs}


# ── Tiny GPT (from-scratch, char-level) ───────────────────────────────────────
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


@app.get("/api/tiny/corpora")
async def tiny_corpora():
    os.makedirs(TINY_DATA_DIR, exist_ok=True)
    files = []
    for f in sorted(os.listdir(TINY_DATA_DIR)):
        path = os.path.join(TINY_DATA_DIR, f)
        if os.path.isfile(path) and f.endswith(".txt"):
            files.append({"name": f, "bytes": os.path.getsize(path)})
    return {"corpora": files}


@app.post("/api/tiny/start")
async def tiny_start(req: TinyStartRequest):
    if not _SAFE_NAME_RE.fullmatch(req.run_name) or req.run_name.startswith("."):
        return JSONResponse({"error": "run_name: letters, digits, _ . - only"}, status_code=400)
    corpus = os.path.join(TINY_DATA_DIR, os.path.basename(req.corpus))
    if not os.path.isfile(corpus):
        return JSONResponse({"error": f"Corpus not found: {req.corpus}"}, status_code=404)
    if req.device not in ("auto", "cuda", "cpu"):
        return JSONResponse({"error": f"invalid device: {req.device}"}, status_code=400)
    if req.n_embd % req.n_head:
        return JSONResponse({"error": "n_embd must be divisible by n_head"}, status_code=400)

    cmd = [VENV_PYTHON, "-m", "tiny_lab.train",
           "--corpus", corpus, "--run_name", req.run_name,
           "--steps", str(req.steps), "--batch", str(req.batch), "--lr", str(req.lr),
           "--block_size", str(req.block_size), "--n_layer", str(req.n_layer),
           "--n_head", str(req.n_head), "--n_embd", str(req.n_embd),
           "--eval_every", str(req.eval_every), "--device", req.device]
    err = _start_job("tiny", req.run_name, cmd, free_vram=req.device != "cpu")
    if err is not None:
        return err
    return {"status": "started", "run_name": req.run_name}


@app.post("/api/tiny/stop")
async def tiny_stop():
    return _stop_job(("tiny",))


@app.get("/api/tiny/status")
async def tiny_status():
    return _job_status()


@app.get("/api/tiny/runs")
async def tiny_runs():
    os.makedirs(TINY_RUNS_DIR, exist_ok=True)
    runs = []
    for name in sorted(os.listdir(TINY_RUNS_DIR)):
        path = os.path.join(TINY_RUNS_DIR, name)
        if not os.path.isdir(path):
            continue
        last_train = last_val = last_step = None
        try:
            with open(os.path.join(path, "metrics.jsonl")) as f:
                for line in f:
                    m = json.loads(line)
                    last_step = m.get("step", last_step)
                    if m.get("loss") is not None:
                        last_train = m["loss"]
                    if m.get("val_loss") is not None:
                        last_val = m["val_loss"]
        except (OSError, json.JSONDecodeError):
            pass
        runs.append({
            "name": name,
            "has_ckpt": os.path.exists(os.path.join(path, "ckpt.pt")),
            "mtime": os.path.getmtime(path),
            "step": last_step,
            "train_loss": last_train,
            "val_loss": last_val,
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return {"runs": runs}


@app.post("/api/tiny/sample")
async def tiny_sample(req: TinySampleRequest):
    """Short in-process CPU generation. Refused while any job holds the slot."""
    if _job_running():
        return JSONResponse({"error": JOB_BUSY_ERROR}, status_code=503)
    if not _SAFE_NAME_RE.fullmatch(req.run_name) or not os.path.exists(
            os.path.join(TINY_RUNS_DIR, req.run_name, "ckpt.pt")):
        return JSONResponse({"error": f"No checkpoint for run: {req.run_name}"}, status_code=404)
    n = max(1, min(req.n_chars, 2000))
    if LAB_ROOT not in sys.path:
        sys.path.insert(0, LAB_ROOT)
    from tiny_lab.sample import sample as tiny_sample_fn
    text = await asyncio.to_thread(
        tiny_sample_fn, req.run_name, n, req.temperature, req.prompt, "cpu")
    return {"run_name": req.run_name, "text": text}

