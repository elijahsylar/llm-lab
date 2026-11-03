#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

# Optional local settings (see .env.example)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

mkdir -p logs

# ── Kill any existing backend ──────────────────────────────────────────────────
if pgrep -f "uvicorn backend.main" > /dev/null 2>&1; then
    echo "Stopping existing backend..."
    pkill -f "uvicorn backend.main" || true
    sleep 1
fi

# ── GPU status ─────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo "  LLM LAB  //  Neural Interface"
echo "═══════════════════════════════════════════"
nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F',' '{
        printf "  GPU:   %s\n", $1
        printf "  VRAM:  %sMB used / %sMB free\n", $2, $3
        printf "  Temp:  %s°C   Util: %s%%\n", $4, $5
    }'
echo ""

# ── Model count ────────────────────────────────────────────────────────────────
OLLAMA_COUNT=$(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null || echo "?")
GGUF_COUNT=$(find models -name '*.gguf' 2>/dev/null | wc -l)
echo "  Ollama models: $OLLAMA_COUNT"
echo "  Local GGUFs:   $GGUF_COUNT"
echo ""

# ── Start backend ──────────────────────────────────────────────────────────────
echo "Starting backend on :8000 ..."
nohup uvicorn backend.main:app --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "  PID: $BACKEND_PID"

# Wait for backend to come up
for i in {1..15}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "  Backend ready."
        break
    fi
    sleep 1
done

# ── Open frontend ──────────────────────────────────────────────────────────────
FRONTEND="$SCRIPT_DIR/frontend/index.html"
echo "Opening frontend: $FRONTEND"
xdg-open "$FRONTEND" 2>/dev/null || open "$FRONTEND" 2>/dev/null || \
    echo "  (open manually: file://$FRONTEND)"

echo ""
echo "  Backend log: $SCRIPT_DIR/logs/backend.log"
echo "  Stop:  pkill -f 'uvicorn backend.main'"
echo ""
