#!/usr/bin/env bash
# serve.sh [qwen38-27b|inkling2-mini] -- start llama-server cleanly in the background.
set -euo pipefail

TARGET="${1:-}"
[[ -n "$TARGET" ]] || { echo "Usage: $0 {qwen38-27b|inkling2-mini}" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$REPO_ROOT/build/bin/llama-server}"
HOST="${HOST:-127.0.0.1}"

[[ -x "$LLAMA_SERVER_BIN" ]] || { echo "llama-server binary not found or not executable: $LLAMA_SERVER_BIN (run scripts/build-engine.sh first or set LLAMA_SERVER_BIN)" >&2; exit 1; }

exec 9>"$REPO_ROOT/.gpu.lock"
flock -n 9 || { echo "GPU lock held (another launch active)" >&2; exit 2; }

"$REPO_ROOT/scripts/stop.sh"
mkdir -p "$REPO_ROOT/logs"

EXTRA_ARGS=()
case "$TARGET" in
  qwen38-27b)
    PORT="${QWEN_PORT:-8081}"
    ALIAS="qwen38-27b"
    MODEL="${QWEN_MODEL:-$MODELS_DIR/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp1.gguf}"
    if [[ ! -f "$MODEL" && -f "$MODELS_DIR/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp.gguf" ]]; then
      MODEL="$MODELS_DIR/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp.gguf"
    fi
    ARGS_FILE="$REPO_ROOT/configs/qwen38-27b.args"
    ENV_FILE="$REPO_ROOT/configs/qwen38-27b.env"
    ;;
  inkling2-mini)
    PORT="${INKLING_PORT:-8082}"
    ALIAS="inkling2-mini"
    MODEL="${INKLING_MODEL:-$MODELS_DIR/Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf}"
    DRAFT="${INKLING_DRAFT_MODEL:-$MODELS_DIR/dspark-Inkling-2-Mini-Q8_0-ista.gguf}"
    [[ -f "$DRAFT" ]] || { echo "DSpark draft model not found: $DRAFT" >&2; exit 1; }
    EXTRA_ARGS=(-md "$DRAFT")
    ARGS_FILE="$REPO_ROOT/configs/inkling2-mini.args"
    ENV_FILE="$REPO_ROOT/configs/inkling2-mini.env"
    ;;
  *)
    echo "Unknown target: $TARGET (expected qwen38-27b or inkling2-mini)" >&2
    exit 1
    ;;
esac

[[ -f "$MODEL" ]] || { echo "Model file not found: $MODEL" >&2; exit 1; }
mapfile -t ARGS < "$ARGS_FILE"

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done

setsid nohup "$LLAMA_SERVER_BIN" \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" -a "$ALIAS" --jinja \
  "${EXTRA_ARGS[@]}" \
  "${ARGS[@]}" > "$REPO_ROOT/logs/$TARGET.log" 2>&1 < /dev/null 9>&- &

echo "started $TARGET on http://$HOST:$PORT (pid $!, log: $REPO_ROOT/logs/$TARGET.log)"
