#!/usr/bin/env bash
# switch-model.sh [qwen|inkling|stop|status] -- cleanly switch the single-GPU local model server.
set -euo pipefail

CMD="${1:-status}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_PORT="${QWEN_PORT:-8081}"
INKLING_PORT="${INKLING_PORT:-8082}"

vram_gib() {
  local b
  b=$("$REPO_ROOT/scripts/mem_probe.sh" 2>/dev/null || echo 0)
  python3 -c "print(f'{int(${b:-0})/2**30:.2f}')"
}

wait_health() {
  local port="$1"
  local t0 t1
  t0=$(date +%s.%N)
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      t1=$(date +%s.%N)
      python3 -c "print(f'Ready on http://127.0.0.1:${port} in {float($t1)-float($t0):.1f}s (VRAM: $(vram_gib) GiB)')"
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for :${port} to become healthy (check $REPO_ROOT/logs/)" >&2
  return 1
}

case "$CMD" in
  qwen|qwen38|27b)
    echo "Switching GPU to Qwen 3.8 27B (port $QWEN_PORT)..."
    "$REPO_ROOT/scripts/serve.sh" qwen38-27b
    wait_health "$QWEN_PORT"
    echo "OpenCode model: llamacpp/qwen38-27b"
    ;;
  inkling|inkling2|mini)
    echo "Switching GPU to Inkling 2 Mini (port $INKLING_PORT)..."
    "$REPO_ROOT/scripts/serve.sh" inkling2-mini
    wait_health "$INKLING_PORT"
    echo "OpenCode model: inkling2/inkling2-mini"
    ;;
  stop|off)
    echo "Stopping local llama-server..."
    "$REPO_ROOT/scripts/stop.sh"
    echo "GPU idle (VRAM: $(vram_gib) GiB)"
    ;;
  status)
    echo "VRAM used: $(vram_gib) GiB"
    ps -eo pid=,args= | awk '/llama-server/ && !/awk/ {print "Running PID " $1 ": " $0}' || true
    for p in "$QWEN_PORT" "$INKLING_PORT"; do
      if curl -sf "http://127.0.0.1:${p}/v1/models" >/dev/null 2>&1; then
        m=$(curl -sf "http://127.0.0.1:${p}/v1/models" | python3 -c "import json,sys; d=json.load(sys.stdin); print(','.join(x['id'] for x in d.get('data',[])))" 2>/dev/null || echo "ok")
        echo "  :${p} -> UP (${m})"
      else
        echo "  :${p} -> down"
      fi
    done
    ;;
  *)
    echo "Usage: $0 {qwen|inkling|stop|status}" >&2
    exit 1
    ;;
esac
