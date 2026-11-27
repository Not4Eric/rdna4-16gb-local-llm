#!/usr/bin/env bash
# Gracefully stop any local llama-server on :8081/:8082 and wait for VRAM release.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

vram_used() { "$REPO_ROOT/scripts/mem_probe.sh"; }

PIDS=$(ps -eo pid=,args= | awk '/llama-server/ && /--port 808[12]/ && !/awk/ {print $1}')
if [[ -n "${PIDS:-}" ]]; then
  for p in $PIDS; do kill -TERM "$p" 2>/dev/null || true; done
  for i in $(seq 1 60); do
    STILL=$(ps -eo pid=,args= | awk '/llama-server/ && /--port 808[12]/ && !/awk/ {print $1}')
    [[ -z "$STILL" ]] && break
    sleep 1
  done
  STILL=$(ps -eo pid=,args= | awk '/llama-server/ && /--port 808[12]/ && !/awk/ {print $1}')
  if [[ -n "$STILL" ]]; then
    echo "WARNING: llama-server still running after 60s TERM: $STILL" >&2
  fi
fi

# Wait for freed VRAM pages to settle (leaves headroom for desktop compositor)
for i in $(seq 1 60); do
  u=$(vram_used 2>/dev/null || echo 0)
  [[ -n "$u" && "$u" -lt 2362232012 ]] && break
  sleep 1
done
true
