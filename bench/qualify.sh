#!/usr/bin/env bash
# Run all 6 qualification gates (load, chat, reasoning, tools, 100k needles, determinism).
# Usage: ./bench/qualify.sh [qwen|inkling]
set -euo pipefail

MODE="${1:-qwen}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$MODE" == "qwen" ]]; then
  PORT="${QWEN_PORT:-8081}"
  MODEL="qwen38-27b"
else
  PORT="${INKLING_PORT:-8082}"
  MODEL="inkling2-mini"
fi

echo "=== GATE 1: LOAD ($MODE) ==="
"$REPO_ROOT/scripts/switch-model.sh" "$MODE"

python3 - "$PORT" "$MODEL" "$MODE" << 'EOF'
import json, sys, urllib.request

port, model, mode = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = f"http://127.0.0.1:{port}"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}} if mode == "qwen" else {"chat_template_kwargs": {"reasoning_effort": "none"}}
THINK = {"chat_template_kwargs": {"enable_thinking": True}} if mode == "qwen" else {"chat_template_kwargs": {"reasoning_effort": "medium"}}

def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

print("=== GATE 2: CHAT ===")
r = post({"model": model, "messages": [{"role": "user", "content": "What is the capital of France? One sentence."}],
          "max_tokens": 64, "temperature": 0.0, **NO_THINK})
ch = r["choices"][0]
print("finish_reason:", ch["finish_reason"], "| content:", repr(ch["message"].get("content")))
assert ch["finish_reason"] == "stop" and "Paris" in (ch["message"].get("content") or "")

print("=== GATE 3: REASONING SEPARATION ===")
r = post({"model": model, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
          "max_tokens": 512, "temperature": 0.0, **THINK})
ch = r["choices"][0]
rc = ch["message"].get("reasoning_content") or ""
ct = ch["message"].get("content") or ""
print("finish_reason:", ch["finish_reason"], "| reasoning_len:", len(rc), "| content:", repr(ct))
assert len(rc) > 10 and "391" in ct

print("=== GATE 4: TOOL-CALL ROUND TRIP ===")
tools = [{"type": "function", "function": {
    "name": "get_vault_code", "description": "Retrieve the secret verification code for a vault ID.",
    "parameters": {"type": "object", "properties": {"vault_id": {"type": "string"}}, "required": ["vault_id"]}}}]
msgs = [{"role": "user", "content": "Use get_vault_code to look up vault V-42 and tell me the code."}]
r1 = post({"model": model, "messages": msgs, "tools": tools, "max_tokens": 1024, "temperature": 0.0, **NO_THINK})
m1 = r1["choices"][0]["message"]
tcs = m1.get("tool_calls") or []
print("turn1 tool_calls:", tcs)
assert len(tcs) >= 1 and tcs[0]["function"]["name"] == "get_vault_code"
msgs.append(m1)
msgs.append({"role": "tool", "tool_call_id": tcs[0]["id"], "name": "get_vault_code", "content": json.dumps({"code": "ZEBRA-9184"})})
r2 = post({"model": model, "messages": msgs, "tools": tools, "max_tokens": 512, "temperature": 0.0, **NO_THINK})
ct2 = r2["choices"][0]["message"].get("content") or ""
print("turn2 content:", repr(ct2))
assert "ZEBRA-9184" in ct2
print("GATES 2, 3, 4 PASSED!")
EOF

echo "=== GATE 5: 100K NEEDLE RETRIEVAL & DECODE ==="
python3 "$REPO_ROOT/bench/longtest.py" --base-url "http://127.0.0.1:${PORT}" --model "$MODEL" --mode "$MODE" --tokens 100000 --gen 400 --reps 3

echo "=== GATE 6: TEMP=0 DETERMINISM ==="
python3 "$REPO_ROOT/bench/equiv.py" --base-url "http://127.0.0.1:${PORT}" --model "$MODEL" --mode "$MODE"
