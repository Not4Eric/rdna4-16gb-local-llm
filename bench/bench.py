#!/usr/bin/env python3
"""Portable 6-prompt hill-climb benchmark for Qwen 3.8 27B (:8081) or Inkling 2 Mini (:8082).
Standard library only. Token counts always come from server usage stats.

Usage:
  python3 bench/bench.py --base-url http://127.0.0.1:8081 --model qwen38-27b --mode qwen
  python3 bench/bench.py --base-url http://127.0.0.1:8082 --model inkling2-mini --mode inkling
"""
import argparse, json, os, random, statistics, string, subprocess, sys, threading, time, urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = [
    "Write a Python implementation of an LRU cache with O(1) get and put, with docstrings and tests.",
    "Explain, step by step, how TCP congestion control works, including slow start and fast recovery.",
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle.",
    "Compare B-trees and LSM-trees for a write-heavy database workload. Use a table and give a recommendation.",
    "Write a bash script that backs up a directory to a timestamped tar.gz and keeps only the last 7 backups.",
    "Summarise the causes and consequences of the 1929 stock market crash in five paragraphs.",
]
WORDS = ("river stone paper cloud engine garden silver market winter orange signal harbor violet "
         "candle forest mirror thunder copper meadow lantern").split()

def document(tokens, seed, needles=3):
    rng = random.Random(seed)
    n_words = int(tokens / 1.052)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    codes = ["".join(rng.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(needles)]
    for i, c in enumerate(codes):
        pos = int(n_words * (i + 1) / (needles + 1))
        words.insert(pos, f". The secret code number {i + 1} is {c}. ")
    head = f"[doc {seed}-{time.time_ns()}] "
    q = "\n\nList every secret code number and its value that appears in the document above. Answer only with the codes."
    return head + " ".join(words) + q, codes

def looping(text, n=8, thresh=0.5):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(max(0, len(w) - n))]
    return len(grams) > 50 and 1 - len(set(grams)) / len(grams) > thresh

def chat(base_url, model, prompt, max_tokens, extra_body, temperature=0.6, timeout=7200):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "temperature": temperature}
    body.update(extra_body)
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); tf = None; text = []; usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    if tf is None:
                        tf = time.monotonic()
                    text.append(piece)
    te = time.monotonic()
    ct, pt = usage.get("completion_tokens"), usage.get("prompt_tokens")
    if tf is None:
        tf = te
    dec = (ct - 1) / (te - tf) if ct and ct > 1 and te > tf else None
    return {"ttft": tf - t0, "decode": dec, "completion_tokens": ct, "prompt_tokens": pt, "text": "".join(text)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--mode", choices=["qwen", "inkling"], default="qwen")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--prefill", nargs="*", type=int, default=[4096, 32768])
    args = ap.parse_args()

    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.mode == "qwen" else {
        "chat_template_kwargs": {"reasoning_effort": "none"}}

    peak = 0
    stop = threading.Event()
    def watch():
        nonlocal peak
        probe = os.path.join(REPO_ROOT, "scripts", "mem_probe.sh")
        while not stop.wait(2.0):
            try:
                b = int(subprocess.run([probe], capture_output=True, text=True, timeout=10).stdout.strip() or 0)
                peak = max(peak, b)
            except Exception:
                pass
    threading.Thread(target=watch, daemon=True).start()

    chat(args.base_url, args.model, "Say hello.", 16, extra)  # warm-up
    r = chat(args.base_url, args.model, "Reply with the single word: ready", 8, extra)
    res = {"model": args.model, "ttft_short_ms": round(r["ttft"] * 1000, 1), "flags": []}

    c1 = []
    for i in range(args.reps):
        r = chat(args.base_url, args.model, PROMPTS[i % len(PROMPTS)], 512, extra)
        if r["decode"]:
            c1.append(round(r["decode"], 2))
        if looping(r["text"]):
            res["flags"].append("LOOP")
    res["c1_decode"] = round(statistics.median(c1), 2) if c1 else None
    res["c1_all"] = c1

    for L in args.prefill:
        prompt, _ = document(L, seed=L)
        r = chat(args.base_url, args.model, prompt, 1, extra, temperature=0.0)
        res[f"prefill_{L}"] = round(r["prompt_tokens"] / r["ttft"], 1) if r["prompt_tokens"] else None
        res[f"prefill_{L}_tokens"] = r["prompt_tokens"]

    stop.set()
    res["vram_peak_gib"] = round(peak / 2**30, 2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
