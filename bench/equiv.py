#!/usr/bin/env python3
"""Temp=0 determinism check: runs 6 fixed prompts twice and checks byte-for-byte reproducibility."""
import argparse, json, urllib.request

PROMPTS = [
    "Write a Python class implementing a binary min-heap with push, pop and peek, with docstrings.",
    "A train leaves at 14:05 travelling 82 km/h; another at 14:20 going 96 km/h on the same track. When does the second catch the first? Show the arithmetic.",
    "List the first 12 prime numbers and explain in one sentence why 1 is not prime.",
    "Write a haiku about winter, then a limerick about debugging.",
    "Summarise the plot of a detective story set in a lighthouse in exactly four sentences.",
    "What are the trade-offs between B-trees and LSM-trees for write-heavy workloads? Two paragraphs.",
]

def run_once(base_url, model, extra):
    out = {}
    for p in PROMPTS:
        body = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": 300,
                "temperature": 0.0, "cache_prompt": False}
        body.update(extra)
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        out[p] = r["choices"][0]["message"].get("content") or ""
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--mode", choices=["qwen", "inkling"], default="qwen")
    args = ap.parse_args()
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.mode == "qwen" else {
        "chat_template_kwargs": {"reasoning_effort": "none"}}
    a = run_once(args.base_url, args.model, extra)
    b = run_once(args.base_url, args.model, extra)
    same = sum(1 for k in a if a[k] == b[k])
    print(f"DETERMINISM: {same}/{len(a)} prompts byte-identical run-to-run at temp=0")
    assert same == len(a), "Determinism check failed"

if __name__ == "__main__":
    main()
