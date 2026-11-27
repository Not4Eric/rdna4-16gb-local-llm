#!/usr/bin/env python3
"""Long-context needle retrieval & decode benchmark (cold + warm prefix reuse)."""
import argparse, os, subprocess, threading
import bench

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--mode", choices=["qwen", "inkling"], default="qwen")
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--gen", type=int, default=400)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.mode == "qwen" else {
        "chat_template_kwargs": {"reasoning_effort": "none"}}
    doc, codes = bench.document(args.tokens, seed=11)
    qs = [
        ("List every secret code number and its value. Then write a detailed 350-word analysis of the document's "
         "vocabulary, mentioning each code again as you go.", codes),
        ("Which secret code appears closest to the middle of the document? Explain your reasoning at length "
         "(about 350 words), quoting surrounding words, and restate the code three times.", [codes[1]]),
        ("Write a 350-word summary of the document, and make sure the summary contains every secret code exactly once.", codes),
    ]

    peak = 0
    stop = threading.Event()
    def watch():
        nonlocal peak
        probe = os.path.join(REPO_ROOT, "scripts", "mem_probe.sh")
        while not stop.wait(1.0):
            try:
                b = int(subprocess.run([probe], capture_output=True, text=True, timeout=10).stdout.strip() or 0)
                peak = max(peak, b)
            except Exception:
                pass
    threading.Thread(target=watch, daemon=True).start()

    for i in range(args.reps):
        q, want = qs[i % len(qs)]
        r = bench.chat(args.base_url, args.model, doc + "\n\n" + q, args.gen, extra, temperature=0.0)
        hits = sum(c in r["text"] for c in want)
        pt = r["prompt_tokens"] or 0
        pps = pt / r["ttft"] if r["ttft"] > 0 else 0
        dec = r["decode"] or 0.0
        print(f"rep{i}: prompt {pt} tok | ttft {r['ttft']:.1f} s ({pps:.0f} tok/s) | "
              f"decode {dec:.2f} tok/s over {r['completion_tokens']} tok | codes {hits}/{len(want)} | "
              f"vram_peak {peak/2**30:.2f} GiB", flush=True)
    stop.set()

if __name__ == "__main__":
    main()
