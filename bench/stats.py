#!/usr/bin/env python3
"""Parse per-request prefill, decode, draft acceptance, and step latency from llama-server logs."""
import re, sys

RE_Q = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_E = re.compile(r"eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_A = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")

for path in sys.argv[1:]:
    q = e = acc = None
    n = 0
    print(f"\n== {path}")
    for line in open(path, errors="replace"):
        if m := RE_Q.search(line):
            q = (float(m[1]), int(m[2]))
        elif m := RE_E.search(line):
            e = (float(m[1]), int(m[2]))
        elif m := RE_A.search(line):
            acc = (float(m[1]), int(m[2]), int(m[3]), float(m[4]))
            if e and e[1] > 1:
                n += 1
                tps = 1000 * e[1] / e[0]
                step = e[0] / (e[1] / acc[3])
                pps = 1000 * q[1] / q[0] if q else 0
                print(f"#{n:2d} prefill {q[1] if q else 0:6d} tok {pps:7.1f} tok/s | decode {e[1]:4d} tok {tps:6.2f} tok/s"
                      f" | acc {acc[0]:.3f} ({acc[1]}/{acc[2]}) mean {acc[3]:.2f} step {step:5.1f} ms")
            e = None
        elif e and "eval time" in line and e[1] == 1:
            e = None
