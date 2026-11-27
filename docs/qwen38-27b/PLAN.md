# Optimising Qwen3.8-27B (heretic, IQ3_S) on a 16 GB RX 9070 XT

Goal (user, m00001/m00055–m00057): **fastest single-stream generation, lossless output, needle-verified at the
working context length, everything resident in the 15.92 GiB VRAM**, tuned for long contexts (100–128k).
Secondary ask: say whether a different model/engine would be a better pick at equal intelligence.

## Winner v2 — KV-streaming fork (2026-11-26)

`configs/fork-128k-final.args` (= `configs/best-config.json`), binary `configs/fork-128k-final.bin`
→ `<KVSTREAM_ROOT>/build/bin/llama-server` (achyudh `feature/adaptive-kv-stream-rocm`, b540a4b,
built locally for gfx1201 with ROCm 7.16 / HIP clang 24).

```
-c 131072 -np 1 -ngl 99 -dev ROCm0 -fa on -fit off -t 6 -b 2048 -ub 512
-ctk q8_0 -ctv q4_0 -ctkd q8_0 -ctvd q4_0
--kv-stream --kv-resident-mib 450 --kv-resident-mib-decode 2400
-ot token_embd.weight=CPU
--spec-type draft-mtp,ngram-mod --spec-rs-replay --spec-draft-n-max 4
--spec-ngram-mod-n-max 16 --spec-ngram-mod-n-min 6 --spec-ngram-mod-n-match 12
--spec-draft-sampling probabilistic
-lm dio
```

| metric | v1 (stock 2.82 + MTP) | **v2 (fork + KV streaming)** |
|---|---|---|
| decode, short prompts (panel median) | 55.6 | **75.9** (90.2/78.3/54.4/73.4/90.0/61.6) |
| decode, 100k prompt, speculative | 29.7 / 36.5 / 41.2 | **88.6 / 65.9 / 96.5** (median 88.6) |
| decode, 100k prompt, **no speculation** | 11.6 | **28.9 / 29.2** (2.5× v1) |
| prefill | 1121 @4.2k, 972 @33k, 701 @100k | **1237 @4.2k, 1210 @33k, 1014 @100k** |
| TTFT at 100k | 149 s cold / 1.4 s warm | **99 s cold / 1.3 s warm** |
| needle @100k / @118k | 3/3 @99.9k | **3/3 @99.9k, 3/3 @117.9k** |
| max context | 106 496 (MTP does not fit at 131072) | **131 072** (user caps use at 128k; 262k retired) |
| VRAM peak | 15.41 GiB | **14.59 GiB** (1.33 free) |
| host RAM | 10.7 GB RSS (`--load-mode none`) | **4.4 GB RSS** (`-lm dio`) |
| startup | 5.0 s (8.2 cold) | **6 s cold**, cache-independent |

Why it works: the full KV cache lives in pinned host RAM (3.4 GB at 128k, q8_0 K + q4_0 V) and only a bounded
window is kept on the GPU (450 MiB during prefill, 2400 MiB during decode), streamed over PCIe. The VRAM the KV
used to hold is free, the model still runs entirely on the GPU, and context is no longer capped by KV size.

**Residency directly buys decode speed** (clean isolation, speculation off, 100k prompt, 300 tokens):
1500 MiB resident → 21.86 / 21.90 tok/s; 2400 MiB resident → 28.92 / 29.21 tok/s (+33 %). With speculation on the
same effect is there but is masked by n-gram luck (see noise note below).

**Load mode — `-lm dio` is mandatory (2026-11-26, bug found in production).** The default `auto`/`mmap` mode
faults the 12.1 GB GGUF through the page cache one 4 KB page at a time. When the file is *cold* and the box is
carrying a desktop (only ~10 GB of cache available vs a 12.1 GB model) that path collapses to ~25 MB/s and
thrashes: a real launch sat at 9.4/12.1 GB for 7.5 min with RSS *falling*, and never became healthy — OpenCode
reported "Failed to load model" because `/health` stayed 503. Observations from that stall:

- the same bytes read at **2.3 GB/s** with O_DIRECT while mmap crawled at 25 MB/s, so the SSD was never the issue
- every earlier fast load (3.5–8.5 s) was a *warm* load; the warm number hid the defect
- `--load-mode none` (`--no-mmap`) also fixes it (6 s) but reads through the page cache and evicts other apps' pages

Three-way test, identical config, only `-lm` changed (cold where applicable): `dio` 6 s / 75.1 tok/s / 13.3 GiB,
`none` 6 s / 73.1 tok/s / 13.7 GiB, `mmap` warm 8 s / 74.8 tok/s / 13.7 GiB, `mmap` cold **never**. All three
produce identical output and acceptance (0.618, mean len 3.47) — it is purely a load-path question. `dio` reads
with O_DIRECT in big sequential blocks, so it needs no host copy and no page cache: 6 s cold, every time.

**262k profile retired** (2026-11-26): the user caps real use at 128k because quality degrades beyond it, and the
262k profile cost 7 GB of pinned KV (its `configs/fork-256k.*` files moved to `configs/retired/`). It did work —
needle 3/3 at 199 898 tokens, prefill 832, decode 22.5 — but 22 tok/s at 200k vs 58–88 at 100–118k was never
going to be worth the RAM.

**Noise note:** long-context decode on this stack is bimodal — answers that revisit document text get long
`ngram-mod` hits and jump from ~55 to ~90 tok/s. Medians of 3–4 reps are the honest number; single reps swing ±40 %.

**Rebuild** (needs the ROCm/HIP toolchain, ~10 min on 6 cores):
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16,q8_0-q4_0" -DGGML_BACKEND_DL=OFF \
  -DGGML_CCACHE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build -j6 --target llama-server
```
Gotcha: `bin/serve.sh` must NOT put the LM Studio backend dir on `LD_LIBRARY_PATH` for the fork binary — it
shadows the fork's own `libllama.so`/`libggml.so` and the process dies with a core dump and an empty log.

## Winner v1 — superseded

`configs/mtp3-104k-ub512.args` (stock llama.cpp 2.82.0 + MTP only)

```
-c 106496 -np 1 -ngl 99 -fa on -t 6 -b 2048 -ub 512 -ctk q4_0 -ctv q4_0
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64
```

| metric | value |
|---|---|
| decode, short prompts | **55.6 tok/s** median (panel: 70.8 / 56.7 / 42.7 / 51.8 / 65.0 / 47.4) |
| decode, 100k-token prompt | **29.7 / 36.5 / 41.2 tok/s** |
| decode, 100k prompt, **no speculation** | 11.6 tok/s → MTP is a **3.0× win at long context** |
| prefill | 1121 tok/s @4.2k, 972 @33k, 701 @100k |
| TTFT at 100k | 149 s cold, **1.4 s warm** (prefix reuse) |
| needle retrieval @99 887 tokens | **3/3**, cold and warm, 400-token answers still coherent |
| VRAM peak | 15.41 GiB of 15.92 → 0.51 GiB free |
| startup | 5.0 s |

## Results table (bench.py panel, 6 samples, median; ctx 98304 unless noted)

| config | decode tok/s | prefill tok/s 4.2k / 33k | VRAM GiB | verdict |
|---|---|---|---|---|
| ctl-128k-q4kv (ub 512) | 34.12 (±0.07) | 1132 / 1087* | 14.8 | control, no spec |
| ctl-96k (ub 256) | 34.05 | 1066 / 914 | 13.86 | control |
| ctl-104k-ub512 | 33.99 | 1157 / 1022 | 14.11 | control |
| **mtp3-96k (ub 256)** | 54.20 | 1009 / 866 | 15.07 | MTP works |
| mtp4-96k | 54.52 | 1004 / 866 | 15.27 | equal to n3 |
| mtp5-96k | 50.83 | 979 / 850 | 15.35 | worse |
| mtp3-96k-ub512 | 53.16 | 1127 / 976 | 15.18 | **ub 512 accepted** (+12 % prefill) |
| **mtp3-104k-ub512** | **55.56** | 1121 / 972 | 15.42 | **winner** |
| mtp3-104k-du8 | 57.05 | 973 / 854 | 15.33 | reject: prefill −13 % |
| mtp3-104k-du4 | 54.19 | 913 / 799 | 15.33 | reject: prefill −18 % |
| ngram-128k | 33.95 | 1053 / 909 | 14.66 | reject: no gain |
| ngram-map-k4v + draft-mtp | 53.42 | 1006 / 866 | 15.23 | reject: no gain over MTP alone |

\*128k control prefill numbers are at 2.8k/21.5k tokens (measured before the document-generator fix).

Long-context behaviour (100k prompt, 3–4 reps, 400-token answers, temperature 0):

| config | cold decode | warm decode | step ms | acceptance | mean len |
|---|---|---|---|---|---|
| no speculation | 11.61 | – | 86 | – | 1.00 |
| **MTP n_max 3** | **29.65** | **36.5 / 41.2** | 82–85 | 0.50–0.84 | 2.5–3.5 |
| MTP n_max 6 | 27.35 | 30.7 | 118 | 0.377 | 3.24 |
| MTP n_max 8 | 14.76 | – | 199 | 0.246 | 2.95 |

KV type A/B at 64k ctx with MTP: q4_0 36.7/38.9 vs q8_0 35.1/38.5 tok/s → **q4_0 KV is free here** (this model
has KV in only 16 of 65 blocks, so the dequant tax the skill warns about does not show up), and it buys ~34k
more context. Memory: 14.17 vs 15.15 GiB.

## What the numbers mean

- Weights = 11.29 GiB and a token needs one full weight sweep, so **no-spec decode is capped at ~41–45 tok/s**
  (measured 414 GB/s effective of 640 GB/s spec): 34 tok/s is 80 % of that ceiling.
- MTP step cost fits `29.3 ms (weights) + 8.7 ms (draft) + ~5.7 ms × n_max (verify rows)` at short ctx.
  `tok/s ≈ 20.4 × mean_accepted_len`. Code prompts accept 4.4–4.8 tokens/step (80 tok/s), prose 2.5–3.5.
- **At 100k the no-spec step costs 86 ms** (attention over 1.9 GiB of q4 KV dominates) while an MTP step costs
  only 82 ms for ~3 tokens — speculation amortises the long-context attention cost, which is why the long-ctx
  win (3.0×) is bigger than the short-ctx win (1.6×).
- n_max above 3 backfires at long context: each extra verify row costs ~12 ms there (vs 5.7 at short ctx),
  acceptance drops, and the extra graph pushes VRAM to 15.8 GiB (0.12 free) where decode collapses.
- Prefill is compute-bound (~50–59 TFLOPS): 100k TTFT ≈ 2.5 min cold, 1.4 s on a warm prefix. Physics, not a flag.

## Fidelity

`equiv.py`, 6 fixed prompts, temperature 0, 300 tokens: MTP vs no-MTP differ on 3/6 replies, but **MTP is
deterministic run-to-run (6/6 identical)** — the divergence is argmax tie-breaking under a different FP
reduction order in the 4-row verify GEMM (e.g. "swore he'd never code again" → "blamed it on the network
again"; both coherent, same length class). Distribution is unchanged; quality gate is the needle/task test,
which passes.

## Gotchas recorded

- **MTP costs 0.87 GiB at 8k, 1.31 GiB at 104k.** MTP + 131072 ctx = 15.8 GiB used and the first request dies
  with `rocblaslt error: Cannot read "TensileLibrary_lazy_gfx1201.dat"` — a *symptom of VRAM exhaustion*
  (the .dat exists in the vendor dir). Same class of failure whenever free VRAM < ~0.15 GiB.
- `-ctkd q4_0 -ctvd q4_0` does not shrink the MTP draft KV.
- Must pass `-np 1` (default is 4 slots × 8192 ctx).
- `-ngl 99` disables llama.cpp's auto-fit, so nothing protects against over-allocation.
- `stop.sh` waits for VRAM to drop below 1 GiB before returning — launching into a dying server's VRAM was the
  original cause of the bogus "missing Tensile library" crash.
- Prefill at long ctx drops (1121 → 701 tok/s at 100k) because attention is quadratic; ub 512 beats ub 256 by
  12 %; shrinking the draft ubatch to 8/4 costs 13–18 % prefill for 0.09 GiB.

## Answer to "is another model/engine a better pick at this intelligence?"

No, not on this machine.
- The only on-disk alternatives are MoE models of 35–58 GB (Flash-Next TQ1_0 58.5 GB, Strata packs 26/35 GB)
  that must page experts from disk; the user's own Strata logs show **7.0–10.5 tok/s** decode plus an HMM
  page-fault deadlock (`svm_range_restore_work → amdgpu_hmm_range_get_pages`). This 27B dense at IQ3_S does
  **35–55 tok/s** fully resident, with 3/3 needle at 100k.
- vLLM/SGLang/TensorRT-LLM are not installable here (no python torch, no docker) and are not faster for a
  single stream on one RDNA4 card; llama.cpp is the only stack with working MTP for `qwen35`.
- The 15.92 GiB budget caps a dense model at ~27–30 B at Q3/IQ3 (this file) or ~14 B at Q6. Anything bigger
  has to page from disk and loses 5–10× throughput, so the IQ3_S 27B is the right point on the curve — the
  remaining speed came from speculation, not from swapping models.

## Left on the table (documented, not worth it now)

- **The collectable items are now done** (v2): KV streaming, 131072/262144 context, `ngram-mod`, 2400 MiB
  resident decode window, `q8_0` K, mmap instead of `--load-mode none`.
- Tried and rejected in v2: `--spec-ngram-mod-n-match 24` (69.92 vs 69.45 tok/s — inside noise, keep 12).
- Untidy leak: `-ot token_embd.weight=CPU` frees ~0.34 GiB of VRAM but costs nothing measurable; `output.weight`
  on CPU would free the same again but is a per-token vocab matmul — do not.
- Small-batch verify rows cost 5.7–12 ms vs 0.97 ms at prefill batch sizes; a better MMQ/GEMV path for
  M = 2..7 would be worth +50–80 % decode, but that is a llama.cpp HIP kernel change, not a config.
- `--kv-resident-mib-decode` above 2400 at 128k pushes VRAM past 14.6 GiB and eats the 1.3 GiB safety margin;
  the 262k profile has 2420 and still peaks at 14.86 GiB.
- Prefill is 1014 tok/s at 100k — a 100k cold TTFT of 99 s is the floor without a smarter chunked prefill.
- Building `stew675/llama-cpp-rdna-boosts` on top of the fork (q4_0 KV fold, +10.8 % tg on q4_0-KV models)
  could stack, but the fork already uses q8_0 K and its own KV kernels; unproven, not attempted.

## Winner v4 — llama-Frankenstein (`rdna-boosts v18-r359` + `J1–J7`) + Safasfly MTP1 transplant, full 248,320 vocabulary (2026-11-27)

Built `<REPO_ROOT>/build/bin/llama-server` (`llama.cpp` `919794500` + `stew675/llama-cpp-rdna-boosts` `v18-919794500-r359` + `overdoingism/llama-cpp-overdoing` `J1–J7`, commit `345fc9b`, with `-DFRANK_HIPBLASLT=ON`) and transplanted the 15 self-distilled `blk.64.*` tensors from `Safasfly/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-P2-MTP1-GGUF` into the unpruned 248,320-vocab Heretic GGUF (`<REPO_ROOT>/models/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp1.gguf`, all 15 tensors SHA-256 verified, 0 non-MTP tensors touched, full 248,320 vocabulary retained).

### Production command (`configs/fork-128k-final.*` / `configs/frank-128k-q40.*`)

```bash
HSA_USERPTR_FOR_PAGED_MEM=0 \
FRANK_SERVER_CACHE_PINNED=0 \
FRANK_SERVER_CKPT_PINNED=0 \
<REPO_ROOT>/build/bin/llama-server \
  -m <REPO_ROOT>/models/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp1.gguf \
  --host 127.0.0.1 --port 8081 -a qwen38-27b --jinja \
  -c 131072 -np 1 -ngl 99 -dev ROCm0 -fa on -fit off -t 6 -b 2048 -ub 512 \
  -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 \
  --kv-window 77824 --spec-draft-kv-window 8192 \
  -ot token_embd.weight=CPU \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 \
  --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 6 --spec-ngram-mod-n-match 12 \
  --spec-draft-sampling probabilistic -lm dio
```

### Progression across all 4 versions (same 6-prompt panel, RX 9070 XT 16 GB)

| Metric | Stock base (`ctl-128k`) | v1 Stock + MTP (`mtp3-104k`) | v2/v3 KV-stream (`b540a4b`) | Option A+B (`82b3314` + MTP1) | **v4 Winner (`frank-128k-q40` + MTP1, full 248k vocab)** |
|---|---:|---:|---:|---:|---:|
| Short decode median (tok/s) | 34.12 | 55.56 | 75.85 | 73.67 (vs 71.25 orig) | **82.53** (w102k) / **80.50** (w74k, up to **108.72**) |
| Per-step time (ms, n_max=4) | 29.3 (1 tok) | ~55 | 41.4–45.4 | 41.0–44.1 | **37.0–37.9** (−12 to −18%) |
| Prefill ~4k (tok/s) | 1131.6 | 1121.0 | 1237.4 | 1250.1 | **1641.7** (+32.7% vs v2) |
| Prefill ~32k (tok/s) | 1086.7 (@21k) | 972.0 | 1210.3 | 1203.9 | **1400.5** (+15.7% vs v2) |
| 100k decode (tok/s) | 11.61 | 29.7–41.2 | 65.9–96.5 | 55.9 / 59.2 / 79.9 | **48.9 / 52.2 / 78.6** (w86k) · **42.5 / 50.9 / 61.4** (w74k) |
| 100k needle retrieval | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** (7/7 across all 3 reps at w74k) |
| Max context | 131,072 | 106,496 | 131,072 | 131,072 | **131,072** (77,824 cells in VRAM + pinned spill) |
| Peak VRAM (GiB / 15.92) | 14.80 | 15.41 | 14.59 | 14.43 | **14.93–15.10** (~0.9–1.0 GiB free) |
| Cold startup (`-lm dio`) | 8.2 s | 41–47 s | 6.0 s | 6.0 s | **6.0–7.0 s** |

### Key findings from Options A + B + C

- **Why `llama-frankenstein` matches/beats vocabulary pruning *without* pruning the vocabulary**: on `82b3314`, `P2G` vocabulary pruning (248,320 → 142,082 tokens) lowered step time from 41.4–45.4 ms to 38.0–41.0 ms (79.98 tok/s) by shrinking `output.weight` from 715 MB to 409 MB. We removed vocabulary pruning per user request. On the full 248,320 vocabulary, `llama-frankenstein` lowers step time even further to **37.0–37.9 ms (80.5–82.5 tok/s)** because `FRANK_HEAD_INT2=1` scores all 248,320 draft tokens using an in-VRAM `int2` sketch (~76 MiB) and exact-recomputes the top 512 rows via `mmvk`, while `FRANK_MMVK=2` keeps 1–8 token activations in registers across weight rows and `FRANK_STEP_LOOP=1` runs sampling + draft acceptance on the GPU.
- **Safasfly `MTP1` self-distilled `blk.64.*` head**: clean A/B on `82b3314` with the full 248,320 vocabulary (`fork-82b3-orig-kvw` vs `fork-82b3-mtp1-kvw`) improved median short decode from **71.25 → 73.67 tok/s (+3.4%)** and on the P2G test improved total accepted/generated draft tokens from **49.63% → 51.17% (77.98 → 79.98 tok/s)** at zero VRAM cost and 100% target verification fidelity.
- **Traps discovered and resolved**:
  1. Windows-ROCm env vars `DEBUG_HIP_FORCE_ASYNC_QUEUE=1`, `DEBUG_HIP_FORCE_GRAPH_QUEUES=1`, and `GGML_CUDA_GRAPH_OPT=1` leak ~150 MiB of HIP graph memory per phase switch on Linux ROCm 7.16 and slow 4k prefill from 1237 → 1005 tok/s. Stripped.
  2. In `llama-frankenstein`, `--spec-ngram-mod-n-max 16` creates a 17-token verify batch (`K=17 > 16`), tripping `GGML_ASSERT(n_tokens_limit == rinfo.head16)` in `gated_delta_net.cu:749` during warmup (`rinfo.head16` assumes `D=16`). Setting `--spec-ngram-mod-n-max 15` (`K=16`) fixes the assertion and keeps the entire verify batch inside the 16-row HIP graph and `J1` band-split verify kernel.
  3. On a 16 GB RAM Linux system, `FRANK_SERVER_CACHE_PINNED=0` and `FRANK_SERVER_CKPT_PINNED=0` must be set so prompt cache and context checkpoints use normal pageable RAM instead of pinning 5.7 GiB of non-swappable host memory.
