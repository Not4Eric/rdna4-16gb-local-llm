# Optimising Inkling 2 Mini (`GSQ-RCO IQ3_S` + `DSpark`) on a 16 GB RX 9070 XT

## Winner (`configs/inkling2-128k-final.*` = `configs/best-config.json`)

- **Binary:** `<REPO_ROOT>/build/bin/llama-server` (branch `frankenstein-inkling2`, commit `d7f7390ce` + uncommitted F16 in-place `kq_b` memory fix in `src/models/inkling2.cpp` and `src/llama-graph.cpp`).
- **Model:** `<REPO_ROOT>/models/Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf` (`11.425 GB`).
- **Draft model:** `<REPO_ROOT>/models/dspark-Inkling-2-Mini-Q8_0-ista.gguf` (`404 MB`).
- **Port / OpenCode ID:** `http://127.0.0.1:8082/v1` (`inkling2/inkling2-mini`).

```bash
HSA_USERPTR_FOR_PAGED_MEM=0 \
FRANK_SERVER_CACHE_PINNED=0 \
FRANK_SERVER_CKPT_PINNED=0 \
LLAMA_SERVER_SPEC_CKPT_DEVICE=0 \
<REPO_ROOT>/build/bin/llama-server \
  -m <REPO_ROOT>/models/Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf \
  --host 127.0.0.1 --port 8082 -a inkling2-mini --jinja \
  -c 131072 -np 1 -ngl 99 -dev ROCm0 -fa on -fit off -t 6 -b 2048 -ub 128 \
  -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 --spec-draft-kv-window 4096 \
  -ot token_embd.weight=CPU \
  -md <REPO_ROOT>/models/dspark-Inkling-2-Mini-Q8_0-ista.gguf \
  --spec-type draft-dspark,ngram-mod --spec-draft-n-max 3 \
  --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-min 6 --spec-ngram-mod-n-match 12 \
  --spec-draft-sampling probabilistic -lm dio
```

### Summary vs Baseline & Ceiling (6-prompt panel, `reasoning_effort=none`, RX 9070 XT 16 GB)

| Metric | Baseline (`ctl-128k-q4kv-ub128`) | **Winner (`inkling2-128k-final`)** | Ceiling (`THEORY.md`) | Notes |
|---|---:|---:|---:|---|
| Short decode median (tok/s) | 35.66 | **52.67** (52.3–54.9 across controls; code peak **68.4**) | ~125–185 (BW-only) | **+48% to +54%** vs baseline; step time 41.0 ms |
| 119k decode (tok/s) | — | **36.33 / 46.14 / 41.03** (bursts **78.3**) | ~100–160 | 100% VRAM-resident KV, zero PCIe spill |
| Prefill @ 4.9k / 38.9k / 119k (tok/s) | 674.1 / 464.9 / — | **666.7 / 461.0 / 269.7** | ~2,200 / 1,800 / 1,400 | Bottlenecked by `-ub 128` + per-layer `kq_b` gather/permute |
| 119k TTFT (cold / warm) | — | **441.4 s cold / 1.4–1.6 s warm** | — | Prefix cache hits cleanly (`f_keep = 0.996`) |
| 119k needle retrieval | — | **3/3** (rep0 & rep2 @ 118,955 tok = 91% of 128k) | — | Exact retrieval with `q4_0` KV |
| Peak VRAM (GiB / 15.92) | 14.14 | **14.80** (`1.12 GiB` free) | <= 15.6 cap | Fits 100% in VRAM with drafter + 128k KV |
| Cold startup (`-lm dio`) | 5.2 s | **4.1 – 5.1 s** | ~4.5 s | Cache-independent O_DIRECT load |

## Full Experiment Ledger (`ledger.jsonl`)

| Config | `c1_decode` | Prefill 4.9k / 38.9k | VRAM peak | Startup | Verdict & Reason |
|---|---:|---:|---:|---:|---|
| `ctl-128k-q4kv` (`-ub 256`) | — | — | 15.89 GiB | 5.7 s | **REJECT (`MEM_CAP`):** `kq_b` F16 + permuted copy = 3.0 GiB at `-ub 256` |
| `ctl-128k-q4kv-ub128` | 35.66 | 674.1 / 464.9 | 14.14 GiB | 5.2 s | **BASELINE:** clean 128k fit, 28.2 ms/tok |
| `dspark-ista-n3` | 50.98 | 670.4 / 460.6 | 15.02 GiB | 5.7 s | **ACCEPT (+43%):** step 41.0 ms, peaks 69.6 tok/s on code |
| `dspark-ista-n4` | 46.61 | 669.6 / 460.9 | 14.79 GiB | 5.2 s | **REJECT:** step grows to 45.5 ms (MoE expert union) for only +0.04 accepted len |
| `dspark-ista-n5` | 45.17 | 668.9 / 461.0 | 14.80 GiB | 4.6 s | **REJECT:** step 49.2 ms, worse median despite 71.6 code peak |
| `dspark-ista-n6` | 45.58 | 666.6 / 460.8 | 14.80 GiB | 4.6 s | **REJECT:** worse median |
| `dspark-ista-n3-kvw` (`--spec-draft-kv-window 4096`) | 52.30 / 54.87 | 668.0 / 461.1 | 14.78 GiB | 4.6 s | **ACCEPT:** −0.24 GiB VRAM, 52.3–54.9 tok/s across interleaved controls |
| `dspark-ista-n2` | 52.44 | 669.0 / 461.3 | 14.77 GiB | 4.6 s | **TIE:** 37.6 ms/step, slightly higher prose floor but lower code ceiling (56–62 vs 64–70) |
| `dspark-mainline-n3` (BF16 Markov) | 51.94 | 669.1 / 460.7 | 14.87 GiB | 4.6 s | **REJECT:** +0.09 GiB VRAM, slightly slower than `ista` Q8_0 |
| `dspark-ista-n3-greedy` | 51.68 | 667.9 / 460.5 | 14.78 GiB | 4.6 s | **REJECT:** slightly slower than `--spec-draft-sampling probabilistic` |
| `dspark-ista-n3-ngram` | 52.67 | 666.7 / 461.0 | 14.79 GiB | 4.1 s | **ACCEPT (WINNER):** zero short-prompt penalty, boosts code/long-ctx (78 tok/s bursts @ 119k) |
| `dspark-ista-n3-fp8b` (`FRANK_KQ_BLAS=fp8b`) | 51.20 | 676.1 / 463.9 | 14.79 GiB | 4.1 s | **REJECT:** +1% prefill inside noise; Inkling 2 prefill is MoE + `kq_b` bound, not dense K-quant |
| `dspark-ista-n3-q8k` (`-ctk q8_0 -ctv q4_0`) | — | — | — | — | **REJECT:** engine requires identical K and V cache dtypes for Inkling 2 |

## Qualification Gates (`bin/qualify.sh inkling2-128k-final`)
1. **Load:** PASS (`5.1 s` cold via `-lm dio`).
2. **Chat (`reasoning_effort=none`):** PASS (`finish_reason=stop`, `"The capital of France is Paris."`).
3. **Reasoning separation (`reasoning_effort=medium`):** PASS (`reasoning_content` 125 chars, `content = "391"`).
4. **Tool-call round trip:** PASS (`get_vault_code({"vault_id":"V-42"})` → `"ZEBRA-9184"`).
5. **119k Long-context needles & decode:** PASS (`3/3` codes on rep0 & rep2 @ 118,955 tokens, `36.3–46.1 tok/s`, warm TTFT `1.4–1.6 s`, `14.80 GiB` peak VRAM).
6. **Temp-0 determinism:** PASS (`6/6` byte-identical run-to-run).

## Gotchas Discovered & Resolved
1. **Variable-size SWA rollback crash (`ggml_tallocr_alloc`):** With `ngram-mod`, the first short n-gram draft allocates a fixed ~10.2 MiB device staging buffer when `LLAMA_SERVER_SPEC_CKPT_DEVICE` is on; later at 119k when the SWA ring is full, the rollback state grows by ~10 KiB and aborts. **Fix:** `LLAMA_SERVER_SPEC_CKPT_DEVICE=0` (host rollback staging, <0.3 ms overhead, zero crashes).
2. **`kq_b` compute buffer cliff at `-ub 256`:** Inkling 2 materializes a per-head relative-position bias tensor `[n_kv, n_ubatch, 24]`. Even with the local F16 in-place patch, `-ub 256` peaks at `15.89 GiB`; `-ub 128` drops peak VRAM to **`14.80 GiB`** (`1.12 GiB` free).
3. **Why short decode plateaus at ~53–55 tok/s vs 120 tok/s BW-only ceiling:** 48 looped layer visits × ~40 small ops/layer (depth-attention `mix`, 4× `convolve`, 3× `clip`, 4× `control`, `kq_b` gather/permute, `SITU` MoE) plus `FRANK_STEP_LOOP` not yet supporting `draft-dspark` (`step_loop_lo: a drafter besides ngram-mod and DFlash2`). Pushing further requires custom RDNA4 fused kernels for Inkling 2's `kq_b` + `shortconv` + `clip` + `mix` ops and extending `J6` `FRANK_STEP_LOOP` to `draft-dspark`.
