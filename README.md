# RDNA4 16 GB Local LLM Inference Stack (`Qwen 3.8 27B` + `Inkling 2 Mini`)

Reproducible, 128k-context local LLM serving recipes tuned for an **AMD Radeon RX 9070 XT (`gfx1201`, 16 GB VRAM) + 16 GB DDR5 RAM** on Linux — running **Qwen 3.8 27B (`IQ3_S` + MTP1 + n-gram)** at **80–108 tok/s** and **Inkling 2 Mini (`IQ3_S` + DSpark + n-gram)** at **53–68 tok/s** with full 128k needle retrieval, tool calling, reasoning separation, and 5–6 s cold startup.

---

## Benchmark Summary (RX 9070 XT 16 GB + 16 GB RAM)

All hill-climb decode numbers are medians over a fixed 6-prompt panel (Python LRU cache, TCP congestion control, 600-word story, B-tree vs LSM table, bash backup script, 1929 crash) at `temperature=0.6` with thinking off (`enable_thinking: false` for Qwen; `reasoning_effort: "none"` for Inkling), measured from server `usage` token counts.

| Metric | Qwen 3.8 27B Baseline (stock, no spec) | **Qwen 3.8 27B Tuned (`qwen38-27b`)** | Inkling 2 Mini Baseline (no spec) | **Inkling 2 Mini Tuned (`inkling2-mini`)** |
|---|---:|---:|---:|---:|
| **Short decode median** | 34.12 tok/s | **80.5 – 82.5 tok/s** (peaks **108.7**) | 35.66 tok/s | **52.7 – 54.9 tok/s** (peaks **68.4**) |
| **Long-ctx decode (100k–119k)** | 11.61 tok/s | **42.5 – 61.4 tok/s** (@100k) | — | **36.3 – 46.1 tok/s** (bursts **78.3** @119k) |
| **Prefill @ ~4k / ~32k / ~100k+** | 1132 / 1087 / 701 | **1642 / 1401 / 946 tok/s** | 674 / 465 / — | **667 / 461 / 270 tok/s** (@119k) |
| **Long-ctx TTFT (cold / warm)** | 132 s / — | **105.6 s cold / 3.0 s warm** | — | **441.4 s cold / 1.4–1.6 s warm** |
| **Needle retrieval (100k–119k)** | 3/3 | **3/3** (7/7 across 3 reps @100k) | — | **3/3** (@118,955 tok = 91% of 128k) |
| **Peak VRAM (of 15.92 GiB)** | 14.80 GiB | **14.93 GiB** (`~1.0 GiB` free) | 14.14 GiB | **14.80 GiB** (`1.12 GiB` free, zero KV spill) |
| **Cold startup (`-lm dio`)** | 8.2 s | **6.0 s** | 5.2 s | **4.1 – 5.1 s** |
| **Context window** | 131,072 | **131,072** (`77,824` cells VRAM + pinned spill) | 131,072 | **131,072** (100% VRAM-resident) |
| **Vocabulary** | 248,320 | **248,320** (unpruned) | 200,058 | **200,058** (unpruned) |

### Which Model Should You Run?
- **Qwen 3.8 27B (`./scripts/switch-model.sh qwen`, port `8081`)**: Fastest overall (80–108 tok/s decode, 3–4× faster cold prefill), lower factual hallucination, heavily accelerated by `rdna-boosts` + `llama-Frankenstein` (`mmvk`, `int2` draft head, GPU step loop, FP8 hipBLASLt prefill, fused Gated DeltaNet).
- **Inkling 2 Mini (`./scripts/switch-model.sh inkling`, port `8082`)**: Stronger on multi-turn agentic/coding suites (SWE-bench Verified 79.2 vs 77.8, DeepSWE 48.2 vs 42.2, MCP Atlas 82.7 vs 77.3) with 100% VRAM-resident 128k KV cache (only 8 full-attention slots + 40 SWA slots).

---

## Requirements

- **GPU:** AMD Radeon RDNA4 (`gfx1201`, e.g., RX 9070 XT 16 GB or Radeon AI PRO R9700). Other RDNA3/RDNA4 GPUs can set `AMDGPU_TARGETS=...` before building, though the configs and hipBLASLt priority lists are tuned for `gfx1201` + 16 GB VRAM.
- **RAM:** ≥ 16 GB system RAM (all configs are specifically hardened for 16 GB RAM Linux desktops).
- **OS & Toolchain (Linux):**
  - ROCm / HIP SDK with `hipBLASLt` (`rocm-hip-sdk`, `rocm-cmake`, `hipblaslt` — e.g., on Arch/CachyOS: `sudo pacman -S rocm-hip-sdk rocm-cmake hipblaslt cmake git python`)
  - `curl`, `ss` (`iproute2`), `flock` (`util-linux`), `python3` (stdlib only — no PyTorch or pip packages required)
- **Disk Space:** ~25 GB for both models + ~5 GB for building `llama-server`.

---

## Setup Guide

### 1. Clone This Repository
```bash
git clone https://github.com/Not4Eric/rdna4-16gb-local-llm
cd rdna4-16gb-local-llm
```

### 2. Build the Pinned Inference Engine (`llama-server`)
`scripts/build-engine.sh` checks out the exact pinned commits:
- `ggml-org/llama.cpp` @ `919794500`
- `stew675/llama-cpp-rdna-boosts` @ `v18-919794500-r359` (`a0e7e62`)
- `overdoingism/llama-cpp-overdoing` @ `1ec4b8c` (`J1`–`J7` + `J2` part b)
- Local patches in [`patches/`](patches/):
  - `0001-inkling2-and-dspark.patch` — Inkling 2 Mini (`inkling2_loop`) + Inkling 2 DSpark speculative decoding support
  - `0002-inkling2-f16-kqb-inplace.patch` — F16 `kq_b` gather + in-place `kq_mask` addition (halves 128k compute scratch VRAM so Inkling 2 fits comfortably in 16 GB)

Run the build (~10–15 min depending on CPU cores):
```bash
# Optional overrides: ROCM_PATH=/opt/rocm AMDGPU_TARGETS=gfx1201 JOBS=6
./scripts/build-engine.sh
```
This produces `build/bin/llama-server`.

> **Memory tip when compiling on 16 GB RAM:** If HIP compilation causes swapping, lower parallelism with `JOBS=4 ./scripts/build-engine.sh` (or `JOBS=2`).

### 3. Download Models into `models/`
Create `models/` and download whichever model(s) you want (e.g., via `hf download` / `huggingface-cli` or browser):

```bash
mkdir -p models
```

#### Option A: Qwen 3.8 27B (`IQ3_S`, 12.1 GB)
1. Download the base full-vocab `IQ3_S` GGUF (`12.12 GB`):
   - Repo: `jegreene/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-GGUF`
   - File: `Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp.gguf` → save into `models/`
2. *(Recommended, +3–4% decode speed, 100% lossless)* Transplant the 15 self-distilled `blk.64.*` MTP1 draft tensors from `Safasfly/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-P2-MTP1-GGUF` while keeping all `blk.0..63` weights and the full 248,320 vocabulary untouched:
   ```bash
   python3 scripts/transplant-mtp1.py \
     --base models/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp.gguf \
     --donor /path/to/Safasfly-P2-MTP1.gguf \
     --out models/Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp1.gguf
   ```
   *(If you skip the transplant, `scripts/serve.sh` automatically falls back to `Swift-1.5-Qwen3.8-27B-heretic-GSQ-RCO-IQ3_S-mtp.gguf`.)*

#### Option B: Inkling 2 Mini (`IQ3_S`, 11.4 GB + 404 MB DSpark drafter)
Download both files from `ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF` into `models/`:
- `Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf` (`11.425 GB`)
- `dspark-Inkling-2-Mini-Q8_0-ista.gguf` (`404 MB` — Q8_0 Markov head; faster and smaller than the BF16 mainline head)

> If your GGUFs already live elsewhere, you can point to them without copying via environment variables:
> `MODELS_DIR=/path/to/models`, `QWEN_MODEL=/path/to/qwen.gguf`, `INKLING_MODEL=/path/to/inkling.gguf`, `INKLING_DRAFT_MODEL=/path/to/dspark.gguf`, `LLAMA_SERVER_BIN=/path/to/llama-server`.

---

## Running & Switching Models

Because both models use ~14.8–15.0 GiB of a 16 GB GPU, only one runs at a time. Use `scripts/switch-model.sh` to drain/stop any active server, wait for VRAM to clear, and launch the target model:

```bash
# Start Qwen 3.8 27B on http://127.0.0.1:8081
./scripts/switch-model.sh qwen

# Start Inkling 2 Mini on http://127.0.0.1:8082
./scripts/switch-model.sh inkling

# Show active model, port status, and VRAM usage
./scripts/switch-model.sh status

# Gracefully stop the server and free the GPU
./scripts/switch-model.sh stop
```

---

## OpenCode Configuration

Merge [`configs/opencode.example.json`](configs/opencode.example.json) into `~/.config/opencode/opencode.json`. Each model uses its own port (`:8081` vs `:8082`) so selecting the inactive model in `/models` fails fast rather than silently routing to the wrong model:

- **`llamacpp/qwen38-27b`** → `http://127.0.0.1:8081/v1`
- **`inkling2/inkling2-mini`** → `http://127.0.0.1:8082/v1`

---

## Verifying & Benchmarking Your Setup

Run the 6-gate qualification suite (load, chat, reasoning separation, tool-call round trip, 100k needle retrieval, temp=0 determinism):

```bash
./bench/qualify.sh qwen
./bench/qualify.sh inkling
```

Or run the 6-prompt panel speed benchmark against whichever server is currently up:

```bash
python3 bench/bench.py --base-url http://127.0.0.1:8081 --model qwen38-27b --mode qwen
python3 bench/bench.py --base-url http://127.0.0.1:8082 --model inkling2-mini --mode inkling
```

Parse per-request step time and draft acceptance directly from server logs:

```bash
python3 bench/stats.py logs/qwen38-27b.log
python3 bench/stats.py logs/inkling2-mini.log
```

---

## Critical Gotchas (Already Baked Into `configs/`)

1. **`-lm dio` (O_DIRECT loading) is mandatory on 16 GB RAM Linux.** Default `mmap` page-faults a 12 GB GGUF through a ~10 GB desktop page cache and collapses to ~25 MB/s thrashing (never becoming healthy on a cold boot). `-lm dio` streams sequentially at ~2.3 GB/s for a **4–6 s cold startup** every time without evicting desktop apps.
2. **`FRANK_SERVER_CACHE_PINNED=0` and `FRANK_SERVER_CKPT_PINNED=0` are mandatory on 16 GB RAM.** Without them, `llama-server` pins ~5.7 GiB of non-swappable host RAM for prompt cache and context checkpoints.
3. **`HSA_USERPTR_FOR_PAGED_MEM=0` on Linux ROCm.** Prevents 5–440 ms tail-latency stalls when Linux swaps or migrates pageable pages registered with ROCm.
4. **Keep `--spec-ngram-mod-n-max 15` (not 16) on Qwen 3.8 27B.** `n-max 16` builds a 17-token verify batch (`K=17 > 16`), tripping an assertion in `gated_delta_net.cu` (`rinfo.head16`). `15` keeps the verify batch inside the 16-row HIP graph band.
5. **`LLAMA_SERVER_SPEC_CKPT_DEVICE=0` is mandatory on Inkling 2 Mini.** With `ngram-mod`, the first short n-gram draft allocates a fixed ~10.2 MiB device staging buffer; at 100k+ context when the SWA ring fills up, rollback state grows by ~10 KiB and triggers `ggml_tallocr_alloc` abort. Host staging eliminates the crash at <0.3 ms cost.
6. **Keep `-ub 128` on Inkling 2 Mini at 128k context.** Inkling 2 materializes an `[n_kv, n_ubatch, 24]` relative-position bias tensor (`kq_b`). At `-ub 256`, VRAM peaks at `15.89 GiB` and trips the 16 GB ceiling; `-ub 128` keeps peak VRAM at **`14.80 GiB`**.
7. **Do not set Windows ROCm graph env vars (`DEBUG_HIP_FORCE_GRAPH_QUEUES=1`, `GGML_CUDA_GRAPH_OPT=1`) on Linux ROCm 7.16.** They leak ~150 MiB of HIP graph memory per phase switch on Linux.
8. **Never run `-np 2` on a 16 GB card at 128k context** with these configs — they are sized for single-stream `-np 1`.

---

## Repository Layout

```text
├── configs/                  # Pinned launch args, env vars, best-config metadata, OpenCode snippet
├── patches/                  # Inkling 2 + DSpark + F16 in-place kq_b patches on top of Frankenstein
├── scripts/                  # Engine builder, MTP1 transplanter, serve/stop/switch-model scripts
├── bench/                    # 6-prompt panel benchmark, 100k needle test, determinism & 6-gate qualifier
└── docs/
    ├── qwen38-27b/           # Full experiment log (PLAN.md), roofline math (THEORY.md), ledger.jsonl
    └── inkling2-mini/        # Full experiment log (PLAN.md), roofline math (THEORY.md), ledger.jsonl
```

## Credits & Upstream Projects

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)
- [overdoingism/llama-cpp-overdoing](https://github.com/overdoingism/llama-cpp-overdoing) (Llama-Frankenstein)
- [lnieuwenhuis/rdna4-27b-inference](https://github.com/lnieuwenhuis/rdna4-27b-inference) & `Safasfly` (self-distilled `MTP1` draft head)
- [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) (GSQ-RCO `IQ3_S` quantizations and `DSpark` Q8_0 drafter)

## License

MIT — see [LICENSE](LICENSE). Upstream repositories and model weights remain under their respective licenses.
