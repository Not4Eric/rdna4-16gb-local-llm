# THEORY — Inkling 2 Mini (`GSQ-RCO IQ3_S` + `DSpark`) on RX 9070 XT 16 GB

## Hardware (measured on this machine, see `../qwen38-27b/THEORY.md`)
- **GPU:** AMD Radeon RX 9070 XT (`gfx1201`, RDNA4), `15.92 GiB` usable VRAM (`17,095,983,104` B; ~1.4 GiB held by desktop/Steam compositor when active).
- **Measured VRAM bandwidth:** `414 GB/s` sustained full-model decode (`500 GB/s` DtoD copy; `640 GB/s` spec).
- **Measured compute:** `~59 TFLOPS` FP16/BF16 GEMM (`~30 TFLOPS` practical 50% ceiling for mixed grouped-MoE + dense prefill).
- **System RAM:** `16 GB` DDR5 (`-lm dio` O_DIRECT loading mandatory to avoid page-cache thrash; `FRANK_SERVER_CACHE_PINNED=0` and `FRANK_SERVER_CKPT_PINNED=0` mandatory to avoid pinning host RAM).

## Model architecture & weight accounting (`Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf`, `11.425 GB` / `10.641 GiB`)
- **Architecture:** `inkling2_loop` — 24 physical blocks (`0..5` prelude, `6..17` looped core × `3` passes, `18..23` coda) = **48 layer visits per token** (`n_slot = 48`).
- **MoE:** `128` experts per physical block, `top-8` active (`6.25%` per single token), `n_embd = 2304`, `n_ff_exp = 1152`.
- **Attention:** `24` query heads, `8` KV heads, `head_dim = 128`, relative-position bias (`n_rel = 16`, `extent = 1024`), short convolutions (`l_cache = 4`).
- **Sliding-window pattern:** `[T, T, T, T, T, F] × 4` → across 48 visits, **40 visits are SWA (`window = 1024`)** and **only 8 visits are full attention** (`blk.5`, `blk.11`×3, `blk.17`×3, `blk.23`).
- **Vocabulary:** `200,058` unpadded (`201,024` padded).

| Component | On-disk / Resident Size | Notes |
|---|---:|---|
| `token_embd.weight` (BF16) | `926.3 MB` (`883.4 MiB`) | Offloaded to CPU (`-ot token_embd.weight=CPU`), 0 decode GPU bandwidth |
| `output.weight` (LM head) | `318.4 MB` (`303.7 MiB`) | Resident on GPU |
| Dense attention + norms + convs (24 phys) | `849.9 MB` (`810.6 MiB`) | Visited `2.0×` across 48 visits (`1,699.9 MB` read/step) |
| Loop controllers (12 core blocks) | `113.7 MB` (`108.4 MiB`) | Active in passes 2 & 3 (`227.4 MB` read/step) |
| Routed experts (24 phys × 128 exps) | `9,523.3 MB` (`9,082.1 MiB`) | Virtual 48-visit pool = `19,046.6 MB`; `8/128` active = `1,190.4 MB`/tok |
| **Total GPU target weights (`token_embd` on CPU)** | **`10,499 MB` (`9.78 GiB`)** | Fits comfortably inside `15.92 GiB` VRAM |
| **DSpark drafter (`ista` Q8_0 vs `mainline`)** | **`404.1 MB` (`0.38 GiB`)** / `503.9 MB` | 3 dense layers + rank-256 semi-autoregressive Markov RNN head (`block_size = 8`) |

## KV cache & scratch budget at `131,072` context
- **Full-attention KV (only 8 slots × 8 KV heads × 128 K/V):**
  - `f16`: `32.0 KiB/tok` → `4.00 GiB` @ 128k
  - `q8_0` K + `q8_0` V: `17.0 KiB/tok` → `2.125 GiB` @ 128k (+ `~85 MiB` for 40 SWA slots)
  - `q8_0` K + `q4_0` V: `13.0 KiB/tok` → `1.664 GiB` @ 128k
  - `q4_0` K + `q4_0` V: `9.0 KiB/tok` → `1.125 GiB` @ 128k (+ `~45 MiB` for 40 SWA slots)
- **Recurrent shortconv state:** `3 × 6,656 × 48 × 4 B ≈ 3.8 MiB/slot`.
- **Relative-position `kq_b` scratch buffer (`[n_kv, n_ubatch, 24]` in F16):**
  - At `n_ctx = 131,072`: `131072 × n_ubatch × 24 × 2 B` = **`0.75 GiB` per live copy at `-ub 128`**, **`1.50 GiB` at `-ub 256`**, **`3.00 GiB` at `-ub 512`**.
  - Because `kq_b` and its permuted copy coexist during `ggml_cont(ggml_permute(...))`, `-ub` directly controls peak compute-buffer VRAM at 128k!
- **Key structural advantage over Qwen 3.8 27B:** The entire 131,072-token KV cache + target weights + DSpark drafter fit **100% in VRAM** with zero PCIe KV spilling!

## Theoretical ceilings (`RX 9070 XT` @ `414 GB/s` sustained)

### 1. Single-stream decode (`c1`)
- **No speculation (`m = 1`):**
  - `B_target = 1.190 (experts) + 1.700 (dense) + 0.227 (loop ctrl) + 0.318 (LM head) = 3.436 GB/tok`.
  - Pure bandwidth ceiling: `414 / 3.436 = 120.5 tok/s` (`145.5 tok/s` @ `500 GB/s`).
  - Including ~3–5 ms launch/routing/shortconv overhead across 48 visits (`8.3 ms` BW + `3–5 ms` overhead = `11.3–13.3 ms`): **practical no-spec ceiling ~75–110 tok/s**.
- **With DSpark (`k` drafts, `m = k + 1` verify tokens, MoE expert union `1 − (15/16)^m`):**
  - `B_verify(m) = 2.245 GB + 19.047 GB × (1 − (15/16)^m)`
  - `B_draft(k, ista) ≈ 0.613 GB (3-layer block + shared LM head) + 0.054 GB × k (Q8_0 markov_w2)`

| Mode | `m = k+1` | Expert union | `B_step` (`ista`) | BW floor @ 414 GB/s | Practical ceiling (`a = 2.8 .. 4.5`) |
|---|---:|---:|---:|---:|---:|
| No speculation | 1 | 6.25% | 3.44 GB | 8.3 ms | **85 – 120 tok/s** |
| DSpark `k = 3` | 4 | 22.75% | 7.35 GB | 17.8 ms | **125 – 185 tok/s** (`a = 2.8–3.5`) |
| DSpark `k = 4` | 5 | 27.58% | 8.33 GB | 20.1 ms | **135 – 195 tok/s** (`a = 3.2–4.2`) |
| DSpark `k = 5` | 6 | 32.11% | 9.24 GB | 22.3 ms | **140 – 200 tok/s** (`a = 3.6–4.6`) |
| DSpark `k = 7` | 8 | 40.33% | 10.92 GB | 26.4 ms | Wins only on high-acceptance code (`a > 5.0`) |

- **At 100k context:** Full-attention KV read per step is only `1.0–1.5 GB` **inside 414 GB/s GDDR6** (`+2.5–3.6 ms/step`, amortized across `a` accepted tokens), so 100k decode ceiling stays **~100–160 tok/s**!

### 2. Prefill
- Active linear FLOPs/token: `2 × (1.30B dense + 0.06B loop + 3.06B active MoE) ≈ 8.84 GFLOP/tok`.
- Attention FLOPs/token: `0.50 GFLOP` (40 SWA @ 1024) + `4.92e-5 × L GFLOP` (8 full-attn @ mean pos `L/2`) → `9.5 GFLOP` @ 4k, `11.0 GFLOP` @ 32k, `14.3 GFLOP` @ 100k.
- At ~25–35 TFLOPS practical grouped-MoE + dense + rel-pos bias throughput: **~2,200–3,600 tok/s @ 4k**, **~1,800–3,000 tok/s @ 32k**, **~1,400–2,400 tok/s @ 100k**.
