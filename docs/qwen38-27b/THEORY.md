# THEORY — where the time goes on this box

## Decode bytes per token (no speculation)

```
weights                 11.29 GiB   (12.12 GB file, all tensors resident, ngl 99)
KV  @128k q4_0           2.25 GiB   (16 attn blocks × 4 kv heads × 256 × 2 × 0.5625 B/token × 131072)
KV  @96k  q4_0           1.69 GiB
SSM state                 ~0.15 GiB  (48 layers, conv d_conv 4 + 128-dim state, f32)
MTP draft ctx (if on)    0.25 GiB KV + ~0.6 GiB extra compute buffers ⇒ **0.87 GiB measured**
```
Measured: baseline 34.12 tok/s at 8k ⇒ 414 GB/s effective (spec 640 GB/s, ~65 %).
Same bytes ⇒ decode tok/s is set by **weight bytes ÷ achievable bandwidth**, and the only lever that does
not touch quality is *reading the weights once for several accepted tokens*.

## Speculative step model (fitted to measurements)

```
step_ms ≈ 29.3            # one weight sweep, the irreducible part
        + 8.7             # MTP draft forward + h_nextn host copy (independent of n_max)
        + 5.7 × n_max     # each extra verify row costs ~5.7 ms (vs 0.97 ms/row at prefill batch sizes)
tok/s   ≈ 1000 / step_ms × mean_accepted_len
```
Consequences:
- The draft is *nearly free* (8.7 ms of a 29.3 ms floor ⇒ 30 % overhead) but the **verify rows are 6× more
  expensive than prefill rows** — the batch>1 GEMM path here is not reusing weights properly at M=2..7.
- Break-even acceptance: with n_max = 3 the step grows 1.67×, so any mean accepted length > 1.67 wins.
  Observed 2.1–5.4 ⇒ always wins on this hardware, but the *optimum* n_max moves with prompt entropy
  (code 5, prose 3).
- If verify rows cost prefill's 0.97 ms, n_max = 5 would give step ≈ 43 ms and mean 4.8 ⇒ 110 tok/s.
  So the single biggest untapped gain on this box is making the small-batch verify path efficient.

## Why quantised KV matters here (and the skill's warning)

Only 16 of 65 blocks have KV, so KV *storage* is small (2.25 GiB even at 128k q4_0) — but that same
asymmetry means the KV read stream is a small fraction of bytes/token, so **q4_0 KV is unlikely to hurt
decode speed** the way it did on the 16-of-64-layer model in references/gotchas.md (45.4 → 27.0 tok/s);
that case had q8_0 *and* different kernels. Verified: q4_0/q4_0 KV decodes at the same speed as f16 at
short ctx (34.12 vs 33.75 ad-hoc), and needle retrieval at 63k is 3/3.

## Long-context step cost (the second regime)

```
100k ctx, no spec : step 86 ms  (~35 ms of it is attention over 1.9 GiB of q4 KV, ~30 ms weights, rest overhead)
100k ctx, MTP n=3 : step 82-85 ms for mean_len 2.5-3.5  -> 30-41 tok/s
100k ctx, MTP n=6 : step 118 ms (marginal row 12 ms)    -> 27-31 tok/s
100k ctx, MTP n=8 : step 199 ms                          -> 15-22 tok/s (VRAM at 0.12 GiB free)
```
So the *fixed* per-step cost grows 29 → 50 ms from 8k to 100k ctx while a whole spec step stays ~82 ms:
amortising that fixed cost is what makes MTP worth 3× at long context versus 1.6× at short.
KV dequant is NOT the cost: q4_0 vs q8_0 at 64k is identical (36.7/38.9 vs 35.1/38.5 tok/s), so the q4 KV
choice is free and buys ~34k more context. The marginal verify row also gets 2× more expensive at long ctx
(5.7 → 12 ms), which is why n_max=3 wins everywhere.

## Prefill

Cold prefill: 1132 tok/s at 2.8k, 1087 at 21.5k, 933 at 32.8k, 733 at 63k with MTP (no-MTP control
1.6k/…). Large-batch path is compute-bound: 27 B params × 2 FLOP × 21 533 tok / 19.8 s ≈ 59 TFLOPS.
TTFT at 90k is therefore ~90–120 s; nothing here will fix that except lower ctx or fewer prefill tokens,
so long-context UX = prefill cost, not decode cost.

## Memory budget (15.92 GiB usable)

```
weights 11.29 + KV 1.69@96k + buffers ~1.15 + MTP 0.87 = 15.0 GiB used  → 0.9 GiB free
weights 11.29 + KV 2.25@128k + buffers ~1.15          = 14.7 GiB used  → 1.2 GiB free
```
To hold MTP **and** ≥1.5 GiB free at 100k+ we must reclaim ~1.3 GiB: candidates are the buffers (ub 128,
b 256), the MTP draft compute buffer (`--spec-draft-n-ubatch`), or dropping to ~80k ctx. Partial CPU
offload is not viable: each offloaded layer costs ~135 MB of PCIe/DDR per token (~15 ms) and the model has
65 blocks.

## v2: KV streaming over PCIe (2026-11-26)

KV is 64 KiB/token at f16, 34 KiB at q8_0-K/q4_0-V (K 17.4 + V 9.2 KiB across the 16 attention layers).
At 128k that is 3.4 GB — more than the VRAM left after 12.1 GB of weights. The fork keeps all of it in pinned
host RAM and leaves a window on the GPU, so each decoded token streams only the part of the cache that is not
resident.

Per-token PCIe demand ≈ (KV_bytes − resident) × 1 token, plus the attention rows the kernel touches. With
2400 MiB resident out of 3.4 GB at 100k, ~1.0 GB streams per step; at ~30 GB/s effective SDMA that is ~33 ms,
which matches the measured 29 tok/s no-spec decode (34 ms/step). Cutting the resident window to 1500 MiB adds
~250 MB, i.e. ~8 ms/step, and the measurement shows exactly that: 21.9 vs 29.1 tok/s.

This is why the curve is depth-shaped: at 16k the resident window covers the whole cache and decode is pure
bandwidth on the weights (~36 tok/s stock); every doubling past that halves what the window can cover.

Draft verification multiplies the KV footprint of a step (M+1 rows) but not its width, so speculation still wins
at depth — it divides the number of streaming steps by the accepted length. 88 tok/s at 100k with ngram hits is
~4× the no-spec streaming rate.

## v3: the load path is part of the design, not an implementation detail (2026-11-26)

`auto`/`mmap` loading assumes the GGUF is in page cache. At 16 GB it usually is not: the model is 12.1 GB and a
desktop leaves ~10 GB of cache, so the loader faults 4 KB pages faster than reclaim, `read_bytes` climbs at
~25 MB/s while RSS *falls*, and the server never becomes healthy. Measured on this box: O_DIRECT read of the same
bytes at 2.3 GB/s, mmap fault path at 25 MB/s — a 90× gap that is entirely page-fault + reclaim latency, not I/O.

`-lm dio` removes the assumption: weights are streamed straight from the SSD into the upload staging buffers,
sequentially, with no host residency afterwards (4.4 GB RSS vs 10.7 GB for `load-mode none`). Time-to-healthy is
6 s cold and insensitive to what else is cached. `none` is equally fast but evicts other processes' pages on the
way in; `mmap` is only fast when warm, which is exactly the case that hides the defect in benchmarking.

Budget after load (128k, dio): 13.3–13.7 GiB VRAM, 4.4 GB host RSS, KV 3.4 GB pinned + 2400 MiB resident window.

## v4: Fixing the small-batch verify bottleneck without pruning vocabulary (2026-11-27)

In v1 we measured that marginal verify rows (`M = 2..7`) cost 5.7 ms/row vs 0.97 ms/row during prefill because standard MMQ/MMVQ kernels re-stream weights or launch separate kernels per layer, and the 248,320-row `output.weight` LM head (`715 MB` in `Q4_K`/`IQ3_S`) runs twice per speculative step (target verify + MTP draft).

Two orthogonal ways to cut that step time were tested:
1. **GGUF vocabulary pruning (`P2G`, 248,320 → 142,082 tokens)** on `82b3314`: saved `2 × 306 MB = 612 MB` of LM-head reads per step (`~1.2 ms` at 500 GB/s), dropping step time from `41.4–45.4 ms` to `38.0–41.0 ms` (`79.98 tok/s` median), subsequently removed per user request to keep the full 248,320 vocabulary.
2. **Kernel-level speculative verify & prefill specialization (`llama-frankenstein` = `rdna-boosts v18-r359` + `J1–J7`) on the full 248,320 vocabulary**:
   - `FRANK_HEAD_INT2=1` scores all 248,320 draft tokens with a 76 MiB `int2` sketch and exact-recomputes the top 512 rows via `mmvk`, cutting draft LM-head bandwidth by ~85% without removing a single token from the model.
   - `FRANK_MMVK=2` keeps 1–8 token activations in registers across weight rows and fuses sibling projection launches (`wq/wk/wv`, `gate/up`, `GDN in_proj`).
   - `FRANK_STEP_LOOP=1` runs sampling, draft verification, acceptance counting, and next-step input assembly on the GPU without host round-trips.
   - `rdna-boosts` Block 10 fuses `ggml_cuda_quantize_q4_0_group` directly into the RoPE and FWHT kernels for `-ctk q4_0 -ctv q4_0` (`18.0 KiB/cell`, nearly half the footprint of `q8_0r`'s `34.0 KiB/cell`), allowing `77,824` cells (`1,368 MiB`) to reside in VRAM inside a `14.93–15.10 GiB` total VRAM footprint and leaving only `504–864 MiB` of in-place PCIe read at 100k–128k context.
   - `J2` `K1` hipBLASLt FP8 outlier-split GEMM + `J1` `F8` prefill FA + `FRANK_GDN_OWN` chunked WMMA Gated DeltaNet prefill raise 4k prefill from `1237.4 → 1641.7 tok/s (+32.7%)` and 32k prefill from `1210.3 → 1400.5 tok/s (+15.7%)`.
   - Net result on the full 248,320 vocabulary: step time drops to **`37.0–37.9 ms/step`** and median short decode reaches **`80.5–82.5 tok/s`** (peaks `104–108.7 tok/s`), 2.42× the original 34.12 tok/s baseline.
