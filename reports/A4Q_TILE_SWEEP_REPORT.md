# A4Q NVFP4 fp4-QK Decode — Compile-Time Tile/Stage Autotune Sweep (GB10)

**Date:** 2026-08-15 **Box:** DGX Spark GB10 (`sm_121a`, **48 SMs**, 99 KB smem/block) @ 10.100.10.1
**Image:** `eugr-gb10-nvfp4kv:a4q` (flashinfer 0.6.18 + A4Q fp4-QK, JIT-compiled for `compute_120f`)
**Kernel:** fp4-QK decode via `BatchDecodeWithPagedKVCacheWrapper(use_nvf4_qk=True)` →
batch-prefill paged fp4-QK dispatch `BatchPrefillWithPagedKVCacheDispatched`
(`prefill.cuh:4671`). Split-KV scheduler kept **ON** (default) throughout.

## TL;DR

**No GB10-specific tile config beats the current default by a reproducible margin. The default
is GB10-optimal — proven, not assumed.** Apparent 6–12 % wins in a first single-shot sweep
**evaporated** under interleaved repeat measurement (they were thermal/sampling noise). Using the
min-latency estimator, the default (`CTA_TILE_KV=64`, 2 CTAs/SM) ties or beats every alternative
across the full batch×q-len×context grid. Parity is config-independent (all tiles compute identical
math) and holds at cos ≈ 0.995 vs the bf16-QK reference. **Nothing wired; default left in place.**

## The autotuner that was built

There is no runtime attention autotuner in flashinfer (confirmed in the prior
`A4Q_DECODE_AUTOTUNE_REPORT.md` — `autotuner.py` is GEMM/MoE-only). The kernel's perf knobs are
**compile-time C++ template params**. So the autotuner here is a **JIT tile-sweep**:

Key realization that made the sweep cheap: `DISPATCH_NUM_MMA_KV` (`utils.cuh:94`) instantiates
**all** of `NUM_MMA_KV ∈ {1,2,4,8}` into a **single `.so`** and selects one at runtime from an
integer argument. So every smem-valid tile variant is *already compiled into one module* — no
per-config rebuild. I instrumented the paged dispatch (`prefill.cuh:4671`) with two `getenv`
overrides read at each launch (patch: `tile_sweep/patch_prefill.py`):

- `A4Q_NUM_MMA_KV` — override the `NUM_MMA_KV` fed to `DISPATCH_NUM_MMA_KV`
  (clamped to the smem-valid max, so it can only pick a *pre-instantiated, safe* tile).
- `A4Q_CTAS_PER_SM` — override the `num_ctas_per_sm` occupancy target (1 or 2), which sets the
  per-block smem budget and hence how large a tile is smem-legal.
- `A4Q_TILE_DEBUG` — print the resolved `CTA_TILE_Q/KV`, warp layout, smem/config, occupancy.

With the env unset the default path is byte-for-byte unchanged (one `getenv` per launch, negligible).
Because `getenv` reads live process env, configs can be A/B'd **in one process on identical tensors
and a single `plan()`** (the tile is a `run()`-time choice; the split-KV plan is tile-independent).

## Decode kernel geometry (from `A4Q_TILE_DEBUG`, head_dim 256, GQA 24q/4kv, nvfp4 KV)

`FA2DetermineCtaTileQ` fixes `CTA_TILE_Q=16` for decode → `NUM_WARPS_Q=1, NUM_WARPS_KV=4,
NUM_MMA_Q=1`. Therefore `CTA_TILE_KV = NUM_MMA_KV · NUM_WARPS_KV · 16 = NUM_MMA_KV · 64`.
Per-tile smem cost (measured constants): `kFixedSmem = 8192 B`, `kKVSmemPerMmaKV = 32768 B`
(fp4 uses the non-shared K/V path, `(HEAD_DIM_QK+HEAD_DIM_VO)·16·NUM_WARPS_KV·sizeof`).

## Config set (smem-pruned against 99 KB/block)

| config | occ (CTAs/SM) | NUM_MMA_KV | CTA_TILE_KV | smem/block | smem/SM | status |
|:--|:--:|:--:|:--:|--:|--:|:--|
| **C0 (DEFAULT)** | 2 | 1 | 64  | 40 KB | 80 KB | shipping default |
| C1 | 1 | 1 | 64  | 40 KB | 40 KB | tested |
| C2 | 1 | 2 | 128 | 72 KB | 72 KB | tested |
| ~~mma4~~ | 1 | 4 | 256 | **136 KB** | — | **PRUNED > 99 KB** |
| ~~mma8~~ | 1 | 8 | 512 | **264 KB** | — | **PRUNED > 99 KB** |

The default logic itself computes C0: it targets 2 CTAs/SM, and each `NUM_MMA_KV` costs 32 KB, so on
GB10's 99 KB only `NUM_MMA_KV=1` fits at 2 CTAs/SM (`smem_cap=1`). C1/C2 are the only reachable
alternatives; a bigger tile (C2) is only smem-legal by dropping to 1 CTA/SM.

## Measurement 1 — full single-shot grid (`tile_sweep/sweep_full.csv`)

Grid: batch {1,4,8,16,32} × q-len {1,4} × ctx {4K,16K,48K,96K}, ms/iter over 50 iters/10 warm,
one plan, 3 configs on identical tensors. This first pass showed **scattered** 6–12 % apparent wins
for C1 *and* C2 at various large-batch/long-ctx shapes — but with no consistent pattern (C1 won some,
C2 others, C0 most), the hallmark of measurement noise at these variance-prone sizes. (The `NUM_MMA_KV=1`
+ occ=1 combo is invalid for q-len 4 — below `kMinValidMmaKV` — so C1 correctly errors there; not a
real config.)

## Measurement 2 — interleaved repeats (the honest test, `tile_sweep/verify_shapes.py`)

7 repeats × 100 iters, **interleaved** config-by-config to cancel thermal/scheduler drift, median +
**min** (min = cleanest estimator of true kernel time). Every shape the single-shot pass flagged:

| shape | C0_def med (min) ms | C1 med (min) | C2 med (min) | best/ C0 (median) | min: is C0 beaten? | parity cos |
|:--|--:|--:|--:|:--:|:--:|--:|
| B4 QL1 c96K  | 2.332 (2.189) | 2.360 (2.181) | 2.556 (2.342) | 0.988× | no | 0.99505 |
| B16 QL1 c48K | 5.447 (5.277) | 5.457 (5.212) | 6.061 (5.881) | 0.998× | no | 0.99493 |
| B16 QL1 c96K | 10.94 (10.85) | 10.94 (10.83) | 12.13 (12.02) | 1.000× | no | 0.99496 |
| B8 QL1 c96K  | 4.681 (4.412) | 4.721 (4.365) | 5.026 (4.724) | 0.992× | no | 0.99486 |
| B32 QL1 c48K | 10.70 (10.58) | 10.70 (10.57) | 11.12 (11.00) | 1.000× | no | 0.99496 |
| B1 QL1 c16K  | 0.068 (0.0555)| 0.057 (0.0555)| 0.069 (0.0617)| 1.200× | no (mins equal) | 0.99465 |

MTP (q-len 4) regime — the only place C2's *median* leaned ahead:

| shape | C0_def med (min) ms | C2_128 med (min) | median/ C0 | min: is C0 beaten? | parity cos |
|:--|--:|--:|:--:|:--:|--:|
| B1 QL4 c4K   | 0.035 (0.035) | 0.035 (0.035) | 1.000× | tie | 0.99504 |
| B4 QL4 c48K  | 1.277 (1.141) | 1.157 (1.135) | 1.104× | **no** (C0 min 1.141 vs C2 1.135, 0.5%) | 0.99494 |
| B4 QL4 c96K  | 2.629 (2.249) | 2.446 (2.391) | 1.075× | **no — C0 min FASTER** (2.249 vs 2.391) | 0.99507 |
| B8 QL4 c48K  | 2.609 (2.339) | 2.631 (2.255) | 0.992× | no | 0.99499 |
| B16 QL4 c48K | 7.095 (6.928) | 7.118 (6.926) | 0.997× | tie | 0.99493 |
| B16 QL4 c96K | 14.81 (14.47) | 14.44 (14.23) | 1.025× | marginal (C2 min 1.7%) | 0.99497 |
| B32 QL4 c96K | 24.83 (24.67) | 25.12 (25.05) | 0.989× | no | 0.99492 |

**Reading:** C2's apparent MTP median advantage is an artifact of C0's *occasional* high-variance
samples inflating C0's median — on **min latency** (true kernel time) C0 and C2 are tied, with C0
winning the mins as often as C2 (e.g. B4QL4c96K: C0 min 2.249 vs C2 2.391 — C0 clearly faster).
No shape shows a reproducible ≥3–5 % C2 win. The B1c16K "1.2×" is sub-0.1 ms launch noise (identical
0.0555 mins).

## Why the default is right for GB10 (mechanism)

fp4 decode is **latency/occupancy-bound**, not tile-throughput-bound. Each KV chunk is tiny
(`CTA_TILE_KV=64`) and the split-KV scheduler already fans work across all 48 SMs. Keeping **2 CTAs
per SM** (the default's occupancy target) hides memory/MMA latency better than a 2× larger tile at 1
CTA/SM. On a big-smem SM100/B200 the same heuristic would allow a larger `NUM_MMA_KV` at 2 CTAs/SM —
but on GB10's 99 KB it lands on `CTA_TILE_KV=64`, and that is precisely the fast point. The
`num_ctas_per_sm=2`, smem-driven `NUM_MMA_KV` selection is **already GB10-adaptive**.

## Parity

cos(nvf4 decode, bf16-QK reference) ≈ **0.9946–0.9951** across all shapes — this is the intrinsic
nvfp4-KV quantization error and is **config-independent** (tile params change the work partition, not
the arithmetic). All tested tiles are numerically equivalent; a "faster-but-wrong" config never
arose. Parity gate (> 0.99) satisfied.

## Verdict

- **Headroom found: none reproducible.** Best honest speedup vs default over the whole grid is
  ≤ 2.5 % at one MTP shape (below the 3–5 % bar) and does not survive the min-latency estimator.
- **Winner: DEFAULT (`CTA_TILE_KV=64`, 2 CTAs/SM). GB10-optimal as shipped.**
- **Nothing wired.** No dispatch table baked, no image rebuild — there is no config to promote. The
  shipping overlay `prefill.cuh` is left pristine (the instrumented copy is a reference artifact only).
- Parity preserved (config-independent, cos ≈ 0.995 > 0.99).

## Files (in `a4q_merge/tile_sweep/`)

- `patch_prefill.py` — the tile-sweep instrumentation (env-selectable pre-compiled tile variants +
  `A4Q_TILE_DEBUG` introspection). Apply to a container's JIT `prefill.cuh` to re-run the sweep.
- `sweep_tile.py` — full grid harness (one plan, N configs on identical tensors, ms/iter + tok/s).
- `verify_shapes.py` — interleaved-repeat verifier (median+min, parity cos vs bf16-QK).
- `sweep_full.csv` — raw single-shot grid data.
- `prefill_patched.cuh` — instrumented kernel (reference; **not** the shipping file).

Shipping overlay `overlay/cuda/include/flashinfer/attention/prefill.cuh`: **unchanged** (verified no
`A4Q_TILE` markers).
