# A4Q "Side 2" NVFP4 Decode Autotune — Investigation & Result

**Date:** 2026-08-15  **Box:** DGX Spark GB10 (sm_121, **48 SMs**, 99KB smem) @ 10.100.10.1
**Image:** `eugr-gb10-nvfp4kv:a4q` (flashinfer 0.6.18 + A4Q fp4-QK)

## TL;DR

**The requested "enable the NVFP4 decode autotune" cannot be done as specified — not because
eugr is missing `autotuner.py`, but because flashinfer's autotuner framework does NOT and CANNOT
tune the fa2 attention decode kernel. The jethac fork does not wire it to the decode path either.**
Porting `autotuner.py` yields a cutlass-GEMM/MoE/MLA tactic selector that the nvf4 decode path
never calls.

The good news, verified by measurement: **the nvf4 decode path is already fast.** The real
GB10-sensitive lever — the plan-time, num_sm-driven **split-KV scheduler** — is already engaged by
default and already adapts to GB10's 48 SMs. A4Q nvf4 decode already runs **2–3.5× faster than the
16-bit-KV reference** at 16K–96K context. There is no "untuned default config" leaving 30% on the
table via an autotuner; the decode kernel is a single JIT template whose tile/stage/split are fixed
at compile time and whose work partitioning is already SM-adaptive at plan time.

## What "the autotune framework" actually is (evidence)

`flashinfer/autotuner.py` (fork 0.6.13, 1955 lines) implements a **runtime tactic selector** for
kernels that ship *multiple selectable implementations in one `.so`*:

- `TunableRunner` ABC requires `get_valid_tactics(inputs, profile) -> List[int]` and
  `forward(inputs, tactic=int)` that dispatches to a *different kernel per integer tactic*
  (`autotuner.py:381-465`). `AutoTuner.choose_one()` benchmarks the tactics and caches the best.
- Every consumer in the fork is a cutlass / cute-DSL / cuBLASLt path:
  `grep -rln TunableRunner` → `gemm/gemm_base.py`, `fused_moe/*`, `mla/_core.py`,
  `mla/_sparse_mla_sm120.py`, `trtllm_low_latency_gemm.py`. Example: the GEMM runner's
  `get_valid_tactics` returns `range(count)` of enumerated cuBLASLt algos and `forward(tactic=i)`
  calls `bmm_fp8_run_with_algo(..., tactic)` (`gemm_base.py:181-230`).

**The fa2 attention path is not in that list.** In the fork:
- `flashinfer/decode.py` — `grep -c "autotun|TunableRunner|get_valid_tactics|tuning_config"` = **0**.
  It only threads `use_nvf4_qk` through to the JIT kernel.
- `flashinfer/prefill.py` — same: no autotuner, only `use_nvf4_qk` passthrough.
- `jit/attention/modules.py` — the only "tile" is `qo_tile_len = 128 if sm>=9 else 64`
  (`modules.py:252`), a compile-time choice for the *MLA* decode kernel, identical in eugr.

Confirmed on the live eugr image: `decode.py` has 0 tuner refs / 9 `use_nvf4_qk` refs;
`get_valid_tactics` exists only under `fused_moe/` and `grouped_mm/_sm120_moe_autotune.py`
(eugr already ships MoE autotuning) — **never for attention.**

## Why the decode kernel can't be tactic-tuned

The fp4-QK decode reuses the batch-prefill fa2 tensor-core dispatch. That kernel's performance
knobs are **compile-time C++ template parameters**, not runtime tactics
(`overlay/cuda/include/flashinfer/attention/prefill.cuh`):
`CTA_TILE_Q`, `CTA_TILE_KV`, `NUM_MMA_{Q,KV,D}`, `NUM_WARPS_{Q,KV}`, `NUM_STAGES` (=1 for
BatchAttention), and the `kVShareActive` 99KB-smem path (explicitly gated for
"SM86/89/120/121", i.e. GB10, at `prefill.cuh:156-159`). These are baked by `KernelTraits` at JIT
time. There is exactly one kernel per shape in the `.so` — nothing for an integer `tactic` to select.

To "autotune" tile/stages you would have to JIT-compile multiple kernel variants and benchmark them
— a bespoke offline sweep, which is **not** what `autotuner.py` does and is not wired anywhere in
either build.

**What would be missing to make the decode path autotunable (none of it exists in the fork):**
1. An attention `TunableRunner` subclass with `get_valid_tactics()` / `forward(tactic=...)`.
2. A decode kernel `.so` containing multiple runtime-selectable implementations (multi-tile/stage),
   or a launcher that picks compiled variants by integer.
3. A `plan()`-time call into `AutoTuner.choose_one()` for the decode custom op.

## The real GB10 lever, and it's already engaged: plan-time split-KV

The fa2 decode plan runs a C++ scheduler (`plan_info_vec`) that partitions KV work across the
device SM count. This is the genuine num_sm-sensitive knob, and it reads GB10's real 48 SMs at
plan time. The only runtime handle exposed to Python is `disable_split_kv` (default False = split on).

Measurement confirms split-KV is doing the heavy lifting and is already in the right position:
disabling it is **2–10× worse** at long context for low batch (where a single unsplit decode
starves GB10's 48 SMs). For large batch the batch already fills the SMs, so it's a wash.

## Measurements (ms/iter over 50 iters, 10 warm; head_dim 256, 24q/4kv, nvfp4 paged KV)

Harness: `overlay/decode_bench.py` (on box at `~/a4q-dbg/a4q/overlay/decode_bench.py`).
Columns: **a4q_def** = nvf4 decode, split-KV on (current default) · **a4q_nosplit** =
`disable_split_kv=True` · **bf16ref** = same wrapper, 16-bit KV tensor-core decode (higher-precision
reference — the "cost to recover toward"). Ratio = a4q_def / bf16ref (lower = a4q faster).

### B=1, QL=1 (single-token decode)
| ctx | a4q_def ms | a4q_nosplit ms | bf16ref ms | a4q vs bf16 |
|----:|----:|----:|----:|----:|
| 4K  | 0.160 | 0.133 | 0.063 | 2.52× (overhead-bound) |
| 16K | 0.147 | 0.469 | 0.305 | **0.48×** |
| 48K | 0.297 | 1.465 | 0.855 | **0.35×** |
| 96K | 0.555 | 3.243 | 1.966 | **0.28×** |

### B=1, QL=4 (MTP)
| ctx | a4q_def ms | a4q_nosplit ms | bf16ref ms | a4q vs bf16 |
|----:|----:|----:|----:|----:|
| 4K  | 0.145 | 0.370 | 0.020 | 7.3× (overhead-bound) |
| 16K | 0.106 | 1.457 | 0.349 | **0.30×** |
| 48K | 0.356 | 4.327 | 1.018 | **0.35×** |
| 96K | 0.690 | 7.831 | 1.972 | **0.35×** |

### B=8, QL=1
| ctx | a4q_def ms | a4q_nosplit ms | bf16ref ms | a4q vs bf16 |
|----:|----:|----:|----:|----:|
| 4K  | 0.178 | 0.192 | 0.638 | **0.28×** |
| 16K | 0.842 | 0.832 | 2.372 | **0.36×** |
| 48K | 2.205 | 2.186 | 7.535 | **0.29×** |
| 96K | 4.309 | 4.868 | 13.973 | **0.31×** |

### Decode throughput (tok/s = 1000·B·QL / ms)
| shape | ctx | a4q_def | bf16ref | a4q speedup |
|:--|--:|--:|--:|--:|
| B1 QL1 | 96K | **1802** | 509 | 3.5× |
| B1 QL1 | 48K | **3367** | 1170 | 2.9× |
| B8 QL1 | 96K | **1856** | 573 | 3.2× |
| B8 QL1 | 48K | **3628** | 1062 | 3.4× |

**Reading:** at every context that matters (16K+), nvf4 decode is already 2–3.5× faster than the
16-bit-KV reference — the nvfp4-KV bandwidth win is fully realized, driven by the SM-adaptive
split-KV scheduler that is already on by default. The only place nvf4 loses is the tiny
4K/B=1 regime, which is pure launch/quant overhead and irrelevant to long-context decode.

## Verdict on the task gate

- "Autotuned A4Q decode measurably faster than untuned": **not achievable via an autotuner** —
  there is no decode autotuner to enable, in eugr *or* the fork. The premise ("no autotuner.py → single
  untuned config, 30% left on table") does not hold: the decode kernel is a single compile-time
  template and its num_sm-adaptive split-KV scheduling is already active and already recovering the
  long-context cost (2–3.5× over 16-bit KV).
- Parity: unchanged — nothing was modified, so the previously-verified cos 0.99971 stands. No tuning
  touched numerics.
- Recovery is *already present* by default; it is not gated behind an autotuner that eugr lacks.

## If a further decode speedup is genuinely wanted (not "autotune")

The only remaining headroom is a **compile-time kernel-tile sweep** for GB10: JIT-build the nvf4
batch-prefill kernel with alternative `CTA_TILE_KV` / `NUM_WARPS_KV` / `NUM_STAGES` / `kVShareActive`
settings tuned to 48 SMs + 99KB smem, benchmark each, and hardcode GB10's winner into the
`KernelTraits` dispatch. That is a bespoke C++/JIT sweep + image rebuild, orthogonal to
`autotuner.py`. It was not attempted here because it is a multi-hour build loop with uncertain
payoff given decode is already 2–3.5× over the 16-bit path.

## Files
- `overlay/decode_bench.py` — timing harness (new; also on box at `~/a4q-dbg/a4q/overlay/`).
- No source changes made to decode.py / modules.py / autotuner.py: porting `autotuner.py` would not
  affect the decode path, so it was (correctly) not integrated.
