# A4Q (native NVFP4 fp4-QKᵀ) port into eugr FlashInfer 0.6.18 — merge plan

**Verdict: (B)** — needs jethac's CUDA/headers + JIT-recompile for sm_121a
(Python plumbing **and** csrc/include overlay), **no full source wheel rebuild.**
The fp4-QK kernel does **not** exist in eugr 0.6.18's CUDA and cannot be JIT-dropped
from files already present (rules out A); but FlashInfer JIT-compiles the attention
kernels from `flashinfer/data/{csrc,include}` at first use, so overlaying the A4Q
CUDA into the installed `data/` tree + adding Python plumbing is sufficient — no
`pip wheel` / setup.py rebuild (rules out C). This is the same JIT-from-csrc
mechanism the EXP4 port already proved (`EXP4_PORT_LOG.md`: "the fp4 math lives in
the FlashInfer fork (JIT)", fixed by uninstalling `flashinfer-jit-cache` so JIT
compiles from fork csrc).

---

## 1. Which flashinfer path does eugr 0.6.18 use for nvfp4-KV decode/prefill @ head_dim 256

**The paged `BatchPrefillWithPagedKVCacheWrapper` / tensor-core `BatchDecodeWithPagedKVCacheWrapper`
(prefill.py / decode.py) — NOT the dense `nvfp4_attention_sm120`.**

Evidence:
- `use_nvf4_qk`, `maybe_q_sf`, `nvfp4_quantize_q_cuda` appear **only** in
  `decode.py`, `prefill.py`, `jit/attention/modules.py` in the fork. They appear in
  **zero** lines of `nvfp4_attention_sm120.py` (either version).
- eugr018 `nvfp4_attention_sm120.py` is a **separate dense** kernel:
  `_SUPPORTED_HEAD_DIMS = (64, 128)`, `[batch, num_heads, seq_len, head_dim]` dense
  layout, its own quantize helpers. It is capped at head_dim {64,128}, has no paged
  KV, no `use_nvf4_qk`. It is a **red herring** for this port and is not what vLLM calls.
- Our vLLM-0.27 A4Q backend calls the **paged wrapper**:
  `~/a4q-lab/vllm-a4q/vllm/v1/attention/backends/flashinfer.py`
  `prefill_wrapper.plan(..., **({"use_nvf4_qk": True} if self.a4q_prefill else {}))`
  (L1748) and the decode wrapper `.plan(... {"use_nvf4_qk": True})` (L1363).

**Where `use_nvf4_qk` is consumed (paged path):**
- `plan(use_nvf4_qk=...)` → validates (fa2 only, posenc NONE, head_dim∈{128,256}) →
  `get_batch_prefill_module(..., use_nvf4_qk)` (decode's tensor-core route **reuses
  the batch-prefill module**, fork decode.py L1224‑1235).
- The module builder threads `use_nvf4_qk` into the **URI** (`_nvf4qk_True` tag), adds
  `maybe_q_sf` (uint8) as an additional tensor, sets the jinja var `use_nvf4_qk=true`,
  and switches compile flags to **sm_120f + `-DFLASHINFER_ENABLE_NVF4_QK_MMA`**.
- `run()` (fork decode.py L1484‑1500 / prefill.py L2524‑2540): if q is bf16/fp16 and
  `q_sf is None`, quantizes on the fly via `nvfp4_quantize_q_cuda(q)`; requires q as
  packed uint8 e2m1 + q_sf uint8; appends `q_sf` as the trailing `maybe_q_sf` run-arg
  (decode L1761, prefill L2798).
- Kernel: the fp4-QKᵀ MMA lives in `include/flashinfer/attention/prefill.cuh`
  (`compute_qk_nvf4`, `mma_nvf4_m16n8k64`), i.e. the **FA2 prefill kernel** — decode
  reaches it because A4Q decode is forced onto the tensor-core (=batch-prefill) route.

---

## 2. Exact A4Q additions (minimal file / symbol set) — fork013 vs eugr018

### 2a. Python plumbing (missing entirely from eugr018 — grep `use_nvf4_qk`/`nvf4` = 0 hits)

**`jit/attention/modules.py`** (eugr: `jit_attention_modules.py`):
- `get_single_prefill_uri` / `get_batch_prefill_uri`: add `use_nvf4_qk: bool=False` param
  + `("_nvf4qk_True" if use_nvf4_qk else "")` URI suffix (fork L337/349, L396/409).
- `gen_single_prefill_module` / `gen_batch_prefill_module`: add `use_nvf4_qk` param;
  in the fa2 branch append `maybe_q_sf`/`"uint8_t"` to additional tensors when set
  (fork L545‑548, L1046‑1048); pass through to `gen_customize_*`.
- `gen_customize_single_prefill_module` / `gen_customize_batch_prefill_module`:
  add `use_nvf4_qk` param; put `"use_nvf4_qk": str(use_nvf4_qk).lower()` in jinja kwargs
  (fork L1354, L1639); validate (fp4 KV required + maybe_q_sf required, fork L1605‑1611);
  **the A4Q compile gate** (fork L1416‑1421 single, L1718‑1723 batch):
  ```python
  if use_nvf4_qk:
      extra_cuda_cflags = sm120f_nvcc_flags + ["-DFLASHINFER_ENABLE_NVF4_QK_MMA"]
  ```
  ⚠ **0.6.18 adaptation:** eugr's modules.py does NOT import `sm120f_nvcc_flags`; it
  uses `current_compilation_context.get_nvcc_flags_list(...)`. The port must obtain the
  compute_120f gencode via 0.6.18's `current_compilation_context` (or add a local
  `sm120f_nvcc_flags = ["-gencode=arch=compute_120f,code=sm_120f"] + common_nvcc_flags`,
  which is exactly the 0.6.13 core def) and still append `-DFLASHINFER_ENABLE_NVF4_QK_MMA`.

**`decode.py`** — `BatchDecodeWithPagedKVCacheWrapper`:
- `plan()`: add `use_nvf4_qk: bool=False`; gate `use_tensor_cores=True` (L1014‑1017),
  fa2-only + posenc NONE + head_dim∈{128,256} (L1208‑1222); pass `use_nvf4_qk` into
  `get_batch_prefill_module(...)` (L1235); store `self._use_nvf4_qk` (L1303).
- `run()`: add `q_sf` param (L1396); the A4Q block L1484‑1500 (on-the-fly
  `nvfp4_quantize_q_cuda`, dtype checks, `q.view(cached_q_data_type)`); append
  `q_sf` as trailing `maybe_q_sf` run-arg (L1761).

**`prefill.py`** — `BatchPrefillWithPagedKVCacheWrapper` (+ single/ragged helpers):
- module-level `nvfp4_quantize_q(q)` reference helper (L107).
- `_paged_run` / `single_prefill_run` helpers: add `use_nvf4_qk` positional (backward-
  compat via `args[9]/args[10]`, L378‑379/L517‑518), `maybe_q_sf` param, append to
  `fa2_args` after KV SF tensors (L466, L594, L824).
- `plan()`: `use_nvf4_qk` param (L1946) + validation (fa2, posenc NONE,
  head_dim_qk∈{128,256,512}, L2231‑2242) + module build (L2256) + `self._use_nvf4_qk`
  (L2339); note split-KV re-enabled behind A4Q (L2319‑2326).
- `run()`: `q_sf` param (L2435) + A4Q quant block (L2524‑2540) + `"maybe_q_sf": q_sf`
  into module kwargs (L2730) and trailing run-arg (L2798).
- Single/ragged `single_prefill_with_kv_cache` path: `use_nvf4_qk` (L1256) + quant
  block (L1401‑1420) + arg (L1533).

**`quantization/fp4_quantization.py`** (the real impl; the top-level
`fp4_quantization.py` is only a re-export stub):
- Add the **`nvfp4_q_quant` custom op** + `nvfp4_quantize_q_cuda` wrapper
  (fork quantization/fp4_quantization.py L1858‑1877): registers
  `flashinfer::nvfp4_q_quant`, calls `module.nvfp4_q_quant(input, fp4_output, block_scales)`,
  returns `SimpleNamespace(nvfp4_kv_quant=..., nvfp4_q_quant=...)`.
- Re-export `nvfp4_quantize_q_cuda`/`nvfp4_quantize_q` from the stub `fp4_quantization.py`
  (cosmetic; call sites import from `.quantization.fp4_quantization` directly).

### 2b. CUDA / headers (the actual fp4-QK kernel + Q-quant kernel — absent from stock)

Confirmed absent from stock 0.6.4/0.6.12/0.6.13 (`grep -c USE_NVF4_QK|compute_qk_nvf4|mma_nvf4` = **0**);
present only in the fork. Files carrying A4Q (fork wheel `flashinfer/data/`):

| File | A4Q addition | scope |
|---|---|---|
| `include/flashinfer/attention/prefill.cuh` | `#include <cuda_fp4.h>`; `is_fp4_type`/`is_fp4_type_v`; `USE_NVF4_QK_` KTraits (+`kDenseKFp4` dense-pack); `load_q_nvf4` (packed-e2m1 Q + ue4m3 SF → smem); `mma_nvf4_m16n8k64` (inline PTX `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…e2m1.e2m1`); `compute_qk_nvf4` (replaces dequant+HMMA); producer dense-pack of fp4 K tile | **~+420 lines** (3788→4208 vs stock 0.6.13) |
| `csrc/batch_prefill.cu` | `bool USE_NVF4_QK=false` template param + dispatch (L28, L325) | 2 hunks |
| `csrc/single_prefill.cu` | same (L29, L106) | 2 hunks |
| `csrc/batch_prefill_customize_config.jinja` | `constexpr bool USE_NVF4_QK = {{ use_nvf4_qk }};` (L37) | 1 line |
| `csrc/single_prefill_customize_config.jinja` | same (L30) | 1 line |
| `csrc/batch_prefill_paged_kernel_inst.jinja` | `{{ variant_name }}, PagedParams, {{ use_nvf4_qk }}>` (L13) | 1 line |
| `csrc/single_prefill_kernel_inst.jinja` | same for single | 1 line |
| `csrc/fp4_kv_quantization.cu` | `nvfp4_q_quant_kernel` (L273) + `nvfp4_q_quant(...)` (L322) + `TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_q_quant, nvfp4_q_quant)` (L410) | ~+140 lines in a file eugr already has (it has `nvfp4_kv_quant`) |

**Self-containment (good news):** the fp4-QK MMA in `prefill.cuh` adds **no new
`#include`** beyond `cuda_fp4.h` and does **not** depend on the
`nvfp4_attention_sm120/*` dense headers or `fp4_convert.cuh` — it is inline-PTX +
existing smem primitives. So the dense `nvfp4_attention_sm120/*` tree does **not**
need to be overlaid for this path.

---

## 3. Feasibility verdict — **(B)**, with evidence

- **Not (A):** the QKᵀ fp4 MMA (`compute_qk_nvf4`/`mma_nvf4_m16n8k64`) and the Q-quant
  kernel (`nvfp4_q_quant`) are **not present in any stock/eugr CUDA source** (0 grep
  hits in 0.6.4/0.6.12/0.6.13 `prefill.cuh`; eugr `fp4_kv_quantization.cu` has only
  `nvfp4_kv_quant`). Python plumbing alone would call an op that doesn't exist and a
  `USE_NVF4_QK=true` template that instantiates a code path with no MMA. Python-only
  overlay is impossible.
- **Not (C):** FlashInfer's JIT (`gen_customize_*_module`) reads the templates and
  kernels **from `flashinfer/data/{csrc,include}` of the installed package** and
  `nvcc`-compiles them on first `plan()`. Overlaying the A4Q files into that tree +
  the Python plumbing is picked up automatically — **no setup.py / wheel rebuild**.
  EXP4_PORT_LOG proves the JIT-from-csrc route works end-to-end (uninstall
  `flashinfer-jit-cache` so the AOT `.so` doesn't shadow the JIT build).
- **(B) is the fit:** overlay the 8 CUDA files (transplant the hunks — see risk) into
  eugr 0.6.18's `data/` tree + apply the Python plumbing patches; JIT recompiles for
  sm_120f/121a on first use.

**Critical (B) caveat — transplant, don't blind-copy:** the fork CUDA is **0.6.13**;
eugr is **0.6.18**. `prefill.cuh` diverged (stock 0.6.13 = 3788L, 0.6.18 newer). Dropping
the fork's whole `prefill.cuh`/`batch_prefill.cu` would revert eugr's kernel to 0.6.13
and can break the **Stage‑1 0.6.18 nvfp4-KV path** (head_dim 256). The A4Q hunks must be
**ported into eugr 0.6.18's own** `prefill.cuh`/csrc, keeping eugr's KV staging. The
alternative Keys already validated — ship the **whole fork 0.6.13 wheel** — is simpler
but abandons eugr 0.6.18's KV path, which is the thing this task is trying to preserve.

See `patch/` for the Python overlay fragments and the CUDA/Docker overlay recipe.

---

## 4. Biggest MIS-COMPUTE risk (silent wrong output, not a crash)

**Scale-factor + dense-packed-K smem layout mismatch between the fork's
`compute_qk_nvf4` and eugr 0.6.18's own fp4-KV producer staging, at head_dim 256.**

`compute_qk_nvf4` (prefill.cuh L1229‑1349) does **not** read K/Q/SF the way the stock
fp4-KV path stages them. It assumes three fork-specific, tightly-coupled layouts:
1. **J‑5 dense-packed K tile**: `ROW_BYTES_K = HEAD_DIM_QK/2`, row-major, *no swizzle*
   (the fork **rewrote the producer** `produce_kv`/`load_k` at L383‑500 with
   `kDenseKFp4` to match this). eugr 0.6.18's KV path stages fp4 K in the **k128B
   half-empty swizzle**; if the producer rewrite is not transplanted *together with*
   the consumer, `b[j]` reads the wrong bytes.
2. **Q SF smem layout** `q_sf_smem + (row+am)*SF_COLS + kb*4` and **K SF** `(p*32+bn)*SF_COLS + kb*4`
   with `SF_COLS = HEAD_DIM/16`, feeding the `scale_vec::4X m16n8k64` block-scale MMA.
   The ue4m3 per-16 SF must be in exactly this interleave; eugr's Stage‑1 KV SF swizzle
   (block_scale_interleave) may differ.

At **head_dim 256** this is the sharpest edge: 256 → 4 `k64` blocks + `NUM_MMA_KV`
tail path (odd-tail sub-MMA, L1326‑1347) that upstream 0.6.18 never exercised for a QK
MMA. Any off-by-one in `SF_COLS`/`kb*4` indexing, or a K-tile stride mismatch, makes the
`block_scale` MMA consume **valid-but-wrong** scale factors / K bytes and emit
**plausible, silently-degraded attention logits** — no illegal address, no NaN, just
wrong numbers. This is strictly worse than the sm_120a "no kernel image" crash the fork
already worked around by compiling for the **sm_120f family** (that one is loud).

**Mitigation:** transplant the producer-side dense-pack + SF staging *and* the consumer
together; validate with a bf16-QK vs fp4-QK reference on head_dim 256 (max-abs logit
diff + end-to-end output parity) before trusting it — exactly the kind of numeric A/B the
fork's own `a4q_test_qquant.py` / `a4q_bench.py` do.
