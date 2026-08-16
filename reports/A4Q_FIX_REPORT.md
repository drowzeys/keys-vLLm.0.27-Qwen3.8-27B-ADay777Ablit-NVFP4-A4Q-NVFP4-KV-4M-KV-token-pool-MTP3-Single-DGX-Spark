# A4Q head_dim-256 per-request NaN — FIX report

Date: 2026-08-15 · Node: gx10 (10.100.10.1, GPU-exclusive DGX Spark GB10, sm_121a)
Image: `eugr-gb10-nvfp4kv:a4q` · CUDA 13.0.2 · compute-sanitizer clean

## Summary

The reported "A4Q fp4-QK per-request NaN in batched paged prefill at head_dim 256"
(req0 ALL NaN, req1 HALF NaN, req2 clean) was **two independent bugs**, neither in the
dense-K / SF addressing that was originally suspected. Both are fixed; the native
`compute_qk_nvf4` path now runs, is memcheck-clean, and matches the ideal fp4-Q attention
to **cos = 0.999997** (parity vs the bf16-QK reference: **cos = 0.9997**, gate ≥ 0.99).

Crucially, on-GPU instrumentation proved the dense-K K bytes (`b`), Q scale (`sfa`), and
K scale (`sfb`) the kernel loads are all **byte-exact correct** — the transplanted
dense-K producer/consumer and the ue4m3 SF layout were never wrong. The A4Q_DEBUG_REPORT's
"dense-K addressing / SF interleave" suspicions are ruled out by direct measurement.

## Bug 1 — paged/ragged kernel silently compiled with `USE_NVF4_QK = false`

`plan(use_nvf4_qk=True)` correctly builds the `..._nvf4qk_True` module (config renders
`constexpr bool USE_NVF4_QK = true;`, and the paged inst is `...PagedParams, true>`), **but**
inside `prefill.cuh` the paged and ragged dispatch functions built `KernelTraits<...>`
**without the trailing `USE_NVF4_QK` template argument**, so it fell back to the
struct default `false`. The single-prefill dispatch (L2929) forwarded it correctly; paged
and ragged did not.

Consequence: the paged kernel ran the **bf16-dequant** QK path (`compute_qk`, `load_q_global_smem`),
NOT `compute_qk_nvf4`. Meanwhile `run()` had already quantized Q to packed uint8 e2m1 and
handed it to a kernel expecting bf16 Q → garbage Q → huge scores → softmax `inf/inf` = **NaN**.
An entry `printf` proved the running paged kernel reported `nvf4=0`.

### Fix (file: `overlay/cuda/include/flashinfer/attention/prefill.cuh`)

In **both** `BatchPrefillWithPagedKVCacheDispatched` and
`BatchPrefillWithRaggedKVCacheDispatched`, the `using KTraits = KernelTraits<...>` line:

```diff
             KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV, NUM_MMA_D_QK, NUM_MMA_D_VO,
                          NUM_WARPS_Q, NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO,
-                         DTypeQKAccum, typename Params::IdType, AttentionVariant>;
+                         DTypeQKAccum, typename Params::IdType, AttentionVariant, USE_NVF4_QK>;
```

(Now all 3 dispatch fns — single/ragged/paged — forward `USE_NVF4_QK`.)

Result after Bug 1 alone: `nan = 0` for all requests, but parity **cos = 0.951** (a second bug).

## Bug 2 — packed Q read at 2× stride (missing `q.view(dtype)`)

With the native path now active, `compute_qk_nvf4` was numerically wrong (cos 0.951, ~31 %
error). On-GPU byte dump (block0, kv-head0, lane0, kb0, p0) vs the known quantized tensors:

| operand | kernel | expected | match |
|---|---|---|---|
| `b[0..7]` (K data) | 1ed25c69 84a1f6af … | 1ed25c69 84a1f6af … | ✅ all 8 |
| `sfa` (Q SF) | 26282421 | 26282421 | ✅ |
| `sfb` (K SF) | 25232527 | 25232527 | ✅ |
| `a[0]`,`a[2]` (Q row r0) | 6eaedc4f, 5b520959 | idem | ✅ |
| `a[1]`,`a[3]` (Q row r0+8) | 71913a03, 767cd662 | 51f9beed, 7db8c736 | ❌ |

`a[1]/a[3]` read packed index **16** (token2,head4) instead of index **8** (token1,head2).
A Q-smem slot dump showed slot `s` held packed index **`2·s`** (slot1→idx2, slot2→idx4,
slot8→idx16) — a consistent **2× doubling**.

Root cause: `run()` replaced `q` with the packed **uint8** `[T, H, HEAD_DIM/2]` tensor but
never viewed it back to the Q dtype, while `load_q_packed_global_smem` (and `q_ptr_base`)
compute offsets on a `DTypeQ*` (bf16) pointer using the tensor's element strides. With uint8
(1-byte) strides applied to a 2-byte pointer, every `(token, head)` Q row is addressed at
**2×** its byte offset. The fork's decode `run()` does `q.view(cached_q_data_type)`; the
prefill overlay was missing it.

### Fix (file: `overlay/py/prefill.py`, paged `run()`)

```diff
                 if q.dtype in (torch.bfloat16, torch.float16):
-                    q, maybe_q_sf = nvfp4_quantize_q_cuda(q)
+                    _q_view_dtype = q.dtype
+                    q, maybe_q_sf = nvfp4_quantize_q_cuda(q)
+                    # Present packed uint8 Q (HEAD_DIM/2 bytes/row) as a DTypeQ view so the
+                    # fp4-QK kernel's DTypeQ-pointer element strides land on the right rows.
+                    q = q.view(_q_view_dtype)
```

## Validation (on GB10 sm_121a, GPU-exclusive)

Aligned repro (`diag.py a4q_only`, BATCH=3, 32 q-rows/req, GQA 24q/4kv, head_dim 256):

```
before: a4q nan=294912   req0=196608  req1=98304  req2=0
after : a4q nan=0        req0=0       req1=0      req2=0     (min -0.328, max 0.369)
```

Parity gate (`parity_test.py`, A4Q vs bf16-QK reference):

| causal | cos | max_abs | gate |
|---|---|---|---|
| 1 | 0.999722 | 1.07e-02 | A4Q_PARITY_PASS |
| 0 | 0.999721 | 8.79e-03 | A4Q_PARITY_PASS |

Kernel-vs-ideal (correct token-indptr pure-torch fp4-Q reference):
- a4q(kernel) vs torch-fp4Q(ideal): **cos 0.999997**, max_abs 1.33e-3
- torch-fp4Q vs torch-bf16Q (true fp4-Q cost): cos 0.999727 → the kernel is at the fp4-Q
  quantization floor, i.e. numerically correct.

memcheck (`compute-sanitizer --tool memcheck`, a4q_only): **ERROR SUMMARY: 0 errors**, nan=0.

Ragged + partial-last-page (`probe_ragged.py`, QO=[16,24,8], KV=[40,48,24] → last_page_len
[8,16,8], kv_len not a multiple of PAGE_SIZE): nan=0 all requests, cos 0.9997 (causal on/off).
This also resolves the vLLM-serve ragged-tail illegal-access class.

## Files changed (HEAD tree `a4q_merge/overlay/`)
- `overlay/cuda/include/flashinfer/attention/prefill.cuh` — Bug 1: forward `USE_NVF4_QK`
  to `KernelTraits` in the paged and ragged dispatch functions.
- `overlay/py/prefill.py` — Bug 2: `q = q.view(dtype)` after on-the-fly Q quant in paged `run()`.
- `overlay/parity_test.py` — test convenience only: honor `CAUSAL` env (default 1).

## Notes / follow-ups
- `decode.py` A4Q `run()` already `.view`s Q (fork parity); the paged-prefill `run()` was the
  gap. If ragged-prefill or single-prefill `run()` paths gain an on-the-fly Q-quant block,
  apply the same `.view` there.
- The dense-K producer/consumer byte layout `row*(HEAD_DIM/2)+(kb*2)*16+tr*4` and the linear
  ue4m3 SF layout `SF_COLS=HEAD_DIM/16` were verified correct by direct byte comparison — no
  change needed.
