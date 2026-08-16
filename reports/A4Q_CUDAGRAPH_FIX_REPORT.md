# A4Q cudagraph-capture crash — FIX report

Date: 2026-08-15 · Node: gx10 (10.100.10.1, GPU-exclusive DGX Spark GB10, sm_121a)
Image: `eugr-gb10-nvfp4kv:a4q2` (fixed A4Q flashinfer + A4Q-wired vLLM backend) · CUDA 13.0.2

## Summary

The full vLLM serve crashed during startup with `CUDA error: an illegal memory access`
during **cudagraph capture** (the FULL decode graphs). Root cause: the **A4Q decode
`run()` path was missing the `q = q.view(dtype)` reinterpret after on-the-fly Q
quantization** — the exact analog of prefill "Bug 2", which the prior fix report had
*assumed* decode already carried ("decode.py A4Q run() already `.view`s Q"). It did not;
the decode A4Q Q-quant block was hand-added during the port and never had the view.

One-line fix in `overlay/py/decode.py`. After it:
- decode parity vs bf16-QK: **cos 0.9997** at every batch size (1,2,4,8,16,32,64) and
  both q-len 1 and q-len 4 (MTP verify) — up from **cos 0.93** unfixed.
- compute-sanitizer memcheck: **0 errors** on decode (incl. MTP q-len 4) and prefill.
- the full serve **BOOTS WITH CUDAGRAPH ON** (no `--enforce-eager`), reaches health,
  and answers coherently, with `VLLM_NVFP4_A4Q=1`.

## The faulting access (ground truth, unfixed serve)

Reproduced with the task's serve command (reduced `--max-model-len 8192`,
`--max-num-seqs 16` for fast iteration; cudagraph ON, MTP `num_speculative_tokens=3`):

```
INFO gpu_model_runner: Profiling CUDA graph memory: PIECEWISE=32 (largest=128), FULL=16 (largest=64)
...
File ".../vllm/utils/flashinfer.py", line 605, in flashinfer_mm_fp4
File ".../flashinfer/gemm/gemm_base.py", line 6930, in mm_fp4
File ".../flashinfer/gemm/gemm_base.py", line 1791, in forward
RuntimeError: [FP4 gemm Runner] Failed to initialize cutlass FP4 gemm on sm120/sm121. Error: Error Internal
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: an illegal memory access was encountered
Exception raised from currentStreamCaptureStatusMayInitCtx at .../CUDAGraphsC10Utils.h:73
(APIServer) RuntimeError: Engine core initialization failed.
```

The `mm_fp4` / "Failed to initialize cutlass FP4 gemm" line is a **red herring**: CUDA
errors are asynchronous, and `currentStreamCaptureStatusMayInitCtx` is simply the *next*
CUDA API call to inspect the stream after the fault. The illegal access itself happened
**earlier in the same FULL-graph capture — in the A4Q decode attention kernel**. Evidence
that this is the true cause, not a real cutlass-gemm failure:
- with `VLLM_NVFP4_A4Q=0` the identical `mm_fp4` path runs fine (control below), so the
  FP4 gemm is not broken; only the A4Q decode kernel corrupts the context.
- with the fix, the very same `fp4_gemm` autotuner now completes 100% across all shapes
  and the capture finishes — the context is no longer poisoned.

## Root cause — decode A4Q packed-Q read at 2× stride

`BatchDecodeWithPagedKVCacheWrapper` with `use_nvf4_qk=True` forces `use_tensor_cores=True`
and reuses the batch-prefill fp4-QK MMA kernel. Its `run()` quantizes Q on the fly:

```python
q, maybe_q_sf = nvfp4_quantize_q_cuda(q)   # q now packed uint8 [..., HEAD_DIM/2]
```

`nvfp4_quantize_q_cuda` returns **packed uint8** `[T, H, HEAD_DIM/2]` (1 byte holds two
e2m1 codes). The fp4-QK kernel addresses Q rows off a `DTypeQ*` (bf16, 2-byte) pointer
using the *tensor's element strides*. With uint8 (1-byte) element strides applied to a
2-byte pointer, every `(token, head)` Q row is addressed at **2× its true byte offset**,
so the read walks ~1× the tensor size past the end.

- In **eager** mode this OOB read lands inside the CUDA caching-allocator pool block, so
  it returns garbage (parity cos ≈ 0.93) but does **not** fault — which is why a plain
  eager decode never crashed and masked the bug.
- Under **cudagraph capture** the allocation layout is tight/graph-owned, so the same 2×
  read hits an unmapped page → `cudaErrorIllegalAddress`, surfaced at the next CUDA call.

Prefill's paged `run()` already carried the corrective `q = q.view(dtype)` (Bug 2 fix);
decode was the remaining gap. The run-args padding (`..., None, maybe_q_sf`) and the
`fast_decode_plan` cudagraph path were both verified correct — they reuse the a4q
`_cached_module` and forward `maybe_q_sf` at the right positional slots (46/47 of
`paged_run`); no change needed there.

## The fix (`overlay/py/decode.py`, decode `run()`)

```diff
             if q.dtype in (torch.bfloat16, torch.float16):
-                q, maybe_q_sf = nvfp4_quantize_q_cuda(q)
+                _q_view_dtype = q.dtype
+                q, maybe_q_sf = nvfp4_quantize_q_cuda(q)
+                # Present packed uint8 Q ([..., HEAD_DIM/2] bytes/row) as a DTypeQ view so
+                # the fp4-QK kernel's DTypeQ-pointer element strides land on the right rows.
+                q = q.view(_q_view_dtype)
             else:
                 raise ValueError(
                     "A4Q run() expects bf16/fp16 q for on-the-fly nvf4 Q quantization"
                 )
```

## Validation (on GB10 sm_121a, GPU-exclusive)

### a. Decode-path parity harness (`decode_repro.py`, A4Q vs bf16-QK reference)

`BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True).plan(use_nvf4_qk=True)` +
`run(maybe_q_sf)`, GQA 24q/4kv, head_dim 256, NVFP4 paged KV, single-token-per-seq and
MTP q-len 4:

| q_len | batch sizes 1..64 | unfixed cos | fixed cos | nan |
|---|---|---|---|---|
| 1 (decode) | 1,2,4,8,16,32,64 | 0.932–0.948 FAIL | **0.99971** PASS | 0 |
| 4 (MTP verify) | 1,2,4,8,16,32,64 | 0.935–0.941 FAIL | **0.99972** PASS | 0 |

Worst fixed cos = 0.999713 (at the fp4-Q quantization floor, matching prefill).

### b. compute-sanitizer memcheck (`--tool memcheck --error-exitcode 99`)

- decode repro `one 32 4` (MTP) → `ERROR SUMMARY: 0 errors`, nan=0
- decode repro `one 8 1`        → `ERROR SUMMARY: 0 errors`, nan=0
- prefill `diag.py a4q_only`     → `ERROR SUMMARY: 0 errors`, nan=0 (all 3 reqs)

### c. THE REAL GATE — full serve boots with cudagraph ON

`vllm serve /models/aday777 --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.85
--max-model-len 8192 --max-num-seqs 16 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
(NO `--enforce-eager`), `VLLM_NVFP4_A4Q=1`:

```
INFO flashinfer.py:876 A4Q: nvf4 block-scaled QK MMA enabled for FA2 NVFP4 prefill (decode=True, head_dim=256).
INFO gpu_model_runner:6971 Graph capturing finished in 13 secs, took 1.85 GiB
INFO api_server:680 Supported tasks: ['generate']
INFO Application startup complete.
health_http=200
completion: "The capital of France is" -> " Paris, a city that has long been a global hub for art, fashion, and culture. The city is home to"
```

### d. No-regression control — `VLLM_NVFP4_A4Q=0`

Same command, A4Q disabled: boots with cudagraph, health 200, coherent completion.
(Confirms the FP4 gemm path is healthy and the fix introduces no regression.)

## Files changed (HEAD tree `a4q_merge/overlay/`)
- `overlay/py/decode.py` — decode `run()`: `q = q.view(dtype)` after on-the-fly Q quant
  (the analog of the prefill Bug-2 fix). This is the entire fix.

Editable source on 10.100.10.1 (`~/a4q-dbg/a4q/overlay/py/decode.py`) updated identically;
images `eugr-gb10-nvfp4kv:a4q` and `:a4q2` rebuilt to bake the fix in.

## Notes
- The prefill fixes (USE_NVF4_QK forwarding in prefill.cuh; prefill Bug-2 view) were
  already present and are shared by the decode kernel (decode reuses the batch-prefill
  paged fp4-QK dispatch), so no CUDA/header change was needed for the cudagraph fix.
- The `vllm_gb10_nvfp4kv` overlay-extension build WARNING in the logs
  (`aoti_torch_get_current_cuda_stream undefined`) is pre-existing (stage1) and unrelated;
  it falls back cleanly and does not affect the A4Q path.
