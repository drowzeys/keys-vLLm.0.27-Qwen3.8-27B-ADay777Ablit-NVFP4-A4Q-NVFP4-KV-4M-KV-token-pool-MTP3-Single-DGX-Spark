# A4Q parity test — the numeric gate

`parity_test.py` is the gate that decides whether the fp4-QK transplant is trustworthy.
Silent wrong output is the enemy: an A4Q kernel that reads the wrong dense-K byte, the
wrong scale-factor interleave, or maps the m16n8k64 fragments wrong at the head_dim-256
tail will still *run* and emit *plausible* logits — no crash, no NaN, just wrong numbers.
Only a numeric A/B catches that.

## What it does

It builds ONE paged NVFP4 KV cache (packed e2m1 uint8 + flat per-16 ue4m3 scale factors)
and ONE bf16 query, head_dim 256, GQA 24 q-heads / 4 kv-heads, then runs the identical
`BatchPrefillWithPagedKVCacheWrapper` twice, changing only one flag:

| run | `use_nvf4_qk` | Q.K^T compute |
|-----|---------------|---------------|
| REFERENCE | `False` | Stage-1 nvfp4-KV path: K dequantized fp4→bf16 in smem, bf16 HMMA; Q stays bf16 |
| A4Q       | `True`  | Q quantized on the fly to packed e2m1 (+per-16 SF); native sm120 `mma.sync…kind::mxf4nvf4.block_scale…m16n8k64` |

Because the fp4 K/V bytes and the KV scale factors are **byte-identical** between the two
runs, the output delta isolates exactly the fp4-QK-vs-bf16-QK numerics (the added error is
just the fp4 quantization of Q). A third, backend-independent **pure-torch** reference is
computed from the dequantized fp4 Q and fp4 K/V to bound what "correct fp4 attention"
should look like.

The KV cache and Q are quantized with `nvfp4_quantize_q_cuda` — the same op the A4Q Q path
uses — so the scale-factor layout produced here is exactly the flat per-16 layout the
attention kernel reads (`produce_kv_sf` / `compute_qk_nvf4`). This keeps the A/B honest.

## How to run

Needs a GPU (sm_120a RTX PRO 6000, or sm_121a / GB10 Spark). Inside the A4Q overlay image:

```
docker run --rm --gpus all <a4q-overlay-image> python3 /opt/a4q/parity_test.py
```

or against a dev checkout with the overlay applied:

```
python3 overlay/parity_test.py
```

## Pass / fail

Printed metrics (A4Q vs the bf16-QK reference): `max_abs`, `mean_abs`, `max_rel`, `cos`.

The gate passes when the A4Q output stays highly correlated with the bf16-QK reference and
within a small absolute band:

```
cos(A4Q, ref) > 0.99   AND   max_abs(A4Q, ref) < 0.15*mean|ref| + 0.05
```

* **PASS** (`A4Q_PARITY_PASS`): the fp4-QK MMA computes the same attention as the bf16 path
  up to fp4-Q quantization noise. Trust it.
* **FAIL** (`A4Q_PARITY_FAIL`, especially `cos` far below 1.0): the transplant miscomputes.
  The most likely culprits, in order:
  1. **Dense-K / SF smem layout disagreement** at head_dim 256 — the producer dense-pack
     (`kDenseKFp4` branch in `produce_kv` / `page_produce_kv` / `page_produce_kv_on_the_fly`)
     and the consumer (`compute_qk_nvf4`) must agree byte-for-byte. The K scale factors are
     read warp-local `(p*32+bn)*SF_COLS + kb*4`; the dense K bytes at `row*(HEAD_DIM/2) +
     (kb*2)*16 + tr*4`.
  2. **NUM_MMA_KV odd tail** (head_dim 256 with a 1-KV-fragment step): the `NUM_MMA_KV % 2
     == 1` tail path in `compute_qk_nvf4` runs only sub-MMAs 0/1 with the clamped SFB read.
  3. **s_frag fragment order** feeding `logits_transform` — must match eugr's downstream
     softmax fragment convention (shared FA2 ancestry says it does, but this is the check).

## Tuning notes

- If `max_rel` is large only where `|ref|` is tiny, that is expected (division by ~0);
  trust `cos` and `max_abs` there.
- To sweep shapes, edit `NUM_QO_HEADS / NUM_KV_HEADS / HEAD_DIM / KV_LEN_PER_REQ`. head_dim
  128 and 256 are the validated A4Q head dims; 256 is the sharpest edge (4 k64 blocks + the
  odd-tail path) and is the default here.
- A large gap between the **bf16-QK reference** and the **pure-torch** reference (last row)
  indicates a KV-cache-construction mismatch in the test harness itself (paged layout / SF
  layout), not an A4Q bug — fix the harness before trusting the A4Q verdict.
