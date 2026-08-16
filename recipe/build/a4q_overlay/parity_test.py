#!/usr/bin/env python3
"""A4Q numeric-parity gate: native fp4 QK^T MMA vs bf16 QK^T, head_dim 256 GQA.

Silent-wrong-output is the enemy. This script runs the SAME paged prefill attention
on the SAME fp4 KV cache and the SAME query twice:

  * REFERENCE : wrapper.plan(use_nvf4_qk=False) + run(q_bf16)
                -> Stage-1 nvfp4-KV path: K is dequantized fp4->bf16 in smem, QK done
                   with the bf16 HMMA. Q stays bf16.
  * A4Q       : wrapper.plan(use_nvf4_qk=True)  + run(q_bf16)
                -> Q is quantized on the fly to packed e2m1 (+per-16 ue4m3 SF) and the
                   QK^T is computed with the native sm120 nvf4 block-scale MMA.

The ONLY difference between the two runs is `use_nvf4_qk`; the fp4 K/V bytes and the KV
scale factors are byte-identical. So the output delta isolates exactly the fp4-QK vs
bf16-QK numerics (the added error is the fp4 quantization of Q). If the transplant
miscomputes (wrong dense-K byte, wrong SF interleave, wrong fragment mapping at the
head_dim-256 tail), the A4Q output diverges WILDLY from the reference and this gate fails.

A third, backend-independent PURE-TORCH reference is also computed from the dequantized
fp4 Q and fp4 K/V, bounding what "correct fp4 attention" should look like.

Requires a GPU (sm_120a / sm_121a). Run inside the A4Q overlay image:
    docker run --rm --gpus all <image> python3 /opt/a4q/parity_test.py
"""
import math
import os
import sys

import torch

import flashinfer
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.quantization.fp4_quantization import nvfp4_quantize_q_cuda

# ----------------------------- problem shape --------------------------------------
NUM_QO_HEADS = 24
NUM_KV_HEADS = 4          # GQA group size 6
HEAD_DIM = 256
PAGE_SIZE = 16
KV_LEN_PER_REQ = 64       # multiple of PAGE_SIZE
QO_LEN_PER_REQ = 32
BATCH = 3
DEVICE = "cuda"
DTYPE = torch.bfloat16
SF_BLOCK = 16             # NVFP4_SF_VEC_SIZE
SEED = 1234


def _e4m3_to_f32(x_u8: torch.Tensor) -> torch.Tensor:
    """Reinterpret a uint8 e4m3 (ue4m3) byte tensor as float32."""
    return x_u8.view(torch.float8_e4m3fn).to(torch.float32)


def dequant_nvfp4(packed: torch.Tensor, sf_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize (packed e2m1 [.., D/2], linear ue4m3 SF [.., D/16]) -> bf16 [.., D].

    Matches nvfp4_quantize_q_cuda's semantics: value = e2m1(code) * float(e4m3(sf)).
    Used only to build the PURE-TORCH reference; the kernel does the real work.
    """
    *lead, half = packed.shape
    D = half * 2
    # unpack nibbles -> e2m1 code 0..15 -> float via the e2m1 grid.
    lo = (packed & 0x0F).to(torch.int64)
    hi = (packed >> 4).to(torch.int64)
    codes = torch.stack([lo, hi], dim=-1).reshape(*lead, D)  # even=low nibble
    E2M1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=packed.device, dtype=torch.float32,
    )
    vals = E2M1[codes]
    sf = _e4m3_to_f32(sf_u8)                                  # [.., D/16]
    sf = sf.repeat_interleave(SF_BLOCK, dim=-1)               # [.., D]
    return (vals * sf).to(DTYPE)


def make_fp4_row(x_bf16: torch.Tensor):
    """Quantize a [..., D] bf16 tensor to (packed uint8 [.., D/2], sf uint8 [.., D/16]).

    Uses the SAME kernel op the A4Q Q path uses, so the K/V SF layout produced here is
    exactly the flat per-16 layout the attention kernel reads (produce_kv_sf /
    compute_qk_nvf4). Building both runs' cache this way keeps the A/B clean.
    """
    return nvfp4_quantize_q_cuda(x_bf16)


def pure_torch_attention(q_bf16, k_bf16, v_bf16, qo_indptr, kv_indptr, sm_scale, causal):
    """Reference variable-length causal/non-causal attention in fp32, per request."""
    outs = []
    for b in range(BATCH):
        qs, qe = qo_indptr[b].item(), qo_indptr[b + 1].item()
        ks, ke = kv_indptr[b].item(), kv_indptr[b + 1].item()
        q = q_bf16[qs:qe].to(torch.float32)                  # [Lq, Hq, D]
        k = k_bf16[ks:ke].to(torch.float32)                  # [Lk, Hkv, D]
        v = v_bf16[ks:ke].to(torch.float32)
        Lq, Hq, D = q.shape
        Lk, Hkv, _ = k.shape
        g = Hq // Hkv
        k = k.repeat_interleave(g, dim=1)                    # [Lk, Hq, D]
        v = v.repeat_interleave(g, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * sm_scale
        if causal:
            # align the last query to the last kv position (append semantics)
            q_pos = torch.arange(Lq, device=q.device) + (Lk - Lq)
            k_pos = torch.arange(Lk, device=q.device)
            mask = k_pos[None, :] > q_pos[:, None]
            scores.masked_fill_(mask[None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("hqk,khd->qhd", p, v)              # [Lq, Hq, D]
        outs.append(o)
    return torch.cat(outs, dim=0).to(DTYPE)


def main():
    if not torch.cuda.is_available():
        print("PARITY_SKIP: no CUDA device", flush=True)
        return 2
    torch.manual_seed(SEED)
    cc = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}", flush=True)

    causal = os.environ.get("CAUSAL", "1") == "1"
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # ---- ragged Q ----
    qo_indptr = torch.arange(0, BATCH + 1, device=DEVICE, dtype=torch.int32) * QO_LEN_PER_REQ
    nnz_q = BATCH * QO_LEN_PER_REQ
    q_bf16 = torch.randn(nnz_q, NUM_QO_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE) * 0.5

    # ---- paged fp4 KV cache ----
    pages_per_req = KV_LEN_PER_REQ // PAGE_SIZE
    total_pages = BATCH * pages_per_req
    # paged_kv_indptr indexes the kv_indices (PAGE) array — cumulative PAGE counts per
    # request, NOT token counts. (Was *KV_LEN_PER_REQ, which walked far past the
    # total_pages-length kv_indices array -> garbage page ids -> OOB KV reads: NaN on the
    # bf16-QK reference path and an illegal access on the A4Q dense-K producer.)
    kv_indptr = torch.arange(0, BATCH + 1, device=DEVICE, dtype=torch.int32) * pages_per_req
    kv_indices = torch.arange(0, total_pages, device=DEVICE, dtype=torch.int32)
    kv_last_page_len = torch.full((BATCH,), PAGE_SIZE, device=DEVICE, dtype=torch.int32)

    # random bf16 K,V then quantize to fp4 (packed uint8 + flat per-16 SF)
    k_bf16 = torch.randn(BATCH * KV_LEN_PER_REQ, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE) * 0.5
    v_bf16 = torch.randn(BATCH * KV_LEN_PER_REQ, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE) * 0.5
    k_packed, k_sf = make_fp4_row(k_bf16)   # [.,Hkv,D/2] uint8, [.,Hkv,D/16] uint8
    v_packed, v_sf = make_fp4_row(v_bf16)

    # NHD paged layout: [num_pages, page_size, num_kv_heads, feat]
    def to_paged(x, feat):
        return x.reshape(total_pages, PAGE_SIZE, NUM_KV_HEADS, feat).contiguous()

    k_cache = to_paged(k_packed, HEAD_DIM // 2)
    v_cache = to_paged(v_packed, HEAD_DIM // 2)
    k_cache_sf = to_paged(k_sf, HEAD_DIM // SF_BLOCK)
    v_cache_sf = to_paged(v_sf, HEAD_DIM // SF_BLOCK)

    # dequantized fp4 tensors for the pure-torch reference (what the kernel "sees")
    k_deq = dequant_nvfp4(k_packed, k_sf)
    v_deq = dequant_nvfp4(v_packed, v_sf)
    # A4Q also quantizes Q to fp4; dequantize it back for the pure-torch reference.
    q_packed, q_sf = make_fp4_row(q_bf16)
    q_deq = dequant_nvfp4(q_packed, q_sf)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    def run(use_nvf4_qk):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            causal=causal,
            pos_encoding_mode="NONE",
            q_data_type=DTYPE,
            kv_data_type=torch.uint8,   # packed NVFP4
            o_data_type=DTYPE,
            use_nvf4_qk=use_nvf4_qk,
        )
        out = wrapper.run(
            q_bf16,
            (k_cache, v_cache),
            kv_cache_sf=(k_cache_sf, v_cache_sf),
        )
        torch.cuda.synchronize()
        return out.to(torch.float32)

    print("running REFERENCE (bf16 QK, use_nvf4_qk=False) ...", flush=True)
    ref = run(False)
    print("running A4Q      (fp4  QK, use_nvf4_qk=True ) ...", flush=True)
    a4q = run(True)

    # pure-torch fp4 reference (ideal fp4-input attention, backend independent)
    pt = pure_torch_attention(q_deq, k_deq, v_deq, qo_indptr, kv_indptr, sm_scale, causal).to(torch.float32)

    def stats(name, a, b):
        diff = (a - b).abs()
        denom = b.abs().clamp_min(1e-3)
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        max_rel = (diff / denom).max().item()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        print(f"  {name:28s} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} "
              f"max_rel={max_rel:.3e} cos={cos:.6f}", flush=True)
        return max_abs, cos

    print("\n=== parity ===", flush=True)
    a4q_vs_ref_abs, a4q_vs_ref_cos = stats("A4Q vs bf16-QK reference", a4q, ref)
    stats("A4Q vs pure-torch fp4 ref", a4q, pt)
    stats("bf16-QK ref vs pure-torch", ref, pt)

    # ---- GATE ----
    # fp4-QK adds only Q quantization error on top of the bf16-QK path. Empirically the
    # per-element output should stay within a small band and remain highly correlated.
    # A miscompute (wrong K byte / SF interleave / fragment map) destroys correlation.
    ref_scale = ref.abs().mean().item()
    ok = (a4q_vs_ref_cos > 0.99) and (a4q_vs_ref_abs < 0.15 * max(ref_scale, 1e-3) + 5e-2)
    print("\nRESULT:", "A4Q_PARITY_PASS" if ok else "A4Q_PARITY_FAIL",
          f"(cos={a4q_vs_ref_cos:.6f}, max_abs={a4q_vs_ref_abs:.4e}, ref_scale={ref_scale:.4e})",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
