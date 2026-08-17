# A4Q overlay: transplant jethac's native NVFP4 fp4-QK^T MMA into eugr FlashInfer
# 0.6.18, JIT-recompiled for the compute_120f family (runs on sm_120a AND sm_121a/GB10).
#
# Verdict (B): Python plumbing + csrc/include overlay, NO full wheel rebuild. FlashInfer
# JIT-compiles the attention kernels from flashinfer/data/{csrc,include} on first
# plan(use_nvf4_qk=True); overlaying the transplanted files + Python plumbing is picked
# up automatically.
#
# Design guarantee: every A4Q code path is gated behind `if constexpr (USE_NVF4_QK)` /
# `if use_nvf4_qk`. With VLLM_NVFP4_A4Q unset (i.e. plan(use_nvf4_qk=False), the default),
# behaviour is byte-identical to stage1 and the Stage-1 nvfp4-KV path is preserved
# (verified: the overlaid prefill.cuh still compiles the fp4-KV module with no _nvf4qk tag).

ARG BASE=eugr-gb10-nvfp4kv:stage1
FROM ${BASE}

# Installed flashinfer inside the eugr stage1 image.
ENV FI=/usr/local/lib/python3.12/dist-packages/flashinfer

# --- 1. Python plumbing overlay (transplanted A4Q, per patch/01) --------------------
COPY overlay/py/decode.py                        ${FI}/decode.py
COPY overlay/py/prefill.py                       ${FI}/prefill.py
COPY overlay/py/jit/attention/modules.py         ${FI}/jit/attention/modules.py
COPY overlay/py/quantization/fp4_quantization.py ${FI}/quantization/fp4_quantization.py
COPY overlay/py/fp4_quantization.py              ${FI}/fp4_quantization.py
# jit/fp4_kv_quantization.py is unchanged from stage1 (kept here for provenance only).

# --- 2. CUDA/header overlay: A4Q hunks transplanted into eugr 0.6.18's OWN files -----
#     prefill.cuh carries the fp4-QK MMA (compute_qk_nvf4 + dense-K producer + load_q);
#     the *.cu/*.jinja carry the USE_NVF4_QK dispatch; fp4_kv_quantization.cu gains
#     nvfp4_q_quant. These are MERGED files, not raw fork copies.
COPY overlay/cuda/include/flashinfer/attention/prefill.cuh \
     ${FI}/data/include/flashinfer/attention/prefill.cuh
COPY overlay/cuda/csrc/batch_prefill.cu                       ${FI}/data/csrc/batch_prefill.cu
COPY overlay/cuda/csrc/single_prefill.cu                      ${FI}/data/csrc/single_prefill.cu
COPY overlay/cuda/csrc/batch_prefill_customize_config.jinja   ${FI}/data/csrc/batch_prefill_customize_config.jinja
COPY overlay/cuda/csrc/single_prefill_customize_config.jinja  ${FI}/data/csrc/single_prefill_customize_config.jinja
COPY overlay/cuda/csrc/batch_prefill_paged_kernel_inst.jinja  ${FI}/data/csrc/batch_prefill_paged_kernel_inst.jinja
COPY overlay/cuda/csrc/single_prefill_kernel_inst.jinja       ${FI}/data/csrc/single_prefill_kernel_inst.jinja
COPY overlay/cuda/csrc/fp4_kv_quantization.cu                 ${FI}/data/csrc/fp4_kv_quantization.cu

# --- 3. kill any AOT jit-cache that would shadow the JIT build (EXP4 wall) -----------
#     A stale AOT .so lacking nvfp4_q_quant / the USE_NVF4_QK instantiation would be
#     loaded before the JIT build. Also drop stale JIT caches + bytecode.
RUN pip uninstall -y flashinfer-jit-cache 2>/dev/null || true ; \
    rm -rf /root/.cache/flashinfer ~/.cache/flashinfer ; \
    find ${FI} -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# --- 4. sm_120f toolchain: nvcc must accept compute_120f (CUDA 12.9+/13.x) -----------
#     The A4Q modules compile with sm120f_nvcc_flags + -DFLASHINFER_ENABLE_NVF4_QK_MMA,
#     set automatically by gen_customize_batch_prefill_module when use_nvf4_qk=True.
#     (Verified: nvcc 13.0 accepts -gencode=arch=compute_120f,code=sm_120f for the
#      mma.sync…kind::mxf4nvf4.block_scale MMA; the -arch=sm_120f shorthand does NOT.)
ENV TORCH_CUDA_ARCH_LIST="12.1a"

# --- 5. (recommended) NUMERICALLY VALIDATE the JIT before serving --------------------
#     The #1 residual risk is SILENT miscompute at head_dim 256 (SF / dense-K layout).
#     Run parity_test.py on the GB10 (needs a GPU) before trusting A4Q:
#       docker run --rm --gpus all <image> python3 /opt/a4q/parity_test.py
COPY overlay/parity_test.py /opt/a4q/parity_test.py
# --- GB10 vLLM spin-wait fix -------------------------------------------------
# https://nacyot.github.io/artifacts/vllm-spin-wait-gb10/  (see GB10_SPIN_WAIT_PATCH.md)
# SpinCondition.wait() busy-loops 1s before sleeping; decode messages arrive every
# few ms, so CPU cores spin at max clock burning 10-20W. On GB10 the CPU and GPU
# share one SoC thermal budget, so that heat throttles the GPU. 1s -> 2ms.
# Guarded: never fails the build if this image has no vLLM at that path.
RUN f=$(find /usr/local/lib -name shm_broadcast.py -path "*device_communicators*" 2>/dev/null | head -1); \
    if [ -n "$f" ]; then sed -i 's/busy_loop_s: float = 1,/busy_loop_s: float = 0.002,/' "$f"; fi
