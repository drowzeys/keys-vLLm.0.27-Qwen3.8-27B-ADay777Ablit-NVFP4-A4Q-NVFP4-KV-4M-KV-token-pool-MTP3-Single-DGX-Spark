"""GB10 (sm_121a) NVFP4 KV-cache writer overlay.

Importing this module:
  1. JIT-compiles the PATCHED nvfp4_kv_cache_kernels.cu (+ overlay_binding.cpp)
     for sm_121a and registers torch.ops.vllm_gb10.reshape_and_cache_nvfp4.
  2. Monkeypatches torch.ops._C_cache_ops.reshape_and_cache_flash so that, on
     device-capability-family 120 with kv_cache_dtype in {"nvfp4","nvfp4_4over6"},
     the write is routed to the overlay op (which writes LINEAR V scales that
     the FlashInfer FA2 sm12x paged reader can consume).

Build once at image-build time (AOT) by setting VLLM_GB10_NVFP4KV_AOT=1 and
running `python -c "import vllm_gb10_nvfp4kv"` in the Dockerfile, so the first
real request does not pay JIT latency.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# --- locations (overridable via env) ---------------------------------------
_HERE = Path(__file__).resolve().parent
# csrc include root of a vLLM checkout matching the image's vLLM commit, so
# that `#include "libtorch_stable/..."` and `#include "core/registration.h"`
# resolve. Set by the Dockerfile.
_VLLM_CSRC = Path(os.environ.get("VLLM_GB10_CSRC", "/opt/vllm-src/csrc"))

# sm_121a for GB10. flashinfer 0.6.18 uses the identical gencode for its own
# nvfp4 sm120 module, and the stock eugr _C_stable_libtorch.abi3.so is compiled
# with `arch = sm_121a`, so this kernel is proven to build for this target.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")

_EXTRA_CUDA_FLAGS = [
    "-O3",
    "-gencode=arch=compute_121a,code=sm_121a",
    "-DNVFP4_ENABLE_ELTS16=1",
    "-DENABLE_NVFP4_SM120",  # defensive; the .cu itself is unconditional
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]


def _build():
    load(
        name="vllm_gb10_nvfp4kv",
        sources=[
            str(_HERE / "nvfp4_kv_cache_kernels.cu"),
            str(_HERE / "overlay_binding.cpp"),
        ],
        extra_include_paths=[str(_VLLM_CSRC)],
        extra_cuda_cflags=_EXTRA_CUDA_FLAGS,
        extra_cflags=["-O3"],
        is_python_module=False,  # register ops globally, no py module returned
        verbose=bool(int(os.environ.get("VLLM_GB10_VERBOSE", "0"))),
    )


_build()


def _is_family_120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


# --- route NVFP4 KV writes to the linear-V-scale overlay op on GB10 ---------
if _is_family_120():
    _cache_ops = torch.ops._C_cache_ops
    _orig_flash = _cache_ops.reshape_and_cache_flash

    def _reshape_and_cache_flash_gb10(
        key, value, key_cache, value_cache, slot_mapping,
        kv_cache_dtype, k_scale, v_scale,
    ):
        if kv_cache_dtype in ("nvfp4", "nvfp4_4over6"):
            return torch.ops.vllm_gb10.reshape_and_cache_nvfp4(
                key, value, key_cache, value_cache, slot_mapping,
                kv_cache_dtype, k_scale, v_scale,
            )
        return _orig_flash(
            key, value, key_cache, value_cache, slot_mapping,
            kv_cache_dtype, k_scale, v_scale,
        )

    # Overwrite the Python-visible OpOverloadPacket entry. vLLM calls
    # `torch.ops._C_cache_ops.reshape_and_cache_flash(...)` positionally, so a
    # thin Python shim in front of the packet is sufficient. (If a callsite
    # instead uses the `.default` overload, patch that attribute too.)
    _cache_ops.reshape_and_cache_flash = _reshape_and_cache_flash_gb10
