// SPDX-License-Identifier: Apache-2.0
// Overlay op registration for the sm120/sm121 (GB10) NVFP4 KV-cache writer.
//
// The stock eugr image reaches the NVFP4 store through
//   torch.ops._C_cache_ops.reshape_and_cache_flash(..., "nvfp4", ...)
// which links against the PRE-patch reshape_and_cache_nvfp4_dispatch: that
// version ALWAYS writes V scale factors in the SM100 trtllm-gen 4-token
// swizzle. The FlashInfer FA2 sm12x paged reader needs LINEAR V scales.
//
// We cannot re-register _C_cache_ops::reshape_and_cache_flash (duplicate def),
// so we compile the PATCHED nvfp4_kv_cache_kernels.cu (linear V scale on
// device major >= 12) and expose it under a fresh namespace:
//   torch.ops.vllm_gb10.reshape_and_cache_nvfp4(...)
// The Python side (or the bundled sitecustomize monkeypatch) routes NVFP4 KV
// writes to this op on device-capability-family 120.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include "core/registration.h"

#include <string>

// Defined in nvfp4_kv_cache_kernels.cu (this overlay's patched copy).
void reshape_and_cache_nvfp4_dispatch(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    torch::stable::Tensor& slot_mapping, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const std::string& kv_cache_dtype);

// Adapter: identical argument order to reshape_and_cache_flash so callers only
// change the op name. TORCH_BOX passes stable::Tensor by value; we bind the
// lvalue references the dispatch expects to those locals.
void reshape_and_cache_nvfp4(torch::stable::Tensor key,
                             torch::stable::Tensor value,
                             torch::stable::Tensor key_cache,
                             torch::stable::Tensor value_cache,
                             torch::stable::Tensor slot_mapping,
                             std::string kv_cache_dtype,
                             torch::stable::Tensor k_scale,
                             torch::stable::Tensor v_scale) {
  reshape_and_cache_nvfp4_dispatch(key, value, key_cache, value_cache,
                                   slot_mapping, k_scale, v_scale,
                                   kv_cache_dtype);
}

STABLE_TORCH_LIBRARY_FRAGMENT(vllm_gb10, ops) {
  ops.def(
      "reshape_and_cache_nvfp4(Tensor key, Tensor value,"
      "                        Tensor! key_cache, Tensor! value_cache,"
      "                        Tensor slot_mapping, str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(vllm_gb10, CUDA, ops) {
  ops.impl("reshape_and_cache_nvfp4", TORCH_BOX(&reshape_and_cache_nvfp4));
}
