#!/usr/bin/env python3
"""Diagnostic: isolate which tensors are NaN in the parity harness."""
import math, sys, torch
import flashinfer
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.quantization.fp4_quantization import nvfp4_quantize_q_cuda

NUM_QO_HEADS=24; NUM_KV_HEADS=4; HEAD_DIM=256; PAGE_SIZE=16
KV_LEN_PER_REQ=64; QO_LEN_PER_REQ=32; BATCH=3
DEVICE="cuda"; DTYPE=torch.bfloat16; SF_BLOCK=16; SEED=1234

def nan(name,t):
    t=t.float()
    print(f"  {name:26s} shape={tuple(t.shape)} nan={torch.isnan(t).sum().item()} "
          f"inf={torch.isinf(t).sum().item()} min={t.min().item():.3e} max={t.max().item():.3e}", flush=True)

def main():
    torch.manual_seed(SEED)
    cc=torch.cuda.get_device_capability()
    print(f"device {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]}",flush=True)
    import os
    causal = os.environ.get("CAUSAL","1")=="1"
    sm_scale=1.0/math.sqrt(HEAD_DIM)
    qo_indptr=torch.arange(0,BATCH+1,device=DEVICE,dtype=torch.int32)*QO_LEN_PER_REQ
    nnz_q=BATCH*QO_LEN_PER_REQ
    q_bf16=torch.randn(nnz_q,NUM_QO_HEADS,HEAD_DIM,device=DEVICE,dtype=DTYPE)*0.5
    pages_per_req=KV_LEN_PER_REQ//PAGE_SIZE
    total_pages=BATCH*pages_per_req
    kv_indptr=torch.arange(0,BATCH+1,device=DEVICE,dtype=torch.int32)*pages_per_req
    kv_indices=torch.arange(0,total_pages,device=DEVICE,dtype=torch.int32)
    kv_last_page_len=torch.full((BATCH,),PAGE_SIZE,device=DEVICE,dtype=torch.int32)
    k_bf16=torch.randn(BATCH*KV_LEN_PER_REQ,NUM_KV_HEADS,HEAD_DIM,device=DEVICE,dtype=DTYPE)*0.5
    v_bf16=torch.randn(BATCH*KV_LEN_PER_REQ,NUM_KV_HEADS,HEAD_DIM,device=DEVICE,dtype=DTYPE)*0.5
    k_packed,k_sf=nvfp4_quantize_q_cuda(k_bf16)
    v_packed,v_sf=nvfp4_quantize_q_cuda(v_bf16)
    print("=== quant outputs (dtype/shape) ===",flush=True)
    for nm,t in [("k_packed",k_packed),("k_sf",k_sf),("v_packed",v_packed),("v_sf",v_sf)]:
        print(f"  {nm:10s} dtype={t.dtype} shape={tuple(t.shape)}",flush=True)
    def to_paged(x,feat):
        return x.reshape(total_pages,PAGE_SIZE,NUM_KV_HEADS,feat).contiguous()
    k_cache=to_paged(k_packed,HEAD_DIM//2); v_cache=to_paged(v_packed,HEAD_DIM//2)
    k_cache_sf=to_paged(k_sf,HEAD_DIM//SF_BLOCK); v_cache_sf=to_paged(v_sf,HEAD_DIM//SF_BLOCK)
    ws_mb=int(os.environ.get("WS_MB","256"))
    workspace=torch.empty(ws_mb*1024*1024,dtype=torch.uint8,device=DEVICE)
    wrapper=BatchPrefillWithPagedKVCacheWrapper(workspace,kv_layout="NHD")
    def run(use_nvf4_qk):
        wrapper.plan(qo_indptr,kv_indptr,kv_indices,kv_last_page_len,NUM_QO_HEADS,NUM_KV_HEADS,
                     HEAD_DIM,PAGE_SIZE,causal=causal,pos_encoding_mode="NONE",q_data_type=DTYPE,
                     kv_data_type=torch.uint8,o_data_type=DTYPE,use_nvf4_qk=use_nvf4_qk)
        out=wrapper.run(q_bf16,(k_cache,v_cache),kv_cache_sf=(k_cache_sf,v_cache_sf))
        torch.cuda.synchronize(); return out.to(torch.float32)
    mode=sys.argv[1] if len(sys.argv)>1 else "full"
    if mode=="a4q_only":
        print("=== A4Q ONLY run (use_nvf4_qk=True) ===",flush=True)
        a4q=run(True); nan("a4q",a4q)
        for b in range(BATCH):
            s,e=qo_indptr[b].item(),qo_indptr[b+1].item()
            print(f"    req{b} nan={torch.isnan(a4q[s:e]).sum().item()}",flush=True)
        print("A4Q_ONLY_DONE",flush=True); return 0
    print(f"=== REFERENCE run (use_nvf4_qk=False) causal={causal} ===",flush=True)
    ref=run(False); nan("ref",ref)
    # per-request nan
    for b in range(BATCH):
        s,e=qo_indptr[b].item(),qo_indptr[b+1].item()
        seg=ref[s:e]
        print(f"    req{b} rows[{s}:{e}] nan={torch.isnan(seg).sum().item()}",flush=True)
    # per-row nan within request 0
    seg0=ref[0:QO_LEN_PER_REQ]  # [32,24,256]
    rownan=torch.isnan(seg0).any(dim=(1,2))
    print(f"    req0 rows-with-nan: {torch.nonzero(rownan).flatten().tolist()}",flush=True)
    if len(sys.argv)>1 and sys.argv[1]=="ref_only":
        print("REF_ONLY_DONE",flush=True); return 0
    print("=== A4Q run (use_nvf4_qk=True) ===",flush=True)
    a4q=run(True); nan("a4q",a4q)
    print("DIAG_DONE",flush=True); return 0

if __name__=="__main__":
    sys.exit(main())
