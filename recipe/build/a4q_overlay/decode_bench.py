#!/usr/bin/env python3
"""A4Q decode TIMING bench across contexts. Measures decode tok/s for:
   (a) A4Q nvf4 decode (default split-KV)
   (b) A4Q nvf4 decode (disable_split_kv=True)  -- the one accessible runtime knob
   (c) bf16-QK tensor-core decode (non-nvf4 reference, same wrapper)
Reports latency(ms)/tok and effective tok/s. Also prints the plan-selected
split-KV scheduling so we can see how GB10's SM count drives it.
"""
import os, sys, time, torch
import flashinfer
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.quantization.fp4_quantization import nvfp4_quantize_q_cuda

NQ=24; NKV=4; HEAD_DIM=256; PAGE_SIZE=16; SF_BLOCK=16
DEV="cuda"; DT=torch.bfloat16; SEED=1234
ITERS=50; WARM=10

def build(B, QL, KV_LEN):
    torch.manual_seed(SEED)
    ppr=KV_LEN//PAGE_SIZE
    total=B*ppr
    kv_indptr=torch.arange(0,B+1,device=DEV,dtype=torch.int32)*ppr
    kv_indices=torch.arange(0,total,device=DEV,dtype=torch.int32)
    kv_last=torch.full((B,),PAGE_SIZE,device=DEV,dtype=torch.int32)
    q=torch.randn(B*QL,NQ,HEAD_DIM,device=DEV,dtype=DT)*0.5
    k=torch.randn(total*PAGE_SIZE,NKV,HEAD_DIM,device=DEV,dtype=DT)*0.5
    v=torch.randn(total*PAGE_SIZE,NKV,HEAD_DIM,device=DEV,dtype=DT)*0.5
    kp,ksf=nvfp4_quantize_q_cuda(k); vp,vsf=nvfp4_quantize_q_cuda(v)
    def pg(x,f): return x.reshape(total,PAGE_SIZE,NKV,f).contiguous()
    kc=pg(kp,HEAD_DIM//2); vc=pg(vp,HEAD_DIM//2)
    kcsf=pg(ksf,HEAD_DIM//SF_BLOCK); vcsf=pg(vsf,HEAD_DIM//SF_BLOCK)
    # bf16 paged kv for reference path
    kbf=k.reshape(total,PAGE_SIZE,NKV,HEAD_DIM).contiguous()
    vbf=v.reshape(total,PAGE_SIZE,NKV,HEAD_DIM).contiguous()
    return q,(kc,vc),(kcsf,vcsf),(kbf,vbf),kv_indptr,kv_indices,kv_last

def time_run(w, args, kwargs):
    for _ in range(WARM): w.run(*args, **kwargs)
    torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(ITERS): w.run(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/ITERS*1e3  # ms/iter

def bench(B, QL, KV_LEN):
    q,kv,kvsf,kvbf,kv_indptr,kv_indices,kv_last=build(B,QL,KV_LEN)
    ws=torch.empty(512*1024*1024,dtype=torch.uint8,device=DEV)
    res={}
    # (a) A4Q nvf4 default
    w=BatchDecodeWithPagedKVCacheWrapper(ws,kv_layout="NHD",use_tensor_cores=True)
    w.plan(kv_indptr,kv_indices,kv_last,NQ,NKV,HEAD_DIM,PAGE_SIZE,
           q_data_type=DT,kv_data_type=torch.uint8,o_data_type=DT,
           q_len_per_req=QL,use_nvf4_qk=True)
    res['a4q_def']=time_run(w,(q,kv),{'kv_cache_sf':kvsf})
    # (b) A4Q nvf4 disable_split_kv
    try:
        w2=BatchDecodeWithPagedKVCacheWrapper(ws,kv_layout="NHD",use_tensor_cores=True)
        w2.plan(kv_indptr,kv_indices,kv_last,NQ,NKV,HEAD_DIM,PAGE_SIZE,
               q_data_type=DT,kv_data_type=torch.uint8,o_data_type=DT,
               q_len_per_req=QL,use_nvf4_qk=True,disable_split_kv=True)
        res['a4q_nosplit']=time_run(w2,(q,kv),{'kv_cache_sf':kvsf})
    except Exception as e:
        res['a4q_nosplit']=f"ERR:{type(e).__name__}"
    # (c) bf16-QK reference
    try:
        w3=BatchDecodeWithPagedKVCacheWrapper(ws,kv_layout="NHD",use_tensor_cores=True)
        w3.plan(kv_indptr,kv_indices,kv_last,NQ,NKV,HEAD_DIM,PAGE_SIZE,
               q_data_type=DT,kv_data_type=DT,o_data_type=DT,q_len_per_req=QL)
        res['bf16']=time_run(w3,(q,kvbf),{})
    except Exception as e:
        res['bf16']=f"ERR:{type(e).__name__}"
    return res

def main():
    cc=torch.cuda.get_device_capability()
    print(f"device {torch.cuda.get_device_name()} sm_{cc[0]}{cc[1]} SMs={torch.cuda.get_device_properties(0).multi_processor_count}",flush=True)
    B=int(os.environ.get("B","1")); QL=int(os.environ.get("QL","1"))
    print(f"B={B} QL={QL}  (ms/iter, lower=better)")
    print(f"{'ctx':>7} {'a4q_def':>10} {'a4q_nosplit':>12} {'bf16ref':>10} {'a4q_vs_bf16':>12}")
    for KV_LEN in (4096,16384,49152,98304):
        try:
            r=bench(B,QL,KV_LEN)
            a=r['a4q_def']; b=r.get('a4q_nosplit'); c=r.get('bf16')
            ratio = f"{a/c:.3f}x" if isinstance(c,float) else "n/a"
            bs = f"{b:.3f}" if isinstance(b,float) else str(b)
            cs = f"{c:.3f}" if isinstance(c,float) else str(c)
            print(f"{KV_LEN:>7} {a:>10.3f} {bs:>12} {cs:>10} {ratio:>12}",flush=True)
        except Exception as e:
            print(f"{KV_LEN:>7}  ERR {type(e).__name__}: {e}",flush=True)
    print("DONE")

if __name__=="__main__": main()
