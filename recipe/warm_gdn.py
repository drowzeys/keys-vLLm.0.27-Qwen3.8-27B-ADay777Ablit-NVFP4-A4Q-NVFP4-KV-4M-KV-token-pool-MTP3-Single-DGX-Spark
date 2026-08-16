import json,sys,time,urllib.request
base=sys.argv[1] if len(sys.argv)>1 else "http://localhost:8078"
def req(ntok,maxtok,label):
    p=("Unified memory bandwidth bounds decode throughput on edge accelerators today. "*max(1,ntok//12))+"\nSummarize in one sentence."
    b=json.dumps({"model":"qwen38-nvfp4","messages":[{"role":"user","content":p}],"max_tokens":maxtok,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    t0=time.time()
    try:
        urllib.request.urlopen(urllib.request.Request(base+"/v1/chat/completions",data=b,headers={"Content-Type":"application/json"}),timeout=300).read()
        print(f"  warm {label}: {time.time()-t0:.1f}s")
    except Exception as e: print(f"  warm {label} ERR: {e}")
# sweep prefill-conv shape buckets + a multi-token gen to warm decode-side GDN
for n,mt,l in [(400,8,"0.5K"),(4000,8,"4K"),(12000,8,"12K"),(26000,8,"26K"),(400,200,"decode-200tok")]:
    req(n,mt,l)
print("GDN warmup sweep done")
