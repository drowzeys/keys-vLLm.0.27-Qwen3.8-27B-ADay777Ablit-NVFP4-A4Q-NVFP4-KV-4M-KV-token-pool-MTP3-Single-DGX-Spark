#!/usr/bin/env bash
# Launch the nvfp4-KV champion AND warm the large-prefill path so Hermes's
# ~20K first-load dump never hits a cold serve (the "first stuck prompt" fix).
set -euo pipefail
PORT="${PORT:-8078}"
docker rm -f qwen38-nvfp4kv-champ >/dev/null 2>&1 || true
docker run -d --restart unless-stopped --name qwen38-nvfp4kv-champ --gpus all --ipc=host --network host \
  -v $HOME/models-local-qwen38:/models \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  eugr-gb10-nvfp4kv:stage1 \
  vllm serve /models/Qwen3.8-27B-ARA-abliterated-NVFP4-MTP --served-model-name qwen38-nvfp4 \
    --host 0.0.0.0 --port "$PORT" --max-model-len 262144 --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.90 \
    --enable-flashinfer-autotune --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3}" >/dev/null
echo "launched; waiting for health then warming large-prefill (26K)..."
python3 - "$PORT" <<PY
import json,sys,time,urllib.request
port=sys.argv[1]; base=f"http://localhost:{port}"
for _ in range(180):
    try: urllib.request.urlopen(base+"/v1/models",timeout=3); break
    except Exception: time.sleep(5)
p=("Unified memory bandwidth bounds decode throughput on edge accelerators today. "*2200)+"\nReply with: OK"
b=json.dumps({"model":"qwen38-nvfp4","messages":[{"role":"user","content":p}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}).encode()
try:
    urllib.request.urlopen(urllib.request.Request(base+"/v1/chat/completions",data=b,headers={"Content-Type":"application/json"}),timeout=300).read()
    print("  warmup ok (~26K prefill compiled) — Hermes 20K first-load will be fast")
except Exception as e: print("  WARN warmup:",e)
PY

# GDN shape-sweep warmup (causal_conv1d + attn buckets, avoid mid-inference JIT spikes)
python3 /home/keyspark/warm_gdn.py "http://localhost:${PORT:-8078}"
