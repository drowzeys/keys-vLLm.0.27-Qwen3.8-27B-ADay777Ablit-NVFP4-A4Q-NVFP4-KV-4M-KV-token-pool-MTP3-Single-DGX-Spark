# Building the images (reproducibility)
FROM `eugr/spark-vllm-b12x:nightly-20260813` (GB10 base):
1. `stage1/`  -> `eugr-gb10-nvfp4kv:stage1` : NVFP4-KV FA2 back-port (PR #49891) + writer overlay
   `docker build -f stage1/Dockerfile.combined -t eugr-gb10-nvfp4kv:stage1 stage1/`
2. `a4q_overlay/` (FROM :stage1) -> `:a4q` : A4Q fp4-QK MMA transplant into FlashInfer 0.6.18
   `docker build -f a4q_overlay/Dockerfile -t eugr-gb10-nvfp4kv:a4q a4q_overlay/`
3. `:a4q` + `a4q2_backend_flashinfer.py` -> `:a4q2` : the A4Q-wired vLLM backend
Prebuilt mirrors: ghcr.io/drowzeys/eugr-gb10-nvfp4kv:{stage1,a4q2}
