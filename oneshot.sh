#!/usr/bin/env bash
# =============================================================================
# ONE-SHOT: Qwen3.8-27B ADay777-Ablit · NVFP4 · A4Q · NVFP4-KV · MTP-3 on GB10
# =============================================================================
# Idempotent; re-run any time (skips finished steps).
#   bash oneshot.sh              # DEFAULT: Profile B — 1M context, c~4  (4.34M pool)
#   PROFILE=A bash oneshot.sh    # Profile A — 256K, 15x concurrency     (3.93M pool)
# Requires: a DGX Spark (GB10/sm_121a), Docker + NVIDIA runtime, ~60 GB free disk.
#
# ⚠️  CLIENT NOTE: Qwen3.8-27B (this abliterated build) is VERY wordy — set your
#     client's per-request max_tokens to >= 20000, or replies truncate
#     (finish_reason=length) and agent tool-calls retry-then-bail. The serve
#     imposes no output limit; this is purely a client-side cap. (README "Client note")
# =============================================================================
set -euo pipefail

IMAGE="ghcr.io/drowzeys/eugr-gb10-nvfp4kv:a4q2"            # full stack: eugr 0.27 GB10 + NVFP4-KV + A4Q
IMAGE_FALLBACK="eugr/spark-vllm-b12x:nightly-20260813"    # base GB10 build (won't have nvfp4-KV/A4Q)
MODEL_REPO="aday777/Qwen3.8-27B-ARA-abliterated-NVFP4-MTP"
MODELS_DIR="${MODELS_DIR:-$HOME/models-nvfp4-a4q}"
MODEL_DIR="$MODELS_DIR/aday777"
PORT="${PORT:-8078}"
PROFILE="${PROFILE:-B}"     # B = 1M (default), A = 256K
NAME="qwen38-nvfp4-a4q"

say(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die(){ printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

say "0/5 preflight"
command -v docker >/dev/null || die "docker not installed"
docker info >/dev/null 2>&1 || die "docker daemon unreachable"
arch="$(docker run --rm --gpus all ubuntu:22.04 sh -c 'nvidia-smi --query-gpu=name --format=csv,noheader' 2>/dev/null || true)"
echo "  GPU: ${arch:-unknown}"
case "$arch" in *GB10*|*Spark*) : ;; *) echo "  WARNING: built for GB10 (sm_121a); '$arch' may not run the fp4 kernels";; esac

say "1/5 pull runtime image (pinned GB10 build w/ NVFP4-KV + A4Q)"
docker image inspect "$IMAGE" >/dev/null 2>&1 && echo "  present" || \
  docker pull "$IMAGE" 2>/dev/null || { echo "  mirror unreachable — falling back to base ($IMAGE_FALLBACK); NVFP4-KV/A4Q will NOT be available"; IMAGE="$IMAGE_FALLBACK"; docker pull "$IMAGE"; }

say "2/5 fetch model -> $MODEL_DIR"
if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then echo "  present"; else
  mkdir -p "$MODEL_DIR"
  python3 - "$MODEL_REPO" "$MODEL_DIR" <<'PY' || die "download failed (pip install -U huggingface_hub)"
import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2]); print("  downloaded")
PY
fi

say "3/5 launch (profile $PROFILE)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
COMMON=(--restart unless-stopped --name "$NAME" --gpus all --ipc=host --network host
  -v "$MODELS_DIR":/models
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e VLLM_NVFP4_A4Q=1 "$IMAGE"
  vllm serve /models/aday777 --served-model-name qwen38-nvfp4 --host 0.0.0.0 --port "$PORT"
  --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.90 --enable-flashinfer-autotune
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}')
if [ "$PROFILE" = "A" ]; then
  docker run -d "${COMMON[@]}" --max-model-len 262144 >/dev/null || die "launch failed"
  echo "  Profile A: 256K, 15x @ 256K"
else
  docker run -d "${COMMON[@]}" --max-model-len 1048576 --enable-prefix-caching --max-num-batched-tokens 4096 \
    --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}' >/dev/null || die "launch failed"
  echo "  Profile B: 1M context (YaRN 4x), c~4.14 @ 1M"
fi

say "4/5 wait for health (first run compiles FP4 kernels; up to ~12 min)"
for i in $(seq 1 160); do curl -sf -m3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "  healthy"; break; }
  [ "$i" = 160 ] && { docker logs --tail 40 "$NAME"; die "not healthy"; }; sleep 5; done

say "5/5 warmup (GDN + large-prefill) + smoke"
[ -f "$(dirname "$0")/recipe/warm_gdn.py" ] && python3 "$(dirname "$0")/recipe/warm_gdn.py" "http://localhost:$PORT" || echo "  (warm_gdn.py not found; skipping)"
out=$(curl -s -m60 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-nvfp4","messages":[{"role":"user","content":"Reply with exactly: READY"}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null || true)
[ -n "$out" ] || die "smoke test no response"
printf '\n\033[1;32m✅ DONE.\033[0m serving at http://localhost:%s/v1  (model: qwen38-nvfp4) — replied: %s\n' "$PORT" "$out"

# --- CLIENT REMINDER (do not skip) ---------------------------------------------
printf '\n\033[1;33m⚠️  CLIENT max_tokens ≥ 20000\033[0m\n'
printf '   Qwen3.8-27B (this abliterated build) is VERY wordy. With a low client\n'
printf '   max_tokens it truncates (finish_reason=length); mid-tool-call that makes\n'
printf '   agents retry-then-bail ("Response truncated due to output length limit").\n'
printf '   Set your client cap to >= 20000 (Hermes: model.max_tokens: 20000 in\n'
printf '   ~/.hermes/config.yaml, then restart the gateway). The serve imposes no such\n'
printf '   limit — 1M max-model-len easily covers it. See README "Client note".\n'
