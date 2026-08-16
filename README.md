# keys-vLLm.0.27 — Qwen3.8-27B ADay777-Ablit · NVFP4 · A4Q · NVFP4-KV (4M+ pool) · MTP-3 — Single DGX Spark

The complete **4-bit inference pathway** for **abliterated Qwen3.8-27B** on **one NVIDIA DGX Spark (GB10, sm_121a, 121 GB UMA, ~273 GB/s, 48 SMs)**, on the **eugr `spark-vllm-b12x` 0.27 nightly** GB10 build.

**The whole point:** back-porting the upstream **FA2 NVFP4-KV path to GB10** takes the KV pool to **>4M tokens** — which is what makes **1M context at c≈4 fit on a *single* Spark** (the default profile here). **A4Q** native fp4-QKᵀ then adds prefill/TTFT on top. Every number is measured on the hardware and backed by a log in [`bench/`](bench/); the prebuilt images are mirrored so this is self-contained.

> **Verify-first:** claims → `bench/*.log`; engineering record incl. dead-ends → `reports/*.md`; corrections are stated in the open (see [Honest findings](#honest-findings)).

---

## TL;DR — measured on GB10

| Result | Number | Evidence |
|---|---|---|
| **1M context, single Spark (default)** | **4.14× @ 1,048,576 tok** · **4.34M-token pool** | `bench/1m_profileB_serve_evidence.log` |
| **256K profile** | **15× @ 262,144 tok** · 3.93M pool | `bench/mtp_nvfp4kv_bench.log` |
| **nvfp4 KV vs fp8 (same model)** | decode **~neutral (±10%)**, **+64% pool** | `bench/aday_fp8_clean.log` |
| **A4Q fp4-QK prefill** | **+8–10% prefill / −7–9% TTFT @ 48–96K** | `bench/a4q_on_ctx.log` vs `a4q_off_ctx.log` |
| **A4Q correctness** | parity **cos 0.99972** vs bf16-QK, memcheck-clean | `reports/A4Q_FIX_REPORT.md` |
| **nvfp4-KV long-ctx quality** | **6/6 passkey** to 96K | §Quality |
| **MTP depth** | **n ∈ {2,3}** (n≥4 crash: 1 MTP layer) | `bench/mtp_ns*.log` |
| **vs AEON-7 BF16** | this stack is **~2.1× faster** decode | `bench/aeon7_bench.log` |

---

## 🚀 One-shot

```bash
git clone https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark.git
cd keys-vLLm.0.27-*
bash oneshot.sh              # DEFAULT: Profile B — 1M context, c≈4
PROFILE=A bash oneshot.sh    # Profile A — 256K, 15× concurrency
```
One idempotent script: preflight → pull the pinned GB10 image (mirror) → download the model → launch the chosen profile → health-wait → GDN+large-prefill warmup → smoke test.

---

## The stack

`aday777/Qwen3.8-27B-ARA-abliterated-NVFP4-MTP` (uniform NVFP4, abliterated, MTP, head_dim 256) on `eugr vLLM 0.27 / GB10`, with:

- **NVFP4 KV cache** — the FA2 sm120/121 path back-ported from vLLM PR **#49891** onto the eugr build (upstream ships it SM100-only; GB10 is sm_121a). **+64% pool → 4M+ tokens**, ~neutral decode.
- **A4Q** — jethac's native NVFP4 fp4-QKᵀ block-scale MMA transplanted into eugr's FlashInfer 0.6.18 (prefill + decode), numerically validated (cos 0.99972). Prefill/TTFT win, decode-neutral. ⚠️ **model-sensitive** (see caveat).
- **MTP-3** self-speculation · **GDN-hybrid warmup** · **tool-use** (`qwen3_xml`).

### Prebuilt images (mirrored, self-contained)
```
ghcr.io/drowzeys/eugr-gb10-nvfp4kv:a4q2      # eugr 0.27 GB10 + NVFP4-KV back-port + A4Q overlay  (the full stack)
ghcr.io/drowzeys/eugr-gb10-nvfp4kv:stage1    # eugr 0.27 GB10 + NVFP4-KV back-port (no A4Q)
```
Both build from `eugr/spark-vllm-b12x:nightly-20260813`. Full build in [`recipe/`](recipe/) (`Dockerfile`, overlay `.cu`/`.py`, and the reconciled FlashInfer backend).

---

## The recipe

Env: `VLLM_NVFP4_A4Q=1` (A4Q on). Model mounted at `/models/aday777`.

### ⭐ Profile B — 1M context (DEFAULT) → 4.34M pool, 4.14× @ 1M
```bash
vllm serve /models/aday777 --served-model-name qwen38-nvfp4 \
  --host 0.0.0.0 --port 8078 --max-model-len 1048576 \
  --enable-prefix-caching --max-num-batched-tokens 4096 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}' \
  --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.90 --enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```
**Three GB10 + Mamba-hybrid gotchas the 1M profile needs** (each was a boot crash we fixed):
1. **YaRN via `--hf-overrides`** — eugr 0.27 dropped the `--rope-scaling` flag.
2. **`--enable-prefix-caching`** — so the Mamba/GDN state cache and attention KV share allocator pages at long context.
3. **`--max-num-batched-tokens 4096`** — nvfp4-KV block_size is 2848; MTP auto-caps batched tokens at 2048; Mamba-align needs block_size ≤ batched-tokens.

> **1M is a *capacity* result:** the pool holds 4 full-1M sessions concurrently (ideal for long documents already in KV). *Prefilling* 1M tokens is O(n²) attention — minutes-to-hours on one GB10, inherent to attention.

### Profile A — 256K (option) → 3.93M pool, 15× @ 256K
```bash
vllm serve /models/aday777 --served-model-name qwen38-nvfp4 \
  --host 0.0.0.0 --port 8078 --max-model-len 262144 \
  --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.90 --enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Both auto-warm via [`recipe/launch_champion.sh`](recipe/launch_champion.sh) + [`recipe/warm_gdn.py`](recipe/warm_gdn.py) (GDN + large-prefill; avoids first-request JIT spikes).

---

## Benchmarks (GB10, harness `bench_qwen38_tasks.py`)

**nvfp4 vs fp8 KV — same model, same node** (decode tok/s, only `--kv-cache-dtype` changed):

| task | fp8 | nvfp4 | Δ |
|---|--:|--:|--:|
| count | 26.8 | 29.0 | **+8%** |
| reading | 20.8 | 19.1 | −8% |
| essay | 17.6 | 16.1 | −9% |
| list | 17.0 | 16.7 | −2% |
| long_prose | 16.4 | 16.5 | ~0 |
| **pool** | 2.28M | **3.93M** | **+72%** |

**A4Q prefill on/off** (both nvfp4 KV): 48K +7.8% / 96K +10.1% prefill; TTFT −7 to −9%.
**MTP sweep:** n=2 (4.07M pool), **n=3 (best)**, n=4/5 **crash**.
**Quality:** passkey 8K/32K/96K × {50%,90%} = **6/6 PASS**.
**vs AEON-7 BF16** (AEON's own recipe, GB10): count 12.8 / reading 9.4 / essay 7.6 → **~2.1× slower** than this NVFP4 stack.

---

## How this was built

1. **Diagnosed** that eugr 0.27 serves NVFP4-KV only via SM100 trtllm-gen — GB10 (sm_121a) rejected `--kv-cache-dtype nvfp4` outright.
2. **Back-ported** the FA2 sm120/121 nvfp4-KV routing (vLLM PR #49891) onto the eugr b12x backend → nvfp4 KV boots on GB10, pool doubles.
3. **Transplanted A4Q** (jethac's fp4-QKᵀ MMA) into eugr's FlashInfer 0.6.18 — reconciling the sm120 kernel lineage (tiffany940107) into the 0.6.18 backend; JIT-buildable for sm_121a.
4. **Fixed two cudagraph-blocking bugs** (`reports/`): dispatch not forwarding `USE_NVF4_QK` to `KernelTraits` (→ NaN); `run()` missing `q.view(dtype)` after fp4 Q-quant (→ 2× stride → illegal-address under cudagraph capture).
5. **Built a fp4-decode tile-autotuner** — and *proved the GB10 default optimal* (a rigorous negative result; the split-KV scheduler already SM-adapts).
6. **Validated** end-to-end: parity cos 0.99972, memcheck-clean, 6/6 passkey, and the 1M/c4 capacity on a single Spark. Full record (including what didn't help) in [`reports/`](reports/).

## Honest findings

- **A4Q does not boost *decode*** (measured neutral) — it's a *prefill* accelerator.
- **The nvfp4-KV "30% decode cost" was a measurement artifact** (confounded model comparison); clean same-model A/B is ~neutral.
- **No fp4-decode kernel headroom** — tile-sweep default wins.
- **#41684 (hot-token precision) not warranted** — passkey already 6/6 to 96K.
- ⚠️ **A4Q is model-sensitive.** Validated clean on aday777 (cos 0.99972). A different abliterated NVFP4 checkpoint (Grok's `keys-Qwen3.8-27B-Abliterated`) ran coherently with A4Q **off** but produced garbage with A4Q **on** — its abliteration shifted the attention weight-scale distribution the fp4-QK MMA depends on. **Always validate output per-checkpoint before trusting A4Q.**

## Attribution

This repo is the **GB10 integration + fixes + measurement**; the kernel and model foundations are others' work:
- **tiffany940107** — SM120 NVFP4 attention + Qwen3.5 D256-native-GQA + the perf/N64 pipeline (flashinfer #3640 lineage).
- **hikari07jp** — NVFP4 KV-cache adaptation to SM120 (the KV-cache foundation the GB10 path builds on).
- **Jetha Chan (jethac)** — SM121/GB10 enablement + A4Q native fp4-QKᵀ MMA (the fork this builds on).
- **Aday777** — the abliteration work (`Qwen3.8-27B-ARA-abliterated-NVFP4-MTP`, the model served here).
- **eugr** — the GB10 `spark-vllm-b12x` 0.27 build.
- **vLLM** — the 0.27 base (PR #49891 nvfp4-KV routing by ch2lab; flashinfer #3897 sm121 enablement by bkryu).
- Integration, the FA2 back-port reconciliation, the two cudagraph fixes, the tile-autotuner, and all DGX-Spark benchmarking: **Keys (drowzeys)**, with Claude (Anthropic) as the build/debug pair.

## Reproducing
`bash oneshot.sh` (Profile B default) or `PROFILE=A bash oneshot.sh`. Raw logs behind every table in [`bench/`](bench/); full engineering record in [`reports/`](reports/).
