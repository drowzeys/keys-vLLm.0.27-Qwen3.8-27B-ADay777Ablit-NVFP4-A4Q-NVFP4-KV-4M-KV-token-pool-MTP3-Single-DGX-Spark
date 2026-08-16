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

## How 1M context @ c≈4 was enabled (on a *single* Spark)

The KV cache is a **shared token pool**: any mix of requests fits as long as **Σ(context lengths) ≤ pool**. So "1M context at concurrency C" needs a pool of **C × 1M** tokens. That is exactly the constraint the nvfp4-KV back-port breaks:

| KV dtype | pool (GB10, aday777) | concurrency @ 1M |
|---|--:|--:|
| fp8 (8-bit) | ~2.1M tokens | **~2.0×** |
| **nvfp4 (4-bit)** | **4.34M tokens** | **4.14×** ✅ |

**4-bit KV halves the per-token KV footprint → doubles the pool → doubles the 1M concurrency** (2.0× → 4.14×). That is the entire reason c≈4 @ 1M fits on one 121 GB Spark. Measured: `GPU KV cache size: 4,341,760 tokens` / `Maximum concurrency for 1,048,576 tokens per request: 4.14x` (`bench/1m_profileB_serve_evidence.log`), YaRN output coherent.

**The four things that had to line up** (each was a diagnosed boot failure — see [How this was built](#how-this-was-built)):
1. **NVFP4 KV on GB10** — the FA2 sm120/121 routing back-port (PR #49891); without it GB10 rejects `--kv-cache-dtype nvfp4` and you're stuck at the fp8 ~2.1M pool (~2× @ 1M).
2. **YaRN 4× via `--hf-overrides`** — extends Qwen3.8's 262 144 native window to 1 048 576 (the `--rope-scaling` flag was dropped in 0.27).
3. **`--enable-prefix-caching`** — required so the Mamba/GDN state cache and the attention KV share allocator pages at 1M (Qwen3.8 is a GDN/attention hybrid).
4. **`--max-num-batched-tokens 4096`** — the nvfp4-KV block_size is 2848; MTP auto-caps batched tokens at 2048; the Mamba-align check needs block_size ≤ batched-tokens.

> **This is a *capacity* result** — the pool holds **4 full-1M sessions concurrently** (ideal for long documents already resident in KV). Actually *prefilling* 1M tokens is O(n²) attention (minutes-to-hours on one GB10), inherent to attention, not this stack. For c=8 @ full-1M you'd need an ~8M pool → TP across multiple Sparks.

---

## A4Q + NVFP4-KV: the 4-bit attention pathway, and the autotune closure

**The two halves of 4-bit attention are complementary — and this repo lands both on GB10.**

- **NVFP4-KV (storage side).** K/V are stored in 4-bit (e2m1 packed + block scales) → **halves KV memory/bandwidth** → the 4M+ token pool. Lineage: **hikari07jp** adapted NVFP4 KV to SM120 → this repo back-ports the FA2 routing to **SM121/GB10** (PR #49891) so `--kv-cache-dtype nvfp4` runs on the Spark at all.
- **A4Q (compute side).** The QKᵀ matmul runs in **native 4-bit** (fp4 Q × fp4 K via the `mma.sync…mxf4nvf4.block_scale` MMA), reading the 4-bit K **directly, no dequant**. Lineage: **tiffany940107** (SM120 NVFP4 attention + Qwen3.5 D256-native-GQA + perf/N64 pipeline) → **Jetha Chan** (SM121 + A4Q fp4-QK) → this repo transplants it into eugr's **FlashInfer 0.6.18** and fixes it to boot with cudagraph (parity cos 0.99972).

Together: **4-bit KV storage + 4-bit QK compute = the complete 4-bit attention pathway on GB10.**

**The autotune question — closed on *both* prefill and decode:**

- **Prefill → autotuned (active).** eugr's FlashInfer autotuner runs at boot and caches **~96 GEMM/attention configs** for the running arch (`[Autotuner]: Saved 96 configs … /121a/…/autotune_configs.json`). The prefill path is tuned per-shape out of the box. **A4Q** then adds fp4-QK on top for **+8–10% prefill / −7–9% TTFT at 48–96K** (scales with context).

- **Decode → proven optimal (closed, no autotuner needed).** There is **no runtime *attention* autotuner** anywhere — not in eugr, not in jethac's fork (FlashInfer's autotuner is GEMM/MoE only; the fp4-decode kernel's `CTA_TILE_KV` / `NUM_STAGES` / warps / smem-split are **compile-time template params**). So we **built** a compile-time tile-sweep autotuner (JIT variants, zero-rebuild env-override A/B) and measured rigorously — interleaved **7×100-iter min-latency** repeats: the **GB10 default wins** (best alternative ≤2.5%, inside the noise floor). Mechanism: fp4 decode is **occupancy-bound**, and the `num_sm`-driven **split-KV scheduler already SM-adapts**. The kernel is already at its GB10 optimum. Full method + numbers: [`reports/A4Q_TILE_SWEEP_REPORT.md`](reports/A4Q_TILE_SWEEP_REPORT.md), [`A4Q_DECODE_AUTOTUNE_REPORT.md`](reports/A4Q_DECODE_AUTOTUNE_REPORT.md).

**Net: prefill is autotuned, decode is proven-optimal — the 4-bit pathway is tuned end-to-end on GB10, and the tuning question is closed with evidence** (a real result either way: a win where there was one, a rigorous "already optimal" where there wasn't).

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

**A4Q on vs off** — aday777, both nvfp4 KV, only `VLLM_NVFP4_A4Q` toggled (`bench/a4q_on_ctx.log` vs `a4q_off_ctx.log`):

| ctx | prefill on | prefill off | prefill Δ | TTFT on | TTFT off | TTFT Δ | decode on | decode off |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 4K | 2128 | 2110 | +0.9% | 2.05 s | 2.07 s | −1% | 14.2 | 18.0 |
| 16K | 2013 | 2032 | −0.9% | 8.63 s | 8.55 s | +1% | 17.9 | 17.4 |
| 48K | 1568 | 1455 | **+7.8%** | 33.2 s | 35.8 s | **−7.3%** | 17.1 | 16.0 |
| 96K | 1109 | 1007 | **+10.1%** | 93.9 s | 103.3 s | **−9.2%** | 15.8 | 16.4 |

→ **A4Q's win is prefill/TTFT and it scales with context** (negligible ≤16K, +8–10% / −7–9% at 48–96K, larger at 256K). **Decode is neutral** — the on/off decode deltas are within MTP-acceptance run-to-run noise (4K even shows on *lower*), confirming A4Q is a *prefill* accelerator, not a decode boost.
**MTP sweep:** n=2 (4.07M pool), **n=3 (best)**, n=4/5 **crash**.
**Quality:** passkey 8K/32K/96K × {50%,90%} = **6/6 PASS**.
**vs AEON-7 BF16** (AEON's own recipe, GB10): count 12.8 / reading 9.4 / essay 7.6 → **~2.1× slower** than this NVFP4 stack.

---

## ⚠️ Client note — set a high `max_tokens` (≥ 20K)

**Qwen3.8-27B (especially this abliterated build) is *very* wordy.** With a low client-side `max_tokens`, replies get cut off with `finish_reason='length'` — and when the cutoff lands inside a tool call, agent frameworks retry the truncated call and eventually bail ("Response truncated due to output length limit"). This is **not** a serving bug: the serve is streaming fine, the model just runs past a small ceiling.

- **Fix:** raise your client's per-request output cap to **≥ 20,000** (`max_tokens: 20000`). Bump higher for long-form / heavy tool-loop work. The model's 1M `max-model-len` easily accommodates it.
- **Hermes agent** users specifically: set `model.max_tokens: 20000` (or more) in `~/.hermes/config.yaml`, then restart the gateway so the change is picked up. At the old default (`8192`) this model reliably truncates mid-tool-call.
- This is a *client* setting; the serve imposes no such limit.

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
