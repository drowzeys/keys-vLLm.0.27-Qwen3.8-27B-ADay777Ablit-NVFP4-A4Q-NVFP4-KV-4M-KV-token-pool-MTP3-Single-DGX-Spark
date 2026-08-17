# GB10 vLLM spin-wait patch

**Applies to:** any vLLM served on NVIDIA DGX Spark / GB10 (sm_121a).
**Source:** https://nacyot.github.io/artifacts/vllm-spin-wait-gb10/

## What

vLLM's `SpinCondition.wait()` busy-loops for `busy_loop_s = 1` second before it
falls back to sleeping. During decode, inter-process messages arrive at
**millisecond** intervals, so that 1s threshold is never reached: the sleep path is
never taken and CPU performance cores spin at full clock (3.9 GHz), burning
10–20 W as heat.

On GB10 this matters more than on a discrete-GPU box, because **the CPU and GPU
share one SoC package and one thermal budget** — wasted CPU heat competes directly
with the GPU. Upstream reports the SoC reaching 96 °C (shutdown ~87 °C) while the
GPU itself was only 72.7 °C.

## The fix

One line, in `vllm/distributed/device_communicators/shm_broadcast.py`:

```
busy_loop_s: float = 1,   ->   busy_loop_s: float = 0.002,
```

Upstream reports CPU 333% → 89%, SoC −11 °C, **no throughput change**.

## Honest measurement on our fleet (2026-08-17)

On a **TP=1 single-GPU** serve we measured **no significant change**: container CPU
93.4% (unpatched) vs 92.9% (patched), same load, same container, steady state.

The upstream 333% → 89% figure implies ~3.3 cores spinning, which requires
**multiple ranks** waiting on `shm_broadcast`. At TP=1 there is a single engine-core
process and that ~93% is real scheduling/sampling work, not spin.

**Conclusion:** apply it anyway — it is free and carries no throughput cost — but
expect the actual win on **TP≥2** deployments, not on single-GPU serves.

## How to apply

**Preferred — bake it into the image** (a `docker exec` sed is lost the moment the
container is recreated, which every relaunch does):

```dockerfile
RUN f=$(find /usr/local/lib -name shm_broadcast.py -path "*device_communicators*" 2>/dev/null | head -1); \
    if [ -n "$f" ]; then sed -i 's/busy_loop_s: float = 1,/busy_loop_s: float = 0.002,/' "$f"; fi
```

**Path differs by image:** `dist-packages` on stock `vllm/vllm-openai` and most
community GB10 builds, `site-packages` on `ghcr.io/aeon-7/aeon-vllm-ultimate`.
The `find` above handles both.

**Quick check on a running container:**

```bash
docker exec <container> grep -n 'busy_loop_s: float' \
  $(docker exec <container> find /usr/local/lib -name shm_broadcast.py -path "*device_communicators*" | head -1)
```
