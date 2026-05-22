# Jamba2-Mini LRU Cache + DMA Optimization Plan
Date: 2026-05-22 09:45

## Problem
Jamba2-Mini (51.57B params, IQ2_M) generates at ~2.5 tok/s on GTX 1070 Ti.
The static pin pool (VITRIOL_PIN_FIRST_N_LAYERS) only covers ~17% of expert
tensors (8/48), providing negligible speedup.

## Root Cause
Static pinning allocates a monolithic VRAM pool sized for full MoE tensors
(~343 MiB each). With only ~1 GiB free VRAM after model weights + compute
buffers, the pool can cover at most 2-3 MoE layers. Jamba2's sparse MoE
(16 layers, 2/16 active experts) is poorly suited to full-tensor pinning.

## Solution: LRU Cache + Predictive Prefetch
Instead of statically pinning entire expert tensors, the VITRIOL LRU cache
dynamically caches individual expert slices (~19 MiB each) in VRAM based on
routing patterns. Combined with predictive prefetching, the GPU can prepare
the next token's experts while computing the current one.

## Configuration Changes
| Parameter | Old Value | New Value |
|---|---|---|
| VITRIOL_PIN_FIRST_N_LAYERS | 4 | 0 (disabled) |
| VITRIOL_LRU_MB | unset (default 2048) | 2048 |
| VITRIOL_OUTPUT_CACHE | unset (disabled) | 1 |
| VITRIOL_PREDICTIVE_PREFETCH | unset (disabled) | 1 |
| VITRIOL_MODE | stream | stream |

## Additional: DMA Bandwidth (CAP_IPC_LOCK)
Without CAP_IPC_LOCK, cuMemHostAlloc allocates pageable memory, requiring
an extra kernel copy for DMA transfers. With sudo vitriol setup, the server
binary gets CAP_IPC_LOCK, enabling page-locked DMA for 20-30% higher
PCIe bandwidth. Must be re-run after each build.

## Expected Impact
- LRU cache: 50-70% reduction in expert H2D transfers (covers hot experts)
- CAP_IPC_LOCK: 20-30% DMA bandwidth improvement
- Predictive prefetch: overlaps compute with DMA for next token
- Combined: ~3.5-5 tok/s (from 2.5 tok/s baseline)

## Comparison to Qwen3.6 (12 tok/s)
Qwen achieved 12 tok/s via:
1. Speculative decoding (MTP N=2, 96% acceptance) — halves effective tokens
2. Dense MoE (all layers) — pin pool covered most layers
3. Smaller model (fewer total params transferred)

Jamba2 lacks MTP heads (no speculative path), so cannot match Qwen's
effective-token-halving advantage. However, Jamba2's SSM layers have
no expert weights to transfer, so non-MoE layers are pure compute (fast).
