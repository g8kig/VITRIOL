# Plan: Double-Buffer DMA (Replace LRU)
**Date:** 2026-06-09
**Status:** Proposed

## Objective

Replace the current LRU VRAM cache (2 GB pool, 65536 slots, mutex contention, 25% throughput degradation vs direct reads) with a fixed double-buffer ping-pong prefetch mechanism. Only ~14 MB VRAM needed.

## Current State

**File:** `llama.cpp/ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`

- `g_lru_pool` (line 35): 2 GB VRAM allocation, fixed slot size
- `g_lru_map` (line 56): `std::unordered_map<LRUKey, int>` — mutex-guarded, per-access contention
- `g_lru_order` (line 57): `std::list<LRUKey>` — eviction tracking, O(n) remove
- `vitriol_lru_ensure()` (line 768): On cache miss: allocate/evict → DMA → event record → compute waits
- `vitriol_lru_prefetch_async()` (line 841): Fire-and-forget DMA on predictor hit
- `vitriol_predictor_prefetch()` (line 495): Prefetches union of cross-layer + temporal predictions

**Icarus finding:** LRU degraded throughput to 9.19 tok/s vs direct host reads at 12.25 tok/s. The 2 GB pool with 10240 unique expert-layer slots (256×40) has poor hit rate, and DMA overhead from constant eviction/refill dominates.

## Proposed Architecture

### Data Structures (replace LRU)

```cpp
// Two fixed VRAM buffers, sized for one layer's active expert set
#define DOUBLE_BUF_SLOTS   2
#define DOUBLE_BUF_LAYERS  (VITRIOL_MAX_LAYERS + 2)  // enough for entire model

static CUdeviceptr g_double_buf[DOUBLE_BUF_SLOTS];   // 2 × ~7 MB VRAM
static size_t      g_double_buf_size = 0;             // per-slot size
static int         g_double_buf_current = 0;           // ping-pong index

struct DoubleBufEntry {
    uintptr_t tensor_base;      // which tensor's experts
    int       expert_ids[256];  // loaded expert indices
    int       n_experts;        // count
    CUevent   ready_event;      // signaled when DMA completes
    bool      valid;            // true = buffer has usable data
};

static DoubleBufEntry g_double_buf_slot[DOUBLE_BUF_SLOTS];
static DoubleBufEntry g_double_buf_layer[DOUBLE_BUF_LAYERS]; // predicted entries per layer
```

### Pipeline

```
Per token flow:

for each layer_idx in 0..n_layers-1:
    // 1. Attention (no expert weights needed)
    run_attention(layer_idx)

    // 2. Router — determines which experts are active
    run_router(layer_idx) → active_experts[8]

    // 3. Ensure active experts are in the current buffer
    current_buf = g_double_buf[g_double_buf_current]
    if current_buf.tensor_base == tensor_base_of(layer_idx)
       && sets_match(current_buf.expert_ids, active_experts):
        // HIT: experts already resident from prefetch
        cuStreamWaitEvent(compute_stream, current_buf.ready_event, 0)
    else:
        // MISS: synchronously load
        pack_experts(staging_buf, active_experts)         // CPU memcpy to staging
        cuMemcpyHtoDAsync(current_buf.addr, staging_buf, size, stream)
        cuStreamSynchronize(stream)

    // 4. Run MoE FFN from current buffer (all VRAM, no PCIe)
    run_moe_ffn(layer_idx, current_buf.addr, active_experts)

    // 5. Prefetch next layer's predicted experts into the other buffer
    next_buf_idx = (g_double_buf_current + 1) % 2
    predicted = predictor_predict_next(layer_idx, active_experts)
    pack_experts(staging_buf2, predicted.expert_ids)
    cuMemcpyHtoDAsync(g_double_buf[next_buf_idx].addr,
                      staging_buf2, size, g_prefetch_stream)
    cuEventRecord(g_double_buf[next_buf_idx].ready_event, g_prefetch_stream)

    g_double_buf_current = next_buf_idx
```

### Key Changes to `vitriol-cuda-integration.cpp`

| Function | Change |
|----------|--------|
| `vitriol_init()` (line ~280) | Allocate 2 × ~7 MB VRAM buffers instead of 2 GB LRU pool. Create prefetch stream + events. |
| `vitriol_lru_ensure()` → `vitriol_double_buf_ensure()` | Replace LRU lookup with simple slot check. On hit: wait for event. On miss: sync DMA + compute. Return VRAM address of packed experts. |
| `vitriol_lru_prefetch_async()` → `vitriol_double_buf_prefetch()` | Replace LRU prefetch with double-buffer async DMA into the "other" slot. Fire event when done. |
| `vitriol_predictor_prefetch()` | No change to predictor logic, but route prefetches to double-buf instead of LRU. |
| `vitriol_cuda_cleanup_vram()` | Free 2 small buffers instead of 1 large pool. |
| `vitriol_pin_ensure()` | No change — pinned layers bypass double-buffer entirely. |
| `vitriol_lru_ensure_stream()` → `vitriol_prefetch_stream()` | Rename, create only one prefetch stream (not LRU stream). |
| `vitriol_cuda_print_stats()` | Remove LRU stats. Add: prefetch hit rate, miss count, double-buffer utilization. |

### New Helper: Expert Packing

Add a staging buffer in page-locked host RAM:

```cpp
// In vitriol-cuda-integration.cpp
#define STAGING_BUF_SIZE (8 * 1024 * 1024)  // 8 MB (worst case: 8 experts × 1 MB)

static void* g_staging_buf[2] = {nullptr, nullptr};  // double-buffered staging

// Pack non-contiguous expert slices into a contiguous staging buffer
static size_t pack_experts(void *dst, const void *tensor_base,
                           const int *expert_ids, int n_experts,
                           size_t expert_size) {
    size_t total = 0;
    for (int i = 0; i < n_experts; i++) {
        const void *src = (const char *)tensor_base + expert_ids[i] * expert_size;
        memcpy((char *)dst + total, src, expert_size);
        total += expert_size;
    }
    return total;  // total bytes packed
}
```

### Configuration Changes

| Env Var | Old (LRU) | New (Double-buffer) |
|---------|-----------|---------------------|
| `VITRIOL_LRU_MB` | 2048 (pool size) | **Removed** |
| `VITRIOL_LRU_PREDICTIVE_PREFETCH` | 0/1 | Rename to `VITRIOL_DOUBLE_BUF_PREFETCH` (default=1) |
| (new) `VITRIOL_DOUBLE_BUF_PREDICT_AHEAD` | — | How many layers ahead to prefetch (default=1) |

## Implementation Steps

1. **Remove LRU infrastructure** — delete `g_lru_pool`, `g_lru_map`, `g_lru_order`, `g_lru_mtx`, `g_lru_stats`, `g_lru_stream`, `g_lru_event`. Remove `lru_init_pool()`, `lru_ensure_stream()`, LRU-specific lock/unlock helpers.

2. **Add double-buffer globals** — `g_double_buf[2]`, `g_double_buf_slot[2]`, `g_staging_buf[2]`, `g_prefetch_stream`, prefetch events.

3. **Implement `vitriol_double_buf_init()`** — allocate 2 VRAM buffers + 2 staging buffers. Called once from `vitriol_init()`.

4. **Implement `vitriol_double_buf_ensure()`** — the hot path: check if current slot has the right experts → hit (wait event) or miss (sync DMA).

5. **Implement `vitriol_double_buf_prefetch()`** — pack predicted experts into staging, async DMA into the other slot, record event.

6. **Update `vitriol_predictor_prefetch()`** — route predictions to double-buffer instead of LRU.

7. **Update `vitriol_cuda_cleanup_vram()`** — free small buffers.

8. **Update `vitriol_cuda_print_stats()`** — report prefetch hit rate.

9. **Remove `g_vitriol_config.window_size_mb`** (line 23) — no longer needed.

## Verification

1. Build: `cmake --build . --target llama-server`
2. Run with `VITRIOL_MODE=stream VITRIOL_DOUBLE_BUF_PREFETCH=1` on Qwen3.6-35B
3. Compare tok/s against baseline (12.25 tok/s, direct reads, pin=12)
4. Expected: ≥13.5 tok/s (10% improvement from overlap)
5. Check `VITRIOL stats: prefetch hit rate ≥70%` on steady-state generation

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Predictor accuracy <50% | Prefetch overhead > benefit | Fallback to direct host reads if hit rate < threshold |
| Staging buffer memcpy cost | CPU spends time packing | Use AVX-optimized memcpy, or pack in-place if tensor layout permits |
| Event synchronization bug | GPU hangs or reads stale data | Use `cuEventSynchronize` on miss path; fence with `__threadfence()` in kernels if needed |
| Pinned layers + double-buf conflicts | Attempted double-DMA on pinned layers | Check pin lookup first; skip double-buf if pinned |

## Future Extensions

- **Triple buffering** — 3 slots: current compute, prefetch in flight, next prefetch queued
- **Expert reordering** — Store experts in predicted-access order to make `pack_experts()` a no-op (contiguous read from host)
