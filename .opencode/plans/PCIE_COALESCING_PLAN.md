# Plan: PCIe Transaction Coalescing (Expert Batching)
**Date:** 2026-06-09
**Status:** Proposed

## Objective

Reduce CUDA driver overhead by batching multiple per-expert DMA calls into a single transfer. Currently 8 separate `cuMemcpyHtoDAsync` calls per MoE layer waste ~48 µs on dispatch overhead. Batching reduces this to ~6 µs.

## Current State

**File:** `llama.cpp/ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`

In `vitriol_lru_ensure()` (line 768-838), each expert is transferred individually:

```cpp
// Called once per expert_idx in the active set
r = cuMemcpyHtoDAsync(dst, expert_data, expert_size, g_lru_stream);  // 6µs overhead
r = cuEventRecord(g_lru_event, g_lru_stream);
r = cuStreamWaitEvent(cstream, g_lru_event, 0);
```

Similarly, `vitriol_lru_prefetch_async()` (line 841-892) issues one DMA per expert:

```cpp
// Called per expert_idx in the prediction set
CUresult r = cuMemcpyHtoDAsync(dst, expert_data, expert_size, g_lru_stream);
```

The `vitriol_predictor_prefetch()` (line 495-550) iterates over the prediction set and calls `vitriol_lru_prefetch()` per expert.

**Per-layer overhead**: 8 calls × ~6 µs = 48 µs dispatch. At ~2 ms per layer, that's ~2.4% of layer time. For 28 unpinned layers: ~1.34 ms per token, ~1.6% of total 82 ms.

## Proposed Change

Instead of calling `cuMemcpyHtoDAsync` per expert, pack all active experts for a layer into a contiguous staging buffer, then issue a single DMA.

### Where Changes Apply

This is tightly coupled with the double-buffer design (Plan 1). The staging buffer from Plan 1 *is* the coalescing mechanism. Plan 2 can be implemented independently if the LRU is kept, but batching is simpler with double-buffer.

### Independent Implementation (without double-buffer)

If the LRU is kept, coalescing requires a different approach since each expert lands in a different LRU slot. Options:

**Option A: Contiguous LRU slots** — Allocate LRU slots in groups of 8 so that 8 consecutive slots form a contiguous VRAM region. Pack 8 experts into a staging buffer, then DMA all 8 at once. Slots can still be managed individually (for eviction).

**Option B: Per-layer staging DMA** — Instead of LRU-per-expert, DMA the entire batch of active experts to a fixed VRAM region (like double-buffer) but keep the LRU for subsequent-token reuse.

Option B is the double-buffer approach (Plan 1). For an LRU-only implementation, Option A is:

```cpp
// Modified vitriol_lru_ensure: instead of per-expert DMA,
// batch ALL active experts for this layer into one transfer

struct BatchRequest {
    const void *tensor_base;
    int   expert_ids[256];
    int   n_experts;
    size_t expert_size;
    CUdeviceptr vram_base;  // contiguous VRAM region for this batch
};

static void batch_dma(BatchRequest *req) {
    // Pack all experts into staging buffer
    size_t total = 0;
    for (int i = 0; i < req->n_experts; i++) {
        const void *src = (const char *)req->tensor_base
                        + req->expert_ids[i] * req->expert_size;
        memcpy(g_staging_buf + total, src, req->expert_size);
        total += req->expert_size;
    }
    // Single DMA call
    cuMemcpyHtoDAsync(req->vram_base, g_staging_buf, total, g_lru_stream);
    cuEventRecord(g_lru_event, g_lru_stream);
}
```

### Changes Required (Option A — LRU + Coalescing)

| Function | Change |
|----------|--------|
| `lru_init_pool()` | Allocate contiguous "slot groups" of 8 slots each. All groups are pre-linked. |
| `vitriol_lru_ensure()` | Accept `expert_ids[]` array + count instead of single `expert_idx`. Pack batch → single DMA. Return array of CUdeviceptr (one per expert) or a base pointer + offset calculation. |
| `vitriol_lru_prefetch_async()` | Same: accept batch, single DMA per batch. |
| `vitriol_predictor_prefetch()` | Group predictions per layer, pass batch to prefetch. |

### Caller Changes

The per-expert loop in `ggml-cuda.cu` (which calls `vitriol_lru_ensure` per expert) needs to be restructured to:

1. First collect all active expert indices for the layer
2. Call a new `vitriol_batch_ensure()` once
3. Use the returned VRAM addresses for all experts

This is a deeper change affecting `ggml-cuda.cu`'s MoE dispatch path.

### GPU Kernel Changes

The MMVQ/MMQ/MMF kernels that read per-expert weights need to be able to locate each expert's weights within a contiguous block. Currently each expert has an independent VRAM address. With batching, they'd be at `base + expert_index * expert_size` within the batch region.

**No kernel changes needed** if the batch VRAM layout matches the host layout (expert_size-aligned slices within a contiguous block). The kernels already index by expert_id within a tensor.

### Expected Gain

| Scenario | Overhead per layer | Overhead per token | Gain |
|----------|-------------------|-------------------|------|
| Current (8 DMAs) | 8 × 6µs = 48µs | 28 × 48µs = 1.34ms | — |
| Batched (1 DMA) | 1 × 6µs = 6µs | 28 × 6µs = 168µs | **1.17ms saved** |
| CPU pack cost | 1 × 0.3µs (memcpy 7MB) | 28 × 0.3µs ≈ 8.4µs | — |

Net gain: ~1.16 ms per token → +1.4% throughput on 12.25 tok/s baseline.

### Standalone Implementation Priority

Coalescing alone yields modest gains (1-2%). It's included as a natural component of the double-buffer design (Plan 1), where the staging buffer is already needed. If implementing without double-buffer, **skip standalone coalescing** — the gain doesn't justify the ggml-cuda.cu restructuring.

## Files Modified

| File | Change |
|------|--------|
| `vitriol-cuda-integration.cpp` | New `batch_ensure()` + `batch_prefetch()` functions. Add staging buffer allocation. |
| `vitriol-cuda-integration.h` | Declare batch functions. |
| `ggml-cuda.cu` | Restructure MoE per-expert loop to collect experts first, then batch. |
| `vitriol-buffer.h` | May need buffer-type-specific batch helpers. |

## Verification

1. Build and run baseline (no batching): measure avg µs per `cuMemcpyHtoDAsync`
2. Enable batching: measure avg µs per batched DMA
3. Confirm per-token time decreases by ~1.2 ms
4. Confirm tok/s increases proportionally
