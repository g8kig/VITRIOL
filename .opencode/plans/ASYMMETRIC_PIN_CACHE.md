# Plan: Asymmetric Pinning + Output Cache Hybrid

**Problem:** Expert pinning and output cache target different code paths in `ggml_cuda_mul_mat_id`:
- Pinning: redirects `src0->data` for the **fast path** (MMVQ/MMQ/MMF)
- Output cache: skips matmul entirely in the **per-expert loop**
- With output cache ON, decode always takes the per-expert loop — pinning does nothing
- With pinning ON and output cache OFF, the fast path runs but misses the cache benefit for "lazy" deep layers

**Observation (Hidden State Sluggishness):**
- Layers 0-20: activations change rapidly — output cache has low hit rate, pinning is valuable
- Layers 21-40: activations barely change — output cache has high hit rate, pinning is wasted

## The Hybrid

**Pin layers 0-15** (fast path) + **Output-cache layers 21-40** (per-expert loop)
- Layer 16-20: either (graceful transition zone)
- This requires the `ggml_cuda_mul_mat_id` call to know its layer index and behave differently.

## Implementation

### Approach: Per-Layer Dispatch

In `ggml_cuda_mul_mat_id`, add a layer-aware dispatch:

```cpp
int layer_idx = get_layer_index((uintptr_t)src0->data);

if (layer_idx < g_vitriol_config.pin_first_n_layers) {
    // Pinning path: redirect src0->data, run fast path
    // Temporarily disable output cache for this call
    bool saved_oc = g_vitriol_config.output_cache;
    g_vitriol_config.output_cache = false;
    // ... existing logic (fast path runs) ...
    g_vitriol_config.output_cache = saved_oc;
} else if (layer_idx >= VITRIOL_OC_START_LAYER) {
    // Output cache path: force per-expert loop
    // No pinning redirect, let LRU + output cache handle it
    // ... existing logic with output_cache=true forces per-expert loop ...
}
```

### Config

```ini
[vitriol]
# Pinning range
pin_first_n_layers = 15
# Output cache range (0 = auto, or explicit start layer)
output_cache_start_layer = 20
```

When `output_cache_start_layer > pin_first_n_layers`, the hybrid activates. When both are 0, each setting works independently (current behavior).

## Expected Gain

- Pinned layers 0-15: ~37% of layers run at VRAM speed (~0.15 ms vs 2.5 ms)
- Cached layers 20-40: ~50% of layers skip matmul entirely (0 ms vs 1.5 ms)
- Estimated combined throughput: **13-15 t/s** (vs 10.07 with cache alone, vs ~12 with pinning alone)

## Complexity

- Low: ~50 lines of C++ code
- No kernel changes
- No new infrastructure — just smarter use of existing toggles

## Status

- [ ] Add per-layer gating logic to `ggml_cuda_mul_mat_id`
- [ ] Add `output_cache_start_layer` to config
- [ ] Shell script: config key, TUI, env var
- [ ] Benchmark combinations
