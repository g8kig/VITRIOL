# Plan: Monolithic Pin Pool + Top-K Expert Pruning

**✅ IMPLEMENTED — Both changes are committed and benchmarked.**
- Monolithic VRAM pool: integrated into `vitriol_pin_ensure()` (vitriol-cuda-integration.cpp)
- Top-K Pruning: `VITRIOL_PRUNE_EXPERTS=4` + output cache → **10.71 t/s (+16%)**
- See [MEMORY_OPTIMIZATIONS.md](MEMORY_OPTIMIZATIONS.md) and [COMPUTE_OPTIMIZATIONS.md](COMPUTE_OPTIMIZATIONS.md) for current status.
2. **Top-K Pruning**: Drop bottom-N experts after routing to halve matmul work

---

## Part A: Monolithic VRAM Pool for Expert Pinning

### Problem
Current `vitriol_pin_ensure()` calls `cuMemAlloc` individually for each tensor (up to 30 times for 15 layers × 2 tensors). This fragments VRAM and can cause mid-prefill OOM even when total free VRAM is sufficient.

### Fix
Replace per-tensor allocation with a single pre-allocated pool:

```cpp
static CUdeviceptr g_pin_pool = 0;
static size_t      g_pin_pool_offset = 0;
static size_t      g_pin_pool_total = 0;
```

On first `vitriol_pin_ensure()` call:
```
expected_slots = pin_first_n_layers × pin_tensors_per_layer
pool_size = first_tensor_size × expected_slots × 1.5   // 50% margin
cuMemAlloc(&g_pin_pool, pool_size)
```

Subsequent calls: assign `g_pin_pool + g_pin_pool_offset`, advance offset by tensor size.
If remaining pool space < tensor size: log warning, skip pinning that tensor (graceful fallback).
If initial `cuMemAlloc` fails: disable pinning immediately with error log.

Cleanup: single `cuMemFree(g_pin_pool)` instead of iterating map.

### Benefits
- Single contiguous VRAM block — no fragmentation
- Fail fast at init time, not mid-prefill
- Simpler cleanup
- ~20 lines changed

---

## Part B: Top-K Expert Pruning

### Idea
After the router generates 8 expert IDs per token, drop the bottom N and only compute (8-N). For coding/syntax tasks, the top 4 experts carry ~95% of the signal. Halving the matmul count nearly doubles throughput.

### Implementation

**C/C++:** `ggml_cuda_mul_mat_id`, after `tokens_per_expert[]` built (~line 2617):
```cpp
int vitriol_prune = g_vitriol_config.prune_experts;
if (vitriol_prune > 0 && vitriol_prune < n_expert_used) {
    int keep = n_expert_used - vitriol_prune;
    int kept = 0;
    for (int i = 0; i < ne02 && kept < keep; i++) {
        if (tokens_per_expert[i] > 0) kept++;
    }
    for (; i < ne02; i++) {
        tokens_per_expert[i] = 0;
    }
}
```

No kernel changes. No graph changes. Just zeros out the `tokens_per_expert` entries for dropped experts — the per-expert loop naturally skips them.

**Config:**
- `vitriol_config_t` field: `int prune_experts;` (0-7, default 0)
- Env var: `VITRIOL_PRUNE_EXPERTS=N`
- CLI flag: `--prune-experts N`

**Shell script:**
- Default: `DEFAULT_PRUNE_EXPERTS=0`
- Config key: `vitriol.prune_experts`
- TUI: Model Settings → option **11** (with warning: "Experimental: drops bottom N experts. May affect quality.")
- Env passthrough in run/serve blocks

### Benchmark Plan

| # | Prune | Pin | Expected t/s |
|---|---|---|---|
| 0 | 0 | 0 | ~8.9 (baseline) |
| 1 | 2 | 0 | ~11 (6 experts) |
| 2 | 4 | 0 | ~15 (4 experts) |
| 3 | 4 | 15 | ~15.5 (prune + pin) |
| 4 | 2 | 15 | ~11.5 (combo) |
| 5 | 4 + output cache | 0 | ~17 (prune + cache) |

### Quality Check
After benchmark, run `vitriol run` with prune=4 and generate ~200 tokens of code. If output is coherent, the quality loss is acceptable for coding tasks.

---

## Files to Modify

| File | Changes |
|------|---------|
| `vitriol-cuda-integration.h` | Add `prune_experts` to config struct; `vitriol_pin_active()` already exists |
| `vitriol-cuda-integration.cpp` | Replace per-tensor `cuMemAlloc` with monolithic pool; read `prune_experts` env var |
| `ggml-cuda.cu` (~line 2617) | Add prune logic to `ggml_cuda_mul_mat_id` |
| `scripts/vitriol` | Add prune config key, TUI option, CLI flag, env passthrough |

---

## Timeline

- Monolithic pool: ~15 min
- Top-K pruning C/C++: ~20 min
- Top-K pruning shell: ~20 min
- Build + verify: ~10 min
- Benchmark: ~20 min
- **Total: ~1.5 hours**
