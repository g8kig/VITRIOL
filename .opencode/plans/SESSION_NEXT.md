# Session: 2026-05-15 — CE DMA Pipeline Fix

## Current State

CE DMA works in isolation (all 256 experts load via CE DMA at ~0.06ms each, verified). But crashes during warmup with `illegal memory access` at `ggml_cuda_mul_mat_id` line 2558 (reading `ids` tensor).

## Root Causes (Two Issues)

### Issue 1: `supports_buft` changes graph scheduler behavior
Modifying `ggml_backend_cuda_device_supports_buft` to accept CPU buffer types causes the graph scheduler to route `MUL_MAT_ID` to CUDA backend. This is correct for our purposes, but the scheduler also changes how it partitions the full compute graph, leading to inconsistent tensor allocation (some tensors expected on CPU but actually on GPU, or vice versa). The `ids` tensor (MoE router output) ends up with an invalid device pointer.

**Fix**: Revert `supports_buft` to original. CUDA does not accept CPU buffer types. The scheduler routes MUL_MAT_ID to CPU backend as before.

### Issue 2: VRAM pool allocation during warmup
`ensure_expert_pool()` calls `cuMemAlloc(3420 MB)` inside `vitriol_ensure_expert_loaded`, which is called from `ggml_cuda_mul_mat_id` during warmup. By this time, model tensors are already allocated in VRAM. The large `cuMemAlloc` may cause CUDA's memory manager to invalidate or move existing allocations (like the `ids` tensor).

**Fix**: Move pool allocation to `vitriol_cuda_init()`, which runs during `ggml_backend_cuda_init()` — before any model tensor is allocated. At init time, ~7.9 GB is free. 3.4 GB pool → ~4.5 GB remaining, plenty for the 1.3 GB model.

### Issue 3: Entry point for CE DMA
With `supports_buft` reverted, `ggml_cuda_mul_mat_id` is not called for CPU-resident expert tensors. The CPU backend handles MUL_MAT_ID instead. Our CE DMA call was in the wrong function.

**Fix**: Intercept the CPU backend's MUL_MAT_ID handler. Before it computes an expert slice via CPU matmul, CE DMA the needed slice into the VRAM pool (via bounce buffer), then launch CUDA matmul on the loaded data.

---

## Files Changed

### ggml-cuda.cu (reverts)
1. Remove `supports_buft` CPU-buft addition (restore original)
2. Remove `src0_is_cpu` detection + `vitriol_ensure_expert_loaded` call from `ggml_cuda_mul_mat_id`

### vitriol-cuda-integration.cpp (modifications)
1. Move `ensure_expert_pool()` call from `vitriol_ensure_expert_loaded` to `vitriol_cuda_init()`
2. Keep bounce-buffer CE DMA logic in `vitriol_ensure_expert_loaded` (it works)
3. `vitriol_ensure_expert_loaded` now assumes pool is already allocated

### ggml-cpu backend (new CE DMA interceptor)
1. Find `ggml_compute_forward_mul_mat_id` in the CPU backend
2. At the start, if VITRIOL mode is active and src0 is an expert tensor:
   a. Get the `ids` tensor to find which experts are active
   b. For each active expert not in VRAM cache: CE DMA from CPU data → VRAM pool
   c. After all active experts are in VRAM, launch CUDA kernel for matmul
   d. Copy result back to CPU dst buffer

---

## Implementation Order

1. Write plan document ✓
2. Revert `supports_buft` in ggml-cuda.cu
3. Revert `src0_is_cpu` + CE DMA call in `ggml_cuda_mul_mat_id`
4. Move pool allocation to `vitriol_cuda_init()`
5. Find CPU backend `ggml_compute_forward_mul_mat_id`
6. Add CE DMA interceptor in CPU backend
7. Build and test
