# Plan: Graph Split Optimization + MTP/Prune Stacking

## Background

VITRIOL currently produces **17 graph splits** vs **2 in all-VRAM** baseline (`GGML_SCHED_DEBUG=1`). Each split adds scheduler re-entry overhead and potential cross-backend tensor copies. Fixing this could give 10-15% throughput improvement across ALL configurations.

Based on code analysis (`ggml/src/ggml-backend.cpp` scheduler, Pass 5, lines 1243-1373):

- **`GGML_SCHED_MAX_SPLIT_INPUTS = 30`** is NOT the primary cause (17 < 30)
- VITRIOL buf type IS supported by CUDA's `supports_buft` (line 5308 of `ggml-cuda.cu`)
- **Likely root cause:** VITRIOL-buffered weight tensors get `backend_id = -1` or `CPU` in scheduling passes 1-3 (weight tensors with `GGML_OP_NONE` aren't explicitly assigned). In Pass 5, the CUDA-backed `MUL_MAT_ID` sees its weight source on a different backend → forces a new split.

## Steps

### Step 1: Diagnose (5 min)
Run `GGML_SCHED_DEBUG=1` to see exact split breakdown. Zero code changes needed.

### Step 2: Fix (20 min)

**Approach A (preferred):** In `vitriol-buffer.cpp`, make VITRIOL buffer type share the CUDA host buffer type's `get_name` function pointer. This causes `ggml_backend_buft_is_cuda_host()` to return `true` for VITRIOL buft, telling the scheduler that VITRIOL weights are CUDA-host-compatible.

The check chain:
- `ggml_backend_buft_is_cuda_host(buft)` → compares `buft->iface.get_name` to `ggml_backend_cuda_host_buffer_type_name`
- If VITRIOL's `get_name` matches CUDA host's → scheduler treats VITRIOL weights as CUDA host weights
- CUDA backend already supports CUDA host buft → no split needed

Implementation: Replace VITRIOL's `get_name` function pointer with CUDA host's.

### Step 3: Benchmark (30 min)
Compare before/after with standard configs.

### Step 4: MTP + Prune Stacking (10 min, benchmark only)
Test `--spec-type mtp --spec-draft-n-max 2` with `VITRIOL_PRUNE_EXPERTS=4 VITRIOL_OUTPUT_CACHE=1`.

### Step 5: Quality Check (10 min)
Run `vitriol run` with prune=4, generate ~200 tokens of code, verify coherence.

## Expected Gain

| Config | Before | After (est.) |
|--------|--------|-------------|
| Baseline | 8.94 t/s | ~10 t/s |
| Prune 4 + cache | 10.86 t/s | ~12 t/s |
| + MTP N=2 | unknown | ~13+ t/s |
