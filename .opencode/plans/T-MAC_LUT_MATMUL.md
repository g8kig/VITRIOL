# Plan: T-MAC GPU — Lookup-Table MatMul for Pascal

**Problem:** GTX 1070 Ti (Pascal CC 6.1) peaks at ~21 INT8 TFLOPS. At 10.86 t/s with prune=4 + output cache, we're at ~68% of peak. The remaining 32% cannot be recovered with ALU optimizations — the silicon is saturated.

**Idea:** Bypass the ALU entirely. Replace matrix multiply with **shared memory lookup tables** for low-bit weights (IQ2_M, TQ1_0). Instead of `activation × weight`, do `LUT[weight_bits][activation]` — a shared memory fetch that takes 2 cycles instead of 16+.

## How It Works

For IQ2_M (2 bits/weight, 4 possible values: 00, 01, 10, 11) and INT8 activation (256 values):

```
Standard:       out += dequant(weight) × activation     →  ~20 cycles (dequant + multiply)
T-MAC GPU:      out += LUT[weight_bits][activation]      →  ~2 cycles (SRAM load)
```

The LUT is tiny:
- 256 activation values × 4 weight combos = 1024 entries
- 4 bytes per entry (float32) = 4 KB
- Fits in L1 cache (48 KB on Pascal, 128 KB on newer)

## GPU Architecture

Given a weight matrix of shape `[n_embd, n_ff]` quantized to IQ2_M:

```
For each output element:
  1. Load 2-bit weight from global memory (packed: 16 weights per 32-bit word)
  2. Load activation value from shared memory (already staged)
  3. LUT base = activation_idx × 4 × sizeof(float)
  4. result += LUT[weight_bits] at (LUT_base + weight_bits × 4)
  5. Store accumulated result
```

No multiplication. No dequantization. Just a shared memory load and an addition.

## Implementation Phases

### Phase 1: Research & Understanding (1-2 days)
- [ ] Read T-MAC paper and reference implementation
- [ ] Understand IQ2_M block structure (block_size=32, scale factors, importance maps)
- [ ] Study existing `mmvq.cu` kernel structure for the MoE with `ids` tensor
- [ ] Identify the exact weight layout for IQ2_M in the expert tensor

### Phase 2: Prototype — Standalone LUT Kernel (3-5 days)
- [ ] Write `vitriol_lut_mul_mat_vec_q` — a new CUDA kernel for vector × IQ2_M matrix
- [ ] Kernel signature: `(src0, src1, ids, dst)` matching existing MMVQ conventions
- [ ] LUT computation: precompute all 1024 dot products in shared memory per block
- [ ] Weight loading: extract 2-bit pairs from packed uint32, use as LUT index
- [ ] Accumulation: warp-level reduction for each output element
- [ ] Test standalone: verify against existing MMVQ output for small random inputs

### Phase 3: Integration (1-2 days)
- [ ] Add kernel launch to `ggml_cuda_mul_mat_id` fast-path selection
- [ ] Gate: only use LUT kernel when `ggml_type` is IQ2_M/TQ1_0 and `vitriol_lut_mode` is active
- [ ] Handle the `ids` tensor for MoE expert routing
- [ ] Add config/enable flag (`VITRIOL_LUT_MATMUL=1`)

### Phase 4: Benchmark (1 day)
- [ ] Compare LUT kernel vs standard MMVQ on identical inputs
- [ ] Measure: t/s, FLOP utilization, shared memory usage, register pressure
- [ ] Tune: block size, grid size, LUT placement (shared vs constant memory)

## Key Technical Challenges

1. **IQ2_M block structure**: Weights are not stored as raw 2-bit values — they have shared scale factors per block of 32. The LUT approach needs to account for per-block scaling, not just raw weight values.

2. **Shared memory LUT refresh**: The LUT depends on activation values, which change every token. The LUT needs to be recomputed in shared memory each kernel launch. This takes ~1024 multiply-adds, which is ~2μs — acceptable.

3. **Expert routing (ids tensor)**: The MMVQ with `ids` routes different rows to different experts. The LUT kernel must handle this indirection. Same approach as existing MMVQ: read `ids`, compute expert offset, add to weight base pointer.

4. **Register pressure**: With 4 KB LUT in shared memory and multiple activation values in flight, register usage could be high. May need to tile activation values.

## Dependencies

- `mmvq.cu` — existing MMVQ kernel structure (the file to extend)
- `vitriol-cuda-integration.h/.cpp` — configuration and enable flag
- `ggml-cuda.cu` — kernel dispatch in `ggml_cuda_mul_mat_id`

## Status

- [ ] Phase 1: Research
- [ ] Phase 2: Prototype
- [ ] Phase 3: Integration
- [ ] Phase 4: Benchmark

## Expected Gain

| Metric | Current (prune=4 + cache) | T-MAC (estimated) |
|--------|--------------------------|-------------------|
| Decode throughput | 10.86 t/s | **18-25 t/s** |
| vs theoretical peak | 68% of 16 t/s | **Exceeds Pascal peak** (bypasses ALU) |
| Quality impact | None (prune is approximate) | None (exact math, different implementation) |
