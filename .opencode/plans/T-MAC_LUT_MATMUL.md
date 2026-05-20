# Plan: T-MAC — Lookup-Table MatMul for Sub-2-Bit Models

**Problem:** Pascal GPUs (CC 6.1) peak at ~21 INT8 TFLOPS. For a 35B MoE with 8 active experts × 2 MLP matmuls per layer × 40 layers, the compute floor is ~60 ms/tok. No kernel optimization can beat this — the ALU is saturated.

**Idea:** Replace matrix multiply with **Lookup Tables (LUTs)** on ultra-low-bit weights (TQ1_0, IQ2_M, Q2_K). Instead of multiplying activation × weight, precompute all possible dot products in shared memory and use the weight bits as indices into the table.

## How It Works

For a TQ1_0 weight (ternary: -1, 0, +1) and INT8 activation (256 values):

```
Standard matmul:    out += act[w] * weight[w]     →  ALU operation
T-MAC:              out += LUT[weight_bits][act]   →  SRAM lookup
```

The LUT is tiny (256 × 3 = 768 entries × 4 bytes = 3 KB, fits in L1 cache). The GPU never executes a multiply — it just fetches from the table.

## Expected Gain

- Removes compute bottleneck entirely for sub-2-bit models
- Converts O(n²) multiply into O(n) lookup
- Estimated 2-3× throughput improvement on Pascal (no Tensor Cores needed)
- Works on any GPU with shared memory

## Implementation Path

1. **Research phase**: Study Microsoft's T-MAC paper and reference implementation
2. **Prototype**: Implement a standalone LUT-based `ggml_mul_mat_vec_q` variant for TQ1_0
3. **Integration**: Hook into `ggml_cuda_mul_mat_id` fast-path as an alternative kernel path when the weight type is TQ1_0/IQ2
4. **Benchmark**: Compare against standard MMVQ on same model/config

## Status

- [ ] Research
- [ ] Prototype
- [ ] Integration
- [ ] Benchmark

## Dependencies

- Requires understanding of existing quantized CUDA kernels in `mmvq.cu` / `mmq.cu`
- May need new `ggml_type` or kernel variant
- Low risk to existing code (additive, never replaces existing path)
