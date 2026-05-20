# T-MAC GPU Feasibility Analysis for IQ2_S

**Bottom line:** A full T-MAC lookup-table kernel is **impractical for IQ2_S** but would work brilliantly for **TQ1_0 or BitNet b1.58**. The existing MMVQ kernel for IQ2_S is already doing essentially what T-MAC would do — codebook lookup + dot product.

---

## How IQ2_S Weights Work (the actual type behind IQ2_M)

IQ2_S is **not** raw 2-bit weights. It has a complex multi-layer structure:

```
block_iq2_s (82 bytes = 256 weights):
├── d: fp16 global scale (2 bytes)
├── qs[64]: grid indices + signs
│   ├── qs[0..31]: 8 × 4 grid indices (low 8 bits each) = 32 bytes
│   └── qs[32..63]: 8 × 4 sign bytes = 32 bytes
├── qh[8]: high 2 bits of grid indices (8 bytes)
└── scales[8]: two 4-bit sub-block scales (8 bytes)
```

Each group of 8 weights is decoded from a **10-bit index** into a **1024-entry codebook**, multiplied by sign (±1) and sub-block scale (4-bit).

### Dequantization for one weight:
```
value = d_fp16 × (0.5 + scale_4bit)/4 × codebook[10bit_index][j] × sign_bit
```

### Why T-MAC doesn't work here

A lookup table approach would precompute:
```
LUT[activation][codebook_idx][scale][sign] = activation × codebook_val × scale × sign
```

The LUT size: 256 activations × 1024 codebook entries × 16 scale values × 2 signs = **8,388,608 entries × 4 bytes = 32 MB**. This does NOT fit in shared memory (48 KB on Pascal).

### What the existing kernel already does

The existing `vec_dot_iq2_s_q8_1` kernel:
1. Loads the 8 int8 values from `iq2s_grid[grid_index]`
2. Loads the 8 q8_1 activation values
3. Computes `sum += act[j] × grid_val[j]` (8 multiply-adds)
4. Multiplies result by scale = `d × (0.5 + s4)/4 × sign`
5. Adds to accumulator

This **is** the optimal approach for IQ2_S. There's no faster way to compute `Σ act[j] × codebook[j]` than to iterate 8 values. A lookup table would be faster only if:
- The number of unique (codebook_entry, scale) pairs is small enough to precompute
- It's not — there are 1024 × 16 = 16,384 unique pairs per block

---

## Where T-MAC WOULD Work: TQ1_0 / BitNet b1.58

For a **ternary** format (±1, 0), T-MAC becomes trivial:

```
Standard:    out += act[w] × weight[w]     →  multiply (16 cycles)
T-MAC:       out += LUT[weight][act]        →  SRAM fetch (2 cycles)
```

| Parameter | IQ2_S (current) | TQ1_0 / BitNet |
|-----------|-----------------|----------------|
| Unique weight values | 1024 × 16 × 2 = 32,768 | **3** (-1, 0, +1) |
| LUT size | 32 MB (impossible) | **3 KB** (fits in L1) |
| LUT coverage | Per-block/complex | **Global/simple** |
| Speedup over current | **~0%** (already optimal) | **~3-5×** |

For TQ1_0: LUT[256 activations × 3 weight values] = 768 floats × 4 bytes = **3,072 bytes**. Fits in Pascal's 48 KB L1. The kernel loads a 2-bit weight, uses it as an index, and fetches the precomputed result from shared memory. No multiply, no dequantize.

---

## Recommended Path Forward

### Short-term: Optimize the existing IQ2_S path
- The current `vec_dot_iq2_s_q8_1` is already near-optimal for the IQ2_S format
- The compute bottleneck is the sheer volume of weights (2M weights × 40 layers)
- **Further gains require reducing weight count, not speeding up per-weight math**
- We already did this with Top-K Pruning (-50% weights, +21.5% throughput)

### Medium-term: TQ1_0 format with T-MAC kernel
1. Source or quantize a TQ1_0 version of Qwen3.6-35B-A3B
2. Write a `vitriol_lut_mul_mat_vec_q` CUDA kernel for ternary weights
3. The kernel: 2-bit weight → index into shared memory LUT → accumulate
4. Expected: **3-5× throughput** on Pascal (ALU bypassed entirely)

### Long-term: Hybrid format
- Use IQ2_S for attention/base layers (high precision needed)
- Use TQ1_0 for MoE expert weights (quantization-resistant per MoQE paper)
- Combine with Expert Pinning + Top-K Pruning + Output Cache

---

## Appendix: Expert Tensor Sizes for IQ2_S

| Tensor | Shape | Blocks/row | Row stride | Expert stride | Expert size |
|--------|-------|-----------|------------|---------------|-------------|
| gate_up_exps | {2048, 2048, 256} | 8 | 656 B | **1,343,488 B** | 1.31 MB |
| down_exps | {1024, 2048, 256} | 4 | 328 B | **671,744 B** | 0.66 MB |

Both are 3D: `[n_embd, n_ff, n_expert]`. The `ids` tensor selects `channel_x = expert_id` per token, and the kernel reads from `expert_id × nb02` offset within the tensor.
