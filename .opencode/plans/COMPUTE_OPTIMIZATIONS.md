# Compute Optimizations

**Goal:** Bypass or reduce the ALU bottleneck on Pascal GPUs (CC 6.1) for Qwen3.6-35B-A3B MoE inference.

**Problem statement:** At 10.71 t/s (prune=4 + output cache), we're at ~67% of Pascal's ~16 t/s theoretical compute ceiling. The bottleneck is **ALU saturation**, not memory bandwidth. The remaining compute cannot be recovered with conventional kernel optimizations — the silicon is the limit.

This document consolidates all compute-side optimization plans. See originals at:
- [T-MAC Feasibility Analysis](T-MAC_FEASIBILITY.md)
- [T-MAC GPU Plan](T-MAC_LUT_MATMUL.md)
- [Top-K Pruning](TOP_K_PRUNING.md)

---

## Technique 1: Top-K Expert Pruning (✅ Implemented)

**Drop bottom N of 8 active experts before matmul. Halves compute at N=4.**

### How It Works

After the MoE router generates expert IDs per token, zero out `tokens_per_expert[i]` for the bottom-N experts in `ggml_cuda_mul_mat_id`. The per-expert sorted loop naturally skips experts with 0 tokens. Forces the sorted path (fast-path MMVQ/MMQ/MMF skipped when prune > 0).

### Implementation Config

| Param | Env var | Config key | CLI flag |
|-------|---------|------------|----------|
| Prune count | `VITRIOL_PRUNE_EXPERTS` | `vitriol.prune_experts` | `--prune-experts N` |

### Benchmark Results

| Prune | Output cache | t/s | vs baseline | Notes |
|-------|-------------|-----|-------------|-------|
| 0 | off | 8.94 | — | Fast path, all 8 experts |
| 2 | off | 9.60 | +7.4% | Sorted path, 6 experts |
| **4** | **on** | **10.71** | **+16.3%** | **Best config** |
| 4 | off | 10.10 | +13.0% | Sorted path, 4 experts |
| 4 + pin 15 | off | 10.30 | +15.2% | Pinning doesn't stack |

### Quality Warning

Experimental. Dropping 4 of 8 experts may affect output coherence for creative writing or complex reasoning. For code generation the impact appears minimal. Always verify output quality before production use.

### Original Plan

See [`TOP_K_PRUNING.md`](TOP_K_PRUNING.md) for the original design doc.

---

## Technique 2: T-MAC GPU — Lookup-Table MatMul

**Replace ALU multiply with shared memory lookup table for low-bit weights.**

### Feasibility Verdict: ❌ Impractical for IQ2_S, ✅ Viable for TQ1_0

**For IQ2_S (our current format):** Each weight is a 10-bit index into a 1024-entry codebook with per-sub-block scaling. A full LUT would require 32 MB (256 acts × 1024 entries × 16 scales × 2 signs) — impossible for Pascal's 48 KB shared memory. The existing `vec_dot_iq2_s_q8_1` kernel is already near-optimal for this format.

**For TQ1_0 / BitNet b1.58 (ternary ±1, 0):** Only 3 unique weight values. LUT = 256 × 3 × 4 bytes = **3 KB** (fits in L1). Replaces multiply with a 2-cycle SRAM fetch. Estimated **3-5× throughput**.

### What Would Be Needed

1. **Source or quantize a TQ1_0 version** of Qwen3.6-35B-A3B
2. **Write `vitriol_lut_mul_mat_vec_q`**: a new CUDA kernel for ternary weights
   - Precompute LUT in shared memory (768 floats)
   - Load 2-bit weight → index into LUT → accumulate
   - Handle expert routing via `ids` tensor
3. **Integrate** into `ggml_cuda_mul_mat_id` fast-path selection
4. **Test** output correctness against reference MMVQ kernel

### Implementation Phases

```
Phase 1: Research    ── 1-2 days  — Study T-MAC paper, IQ2_S layout
Phase 2: Prototype   ── 3-5 days  — Standalone LUT CUDA kernel
Phase 3: Integration ── 1-2 days  — Wire into ggml_cuda_mul_mat_id
Phase 4: Benchmark   ── 1 day     — Compare vs standard MMVQ
```

### Key Technical Challenges

1. **IQ2_S block structure**: Re-evaluate if a simplified LUT (scale-only, no codebook) could work
2. **Shared memory LUT refresh**: Must recompute LUT per token (~2μs acceptable)
3. **Register pressure**: 4 KB LUT + activation values → may need activation tiling
4. **Expert routing**: Same `ids` indirection as existing MMVQ

### Original Documents

- [T-MAC Feasibility Analysis](T-MAC_FEASIBILITY.md) — Full IQ2_S block structure analysis
- [T-MAC GPU Implementation Plan](T-MAC_LUT_MATMUL.md) — Original LUT kernel design

---

## Combined Roadmap

| Phase | What | Effort | Gain |
|-------|------|--------|------|
| ✅ Done | Top-K Prune 4 + Output Cache | Implemented | **10.71 t/s (+16%)** |
| 🔜 Next | Source/quantize TQ1_0 model | ~1 week | Unlocks T-MAC |
| 🔬 Future | T-MAC LUT kernel for Pascal | ~2 weeks | **2-3× throughput** |
| 🔬 Future | TQ1_0 experts + IQ2_S attention/base | ~1 week | Best of both formats |
