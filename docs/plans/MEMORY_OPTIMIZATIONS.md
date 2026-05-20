# Memory Optimizations

**Goal:** Minimize PCIe transfer overhead and maximize VRAM utilization for MoE expert weights on VRAM-constrained GPUs.

**Problem statement:** Expert weights live in page-locked host RAM (VITRIOL buffer). The GPU reads them over PCIe DMA on demand. At ~12 GB/s PCIe Gen3 x16, each layer's 2-3 expert matmuls need ~2.5 MB per expert × 8 experts = ~20 MB per layer. For 40 layers, that's ~800 MB per token. With CUDA stream overlap, most of this is hidden, but residual latency costs ~0.5 ms/layer.

This document consolidates all memory-side optimization plans. See originals at:
- [Expert Pinning v2](EXPERT_PINNING_V2.md)
- [Expert Pinning v1](EXPERT_PINNING.md) (superseded by v2)
- [Monolithic Pool + Top-K Prune](MONOLITHIC_POOL_TOP_K_PRUNE.md) (pool part only)
- [On-Demand Expert Plan](ON_DEMAND_EXPERT_PLAN.md)
- [Asymmetric Pin + Cache](ASYMMETRIC_PIN_CACHE.md)

---

## Technique 1: Expert Pinning (✅ Implemented)

**Pre-load full expert weight tensors (all 256 experts) of the first N layers into VRAM at first use.**

### How It Works

Before the fast-path MMVQ/MMQ/MMF kernel checks in `ggml_cuda_mul_mat_id`, create a local `ggml_tensor` copy of `src0` with `.data` pointing to a VRAM buffer containing ALL experts of that tensor. Restore original `src0` before the per-expert loop (LRU/predictor/cache unaffected).

### Implementation Config

| Param | Env var | Config key | CLI flag | TUI |
|-------|---------|------------|----------|-----|
| Pin count | `VITRIOL_PIN_FIRST_N_LAYERS` | `vitriol.pin_first_n_layers` | `--pin-layers N` | VITRIOL Mode → option 5 |

### Benchmark Results

| Pin count | VRAM used | t/s | vs baseline | Notes |
|-----------|-----------|-----|-------------|-------|
| 0 | 0 MB | 8.94 | — | Fast path, no pin |
| 5 | 756 MB | ~8.97 | ~0% | Too few layers |
| 15 | 2,300 MB | 9.30 | +4.0% | Modest gain (compute bound, not PCIe bound) |

### Key Finding

Pinning gives only +4% because the GPU is **compute-bound, not PCIe-bound**. The MMVQ kernel for IQ2_M spends most of its time on dequant + multiply, not waiting for weight fetch. CUDA stream overlap already hides most PCIe latency.

### Safety: Monolithic VRAM Pool

Replaced per-tensor `cuMemAlloc` with a single pre-allocated pool at init time to prevent VRAM fragmentation. If the pool allocation fails, pinning is disabled gracefully.

---

## Technique 2: LRU VRAM Cache (✅ Implemented, inactive for quantized)

**Keep hot experts in a VRAM pool (~512 MB) for faster access.**

**Status:** The LRU cache works correctly but is **never reached** for quantized MoE models (Q2_K_XL, IQ2_M, etc.) because the MMVQ/MMQ/MMF fast path in `ggml_cuda_mul_mat_id` returns before the per-expert LRU loop. The LRU code only activates in the cuBLAS slow path (FP16 only).

---

## Technique 3: Output Cache / Hidden State Caching (✅ Implemented)

**Reuse previous token's expert output for layers where the residual barely changes.**

### How It Works

Per-expert, per-layer output float vector cache. Keyed by `(tensor_base, expert_idx)`. On cache hit (same expert + same layer as previous token), skip the matmul and reuse the cached output vector. Approximate — assumes residual changes slowly across tokens.

### Config

| Param | Env var | Config key | CLI flag | TUI |
|-------|---------|------------|----------|-----|
| Enable | `VITRIOL_OUTPUT_CACHE` | `vitriol.output_cache` | `--output-cache` | Model Settings → option 6 |

### Benchmark

| Config | t/s | vs baseline |
|--------|-----|-------------|
| Baseline | 8.94 | — |
| + Output cache | 10.07 | +12.6% |
| + Prune 4 | **10.71** | **+16.3%** |
| + Pin 15 | 10.30 | +15.2% |

---

## Technique 4: Asymmetric Pin + Cache (☐ Planned)

**Pin early layers (0-15), output-cache late layers (21-40).**

Observations:
- Layers 0-20: activations change rapidly — output cache has low hit rate, pinning is valuable
- Layers 21-40: activations barely change — output cache has high hit rate, pinning is wasted

The hybrid would use a per-layer dispatch in `ggml_cuda_mul_mat_id`: use pinning for early layers, force the sorted path (with output cache) for late layers. This would require splitting the function's behavior based on `get_layer_index()`.

### Original Document

[Asymmetric Pin + Cache Plan](ASYMMETRIC_PIN_CACHE.md)

---

## Technique 5: On-Demand Expert Loading (☐ Planned)

Instead of loading all expert weights into the VITRIOL buffer, load only the experts needed for the current token. This reduces host RAM usage but requires faster page-fault handling or a swap mechanism.

### Original Document

[On-Demand Expert Plan](ON_DEMAND_EXPERT_PLAN.md)

---

## Combined Strategy

| Technique | Status | Gain | When to use |
|-----------|--------|------|-------------|
| **Output Cache** | ✅ Done | +12.6% | Always with prune |
| **Expert Pinning** | ✅ Done | +4% | Only if output cache off |
| **LRU Cache** | ✅ Done | Inactive | Quantized models skip LRU |
| **Asymmetric Pin+CACHE** | 📋 Planned | ~+15-20% | Future work |
| **On-Demand Loading** | 📋 Planned | Host RAM savings | Future work |

**Best current memory config (no VM tricks):**
```
VITRIOL_MODE=stream VITRIOL_OUTPUT_CACHE=1 VITRIOL_PRUNE_EXPERTS=4
```

Expert pinning not needed — the compute bottleneck makes VRAM placement irrelevant for decode.
