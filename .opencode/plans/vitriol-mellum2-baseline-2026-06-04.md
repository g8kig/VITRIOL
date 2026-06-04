# Mellum2 VITRIOL Baseline

**Date:** 2026-06-04 12:35 UTC
**Model:** Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf (7.5 GB, Q4_K_M)
**Hardware:** GTX 1070 Ti (8GB VRAM) + 64GB DDR4
**VITRIOL:** vitriol-mellum2 branch (cherry-picked PR #23966)

---

## Results

| Metric | Value |
|--------|-------|
| Architecture | `mellum` (detected correctly) |
| Layers | 28 |
| Experts | 64 (8 active) |
| Embedding dim | 2304 |
| Expert FFN dim | 896 |
| Attention heads | 32 Q, 4 KV (GQA=8) |
| Sliding window | 1024 (21/28 layers have SWA) |
| Vocab size | 98304 (BPE, mellum2 pre-tokenizer) |
| Context | 32768 tokens |
| Config | `-ngl 20`, VITRIOL DMA mode, LRU=2048 |

## Memory Breakdown

| Pool | Size |
|------|------|
| CUDA model (GPU) | 5171 MiB |
| CPU model (DDR4) | 2522 MiB |
| CUDA KV cache | 446 MiB |
| CPU KV cache | 191 MiB |
| CUDA compute | 264 MiB |
| Host compute | 84 MiB |
| **Total VRAM used** | **5882 MiB / 8112 MiB (72%)** |
| **Total RAM used** | **2797 MiB / 64000 MiB** |

## Performance

| Metric | Value |
|--------|-------|
| Generation speed | **20.40 t/s** |
| Test prompt | 9 tokens |
| Generated toks | 128 tokens |
| Total time | 6.273s |
| Temperature | 0.0 |

## Quality Check

- **Completion:** "What is the capital of France?" → Correctly answered "Paris"
- **Chat:** "Write a haiku about programming." → Produced coherent output: *"Lines of code flow, / Silent logic takes form, / World rewrites itself."*

## Observations

1. **~2.5 GB of expert tensors live in DDR4** — this is exactly where the VITRIOL × Brief VPO LUT engine would run. With hyperfolded LUTs, these expert matmuls would stay entirely in CPU memory, and only small activation tensors would cross PCIe.

2. **64 experts × 8 active means expert hotness varies.** The LRU cache (2048 MB) can hold about 8 experts in VRAM simultaneously. A predictor or pre-baked VPO LUT would dramatically reduce LRU misses.

3. **20 t/s on a 10-year-old GPU** for a 12B MoE model is strong. For comparison, upstream PR #23966 reported **383 t/s on RTX 6000 Pro** — this is roughly what we'd expect given the 32 GB HBM vs 8 GB VRAM gap.
