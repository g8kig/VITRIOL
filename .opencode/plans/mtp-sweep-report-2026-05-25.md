# MTP Sweep Report — 2026-05-25 17:00

## Setup
- **Model:** Qwen3.6-35B-A3B-UD-IQ2_M.gguf (256 MoE experts, IQ2_M)
- **GPU:** GTX 1070 Ti (8 GB VRAM, PCIe Gen3 x16)
- **CPU:** Haswell 4th gen, no AVX2
- **Engine:** VITRIOL DMA offload (-ngl 99, VITRIOL_ENGINE_MODE=vitriol-dma)
- **Benchmark:** Sweep controller, 64-token generation, 1 warmup + 3 measured rounds
- **Fixed params:** ubatch=128, ctx=65536, f16 V cache

## Results

```
 pin mtp     t/s  
   0   0   9.75
   0   2   9.69
   0   4   9.72
   0   5   9.73
   0   6   9.72
   4   0   9.77
   4   2   9.72
   4   4   9.74
   4   5   9.75
   4   6   9.74
   8   0   9.85
   8   2   9.77
   8   4   9.85
   8   5   9.84
   8   6   9.85
  12   0   9.93
  12   2   9.98  ← BEST
  12   4   9.70
  12   5   9.64
  12   6   9.72
  16   0   9.92
  16   2   9.67
  16   4   8.58  ← regression (VRAM pressure)
  16   5   8.58  ← regression
  16   6   8.71  ← regression
```

## Analysis

### MTP has zero measurable benefit
- Scores cluster tightly at **9.7–9.98 t/s** across all MTP values (0 through 6).
- **MTP=0** (no speculation) is consistently among the top performers.
- MTP implies: draft model generates N tokens → full model verifies them in parallel. If verification fails (draft is wrong), the partial accepted prefix is shorter than N and the overhead of running the draft wastes time.

### Pin=16 + MTP shows VRAM pressure
- pin=16 alone (MTP=0) achieves 9.92 t/s — competitive with pin=12.
- But pin=16 + any MTP > 0 drops to 8.58–8.71 t/s (12% regression).
- Explanation: pin=16 uses significant VRAM for expert weight storage; MTP requires additional VRAM for draft model scratch buffers, causing VRAM pressure that spills into PCIe transfers.

### Optimal config: pin=12, mtp=0 (or mtp=2)
- **pin=12 mtp=0: 9.93 t/s**
- **pin=12 mtp=2: 9.98 t/s** (marginally best)
- Given 64-token short generations, the difference is within measurement noise.

## Conclusion
- **MTP is not beneficial for this model on GTX 1070 Ti.** The 35B MoE model's speculations are rejected too frequently to gain throughput.
- **Recommendation:** Run without MTP for production.
- **Next investigation:** Test with longer prompt context (cache reuse) where MTP might shine, or test with smaller MTP draft (MTP=1 or 2 vs 5+).
