# Icarus Profile Validation

**Date:** 2026-05-23 14:50

## Summary

The icarus overclock profile was successfully validated with f16 V cache. q8_0 V cache was confirmed to produce garbage output with VITRIOL DMA path (as warned in the README).

## Results

### Config
| Setting        | Balanced | Icarus   |
|----------------|----------|----------|
| Context        | 136,192  | 65,536   |
| MTP Draft N    | 2        | 5        |
| Pin Layers     | 0        | 12       |
| V Cache Quant  | f16      | f16      |
| UBatch Size    | 256      | 128      |
| K Cache Quant  | q4_0     | q4_0     |

### Benchmark (150-token generation)
| Metric          | Balanced  | Icarus    | Delta  |
|-----------------|-----------|-----------|--------|
| Generation Speed| ~10 t/s   | 12.25 t/s | +22.5% |
| MTP Acceptance  | 91.6%     | 66.7%     | -27%   |
| Prompt Proc     | ~33 t/s   | 29.09 t/s | -12%   |
| VRAM Usage      | ~3,346 MiB| 5,931 MiB | +77%   |
| VRAM Total      | 8,192 MiB | 8,192 MiB | —      |
| VRAM Headroom   | ~4,846 MiB| 2,261 MiB | -53%   |

### Key Findings

1. **q8_0 V cache is broken** with VITRIOL DMA offloading. Only f16 produces sensible output.
2. **ubatch-size 128** is critical for MTP5 efficiency — reduces compute buffer pressure and avoids graph splits.
3. **Pin 12 layers** uses ~2.6 GiB more VRAM but enables faster MTP verification (weights already in VRAM).
4. **MTP5 with 66.7% acceptance** produces ~3.3 accepted tokens per speculation cycle vs MTP2's ~1.83 — the wider speculation window outweighs the lower acceptance rate.
5. **Net speedup: +22.5%** — icarus is definitively faster than the balanced config for this model/hardware.

### Recommendations
- Use `icarus` as the new default profile for little-coder
- Keep `balanced` as fallback for long-context (>65K) tasks
- Keep `little-coder` profile for compatibility when pinning causes issues
- Do not attempt q8_0 V cache again — f16 is the safe minimum
