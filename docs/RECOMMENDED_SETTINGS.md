# VITRIOL Recommended Settings — GTX 1070 Ti (8GB VRAM)

System: 15 GB RAM, NVMe SSD, 4 CPU threads, PCIe Gen3 x16.
**Qwen3.6-35B-A3B-UD-IQ2_M.gguf** (23.3 tok/s verified).

---

## Quick Config (`~/.vitriol/config`)

```ini
[gpu]
device = 0
exclude_secondary = true

[model]
path = /home/randozart/Desktop/Projects/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
context = 8192
threads = 4
ngl = 99
expert_count = 0

[vitriol]
mode = stream
lru_mb = 0
verbose = true
output_cache = off
predictive_prefetch = on
pin_first_n_layers = 8
prune_experts = 0
reasoning = off

[server]
host = 0.0.0.0
port = 8279
parallel = 1           # MTP caps parallel to 1

[memory]
mode = off
semantic_mode = off

[kv]
mode = standard
quant_mode = q4_0       # K cache only; V stays f16
quant_mode_v = f16      # ⚠ Do not change — V quant corrupts output
frozen_prompt = on

[engine]
mode = vitriol-dma

[lookup]
tokens = 0

[spec]
type = mtp
draft_n_max = 2

[chimera]
mode = auto             # Auto-detect CUDA+Vulkan hybrid
```

## CLI Flags (for direct llama-server usage)

```bash
VITRIOL_MODE=stream \
VITRIOL_ENGINE_MODE=vitriol-dma \
VITRIOL_PIN_FIRST_N_LAYERS=8 \
  llama-server \
    -m Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
    -ngl 99 -c 8192 --host 0.0.0.0 --port 8279 \
    --parallel 1 -t 4 -fa on \
    --cache-type-k q4_0 --no-mmap \
    --checkpoint-every-n-tokens 4096 \
    --spec-type mtp --spec-draft-n-max 2
```

**⚠️ V cache:** Do NOT pass `--cache-type-v`. V cache stays at f16.
Quantizing V produces garbage output with VITRIOL (see EXPERIMENT_LOG.md).

## Per-Setting Rationale

| Setting | Value | Why |
|---------|-------|-----|
| `context = 8192` | Matches typical chat use. 256K works but uses more VRAM for KV cache. |
| `threads = 4` | GTX 1070 Ti has 4 scheduler units. `t=8` causes contention (+25% slower). |
| `ngl = 99` | Offload all layers. Weights in host RAM via VITRIOL DMA. |
| `mode = stream` | Only mode that page-locks RAM for VITRIOL DMA. |
| `pin_first_n_layers = 8` | Covers first 8 expert layers in VRAM (+5-10% speed). |
| `parallel = 1` | Required by MTP speculative decoding. |
| `predictive_prefetch = on` | Overlaps expert DMA with compute for next token. |
| `cache-type-k q4_0` | 4-bit K cache saves VRAM. V stays f16. |
| `frozen_prompt = on` | Cache KV prefix across requests, avoids re-prefix. |
| `engine.mode = vitriol-dma` | Enables the CUDA expert intercept layer. |
| `spec.type = mtp` | Multi-Token Prediction: 19/19 draft acceptance (+100%). |
| `chimera.mode = auto` | Auto-detects CUDA+Vulkan backends for hybrid routing. |

## Chimera Dual-Backend

When `chimera.mode = auto` and both CUDA + Vulkan are present:

| Operation | Backend | Benefit |
|-----------|---------|---------|
| MoE expert matmuls | CUDA VITRIOL DMA | Page-locked host RAM, pin pool, predictor |
| SSM scan, attention, norms | Vulkan | Pre-baked command buffers, `VK_EXT_external_memory_host` |
| Cross-backend copies | CPU staging (automatic) | ~0.13% overhead per token |

## Performance

| Config | Gen (tok/s) | vs x8 baseline |
|--------|------------|----------------|
| PCIe x8 (GTX 960 present) | 5.7 | — |
| PCIe x16 (GTX 960 removed) | 8.9 | +56% |
| + IQ2_M + MTP N=2 + pin 8 | 12.82 | +125% |
| **+ Chimera + CAP_IPC_LOCK** | **23.3** | **+309%** |

## Context Size vs KV Cache

| Context | KV Cache (K q4_0 + V f16) | Gen Speed |
|---------|--------------------------|-----------|
| 8,192 | ~28 MiB | 23.3 tok/s |
| 16,384 | ~226 MiB | 20.3 tok/s |
| 32,768 | ~451 MiB | 18.7 tok/s |
| 65,536 | ~902 MiB | 18.3 tok/s |
| 262,144 | ~3.6 GiB | 17.3 tok/s (pin pool disabled) |

