# Mellum2 Optimal Configuration

**Date:** 2026-06-04 12:53 UTC
**Model:** Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf (7.5 GB, Q4_K_M)
**Hardware:** GTX 1070 Ti (8GB VRAM) + 64GB DDR4
**VITRIOL branch:** vitriol-mellum2 (cherry-picked upstream PR #23966)

---

## Optimal Settings

| Setting | Value |
|---------|-------|
| GPU layers (-ngl) | 24 |
| Context (-c) | 32768 |
| Engine mode | vitriol-dma |
| LRU cache | 2048 MiB |
| Expert pinning | 0 (none) |
| MTP | 0 (disabled — native MTP not yet supported) |

### Command

```bash
VITRIOL_ENGINE_MODE=vitriol-dma \
VITRIOL_LRU=2048 \
VITRIOL_PIN=0 \
llama-server \
  -m Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf \
  -ngl 24 \
  -c 32768 \
  --host 0.0.0.0 --port 8080
```

---

## Sweep Results

### Config Sweep

| ngl | ctx | t/s (avg 3 rounds) | VRAM (model) | Status |
|-----|-----|--------------------|-------------|--------|
| 10 | 16384 | 11.06 | 2626 MiB | OK |
| 10 | 32768 | 11.65 | 2626 MiB | OK |
| 15 | 16384 | 15.29 | 3922 MiB | OK |
| 15 | 32768 | 15.96 | 3922 MiB | OK |
| 20 | 16384 | 21.19 | 5172 MiB | OK |
| 20 | 32768 | 18.10 | 5172 MiB | OK |
| **24** | **16384** | **27.74** | **6228 MiB** | **OK** |
| **24** | **32768** | **27.07** | **6228 MiB** | **Optimal** |
| 24 | 65536 | — | — | OOM (compute pp buffers) |
| 25 | 32768 | — | — | OOM (compute pp buffers) |
| 26 | 32768 | — | — | OOM (KV cache) |
| 28 | 32768 | — | — | OOM (model load) |

### Raw Round Data (Optimal: ngl=24 ctx=32768)

| Round | t/s |
|-------|-----|
| 1 | 27.49 |
| 2 | 27.60 |
| 3 | 26.14 |
| **Average** | **27.07** |

### Memory Breakdown (Optimal)

| Pool | Size |
|------|------|
| CUDA model (GPU) | 6228 MiB |
| CPU model (DDR4) | ~1500 MiB |
| CUDA KV cache | ~446 MiB |
| CPU KV cache | ~191 MiB |
| CUDA compute | ~264 MiB |
| **Total VRAM** | **~6938 MiB / 8112 MiB (86%)** |

### Architecture (Detected Correctly)

| Property | Value |
|----------|-------|
| Architecture | `mellum` |
| Layers | 28 |
| Embedding dim | 2304 |
| Attention heads | 32 Q / 4 KV (GQA=8) |
| Expert count (total) | 64 |
| Expert count (active) | 8 |
| Expert FFN dim | 896 |
| Sliding window | 1024 (21/28 layers) |
| Vocab | 98304 (BPE, mellum2 pre-tokenizer) |
| Quality check | Correct answers, coherent chat output |

---

## Quality

- **Completion:** Correctly answered "What is the capital of France?" → "Paris"
- **Chat:** "Write a haiku about programming" → *"Lines of code flow, / Silent logic takes form, / World rewrites itself."*
