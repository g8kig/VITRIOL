# VITRIOL Benchmark Results

## Hardware
- GPU: NVIDIA GeForce GTX 1070 Ti (8112 MiB VRAM, PCIe Gen3 x16)
- RAM: 15 GB
- CPU: 4 threads
- **2026-05-19: GTX 960 removed — PCIe restored from x8 to x16**

## Model
- Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf (12 GB)
- Architecture: qwen35moe, 40 layers, 256 experts (8 active/token)
- VITRIOL mode: stream (page-locked RAM + LRU VRAM cache)
- Frozen prompt: on (permanent after diagnosis)
- KV cache: Q4_0

## Generation Speed Tests — CURRENT (PCIe x16, GTX 960 removed)

### 256k Context — Best Config (Frozen Prompt, Q4_0 KV)
| Length | Gen (tok/s) | Eval (tok/s) | Notes |
|--------|------------|-------------|-------|
| 10     | 9.80       | 57.0        | Short burst peak |
| 50     | 9.10       | 58.2        | Sustained |
| 100    | 8.91       | 58.3        | Sustained |

### 256k Context — MTP Enabled (IQ2_M model, --spec-type mtp --spec-draft-n-max 2)
| Length | Gen (tok/s) | Eval (tok/s) | Draft N | Accepted | Accept Rate |
|--------|------------|-------------|---------|----------|-------------|
| 10     | 11.40      | 58.3        | 8       | 4        | 50% |
| 50     | 10.51      | 58.0        | 48      | 24       | 50% |
| 100    | 10.35      | 57.7        | 98      | 49       | 50% |

**MTP improvement over non-MTP: +15-16% gen speed**
> MTP speculative decoding with `draft-n-max=2` yields 50% acceptance rate
> (1 of 2 drafted tokens accepted per cycle). VITRIOL expert offloading
> handles both main trunk and MTP head seamlessly via the same buffer.
>
> Model: Qwen3.6-35B-A3B-UD-IQ2_M.gguf (IQ2_M quant, 2.6 bpw)

**Improvement vs prior baseline (~5.7 tok/s gen): +56-60%**
> Root cause: PCIe Gen3 bottleneck. GTX 960 in second slot halved primary slot to x8 (7.88 GB/s).
> Removing it restored x16 (15.76 GB/s), nearly doubling expert transfer bandwidth.

### VRAM Usage
- Model weights (GPU): 1337 MiB
- VITRIOL buffer (RAM): 10040 MiB
- KV cache (256k Q4_0): ~1406 MiB (host context allocation)
- Compute/RS buffers: ~556 MiB
- Total VRAM at 256K: ~3921 MiB
- Headroom: ~4191 MiB

---

## Historical Results (GTX 960 present, PCIe x8, v1 configs)

### 100k Context, LRU=512 MB
| Length | Gen (tok/s) | Eval (tok/s) | Prefill (ms) |
|--------|------------|-------------|-------------|
| 10     | 6.19       | 37.51       | 453         |
| 50     | 5.69       | 45.49       | 374         |
| 100    | 5.64       | 44.48       | 382         |

### 150k Context, LRU=2048 MB
| Length | Gen (tok/s) | Eval (tok/s) |
|--------|------------|-------------|
| 10     | 6.19       | 43.90       |
| 50     | 5.70       | 45.97       |
| 100    | 5.63       | 45.53       |

### 150k Context, LRU=4096 MB
| Length | Gen (tok/s) | Eval (tok/s) |
|--------|------------|-------------|
| 50     | 5.69       | 43.67       |
| 100    | 5.63       | 43.11       |

### 150k Context, LRU=1024 MB
| Length | Gen (tok/s) | Eval (tok/s) | Prefill (ms) |
|--------|------------|-------------|-------------|
| 50     | 5.69       | 43.67       | 377         |
| 100    | 5.63       | 43.11       | 377         |

### 254k Context, LRU=1024 MB — MAXIMUM VERIFIED
| Length | Gen (tok/s) | Eval (tok/s) | Prefill (ms) |
|--------|------------|-------------|-------------|
| 10     | 6.20       | 44.57       | 381         |
| 50     | 5.70       | 45.13       | 377         |
| 100    | 5.62       | 45.12       | 377         |

### Frozen Prompt Enabled (500k ctx) — BEST CONFIG
| Config | Length | Gen (tok/s) | Eval (tok/s) | Prefill (ms) |
|--------|--------|------------|-------------|-------------|
| Frozen, LRU=1024, t=4 | 50 | **5.79** | **64.84** | 416 |
| Frozen, LRU=1024, t=4 | 100 | **5.72** | **62.76** | 430 |
| Frozen, LRU=2048, t=4 | 50 | **5.89** | **58.57** | 461 |
| Frozen, LRU=2048, t=4 | 100 | **5.87** | **66.29** | 407 |
| Frozen, LRU=4096, t=4 | 50 | 5.78 | 61.69 | 438 |
| Frozen, LRU=4096, t=4 | 100 | 5.71 | 65.21 | 414 |

> **Note:** LRU cache has zero effect on quantized models — see `docs/LRU_DIAGNOSTIC_FINDING.md`.

### Thread Tuning
| Config | Gen (tok/s) | Notes |
|--------|------------|-------|
| t=4 (baseline) | 5.63-5.70 | **Best** |
| t=8 | 4.14-4.21 | 25% worse — thread contention |

## Conclusions
1. **PCIe x16 is the dominant factor**: removing GTX 960 (which halved the slot to x8) yielded +60% gen speed, from 5.7→9.1 tok/s
2. **Frozen prompt still important**: eval speed ~58-65 tok/s (vs ~45 without) — helps prefill, doesn't affect gen
3. **LRU cache is unreachable** for quantized MoE models — all LRU settings gave identical VRAM and speed (see `docs/LRU_DIAGNOSTIC_FINDING.md`)
4. More threads (t=8) makes things **worse** — contention overhead
5. Q4_0 KV cache enables 256K context with ~1.4 GB host allocation
6. VITRIOL stream mode is required for 35B MoE on 8 GB VRAM
