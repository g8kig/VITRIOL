# RAM Shot: Page-Locked Host RAM for MoE Expert Weights

**Date**: 2026-05-16
**Status**: ✅ WORKING (6.31 tok/s on GTX 1070 Ti, 35B MoE model)

## Architecture

```
Expert weights → VITRIOL buffer (mmap + mlock + cudaHostRegister)
                     ↓
            Page-locked host RAM (10 GB)
                     ↓
            GPU reads over PCIe DMA during MUL_MAT_ID
                     ↓
            No VRAM used for weight storage
```

## Key Files

| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/vitriol-buffer.{h,cpp}` | Custom buffer type: mmap, mlock, cudaHostRegister, is_host=true |
| `ggml/src/ggml-cuda/vitriol-cuda-integration.{h,cpp}` | Init, config, CE DMA stub |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | supports_buft accepts VITRIOL type |
| `ggml/src/llama-model-loader.cpp` | dlsym-based VITRIOL buft auto-apply for expert tensors |

## How to Run

```bash
# One-time privilege grant
sudo setcap cap_ipc_lock=+ep ./build/bin/llama-server

# Run
CUDA_VISIBLE_DEVICES=0 VITRIOL_MODE=stream ./build/bin/llama-server \
  -m model.gguf -ngl 41 -c 2048 --port 8279 --no-warmup

# Inference test
curl -X POST http://127.0.0.1:8279/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Performance

| Metric | Value |
|--------|-------|
| Prompt eval | 33.86 tok/s |
| Generation | 6.31 tok/s |
| Model load time | ~64 s (10 GB memcpy) |
| Graph splits | 17 (was 2) |
| VRAM saved | ~10 GB |

## Trade-offs

**Pros:**
- Enables models that don't fit in VRAM (35B on 8 GB card)
- Only ~3% slower than all-VRAM
- No modification to MUL_MAT_ID kernel
- Simple, clean implementation
- All infrastructure (mmap, mlock, cudaHostRegister) is standard CUDA/Linux

**Cons:**
- Requires `CAP_IPC_LOCK` capability (one-time sudo setup)
- Consumes 10 GB system RAM
- 17 graph splits (vs 2) — scheduler overhead
- HugePages benefit depends on transparent hugepage config

## Future Optimizations

### 1. CE DMA LRU Cache (highest priority)
Use the already-initialized Copy Engine to cache frequently-used experts in a small VRAM pool. On cache hit → VRAM-speed matmul. On miss → PCIe DMA from host RAM.

Estimated gain: 10-50% depending on expert locality.

### 2. Graph Split Reduction
With `is_host=true`, the scheduler creates 17 splits (vs 2). Investigate whether VITRIOL buffer type can be made to look more like a CUDA buffer to reduce splits.

### 3. io_uring + O_DIRECT
Bypass mmap entirely — read expert data directly from GGUF file using io_uring with O_DIRECT. Frees page cache and allows CPU to page out expert data.

### 4. Dual-GPU
With 10 GB freed from VRAM, explore using the second GPU (GTX 960) for additional compute or speculative decoding.

## Previous Approaches (Failed)

1. **supports_buft CPU accept** — CUDA backend accepted CPU buffer types → ROPE crash (GPU accessed unmapped system memory)
2. **VITRIOL buffer + is_host=false** — same ROPE crash (GPU accessed mmap'd system memory without page-locking)
3. **CE DMA on-demand streaming** — complex, invasive MUL_MAT_ID modifications, bounce buffer overhead
4. **GPUDirect RDMA** — blocked by NVIDIA GeForce SKU lockout
5. **PCI BAR1 takeover** — blocked by GMMU page table initialization
6. **Nouveau + DRM** — blocked by nvidia/nouveau mutual exclusion
7. **PAT bypass** — blocked by kernel PAT enforcement on kernel 6.17
