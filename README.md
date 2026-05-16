# VITRIOL

<img src="assets/vitriol_logo.svg" alt="VITRIOL" width="200"/>

*"Visita Interiora Terrae Rectificando Invenies Occultum Lapidem"*

(Visit the Interior of the Earth, by Rectifying you will find the Hidden Stone)

## What It Is

VITRIOL runs large MoE language models on **VRAM-constrained consumer GPUs** by keeping expert weights in **page-locked host system RAM** instead of VRAM. The GPU reads them over PCIe DMA during MUL_MAT_ID — no kernel modules, no PCI unbinding, no display crashes.

**Current approach (RAM Shot):** mmap → mlock → cudaHostRegister → is_host=true. Expert weights live in system RAM, GPU reads them transparently over PCIe. Only ~3% slower than all-VRAM, and enables models that wouldn't otherwise fit.

| Metric | Value |
|--------|-------|
| Model | Qwen3.6-35B-A3B (34.66B, 256 MoE experts) |
| GPU | GTX 1070 Ti (Pascal, 8 GB VRAM) |
| Generation | **6.31 tok/s** |
| VRAM saved | ~10 GB (experts stay in host RAM) |
| System RAM used | 10 GB (page-locked, never swapped) |

**Status:** ✅ Working — `VITRIOL_MODE=stream` on llama-server.

---

## How It Works

```
VITRIOL buffer type (custom ggml_backend_buffer_type)
  │
  ├─ Allocation: mmap(10GB) → madvise(MADV_HUGEPAGE)
  │              → mlock → cudaHostRegister
  │
  ├─ Model load: memcpy from GGUF → VITRIOL buffer (one-time copy)
  │
  └─ Inference:  GPU reads weights over PCIe DMA
                 → 6.31 tok/s on GTX 1070 Ti
```

The VITRIOL buffer type (`vitriol-buffer.cpp`) allocates system RAM, page-locks it for GPU access, and reports `is_host=true` to llama.cpp's graph scheduler. The scheduler routes MUL_MAT_ID to the CUDA backend, which reads the weights transparently from host memory. No VRAM is used for expert weight storage.

---

## Quick Start

### Prerequisites

```bash
# One-time: grant capability for mlock(10GB) + cudaHostRegister(10GB)
sudo setcap cap_ipc_lock=+ep ./build/bin/llama-server

# Build llama.cpp with VITRIOL
cd llama.cpp/build
cmake .. -DGGML_CUDA=ON
make -j$(nproc) llama-server
```

### Run

```bash
CUDA_VISIBLE_DEVICES=0 VITRIOL_MODE=stream ./bin/llama-server \
  -m /path/to/model.gguf \
  -ngl 41 \
  -c 2048 \
  --port 8279
```

### Test

```bash
curl -X POST http://127.0.0.1:8279/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    GPU (GTX 1070 Ti, 8 GB VRAM)              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Base model (1.3 GiB)  │  KV Cache  │  Compute buffers │  │
│  │  Embeddings, Attention,│  (512 ctx) │  (sched: ~215 MB)│  │
│  │  RMS Norm, Output      │  (10 MB)   │                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                    ▲ PCIe DMA (~12 GB/s)                      │
└────────────────────┼─────────────────────────────────────────┘
                     │
┌────────────────────┼─────────────────────────────────────────┐
│           CPU / System RAM (DDR3, ~20 GB/s)                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  VITRIOL buffer (10 GiB, page-locked, never swapped)   │  │
│  │  256 expert weight tensors, 0% VRAM footprint           │  │
│  │  mmap → madvise(HUGEPAGE) → mlock → cudaHostRegister    │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  GGUF mmap (11.44 GiB file, page cache)                │  │
│  └────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  CE DMA (initialized, available for LRU cache)         │  │
│  │  Bounce buffer: 256 MB, pre-pinned, stream ready       │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── llama.cpp/
│   └── ggml/src/ggml-cuda/
│       ├── vitriol-buffer.{cpp,h}         ← RAM Shot buffer type
│       ├── vitriol-cuda-integration.{cpp,h} ← Init, config, CE stub
│       ├── vitriol_copy_engine.{cpp,h}    ← CE DMA implementation
│       └── ggml-cuda.cu                   ← supports_buft + VITRIOL hooks
├── alka-executor/                         ← CE DMA standalone test + executor
├── docs/                                  ← Architecture docs
├── scripts/                               ← Build & benchmark scripts
├── MILESTONE_1.md                         ← Complete failure archive (7 approaches)
├── MILESTONE_2.md                         ← RAM Shot: success report
└── .opencode/plans/                       ← Implementation plans
```

---

## Performance

| Metric | RAM Shot | Notes |
|--------|----------|-------|
| Prompt eval | 33.86 tok/s | 19 tokens in 561 ms |
| Generation | **6.31 tok/s** | 30 tokens in 4.75 s |
| VRAM used | **1.3 GiB** | 8 GiB free for other tasks |
| System RAM | +10 GiB | Page-locked, never swapped |
| Model load | ~64 s | 10 GB memcpy from GGUF |

The model (11.44 GiB) does **not fit** in 8 GB VRAM without VITRIOL.

---

## Hardware Targets

| GPU | VRAM | VITRIOL | Notes |
|-----|------|---------|-------|
| GTX 1070 Ti | 8 GB | ✅ Verified | PCIe 3.0 x16, 6.31 tok/s |
| GTX 960 | 2 GB | ⚠️ Partial | CC 5.2 lacks kernel images for some ops |
| RTX 3060 | 12 GB | ✅ Supported | More VRAM for larger KV cache |
| RTX 4090 | 24 GB | ✅ Supported | PCIe 4.0 x16 → higher bandwidth |

---

## Philosophy

**VITRIOL = Engine.** Raw DMA power. Hard-coded, dangerous, fast.

**Alka = ECU.** Abstraction layer. Safe, repeatable, multi-device.

Build the engine first. Let the friction prove why you need the computer.

---

*"Visita Interiora Terrae Rectificando Invenies Occultum Lapidem"*
