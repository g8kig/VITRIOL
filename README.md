# VITRIOL

<img src="assets/vitriol_logo.svg" alt="VITRIOL" width="200"/>

*"Visita Interiora Terrae Rectificando Invenies Occultum Lapidem"*

(Visit the Interior of the Earth, by Rectifying you will find the Hidden Stone)

## What It Is

VITRIOL runs large MoE language models on **VRAM-constrained consumer GPUs** by keeping expert weights in **page-locked host system RAM** instead of VRAM. The GPU reads them over PCIe DMA during MUL_MAT_ID — no kernel modules, no PCI unbinding, no display crashes.

**RAM Shot:** mmap → mlock → cudaHostRegister → is_host=true. Expert weights live in system RAM, GPU reads transparently over PCIe.

**LRU Cache:** A small VRAM pool (~512 MB) caches hot experts via async cuMemcpyHtoDAsync. Cache hit = native VRAM speed. Cache miss = PCIe DMA from host.

| Metric | Value |
|--------|-------|
| Model | Qwen3.6-35B-A3B (34.66B, 256 MoE experts) |
| GPU | GTX 1070 Ti (Pascal, 8 GB VRAM) |
| Generation (fast path) | **6.9 tok/s** |
| Generation (slow path) | **7.0 tok/s** |
| VRAM saved | ~10 GB (experts stay in host RAM) |
| System RAM used | 10 GB (page-locked, never swapped) |

**Status:** ✅ Working — `vitriol run` with default config.

---

## Quick Start

### Prerequisites

```bash
# llama.cpp is a git submodule — clone with:
git clone --recursive https://github.com/your/vitriol.git
# Or if already cloned:
git submodule update --init --recursive

# Build
cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON && cmake --build build -j$(nproc)

# One-time: grant capability for mlock + cudaHostRegister
vitriol setup
```

### Run

```bash
# Interactive config (set model path, GPU, etc.)
vitriol config

# Launch
vitriol run

# Or one-shot with custom settings
vitriol run -m /path/to/model.gguf -c 4096 -lru 1024 --verbose
```

### Test

```bash
curl -X POST http://127.0.0.1:8279/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

---

## CLI Reference

```
vitriol run [options]    launch inference
vitriol config           interactive configuration TUI
vitriol config show      print current configuration
vitriol config init      create config file with defaults
vitriol config reset     restore defaults
vitriol config edit      open config in $EDITOR
vitriol config set <key> <val>  set a config value
vitriol setup            set CAP_IPC_LOCK capability
vitriol help             this message

Run options:
  -m PATH        model file path
  -c N           context window (tokens)
  -t N           CPU threads
  -ngl N         GPU layers to offload
  -lru MB        LRU VRAM cache size
  -port N        server port
  --verbose      enable debug logging
  --dry-run      print config without launching
```

Config is persisted in `~/.vitriol/config`. Precedence: CLI flag > Config > Env var > Default.

---

## VITRIOL Modes

| Mode | What it does |
|------|-------------|
| **stream** | **(Default)** RAM Shot + LRU VRAM cache. Expert weights in page-locked host RAM, GPU reads over PCIe DMA. Hot experts cached in a 512 MB VRAM pool for native-speed matmul on cache hit. Best perf/VRAM tradeoff. |
| **sync** | Preloads expert data synchronously before each matmul. No LRU cache. Every expert read goes over PCIe DMA. |
| **async** | Double-buffer prefetch on a separate CUDA stream. Hides DMA latency behind computation. |
| **off** | VITRIOL completely inactive. Falls through to normal llama.cpp allocation (likely OOM on 8 GB GPU for a 35B model). |

### LRU Cache

When `mode = stream`, the LRU cache keeps hot expert weights in a small VRAM pool:

- **Cache hit:** Expert accessed from VRAM → native matmul speed
- **Cache miss:** Async `cuMemcpyHtoDAsync` from page-locked host to VRAM pool on a dedicated CUDA stream, synced via `cuStreamWaitEvent` before matmul
- **Eviction:** LRU order (std::list + unordered_map), composite key `(tensor_base_address, expert_idx)` prevents cross-layer collisions
- **Slot sizing:** Fixed on first allocation; larger experts bypass cache and read from host

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
                 → Fast path: MMVQ/MMQ (6.9 tok/s)
                 → Slow path: cuBLAS with LRU cache (7.0 tok/s)
```

The VITRIOL buffer type (`vitriol-buffer.cpp`) allocates system RAM, page-locks it for GPU access, and reports `is_host=true` to llama.cpp's graph scheduler. The scheduler routes MUL_MAT_ID to the CUDA backend, which reads the weights transparently from host memory. No VRAM is used for expert weight storage.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    GPU (GTX 1070 Ti, 8 GB VRAM)              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Base model (1.3 GiB)  │  KV Cache  │  Compute buffers │  │
│  │  Embeddings, Attention,│  (512 ctx) │  (sched: ~215 MB)│  │
│  │  RMS Norm, Output      │  (10 MB)   │                  │  │
│  │                         │            │  LRU pool (opt)  │  │
│  │                         │            │  (512 MB)        │  │
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
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
├── vitriol                  ← CLI entry point (symlink to scripts/vitriol)
├── scripts/
│   └── vitriol              ← Main CLI: config TUI + run + setup
├── assets/
│   ├── vitriol-header.txt   ← ASCII art banner
│   └── vitriol_logo.svg     ← SVG logo
├── llama.cpp/               ← Git submodule (pinned)
│   └── ggml/src/ggml-cuda/
│       ├── vitriol-buffer.{cpp,h}         ← RAM Shot buffer type
│       ├── vitriol-cuda-integration.{cpp,h} ← LRU cache + init + config
│       ├── vitriol_copy_engine.{cpp,h}    ← CE DMA implementation (standalone)
│       └── ggml-cuda.cu                   ← supports_buft + LRU hooks
├── llama.cpp-patches/       ← Tracked diffs for all VITRIOL changes
├── docs/                    ← Architecture docs
├── EXPERIMENT_LOG.md        ← Complete test history (9 experiments)
├── ROADMAP.md               ← Phased development plan
├── MILESTONE_1.md           ← Failed approaches archive (7 approaches)
└── MILESTONE_2.md           ← RAM Shot: success report
```

---

## Performance

| Metric | RAM Shot only | + LRU Cache |
|--------|---------------|-------------|
| Prompt eval (fast path) | 33.86 tok/s | 22.4 tok/s |
| Prompt eval (slow path) | — | 14.3 tok/s |
| Generation | **6.31 tok/s** | **6.9 tok/s** |
| VRAM used | **1.3 GiB** | **~1.8 GiB** |
| System RAM | +10 GiB | +10 GiB |
| Model load | ~64 s | ~64 s |

The model (11.44 GiB) does **not fit** in 8 GB VRAM without VITRIOL.

---

## Hardware Targets

| GPU | VRAM | Status | Notes |
|-----|------|--------|-------|
| GTX 1070 Ti | 8 GB | ✅ Verified | PCIe 3.0 x16, 6.9 tok/s |
| GTX 960 | 2 GB | ⚠️ Limited | CC 5.2 lacks kernel images for some ops |
| RTX 3060 | 12 GB | ✅ Supported | More VRAM for larger KV cache |
| RTX 4090 | 24 GB | ✅ Supported | PCIe 4.0 x16 → higher bandwidth |

---

## Philosophy

**VITRIOL = Engine.** Raw DMA power. Hard-coded, dangerous, fast.

**Alka = ECU.** Abstraction layer. Safe, repeatable, multi-device.

Build the engine first. Let the friction prove why you need the computer.

---

*"Visita Interiora Terrae Rectificando Invenies Occultum Lapidem"*
