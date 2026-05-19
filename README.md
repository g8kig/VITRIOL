# VITRIOL

<img src="assets/vitriol_logo.svg" alt="VITRIOL" width="200"/>

*"Visita Interiora Terrae Rectificando Invenies Occultum Lapidem"*

*(Visit the Interior of the Earth, by Rectifying you will find the Hidden Stone)*


## Quick Start

```bash
# 1. Clone with submodule (llama.cpp is pinned)
git clone --recursive https://github.com/your/vitriol.git
# Or if already cloned: git submodule update --init --recursive

# 2. Build
cd vitriol/llama.cpp && cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  && cmake --build build -j$(nproc)

# 3. One-time capability grant
./vitriol setup

# 4. Configure
./vitriol config

# 5. Run
./vitriol run
```


## Configuration

VITRIOL has several feature flags that control memory, context efficiency, and retrieval. Each has measurable trade-offs between throughput, context size, and recall quality.

| Flag | Effect | tok/s impact | Use Case |
|------|--------|-------------|----------|
| `--spec-type mtp` / `--spec-draft-n-max 2` | MTP speculative decoding (+20% gen) | +20% | Max throughput (requires MTP-capable model) |
| `--cache-type-k q4_0` / `--cache-type-v q4_0` | KV cache quantized to 4-bit | Required at 256K | Reduces host KV from 5000→1406 MiB; without it, RAM exceeds 15 GB |
| `--kv-mode offload` | KV cache in host RAM | Enables 256K context | Long context coding |
| `--kv-mode sparse` | Attention-score eviction (4-8x compression) | Saves host RAM | Extreme context length |
| `--frozen-prompt on` | Cache KV prefix across requests | Prefill saved | Repeated requests, same system prompt |
| `--memory-mode on` | Cross-session persistent memory via SQLite | ~5.0 | Multi-session projects |
| `--semantic-mode on` | Cosine similarity retrieval | ~5.0 | Large memory databases |

All flags can be set via CLI flag, env var, or the TUI (`vitriol config`).

**Configuration defaults guide:** [`docs/CONFIG_DEFAULTS_GUIDE.md`](docs/CONFIG_DEFAULTS_GUIDE.md) — why each default was chosen, measured performance impact, and when to diverge.

**Full reference:** [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) — every flag explained with trade-offs, use cases, and recommended combinations.

**Recommended settings:** [`docs/RECOMMENDED_SETTINGS.md`](docs/RECOMMENDED_SETTINGS.md) — exact optimal config for the GTX 1070 Ti system.

**Optimization catalog:** [`docs/OPTIMIZATIONS_2026-05-19.md`](docs/OPTIMIZATIONS_2026-05-19.md) — prior art, viability analysis, and roadmap for further optimization.

**Findings log:** [`docs/FINDINGS_2026-05-19.md`](docs/FINDINGS_2026-05-19.md) — detailed benchmark sweep results, floundering log, and lessons learned.

**OpenCode setup:** [`docs/OPENCODE_SETUP.md`](docs/OPENCODE_SETUP.md) — configuring VITRIOL as an OpenCode provider, why `vitriol setup` is required, workflow recommendations.

**Test results:** [`docs/TEST_REPORT_2026-05-17.md`](docs/TEST_REPORT_2026-05-17.md) — measured tok/s, VRAM savings, and bug fixes.


## What Is It

VITRIOL is a **VRAM extension layer** for [llama.cpp](https://github.com/ggml-org/llama.cpp) that lets **old consumer GPUs** run modern MoE language models they have no business running.

The problem: the best open-weight models are MoE architectures (Mixture of Experts) with 200+ expert weight matrices. A Qwen3.6-35B-A3B needs ~12 GB VRAM for weights alone. A GTX 1070 Ti has 8 GB. An RTX 3060 has 12. A GTX 960 has 2. These GPUs are in millions of machines — perfectly capable of fast matrix math, but VRAM-starved.

VITRIOL's insight: MoE models only activate ~2-8 out of 256 experts per token. The expert weights don't need to live in VRAM. Keep them in **page-locked system RAM** instead — the GPU reads them over PCIe DMA on demand. The base model, attention weights, KV cache, and compute buffers stay in VRAM. Only the experts are offloaded.

**Result:** 10.96 tok/s on a GTX 1070 Ti (8 GB) with a 34.66B-parameter 256-expert model — **+92% vs the pre-VITRIOL x8 baseline** (5.7 tok/s). The model doesn't fit at all without VITRIOL.

| Metric | Value |
|--------|-------|
| Model | Qwen3.6-35B-A3B (34.66B, 256 MoE experts) |
| GPU | GTX 1070 Ti (Pascal, 8 GB VRAM, PCIe Gen3 x16) |
| CPU | Intel 4th gen (Haswell, no AVX2) |
| Generation (MTP N=2) | **10.96 tok/s** |
| Generation (no MTP) | 9.1 tok/s |
| VRAM saved | ~10 GB (experts stay in host RAM) |
| System RAM used | ~11.5 GiB (10040 MiB buffer + 1406 MiB host KV) |
| VRAM used | ~1.3 GiB (non-MTP) / ~1.6 GiB (with MTP head) |


## How It Works

### The Trick: Making CUDA Think Host Memory Is Fine

CUDA kernels can read from **page-locked host memory** over PCIe DMA transparently — the GPU's memory controller handles the cross-PCIe access as if it were VRAM, just slower (~12 GB/s PCIe 3.0 vs ~256 GB/s GDDR5). llama.cpp normally avoids this because it's bandwidth-inefficient. But for MoE experts — where each matmul uses only 2-8 of 256 experts per layer — the effective bandwidth needed is low enough that PCIe latency is acceptable.

VITRIOL implements this through a custom **ggml backend buffer type**:

```
VITRIOL buffer type
  │
  ├─ 1. Allocation
  │     mmap(10 GB anonymous)         ← reserve address space
  │     madvise(MADV_HUGEPAGE)        ← hint for 2 MB pages (lower GPU TLB pressure)
  │     mlock                         ← pin to RAM, prevent swapping
  │     cudaHostRegister              ← register with CUDA for DMA access
  │
  ├─ 2. Model load
  │     memcpy from GGUF file → VITRIOL buffer (one-time 10 GB copy, ~64 s)
  │
  └─ 3. Inference
        Set is_host=true on the buffer type → llama.cpp scheduler routes
        MUL_MAT_ID to CUDA backend → GPU reads expert weights over PCIe DMA
```

The key flag is **`is_host=true`**. When the graph scheduler sees this on a buffer type it supports (via `supports_buft`), it treats the tensor as host-resident and keeps it in system memory. The CUDA backend accesses `src0->data` directly — the GPU's DMA engine fetches the bytes over PCIe when the kernel reads from that address.

### LRU VRAM Cache (Inactive for Quantized Models)

The LRU cache was designed to keep hot expert weights in a VRAM pool (~512 MB) for faster
access via native GDDR5 bandwidth. **However, it is never reached for quantized MoE models**
(Q2_K_XL, IQ2_M, etc.) because the MMQ fast path in `ggml_cuda_mul_mat_id` returns before
the LRU code. See [`docs/LRU_DIAGNOSTIC_FINDING.md`](docs/LRU_DIAGNOSTIC_FINDING.md) for the
full diagnosis.

Future VITRIOL-level Expert Pinning (see [`docs/OPTIMIZATIONS_2026-05-19.md`](docs/OPTIMIZATIONS_2026-05-19.md))
will provide the same benefit — keeping frequent experts in VRAM — without modifying
llama.cpp's kernel dispatch.

### Fast Path vs Slow Path

llama.cpp's `ggml_cuda_mul_mat_id` has three paths for MoE matmuls:

| Path | Trigger | Expert Data Access | VITRIOL? |
|------|---------|-------------------|----------|
| **MMVQ** | Batch ≤ 8, quantized weights | Reads `src0->data` directly with expert bounds | Yes — host DMA |
| **MMQ** | Large batch, quantized | Reads `src0->data` directly with expert bounds | Yes — host DMA |
| **cuBLAS (slow path)** | Everything else | Per-expert tensor slices | No — FP16 only |

For quantized models, the MMQ/MMVQ fast paths handle all inference. They read expert
weights directly from the page-locked host buffer over PCIe DMA. The cuBLAS slow path
(which has LRU support) is never reached for quantized types.

### Why Not Just Load Everything Into VRAM?

Because it doesn't fit. Qwen3.6-35B-A3B at UD-Q2_K_XL is 11.44 GiB. The GTX 1070 Ti has 8 GiB total. Without VITRIOL, llama.cpp crashes with `cudaMalloc failed: out of memory` at `-ngl 99`.

With VITRIOL:
- Base model weights (non-expert): ~1.3 GiB in VRAM
- Expert weights: 0 GiB in VRAM (host RAM only)
- KV cache + compute: ~1.4 GiB (host, Q4_0) + ~215 MiB (VRAM, compute buffers)
- MTP head (optional): +302 MiB in VRAM
- **Total VRAM: ~1.5 GiB (non-MTP) / ~1.8 GiB (MTP)** — leaving 6.2+ GiB free


## CLI Reference

```
vitriol run [options]      interactive inference session
vitriol serve [options]    persistent HTTP API server
vitriol stop               stop running server
vitriol config             interactive configuration TUI
vitriol config show        print current configuration
vitriol config init        create config file with defaults
vitriol config reset       restore defaults
vitriol config edit        open config in $EDITOR
vitriol config set <key> <val>  set a config value
vitriol setup              set CAP_IPC_LOCK capability
vitriol help               this message

Run options:
  -m PATH        model file path
  -c N           context window (tokens)
  -t N           CPU threads
  -ngl N         GPU layers to offload
  -lru MB        LRU VRAM cache size (inactive for quantized models)
  --memory-mode MODE  emulated memory: on | off (default: off)
  --kv-quant MODE     KV cache quantization: f16 | q8_0 | q4_0 (must be q4_0 for 256K)
  --spec-type TYPE    speculative decoding: mtp | draft (default: disabled)
  --spec-draft-n-max N  tokens to draft per cycle (default: 0, recommended: 2)
  --verbose      enable debug logging
  --dry-run      print config without launching

Serve options:
  -m PATH        model file path
  -c N           context window (tokens)
  -t N           CPU threads
  -ngl N         GPU layers to offload
  -lru MB        LRU VRAM cache size (inactive for quantized models)
  --host ADDR    bind address (default: 127.0.0.1)
  -port N        server port (default: 8279)
  -p N           parallel slots (default: 1)
  --memory-mode MODE  emulated memory: on | off (default: off)
  --kv-quant MODE     KV cache quantization: f16 | q8_0 | q4_0 (must be q4_0 for 256K)
  --spec-type TYPE    speculative decoding: mtp | draft (default: disabled)
  --spec-draft-n-max N  tokens to draft per cycle (default: 0, recommended: 2)
  --detach       run server in background
  --verbose      enable debug logging
  --dry-run      print config without launching
```

Config persisted in `~/.vitriol/config`. Precedence: CLI flag > Config > Env var > Default.

Use `vitriol serve --detach` for background API mode, `vitriol stop` to shut down.

## VITRIOL Modes

| Mode | What it does |
|------|-------------|
| **stream** | **(Default, only active mode.)** RAM Shot + VITRIOL DMA. All expert weights in page-locked host RAM. GPU reads them over PCIe DMA on demand. MTP head (if enabled) loads through the same buffer. |

### Memory Mode (Experimental)

When enabled (`--memory-mode on` or `VITRIOL_MEMORY_MODE=on`), a Python Flask shim intercepts all requests before they reach llama-server. On each request:

1. **Extract** user intent from the last message
2. **Retrieve** relevant context from a project-local SQLite memory database (episodic + semantic scoring with cascading multi-hop retrieval)
3. **Inject** retrieved context as a system message, staying within a token budget
4. **Forward** the compact prompt to llama-server
5. **Store** the response back into the memory database
6. **Update** Hebbian edge weights (post-response connection strengthening)

Ports swap automatically: llama-server moves to `PORT-1` (8278), the shim listens on `PORT` (8279). OpenCode's baseURL never changes.

```
Memory OFF:  llama-server → port 8279
Memory ON:   llama-server → port 8278, shim → port 8279
```

Configure via `vitriol config` (TUI option 4) or `vitriol serve --memory-mode on`.

## Performance

### Current Best: 10.96 tok/s

System: GTX 1070 Ti (PCIe Gen3 x16), 15 GB RAM, IQ2_M model, MTP N=2, 256K context, Q4_0 KV.

| Config | Gen (tok/s) | vs x8 baseline |
|--------|------------|----------------|
| PCIe x8 (GTX 960 present, no VITRIOL) | 5.7 | — |
| PCIe x16 (GTX 960 removed, Q2_K_XL) | 9.1 | +60% |
| + MTP N=2 (IQ2_M model) | **10.96** | **+92%** |

**Key finding:** MTP acceptance rate = exactly `1/N` — N=2 is optimal. Higher draft values waste PCIe bandwidth on rejected tokens.

See full sweep data in [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md#mtp-draft-n-max-sweep-2026-05-19) and [`docs/FINDINGS_2026-05-19.md`](docs/FINDINGS_2026-05-19.md#mtp-draft-n-max-sweep-2026-05-19).

### VRAM at 256K Context

| Component | Non-MTP | With MTP Head |
|-----------|---------|---------------|
| Model weights (GPU) | ~1337 MiB | ~1337 + 302 MiB |
| VITRIOL buffer (RAM) | ~10040 MiB | ~10040 MiB |
| KV cache (host, Q4_0) | ~1406 MiB | ~1406 MiB |
| Compute buffers | ~215 MiB | + overhead |
| **Total VRAM** | **~1.3 GiB** | **~1.6 GiB** |
| **Total system RAM** | **~11.5 GiB** | **~11.8 GiB** |
| **VRAM headroom** | **~6.7 GiB** | **~6.4 GiB** |


## Hardware Targets

> **PCIe warning:** If you have a secondary GPU in the second PCIe slot, the primary slot
> may drop from x16 to x8. This halves PCIe bandwidth and reduces gen speed by ~60%.
> VITRIOL is PCIe-bound — every token transfers ~40 MB of expert weights across the bus.
> x8 bottleneck: ~5.7 tok/s. x16: ~9.1 tok/s (before MTP).

| GPU | VRAM | Status | Notes |
|-----|------|--------|-------|
| GTX 1070 Ti | 8 GB | ✅ Verified | PCIe 3.0 x16, **10.96 tok/s** (MTP N=2) |
| RTX 3060 | 12 GB | ✅ Supported | More VRAM for larger KV cache |
| RTX 4090 | 24 GB | ✅ Supported | PCIe 4.0 x16 → higher bandwidth |

**CPU requirement:** VITRIOL uses the GPU as the primary MoE compute engine (experts
streamed over PCIe DMA). The CPU is only an orchestrator — no AVX2 or fast CPU is
required. This is in contrast to CPU-based expert offloading (e.g., KTransformers),
which depends heavily on CPU vector extensions.

## Compatibility

### Tested
- **GPU:** NVIDIA GeForce GTX 1070 Ti (CC 6.1, 8 GB VRAM, PCIe Gen3 x16)
- **CPU:** Intel 4th gen (Haswell, no AVX2) — only orchestrates, does not compute experts
- **RAM:** 15 GB DDR3 system, NVMe SSD
- **OS:** Linux — Ubuntu 24.04, kernel 6.17+, NVIDIA driver 535.288.01, CUDA 12.2
- **Models:** Qwen3.6-35B-A3B-UD-Q2_K_XL (baseline, ~2.2 bpw) / IQ2_M (MTP-capable, ~2.6 bpw)
- **llama.cpp:** Pinned submodule with VITRIOL CUDA integration
- **Context:** 256,000 tokens at Q4_0 KV quant (—cache-type-k/q4_0 required)

### Likely works
- **NVIDIA GPUs with CC ≥ 5.0:** All Pascal, Turing, Ampere, Ada, Blackwell cards. The `cudaHostRegister` + PCIe DMA path is architecture-agnostic. Higher VRAM GPUs benefit from larger LRU cache or keeping more layers in VRAM.
- **NVIDIA GPUs with CC 5.x (Maxwell):** Some GPU kernel ops may be missing — set `CUDA_VISIBLE_DEVICES` to exclude them or use `vitriol config` to set `exclude_secondary = true`.
- **ROCm/HIP (AMD):** Would need a mechanical port of the CUDA driver API calls to HIP equivalents (`cudaHostRegister` → `hipHostRegister`, `cuMemAlloc` → `hipMalloc`, etc.). llama.cpp already has `GGML_USE_HIP` guards. Estimated ~400 lines changed across 3 files.
- **Windows (NVIDIA):** NVCC + `cudaHostRegister` works identically. Untested but no fundamental blocker. The `vitriol` CLI launcher currently uses bash — would need a PowerShell/batch wrapper.

### Likely won't work
- **Intel GPUs (SYCL/oneAPI):** SYCL doesn't expose page-locked host DMA in the same way. The unified memory model on integrated GPUs also makes VITRIOL's trick unnecessary — there's no discrete PCIe bus to cross.
- **Apple Silicon (Metal):** Unified memory architecture. CPU and GPU share the same physical RAM pool. VITRIOL provides no benefit — the entire model fits in unified memory if it fits at all.
- **Nouveau (open-source NVIDIA driver):** Lacks `cudaHostRegister` equivalent. The GMMU page tables populated by the proprietary NVIDIA RM driver are required for PCIe DMA from system memory.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    GPU (GTX 1070 Ti, 8 GB VRAM)              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Base model (1.3 GiB)  │  KV Cache  │  Compute buffers │  │
│  │  Embeddings, Attention,│  (512 ctx) │  (sched: ~215 MB)│  │
│  │  RMS Norm, Output      │  (10 MB)   │  LRU pool (opt)  │  │
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

## Emulated Memory Architecture (Experimental)

When **memory mode** is enabled (`--memory-mode on`), a Flask proxy shim sits between the client and llama-server. Every request is intercepted, memory is queried for relevant context, and the prompt is compacted *before* reaching the inference engine — eliminating OpenCode's expensive context compaction loop.

```
OpenCode ──POST /v1/chat/completions──► vitriol_shim.py (port 8279)
                                              │
                                        1. Parse X-Project-Id header
                                        2. Extract user intent from last message
                                        3. Query .vitriol/<project>/memory.db
                                           ├─ Scorer: keyword overlap + recency
                                           ├─ Hebbian weight → edge strength
                                           └─ Cascading multi-hop retrieval
                                        4. Inject retrieved context as system msg
                                        5. Forward to llama-server (port 8278)
                                              │
                                              ▼
                                        llama-server (port 8278)
                                        (8192-token context, never compacts)
                                              │
                                        Post-response:
                                        6. Store response as new episode
                                        7. Hebbian weight update on co-occurring edges
```

Ports swap transparently — OpenCode always talks to port 8279:

```
Memory OFF:  llama-server → port 8279
Memory ON:   llama-server → port 8278, shim → port 8279
```

See `docs/EMULATED_MEMORY_ARCHITECTURE.md` for the full design (DB schema, scoring function, spreading activation, token-budgeted compaction, Hebbian updates, consolidation/sleep).

## Project Structure

```
├── vitriol                  ← CLI entry point (symlink to scripts/vitriol)
├── scripts/
│   └── vitriol              ← Main CLI: config TUI + run + serve + stop + setup
├── libvitriol/
│   ├── vitriol_shim.py      ← Flask proxy with memory mode toggle
│   └── memory/              ← Emulated memory subsystem (7 modules)
│       ├── __init__.py
│       ├── db.py            ← SQLite schema + CRUD
│       ├── scorer.py        ← Composite relevance scoring
│       ├── retrieval.py     ← Intent classification + cascading retrieval
│       ├── compact.py       ← Token-budgeted compaction
│       ├── hebbian.py       ← Post-response edge weight updates
│       └── consolidate.py   ← Background summarization + pruning
├── assets/
│   ├── vitriol-header.txt   ← ASCII art banner
│   └── vitriol_logo.svg     ← SVG logo
├── llama.cpp/               ← Git submodule (pinned commit)
├── llama.cpp/ggml/src/ggml-cuda/
│   ├── vitriol-buffer.{cpp,h}              ← RAM Shot buffer type
│   ├── vitriol-cuda-integration.{cpp,h}    ← LRU cache + init + config
│   ├── vitriol_copy_engine.{cpp,h}         ← CE DMA (standalone)
│   └── ggml-cuda.cu                        ← supports_buft + LRU hooks
├── llama.cpp-patches/       ← Tracked diffs for all VITRIOL changes
├── docs/
│   ├── OPTIMIZATION_PLAN.md (V2)           ← 4-layer roadmap with citations
│   ├── OPTIMIZATION_PLAN_V1.md             ← Preserved original
│   ├── OPTIMIZATIONS_2026-05-19.md         ← Optimization catalog with prior art
│   ├── RECOMMENDED_SETTINGS.md             ← Optimal config for this system
│   ├── FINDINGS_2026-05-19.md              ← Benchmark sweeps, floundering log
│   ├── BENCHMARK_RESULTS.md               ← All benchmark data across configs
│   └── EMULATED_MEMORY_ARCHITECTURE.md     ← Memory design doc
├── EXPERIMENT_LOG.md        ← Complete test history (10 experiments)
├── SESSION_LOG_2026-05-17.md ← This session's progress report
├── ROADMAP.md               ← Phased development plan
├── MILESTONE_1.md           ← Failed approaches archive (7 approaches)
└── MILESTONE_2.md           ← RAM Shot: success report
```

## Ars Priori & Acknowledgements

VITRIOL stands on the shoulders of giants. Every core insight — DMA over PCIe, metapage completion signaling, async expert prefetching, extreme quantization on legacy hardware — was reverse-engineered from the following works. We document our debt explicitly.

### Inference Engine

| Project | What We Learned |
|---------|-----------------|
| **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (ggml-org) | The core inference engine. GGUF format, CUDA backend, tensor loading pipeline. The `-ot` (override tensor) flag in PR #11397 was the breakthrough that enabled expert streaming. Our `vitriol-cuda-integration.cpp` hooks into `ggml-cuda.cu` at the tensor-copy boundary. |
| **[GGUF Format](https://github.com/ggerganov/llama.cpp/blob/master/ggml/include/gguf.h)** | Binary model format with tensor offsets accessible via `gguf_get_tensor_offset()`, `gguf_get_tensor_name()`, `gguf_get_tensor_type()` — the foundation of our expert parser. |
| **[PR #11397](https://github.com/ggerganov/llama.cpp/pull/11397)** (slaren) | Added `--override-tensor` (`-ot`) for per-tensor-type buffer placement. The exact mechanism we use: `-ot ".*exps.*=CPU"` keeps 8GB of experts on CPU while attention layers run on GPU. |
| **[PR #11571](https://github.com/ggerganov/llama.cpp/pull/11571)** (fairydreaming) | Load-all-experts-during-warmup; `llama_set_warmup()` API for ensuring all expert tensors are resident before inference. |
| **[PR #6387](https://github.com/ggerganov/llama.cpp/pull/6387)** (slaren) | Changed expert storage from per-expert tensors to a single 3D tensor — critical for our approach since all 256 experts are now in one contiguous block. |

### GPUDirect Storage & DMA

| Project | What We Learned |
|---------|-----------------|
| **[gds-nvidia-fs](https://github.com/NVIDIA/gds-nvidia-fs)** (NVIDIA) | Official GPUDirect Storage source code. We studied `nvfs-core.c`, `nvfs-pci.c`, and `nvfs-dma.c` to understand: kiocb completion callbacks for NVMe, shared metapage (4KB) for fast completion signaling, `wmb()` memory barriers before DMA. |
| **[open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules)** (NVIDIA) | NVIDIA's open kernel module source for PCIe register-level operations — reference for understanding BAR mapping and GPU PCI config space. |
| **[hw-nvdla](https://github.com/NVIDIA/hw-nvdla)** (NVIDIA) | Hardware DLA documentation for understanding direct memory access patterns on NVIDIA silicon. |

### Async Scheduling & MoE Orchestration

| Project | What We Learned |
|---------|-----------------|
| **[KTransformers](https://github.com/kvcache-ai/KTransformers)** (kvcache-ai) | YAML-based layer placement across CPU/GPU, double-buffer prefetch pattern (compute layer N while streaming N+1), MoE-specific async scheduling. KTransformers targets modern CPUs (AMX/AVX512); VITRIOL inverts this — GPU as primary compute, CPU as orchestrator only. |
| **[PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)** (SJTU-IPADS) | Neuron-level offloading with predictor for which neurons will fire — only loads those into GPU. Informs our predictive prefetching approach. |
| **Qwen3.6-35B-A3B MoE** | 256 experts, 8 active per token — the exact sparsity architecture that makes expert streaming viable. The MoE router (`ffn_gate_inp`) determines which 8 experts to load; only those need to be in VRAM. |

### Speculative Decoding

| Paper / Project | What We Learned |
|-----------------|-----------------|
| **[Fate](https://arxiv.org/abs/2502.12224)** — Fang et al. (2025) | Cross-layer expert prefetching: gate inputs from adjacent layers are ~99% correlated, enabling 97%+ prefetch accuracy with zero GPU overhead. Working third-party llama.cpp fork at `github.com/ongunm/llama-moe-cache` reports 1.91× on Qwen3-30B-A3B. |
| **[PreScope](https://arxiv.org/abs/2509.23638)** — Yu et al. (2025) | LLaPor lightweight predictor (0.5-2.8MB, 0.12-0.48ms), AsyncIO optimizer for overlapping PCIe transfers with GPU compute, cross-layer scheduler. 141% throughput improvement on Qwen3-30B-A3B. |
| **[HOBBIT](https://arxiv.org/abs/2411.01433)** — Tang et al. (2024) | Mixed-precision expert offloading on llama.cpp (~8000 lines). Token-level dynamic loading, layer-level adaptive prefetching, multi-dimensional expert cache. Up to 9.93× decoding speedup on edge devices. Code not open-sourced. |
| **[SP-MoE](https://arxiv.org/abs/2510.10302)** — Chen et al. (2025) | First SD-aware expert offloading: uses draft model's attention outputs to predict target model's expert activations. Combines MTP with expert prefetching. 1.07×-3.5× TPOT speedup. |
| **[MTP](https://arxiv.org/abs/2404.19737)** — Gloeckle et al. (Meta, 2024) | Proved that training models to predict N tokens at once improves reasoning and enables parallel decoding. Foundation of our MTP speculative decoding via Unsloth IQ2_M model. |
| **[Speculative Sampling](https://arxiv.org/abs/2211.17192)** — Leviathan et al. (Google, 2022) | Proved that verification of token sequences is parallelizable — checking 5 tokens takes the same time as checking 1. Foundation of all speculative decoding. |
| **[Speculative Sampling](https://arxiv.org/abs/2302.01318)** — Chen et al. (DeepMind, 2023) | Established rejection sampling math ensuring fast/slow model pair output is identical to the slow model alone. |
| **[Medusa](https://github.com/FasterDecoding/Medusa)** — Cai et al. (2024) | Multiple lightweight decoding heads on a single model to predict +1, +2, +3 tokens ahead. No second model needed. |
| **[EAGLE](https://github.com/SafeAILab/EAGLE)** — Li et al. (2024) | Predicts feature vectors (hidden states) instead of tokens — current SOTA for self-speculative decoding. |
| **[Self-Speculative Decoding](https://arxiv.org/abs/2307.13304)** — (2023) | Layer skipping: run a subset of layers for draft generation, full model for verification. Highly relevant for VITRIOL's DMA layer — skip PCIe transfer for 80% of MoE layers during draft phase. |
| **[Mixture of Speculative Experts](https://arxiv.org/abs/2402.13524)** — (2024) | Top-1 expert draft for MoE: generate guesses using 1/8 experts, verify with all 8. Directly applicable to VITRIOL's expert routing. |
| **[Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding)** — Umang (2024) | N-gram matching from existing context — if a token sequence appeared before, reuse it as a draft. Zero extra VRAM, "free" speed on code tasks. |

### KV Cache & Context Management

| Paper / Project | What We Learned |
|-----------------|-----------------|
| **[vLLM PagedAttention](https://arxiv.org/abs/2309.06180)** — Kwon et al. (2023) | Block-level KV cache management enabling near-zero memory waste. Foundation of efficient serving. |
| **[KIVI](https://arxiv.org/abs/2402.02750)** — Liu et al. (2024) | 2-bit KV cache quantization with minimal accuracy loss. Informs `--kv-quant q4_0` and future KV compression. |
| **[StreamingLLM](https://arxiv.org/abs/2309.17453)** — Xiao et al. (2023) | Identified "attention sinks" (first few tokens) that must be preserved for stable long-context generation. Core insight behind sparse KV caching. |

### CPU Offload & Ternary Compute

| Paper / Project | What We Learned |
|-----------------|-----------------|
| **[Fiddler](https://arxiv.org/abs/2402.14103)** — Kamahori, Gu, Zhu, Kasikci (2024) | Demonstrated that moving *activations* to CPU for MoE expert computation can be faster than pulling weights to GPU via PCIe DMA. Informs future `--engine-mode fiddler-cpu`. |
| **[T-MAC](https://github.com/microsoft/T-MAC)** (Microsoft, 2024) | Lookup-table-based CPU inference for low-bit models. Accelerates ternary math on CPUs without AVX-512. |

### Extreme Quantization & Compute

| Paper / Project | What We Learned |
|-----------------|-----------------|
| **[3LTERN](https://github.com/ELX987/3LTERN)** (ELX987) | W1.58A8 (1.58-bit ternary) CUDA kernel for Pascal. 16 weights packed per uint32, branchless decode via `bit0 - bit1`, `__dp4a` instruction on sm_61. Future optimization path for compute-bound layers. |
| **[Unsloth](https://huggingface.co/unsloth)** (Daniel & Michael) | Dynamic quantization formats (UD-Q2_K_XL) that are structurally superior to raw 1.58-bit. Ungated model distribution — their Qwen 3.6 releases don't require HF authentication. The model we target was quantized and distributed by them. |
| **[MoQE](https://arxiv.org/abs/2310.14713)** — Kim, Fahim, Awadalla (Microsoft, 2023) | MoE experts are robust to extreme low-bit quantization (2-bit) without losing base model coherence. Supports our asymmetric quantization approach. |
| **[BitNet b1.58](https://arxiv.org/abs/2402.17764)** — Ma, Wang et al. (Microsoft Research, 2024) | Ternary weights {-1, 0, 1} match FP16 perplexity, eliminating floating-point multiply. Future TQ1_0 format support. |

### Emulated Memory & Context Retrieval

| Paper / Project | What We Learned |
|-----------------|-----------------|
| **[LLM in a Flash](https://arxiv.org/abs/2312.11514)** — Alizadeh, Mirzadeh et al. (Apple, 2023) | Proved that windowing + zero-copy streaming from flash/host memory enables LLM inference on severely memory-limited hardware. Foundation of the RAM Shot base. |
| **[Fiddler](https://arxiv.org/abs/2402.14103)** — Kamahori, Gu, Zhu, Kasikci (2024) | Demonstrated that moving *activations* to CPU for MoE expert computation can be faster than pulling weights to GPU via PCIe DMA. Informs our `fiddler-cpu` mode. |
| **[SnapKV](https://arxiv.org/abs/2404.14469)** — Li et al. (2024) | Attention heads focus on clustered features; safe eviction of filler tokens reduces KV cache 8.2x without accuracy loss. Informs `--kv-mode sparse`. |
| **[H2O](https://arxiv.org/abs/2306.14048)** — Zhang, Sheng et al. (2023) | Pioneered dropping tokens from KV cache by identifying "Heavy Hitter" tokens that contribute most to attention scores. Informs `--kv-mode sparse`. |
| **[GraphRAG](https://arxiv.org/abs/2404.16130)** — Edge, Trinh et al. (Microsoft, 2024) | Replaced flat vector DBs with LLM-derived knowledge graphs for multi-hop retrieval (spreading activation). Informs our cascading memory retrieval. |
| **[Aider](https://github.com/paul-gauthier/aider)** — Paul Gauthier (2023) | Gold standard for tree-sitter AST-based repo mapping. Informs future AST code graphing for context injection. |

See `docs/OPTIMIZATION_PLAN.md` for the full V2 roadmap with implementation phases.