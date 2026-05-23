# VITRIOL Config Profiles & Optimization Plan

**Date:** 2026-05-23 13:31

## Current Optimizations

### GPU Memory
- **RAM Shot** — MoE expert weights in page-locked host RAM; GPU reads over PCIe DMA
- **LRU VRAM Cache** — 2 GB pool (configurable) caches hot expert weights
- **Predictive Prefetching** — cross-layer + temporal heuristic on async CUDA stream
- **Expert Pinning** — first N layers' experts locked in VRAM (currently off)

### Compute
- **Expert Pruning** — drop bottom N of 8 active MoE experts
- **Approximate Output Cache** — reuses expert FFN outputs across consecutive tokens
- **Early Exit** — residual delta stagnation threshold (Qwen3.5 MoE)
- **MTP Speculative Decoding** — Multi-Token Prediction head from GGUF

### Memory Hierarchy
- **Chimera Mode** — hybrid CUDA+Vulkan tensor routing
- **KV Cache Offload** — to page-locked host RAM
- **Sparse KV Eviction** — drop Nth cell when full
- **Disk Offload** — file-backed mmap

### Kernel-Level DMA (vitriol.ko)
- NVMe-to-GPU direct DMA via nvidia P2P cooperative
- Custom Alka bytecode ABI for GPU DMA engine control

### Server
- **Prompt Cache** — full slot state snapshots in RAM
- **KV Checkpoints** — rollback points for speculative decoding
- **Context Shifting** — drop middle context for infinite conversations
- **AST-Based Prompt Compaction** — tree-sitter strips function bodies

### Key Env Vars
`VITRIOL_MODE`, `VITRIOL_LRU_MB`, `VITRIOL_PREDICTIVE_PREFETCH`, `VITRIOL_PIN_FIRST_N_LAYERS`, `VITRIOL_PRUNE_EXPERTS`, `VITRIOL_OUTPUT_CACHE`, `VITRIOL_EARLY_EXIT`, `VITRIOL_KV_MODE`, `VITRIOL_CHIMERA_MODE`, `VITRIOL_DISK_OFFLOAD`, `VITRIOL_EXPERT_COUNT`

## Build: Tree-Sitter Fixes

Resolved build errors in `llama.cpp/tools/server/treesitter/`:
- **ABI field mismatch** — grammar `.version` → `.abi_version` (rename in v0.25+)
- **Missing `TSFieldMapSlice` type** — added `typedef TSMapSlice TSFieldMapSlice` compat alias
- **Upgraded runtime to v0.26.0** — redownloaded core files, ICU/portable headers
- **WASM stubs** — added `wasm_store.c` to build target
- **`le16toh`/`be16toh`** — added `_DEFAULT_SOURCE` to feature test macros

## Little-Coder Integration

- Created `~/.config/little-coder/models.json` override for VITRIOL server at `127.0.0.1:8279`
- Mapped `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` with 136K context + image support

## Config Profile System Design

### Storage
- Profiles stored as full INI files: `~/.vitriol/profiles/<name>/config`
- Metadata: `~/.vitriol/profiles/<name>/meta` (description, timestamp)
- Reuses existing `parse_config()` / `write_config()` functions

### New Subcommands
```
vitriol config save <name> [description]   — save current config as profile
vitriol config load <name>                 — copy profile into main config
vitriol config list                        — list all profiles
vitriol config delete <name>               — remove a profile
vitriol config diff <name>                 — show diff vs current
```

### TUI Menu Additions
```
 9) Save Config Profile
10) Load Config Profile
11) Manage Profiles (list/delete)
```

### Profiles Created

| Profile | Description | Context | Pin Layers | Draft N-Max |
|---------|------------|---------|------------|-------------|
| `balanced` | Default VITRIOL config (136K ctx, MTP2, no pin) | 136192 | 0 | 2 |
| `little-coder` | Optimized for little-coder throughput (65K ctx, pin10, MTP3) | 65536 | 10 | 3 |

### Performance Reasoning

**Context 65536:** Frees ~1300 MiB from KV cache (drops from 1704 MiB to ~820 MiB), giving headroom for pinning.

**Pin 10 layers:** ~2340 MiB VRAM (10 layers × ~234 MiB each). Combined with freed KV cache space, leaves ~3020 MiB free headroom. Eliminates PCIe DMA for first 25% of MoE expert lookups.

**MTP Draft 3:** At ~91.6% per-token acceptance, expected ~2.52 accepted tokens per MTP call vs ~1.76 with draft 2 — ~43% more throughput for ~47% more compute. Sweet spot before diminishing returns at draft 4.
