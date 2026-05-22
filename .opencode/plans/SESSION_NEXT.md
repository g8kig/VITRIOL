# Session Summary — Jamba2 Experiments + Qwen3.6 Return
**Date:** 2026-05-22 09:45-10:15
**Status:** Qwen3.6 verified at 23.5 tok/s, Jamba2 at 2.2 tok/s

---

## What Was Done

### 1. Jamba2-Mini Integration Attempt
   - Confirmed `vitriol_cuda_init` runs via dlopen/dlsym from `libllama.so`
   - VITRIOL pin pool works (4,116 MiB allocated, first 8 expert tensors pinned)
   - **Output cache** (`VITRIOL_OUTPUT_CACHE=1`) corrupts output — disabled
   - **Predictive prefetch** (`VITRIOL_PREDICTIVE_PREFETCH=1`) corrupts output (likely LRU cache returning stale data) — disabled
   - Clean baseline: VITRIOL_MODE=stream only → **2.2 tok/s**, coherent output
   - Root cause: Jamba2 has 16 MoE layers × 3 expert tensors = 48 tensors; pin pool covers only 17%

### 2. CAP_IPC_LOCK Enabled
   - `sudo vitriol setup` → `cap_ipc_lock=ep` set on `llama-server` and `llama-cli`
   - Verified with `getcap`

### 3. Qwen3.6 Return — Verified Config

#### Model File
```
/home/randozart/Desktop/Projects/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
```

#### Environment
| Variable | Value |
|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` |
| `VITRIOL_MODE` | `stream` |
| `VITRIOL_ENGINE_MODE` | `vitriol-dma` |
| `VITRIOL_PIN_FIRST_N_LAYERS` | `8` |
| `VITRIOL_LRU_MB` | `0` |
| `VITRIOL_OUTPUT_CACHE` | `0` |
| `VITRIOL_PREDICTIVE_PREFETCH` | `1` |
| `VITRIOL_VERBOSE` | `1` |
| `LD_LIBRARY_PATH` | `.../build/bin` |

#### Server Arguments
```
llama-server \
  -m Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  -ngl 99 -c 8192 --host 0.0.0.0 --port 8279 \
  --parallel 1 -t 4 -fa on \
  --cache-type-k q4_0 --no-mmap \
  --checkpoint-every-n-tokens 4096 \
  --spec-type mtp --spec-draft-n-max 2
```

#### Benchmark Results
| Metric | Value |
|---|---|
| Prompt speed | 27.16 tok/s |
| Generation speed | 23.49 tok/s |
| Draft tokens | 32 |
| Draft accepted | 32 (100%) |
| Total predicted | 50 tokens in 2,129 ms |

#### ~/.vitriol/config (updated)
```
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
parallel = 1

[engine]
mode = vitriol-dma

[spec]
type = mtp
draft_n_max = 2
```

### 4. Key Code Changes From This Session

- `llama-model-loader.cpp:1187-1220` — VITRIOL init via dlopen/dlsym (debugged, confirmed working)
- `vitriol-cuda-integration.cpp:488-569` — pin pool allocation (debugged, confirmed working)
- `scripts/vitriol:883-890` — `setup_caps()` sets `CAP_IPC_LOCK` via `sudo setcap`
- CAP_IPC_LOCK verified: both `llama-server` and `llama-cli` have `cap_ipc_lock=ep`

### 5. Jamba2 Findings for Future Reference

- Jamba2 GGUF has `build_mamba_layer` (Mamba-1) graph path; weights in Mamba-1 format
- `ssm_in` tensor (shape [4096,16384]) is loaded but **ignored** in Mamba-1 path
- Switching to `build_mamba2_layer` would require model re-quantization (tensor shapes differ)
- 16/32 layers have MoE (odd layers); 16 experts, 2 active per token
- Expert tensors at IQ2_S: ~304 MiB each (3 per MoE layer)
- VITRIOL expert pattern detection works: all 48 expert tensors detected
- VITRIOL buffer type assigned to all expert tensors (system RAM, reported as device)
- Pin pool allocation: 4,116 MiB @ `7964ea000000` (GTX 1070 Ti VRAM address)
- `get_layer_index` assigns sequential IDs 0-45 to expert tensors on first forward pass
- With `pin_first_n_layers=4`: 8 tensors pinned (model_layers 0-3), rest skipped

### 6. Lessons Learned

1. **Output cache corrupts output** — don't use `VITRIOL_OUTPUT_CACHE=1` (LRU key vs output cache key collision)
2. **Predictive prefetch corrupts output** for Jamba2 — likely stale LRU data from wrong expert prediction
3. **Clean VITRIOL_MODE=stream** (no caches) works correctly for both models
4. **CAP_IPC_LOCK** must be re-applied after every rebuild (`sudo vitriol setup`)
5. Jamba2 at IQ2_M (2.7 bpw) generates coherent output but at only 2.2 tok/s on GTX 1070 Ti
6. Qwen3.6 IQ2_M with MTP achieves 23.5 tok/s on same hardware — **10× faster**