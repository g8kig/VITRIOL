# Session Log 2026-05-21

## Objective
Get the custom VITRIOL model (Qwen3.6-35B) serving coherent text through OpenCode via llama-server, with stable streaming and correct output.

## Setup
- **GPU**: NVIDIA GeForce GTX 1070 Ti (8 GB VRAM, CC 6.1)
- **Models**:
  - `/home/randozart/Downloads/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` — IQ2_M, 12 GB, has MTP (`nextn_predict_layers`)
  - `/home/randozart/Desktop/Projects/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` — Q2_K_XL, 12 GB, no MTP
- **Binary**: `llama.cpp/build/bin/llama-server` (b101-e6487cdaf)
- **Provider**: `@ai-sdk/openai-compatible` → `http://127.0.0.1:8279/v1`
- **Config**: `~/.vitriol/config`
- **Script**: `scripts/vitriol`
- **Memory mode**: OFF (direct to llama-server, no Python shim)

---

## Bug 1: Stream Hanging — Dangling Reference Fix

### Location
`llama.cpp/tools/server/server-context.cpp:3531`

### Cause
`should_stop_fn` lambda captured `&req` (dangling reference). `req` is stack-allocated; when the lambda executes later via `next()`, `req` has been destroyed — undefined behavior, stream hangs without `[DONE]`.

### Fix
Changed capture from `&req` to `should_stop_fn = req.should_stop` (copy the `should_stop` flag, not the reference).

### Verification
Stream termination confirmed working — `data: [DONE]` now appears in curl streaming tests.

---

## Bug 2: V Cache Quantization Produces `?` Garbage

### Root Cause
`scripts/vitriol:1327` passed **both** `--cache-type-k q4_0` and `--cache-type-v q4_0` to llama-server when `kv.quant_mode = q4_0`.

### Symptoms
- Model loads, inference runs, but every output token is `?`
- Prompt eval timing looks normal (~50 ms/tok)
- Decode timing also normal (~90-100 ms/tok)
- Tokenizer decodes every generated ID as `?` (meaning IDs are out-of-vocabulary range)

### Isolation
Binary search narrowed the cause step by step:

| Test | Env vars | Server args | Result |
|------|----------|-------------|--------|
| A | minimal VITRIOL env | base args | ✅ Clean |
| B | full VITRIOL env | base args | ✅ Clean |
| C | full + `-fa on` | base + flash attn | ✅ Clean |
| D | full + `--cache-type-k q4_0` | base + K quant only | ✅ Clean |
| E | full + `--cache-type-v q4_0` | base + V quant only | ❌ `?????` |
| F | full + both KV quant | base + K+V quant | ❌ `?????` |
| G | full + `--cache-type-v q8_0` | base + V 8-bit quant | ❌ `?????` |

**Conclusion**: ANY V cache quantization (`q4_0` or `q8_0`) + flash attention + VITRIOL produces garbage. K cache quantization is clean.

### Root Cause Analysis
VITRIOL only intercepts MoE expert tensors (`ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps`), placing them in page-locked host RAM (`CUDA_Host` buffer type). The KV cache remains entirely in GPU VRAM (`CUDA0`). VITRIOL never touches the KV cache.

The bug is likely in **llama.cpp's flash attention V dequantization path for the `qwen35moe` architecture**:
- Model uses Gated Delta Net/SSM layers with `full_attention_interval=4` — only 10/40 layers have KV cache (unusual sparse layout)
- V dequantization path in flash attention kernel may have an architecture-specific bug
- Could not test without VITRIOL to confirm — model OOMs on 8 GB VRAM (needs ~11.4 GiB)

### Fix
`scripts/vitriol:1327` changed from:
```bash
[[ "$KV_QUANT" != "f16" ]] && KV_CACHE_ARGS="--cache-type-k $KV_QUANT --cache-type-v $KV_QUANT"
```
to:
```bash
[[ "$KV_QUANT" != "f16" ]] && KV_CACHE_ARGS="--cache-type-k $KV_QUANT"
```

Only K cache is quantized; V cache stays at f16 precision. Saves ~920 MiB VRAM from K quantization alone.

---

## Bug 3: Shim `max_tokens` Cap

### Location
`libvitriol/vitriol_shim.py:587`

### Cause
Shim capped `max_tokens` at 1024.

### Fix
Raised cap from 1024 to 32768. (Only relevant if memory mode is ON — not currently used.)

---

## Bug 4: Disk Full (Infrastructure)

### Details
Root partition `/dev/sda6` (490 GB) is 100% full (472 GB used). The `~/Desktop/OLD DATA/` directory occupies 344 GB.

### Impact
- Cannot write log files, temp files, or scripts to disk
- Some segfaults during testing may be disk-related
- Running servers still work (they keep model data in RAM/VRAM)

### Status
Identified; pending user decision on cleanup. No action taken.

---

## Optimal Config (Q2_K_XL, Verified Clean)

### `~/.vitriol/config`
```
[gpu]
device = 0
exclude_secondary = true

[model]
path = /home/randozart/Desktop/Projects/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
context = 131072
threads = 4
ngl = 99
expert_count = 0

[vitriol]
mode = stream
lru_mb = 0
verbose = false
output_cache = off
predictive_prefetch = on
pin_first_n_layers = 15
prune_experts = 0
reasoning = off

[server]
host = 0.0.0.0
port = 8279
parallel = 4

[memory]
mode = off
semantic_mode = off

[kv]
mode = standard
quant_mode = q4_0
frozen_prompt = on

[engine]
mode = vitriol-dma

[lookup]
tokens = 0

[spec]
type =
draft_n_max = 0
```

### Resulting Server Args
```
-m Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
-ngl 99 -c 131072 --host 0.0.0.0 --port 8279 --parallel 4
--no-mmap -t 4
--reasoning off
-fa on
--cache-type-k q4_0
```

No `--cache-type-v`. V cache stays f16.

### Performance
- Prompt eval: ~50 ms/tok (20 tok/s)
- Decode: ~100 ms/tok (10 tok/s)
- VRAM model: 1337 MiB (CUDA0) + 10040 MiB (CUDA_Host)
- VRAM KV cache: 1640 MiB (K q4_0: 360 MiB, V f16: 1280 MiB)
- VRAM RS cache: 251 MiB
- VRAM compute: 493 MiB

---

## Next Steps
1. Switch to IQ2_M model (with MTP support)
2. Enable `spec.type = mtp`, `spec.draft_n_max = 2`
3. Test with `--cache-type-k q4_0` fix active
4. Verify MTP works (draft acceptance rate, speedup)
