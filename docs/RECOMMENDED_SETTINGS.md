# VITRIOL Recommended Settings — GTX 1070 Ti (8GB)

System: 15 GB RAM, NVMe SSD, 4 CPU threads, PCIe Gen3 x16.

---

## Quick Config

```ini
[gpu]
device = 0
exclude_secondary = true

[model]
path = /mnt/data/ai/koboldcpp/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
context = 256000
threads = 4
ngl = 99

[vitriol]
mode = stream
lru_mb = 0
verbose = true

[server]
host = 0.0.0.0
port = 8279
parallel = 4

[memory]
mode = off
semantic_mode = off

[kv]
mode = offload
quant_mode = q4_0
frozen_prompt = on

[engine]
mode = vitriol-dma

[lookup]
tokens = 0
```

## CLI Flags (always required)

```
vitriol serve --detach \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -fa on --no-mmap
```

These are automatically wired by the vitriol script from config settings. Pass them explicitly only when bypassing the script.

## MTP (Speculative Decoding)

Only use if your GGUF model has `nextn_predict_layers` metadata (e.g. Unsloth MTP models).

```ini
[spec]
type = mtp
draft_n_max = 2
```

**Why N=2**: Sweep confirmed acceptance rate = exactly `1/N`. N=2 gives 50% acceptance (1 token/cycle). Higher N wastes PCIe bandwidth on rejected draft tokens.

**Detection**: `head -c 1M model.gguf | grep -q nextn_predict_layers`

## Per-Setting Rationale

| Setting | Value | Why not default? |
|---------|-------|-----------------|
| `context = 256000` | Matches Qwen3.6 reported effective context. 500K exceeds RAM. |
| `threads = 4` | GTX 1070 Ti has 4 scheduler units. `t=8` causes contention (+25% slower). |
| `ngl = 99` | Offload all layers. 1337 MiB fits easily in 8 GB VRAM. |
| `mode = stream` | Only mode that page-locks RAM for VITRIOL DMA. Sync/async/off won't work. |
| `lru_mb = 0` | LRU cache is unreachable on quantized MoE models (FP16 only). |
| `kv.mode = offload` | Moves KV cache from VRAM to host, leaving room for compute buffers. |
| `kv.quant_mode = q4_0` | Reduces host KV allocation from 5000 MiB (F16) to 1406 MiB. Without this, total exceeds 15 GB RAM → silent OOM. |
| `frozen_prompt = on` | Keeps system prompt static to avoid re-prefix on each request. |
| `engine.mode = vitriol-dma` | Enables the CUDA expert intercept layer. Native mode = no VITRIOL. |
| `exclude_secondary = true` | Prevents CUDA from seeing the GTX 960 (CC 5.2, no kernel images). |

## Performance

| Config | Gen (tok/s) | vs x8 baseline |
|--------|------------|----------------|
| PCIe x8 (GTX 960 present) | 5.7 | — |
| PCIe x16 (GTX 960 removed) | 9.1 | +60% |
| + MTP N=2 (IQ2_M model) | 10.96 | +92% |
