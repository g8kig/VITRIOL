# Jamba2-Mini Integration: Findings & Plan

**Date:** 2026-05-22
**Status:** Updated with GGUF tensor analysis

---

## 1. What We Know

### 1.1 Model Properties (from GGUF header)

| Property | Value |
|----------|-------|
| Architecture tag | `jamba` |
| Parameters | 52B (12B active, 16 experts, 2 experts/tok) |
| Quantization | IQ2_M |
| File size | 16.2 GB |
| Layers | 32 |
| Embedding dim (`n_embd`) | 4096 |
| FFN dim (`n_ff`) | 14336 |
| SSM conv kernel (`d_conv`) | 4 |
| SSM inner size (`d_inner`) | 8192 (enforced: `2 * n_embd`) |
| SSM state size (`d_state`) | 16 |
| SSM dt rank (`dt_rank`) | 256 |
| Attention heads | 32 |
| KV heads per layer array | `[0,0,0,0,8, 0,0,0,0,0,0,0,8, ...]` |
| Layer pattern | 4× SSM + MoE, 1× Attention + MoE, repeat |
| Attention layers | Layers 4, 12, 20, 28 |
| SSM layers | All other 28 layers |
| KV cache needed | Only 4 layers × attention heads |

Layer arrangement (from `n_head_kv` array):
```
blk.0  blk.1  blk.2  blk.3  | blk.4   | blk.5  blk.6  blk.7  | blk.8   | ...
 SSM    SSM    SSM    SSM    | ATTN    | SSM    SSM    SSM    | ATTN    | ...
 MoE    MoE    MoE    MoE    | MoE     | MoE    MoE    MoE    | MoE     | ...
```

### 1.2 Tensor Analysis: `jamba.cpp` Loader Matches GGUF File

**Crucial finding:** The current `jamba.cpp` at `src/models/jamba.cpp:47-101` already loads all the tensors present in the Jamba2-Mini GGUF, including Mamba-2 specific ones:

| Tensor | Expected Shape (jamba.cpp) | Actual Shape (GGUF) | Match? |
|--------|---------------------------|---------------------|--------|
| `ssm_in.weight` | `{n_embd, 2*d_inner}` = [4096, 16384] | [4096, 16384] | ✅ |
| `ssm_conv1d.weight` | `{d_conv, d_inner}` = [4, 8192] | [4, 8192] | ✅ |
| `ssm_conv1d.bias` | `{d_inner}` = [8192] | [8192] | ✅ |
| `ssm_x.weight` | `{d_inner, dt_rank+2*d_state}` = [8192, 288] | [8192, 288] | ✅ |
| `ssm_dt_norm.weight` | `{dt_rank}` = [256] | [256] | ✅ |
| `ssm_dt.weight` | `{dt_rank, d_inner}` = [256, 8192] | [256, 8192] | ✅ |
| `ssm_dt.bias` | `{d_inner}` = [8192] | [8192] | ✅ |
| `ssm_b_norm.weight` | `{d_state}` = [16] | [16] | ✅ |
| `ssm_c_norm.weight` | `{d_state}` = [16] | [16] | ✅ |
| `ssm_a` | `{d_state, d_inner}` = [16, 8192] | [16, 8192] | ✅ |
| `ssm_d` | `{d_inner}` = [8192] | [8192] | ✅ |
| `ssm_out.weight` | `{d_inner, n_embd}` = [8192, 4096] | [8192, 4096] | ✅ |

**All 531 tensors match.** The model WILL load into `llama_model_jamba`.

### 1.3 The Graph Problem

**`jamba.cpp:128` uses `build_mamba_layer` (Mamba-1)** which:
- Uses `ssm_x`, `ssm_dt`, `ssm_a`, `ssm_d`, `ssm_conv1d`, `ssm_out` as separate projections
- **Ignores `ssm_in`** — the combined Mamba-2 projection tensor is loaded but never used in the graph
- Does NOT use `ssm_b_norm`, `ssm_c_norm`, `ssm_dt_norm`

The Mamba-1 graph projects from `d_inner` (8192) through individual `ssm_x` [8192, 288], `ssm_dt` [256, 8192], `ssm_a` [16, 8192], etc.

The Mamba-2 graph would project from `n_embd` (4096) through combined `ssm_in` [4096, 16384], then use grouped heads with `ssm_b_norm`, `ssm_c_norm`, `ssm_dt_norm`.

**Bottom line:** The model loads. Whether it produces correct output depends on whether bartowski's conversion script produced Mamba-1-compatible weight values in `ssm_x`, `ssm_dt`, etc. Some quantizers do this (normalizing Mamba-2 weights into Mamba-1 format), others don't. The `ssm_in` weights would be completely ignored.

### 1.4 What's Actually Missing for Proper Mamba-2 Support

The `jamba.cpp` loader has Mamba-2 tensors but the codebase is missing:

1. **`ssm_n_group` hparam** — not loaded (Mamba-2 grouped head count). Not in the GGUF either? The file doesn't have this key.
2. **`ssm_dt_b_c_rms` flag** — not loaded. But file has individual `ssm_b_norm`, `ssm_c_norm`, `ssm_dt_norm` so norms are explicit.
3. **Mamba-2 graph path** — `build_mamba2_layer` exists (used by Falcon-H1, Granite) but `jamba.cpp` doesn't use it.

If the Mamba-1 path produces garbled output, the fix is switching `build_mamba_layer` → `build_mamba2_layer` and wiring `ssm_in` into the graph (it's already loaded as `layer.ssm_in`).

---

## 2. Test Results (Pending)

### 2.1 Model Load Test

```
./build/bin/llama-cli --model <path> --no-mmap -ngl 0 -c 512 -p "Hello" -n 10
```

Expected outcomes:
- **Loads clean + coherent output** → jamba.cpp Mamba-1 compat path works. Skip to Phase 2.
- **Loads clean + garbled output** → Need to switch to `build_mamba2_layer`. ~50 lines in `jamba.cpp`.
- **Fails to load** → Tensor mismatch (unexpected given analysis above).

### 2.2 VRAM Test

```
./build/bin/llama-server --model <path> -ngl 99 -c 4096 --no-mmap --parallel 1
```

With 16.2 GB model on 8 GB VRAM, this will OOM unless VITRIOL DMA is active. Need to either:
- Test with `-ngl 20` (partial offload)
- Or use VITRIOL mode directly
- Or test CPU-only with `-ngl 0`

---

## 3. Implementation Plan (Revised)

### Phase A: Verify Current Behavior

1. ✅ GGUF tensor analysis complete — shapes match
2. 🔲 Load test — run `llama-cli` to confirm loading and output quality
3. 🔲 If garbled: switch `build_mamba_layer` → `build_mamba2_layer` in `jamba.cpp:128`

### Phase B: VITRIOL Layer-Type Predictor

The VITRIOL predictor must become layer-type-aware to handle Jamba2:

```
per-layer classifier:
  n_head_kv == 0  → SSM layer       → ALWAYS pin to GPU
  n_head_kv > 0   → Attention layer  → ALWAYS pin to GPU
  ffn_gate_inp    → MoE FFN layer   → VITRIOL DMA candidate
  else            → Dense FFN layer  → PIN to GPU
```

Implementation:
- `vitriol-cuda-integration.cpp`: add a per-layer type table queried at init time
- `scripts/vitriol`: configure pin pool size knowing only MoE layers need DMA (not SSM layers)
- The `llama_model_loader` already exposes per-layer `n_head_kv` so we can determine which layers are SSM vs attention vs MoE

### Phase C: VRAM Budget

| Component | Size | Location |
|-----------|------|----------|
| Model weights (IQ2_M, 16.2 GB) | ~1.6 GB resident | DMA host + GPU pin pool |
| SSM recurrent state (28 layers) | ~200 MB | Pinned GPU |
| KV cache (4 attention layers, 4K ctx) | ~32 MB | Pinned GPU |
| VITRIOL pin pool | ~1.5 GB | GPU VRAM |
| Compute buffers | ~500 MB | GPU VRAM |
| **Total** | **~3.8 GB** | Well within 8 GB |

---

## 4. Key Code References

| Component | File | Notes |
|-----------|------|-------|
| Jamba model file | `src/models/jamba.cpp` | 198 lines total, loader + graph |
| Mamba-1 layer | `src/models/mamba-base.cpp:build_mamba_layer` | Current graph path |
| Mamba-2 layer | `src/models/mamba-base.cpp:build_mamba2_layer` | Used by Falcon-H1, Granite |
| Tensor loader | `src/models/jamba.cpp:56-78` | Loads both Mamba-1 and Mamba-2 tensors |
| Graph builder | `src/models/jamba.cpp:128` | `build_mamba_layer` — line to change |
| Hybrid memory | `src/llama-memory-hybrid.cpp` | SSM + KV combined |
| Recurrent memory | `src/llama-memory-recurrent.cpp` | SSM state with seq_rm limitation |
| VITRIOL predictor | `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp` | Layer-type awareness needed |
| GGUF metadata | `ai21labs_AI21-Jamba2-Mini-IQ2_M.gguf` | 32 layers, 531 tensors |
