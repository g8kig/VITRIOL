# Jamba2-Mini Integration Plan: SSM-MoE for VITRIOL

**Date:** 2026-05-22
**Status:** Research / Pre-Implementation

---

## The Model

| Property | Value |
|----------|-------|
| Base model | [AI21-Jamba2-Mini](https://huggingface.co/ai21labs/AI21-Jamba2-Mini) |
| GGUF | [bartowski/ai21labs_AI21-Jamba2-Mini-GGUF](https://huggingface.co/bartowski/ai21labs_AI21-Jamba2-Mini-GGUF) |
| Parameters | 52B |
| Architecture tag | `jamba` (GGUF metadata) |
| SSM type | Mamba-2 (grouped heads) |
| MoE | Yes |
| Quantization targeted | IQ2_M (16.24 GB) / IQ2_S (14.41 GB) |
| Quantized with | llama.cpp b7652 |
| Our build | b8848 (newer, compatible) |

## Architecture Analysis

Jamba2-Mini is a hybrid SSM-MoE model combining:
- **Mamba-2 SSM layers** (recurrent, no KV cache, linear-time)
- **Attention layers** (standard multi-head, every N-th layer)
- **MoE FFN layers** vs standard FFN (per-layer configurable)

### Mamba-2 vs Mamba-1 (Critical Difference)

The current `jamba.cpp` only supports **Mamba-1** (`build_mamba_layer`). Jamba2-Mini uses Mamba-2:

| Feature | Mamba-1 (current jamba.cpp) | Mamba-2 (Jamba2-Mini) |
|---------|----------------------------|----------------------|
| Head structure | `n_head = d_inner`, `head_dim = 1` | Grouped heads (`n_head × head_dim = d_inner`) |
| Projections | Separate per x/B/C/dt | Combined `ssm_in` tensor |
| Group count (`ssm_n_group`) | Not used | Critical parameter |
| Post-scan norm | No | Yes (grouped norm after scan) |
| Activation | `silu` on x projection | `silu` on x projection |
| Selective scan | `ggml_ssm_scan` | Same, but grouped |

### Layer Arrangement (Data-Driven)

Jamba loads layer types from the GGUF file via per-layer `n_head_kv_arr`:
- `n_head_kv(il) == 0` → recurrent (SSM/Mamba) layer
- `n_head_kv(il) > 0` → attention layer

MoE vs dense FFN is determined by presence of `ffn_gate_inp` tensor per layer.

Canonical Jamba pattern: every 8th layer is attention, rest are Mamba + MoE.

### Key Dimensions (from `load_arch_hparams`)

| Parameter | GGUF Key | Notes |
|-----------|----------|-------|
| `ssm_d_conv` | `ssm.conv_kernel` | Convolution kernel size (typically 4) |
| `ssm_d_inner` | `ssm.inner_size` | Enforced: `== 2 * n_embd` |
| `ssm_d_state` | `ssm.state_size` | SSM state dimension |
| `ssm_dt_rank` | `ssm.time_step_rank` | Time-step projection rank |
| `n_expert` | `expert_count` | Total experts |
| `n_experts_per_tok` | `expert_used` | Active experts per token |

### Missing for Jamba2

These Mamba-2 specific hparams are **not loaded** by the current `jamba.cpp`:
- `ssm_n_group` — group count for Mamba-2 grouped heads
- `ssm_dt_b_c_rms` — whether dt/B/C use RMS norm

## The Problem

The GGUF file has architecture tag `jamba`. Our codebase will attempt to load it with `llama_model_jamba`, which expects Mamba-1 tensor names/layouts. The model will fail to load with missing tensor errors because:

1. Mamba-2 uses `ssm_in` weight (combined x/B/C/dt projection) rather than separate `ssm_x`, `ssm_b`, `ssm_c`, `ssm_dt` weights
2. Mamba-2 requires `ssm_n_group` which is not loaded by the Jamba loader
3. Mamba-2 has grouped norm tensors that don't exist in Mamba-1

## Implementation Plan

### Phase 1: Add Jamba2 Architecture (~3-5 days)

#### 1a. Register New Architecture

**`src/llama-arch.h`:**
```cpp
enum llm_arch {
    // ... existing ...
    LLM_ARCH_JAMBA2,
};
```

**`src/llama-arch.cpp`:**
- Add `"jamba2"` to architecture name mapping
- Add Jamba2 to hybrid architecture list (`llm_arch_is_hybrid`)
- Add `is_recurrent` implementation (same logic: `n_head_kv(il) == 0`)
- Set `supports_recurrent_partial_rollback = false` (same as Jamba1)

#### 1b. Create `src/models/jamba2.cpp`

Model file implementing:
- `load_arch_hparams` — load `ssm_n_group`, `ssm_dt_b_c_rms` in addition to standard Jamba params
- `load_arch_tensors` — Mamba-2 tensor names (`ssm_in`, grouped `ssm_norm` tensors, per-layer `ffn_gate_inp`)
- `graph` — layer loop:
  - If recurrent: `build_mamba2_layer(...)` (reuse from `mamba-base.cpp`)
  - If attention: standard attention (no RoPE, same as Jamba1)
  - If MoE FFN: `build_moe_ffn(...)` (reuse from existing code)
  - If dense FFN: `build_ffn(...)` (reuse)

#### 1c. Wire Up

- Register in `llama_model::load_model` factory
- Add to `llama_model_loader` architecture check
- Register tensors in the tensor name map

### Phase 2: VITRIOL Predictor Adaptation (~2-3 days)

The VITRIOL expert offloading predictor must become layer-type-aware:

#### 2a. Layer Classification

```
For each layer il:
  if recurrent (n_head_kv == 0):
    → SSM layer → ALWAYS pin to GPU (cannot tolerate PCIe latency)
    → No expert offloading (no MoE here)
  if attention (n_head_kv > 0):
    → Attention layer → ALWAYS pin to GPU
    → No expert offloading (dense weights)
  if ffn_gate_inp exists:
    → MoE FFN layer → VITRIOL DMA candidate
    → Expert-level offloading as currently implemented
  else:
    → Dense FFN layer → PIN to GPU
```

#### 2b. MoE Expert Detection

The current VITRIOL MoE detection (looking for `ffn_gate_*` weight patterns) already handles per-layer MoE. Jamba2's per-layer `ffn_gate_inp` is compatible. The predictor just needs to skip non-MoE layers.

#### 2c. SSM State Considerations

- Mamba-2 SSM layers use `llama_memory_recurrent` (same as Qwen3.6 hybrids)
- Partial sequentional removal (`seq_rm` tail) is NOT supported (`supports_recurrent_partial_rollback = false`)
- **Approach E checkpoint fix is critical here** — without partial rollback, the exact-boundary checkpoint is the only way to avoid full re-prefill
- SSM state is NOT offloadable (must stay in GPU VRAM)

### Phase 3: VRAM Budget Planning

Jamba2-Mini at IQ2_S/IQ2_M is ~14-16 GB, far exceeding the GTX 1070 Ti's 8 GB VRAM. VITRIOL's DMA offloading is essential:

| Component | Estimated VRAM | Notes |
|-----------|---------------|-------|
| Model weights (IQ2_S) | ~14.4 GB | DMA offloaded, ~1.5 GB resident |
| SSM recurrent state | ~100-200 MB | Must be pinned in VRAM |
| KV cache (attention layers) | ~500 MB* | Only attention layers, not SSM layers |
| VITRIOL pin pool | ~1.6 GB | Pinned experts + attention weights |
| Compute buffers | ~500 MB | Activation memory |
| **Total VRAM needed** | **~4.3 GB** | With VITRIOL DMA |
| **Available** | **8 GB** | GTX 1070 Ti |

*KV cache is smaller than Qwen3.6 because only attention layers (not SSM layers) need it.

The key advantage: Jamba2-Mini is 52B but only ~1/8 of layers are attention (need KV cache), ~1/8 are dense FFN (pinned), and ~6/8 are Mamba-2 + MoE (DMA-friendly).

## Blockers

- [ ] **Model download** — in progress
- [ ] **Test load** — need to verify GGUF architecture tag and exact tensor names
- [ ] **Disk space** — 344 GB of `~/Desktop/OLD DATA/` needs to be freed for the 14+ GB model and build artifacts
- [ ] **Mamba-2 scan kernel** — verify `ggml_ssm_scan` supports grouped heads (Mamba-2 mode)

## Code References

| Component | File | Notes |
|-----------|------|-------|
| Jamba1 model | `src/models/jamba.cpp` | Template for Jamba2 |
| Mamba-1 SSM | `src/models/mamba-base.cpp:build_mamba_layer` | Current implementation |
| Mamba-2 SSM | `src/models/mamba-base.cpp:build_mamba2_layer` | Reusable for Jamba2 |
| Recurrent memory | `src/llama-memory-recurrent.cpp` | SSM state management |
| Hybrid memory | `src/llama-memory-hybrid.cpp` | Attention + SSM combined |
| Architecture registry | `src/llama-arch.cpp` | Add JAMBA2 here |
| Architecture header | `src/llama-arch.h` | Add enum here |
| MoE FFN | `src/models/models.cpp:build_moe_ffn` | Reusable |
| VITRIOL predictor | `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp` | Needs layer-type awareness |
| Expert pin pool | `scripts/vitriol` | Configuration |
