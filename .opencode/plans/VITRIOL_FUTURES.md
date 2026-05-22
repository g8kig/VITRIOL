# VITRIOL Integration Futures: Vulkan, SSM-MoE, and Architectural Horizons

**Date:** 2026-05-21
**Status:** Research / Exploration

---

## Table of Contents

1. [The Vulkan Chimera](#1-the-vulkan-chimera)
2. [SSM-MoE Hybrids](#2-ssm-moe-hybrids)
3. [Modal Compressed Cross-Conditioning](#3-modal-compressed-cross-conditioning)
4. [Implementation Priority & Assessment](#4-implementation-priority--assessment)

---

## 1. The Vulkan Chimera

### 1.1 The Insight

Vulkan's compute path has a fundamentally different performance profile than CUDA:
- **Generation (decode):** Vulkan wins due to pre-baked command buffers — zero CPU driver overhead per token
- **Prefill (prompt processing):** CUDA wins due to direct memory access — `cudaHostRegister` bypasses staging buffers entirely

VITRIOL's current approach (CUDA-only) has the generation weakness. A Vulkan backend with VITRIOL's memory management would have both strengths.

### 1.2 Current Vulkan Backend State

The llama.cpp Vulkan backend (`ggml-vulkan.cpp`, **16,939 lines**, ~70+ GLSL shaders) already has:

| Feature | Location | Status |
|---------|----------|--------|
| `VK_EXT_external_memory_host` detection | `ggml-vulkan.cpp:4964` | ✅ Complete |
| Extension enablement | `ggml-vulkan.cpp:5261-5262` | ✅ Complete |
| Alignment query (`minImportedHostPointerAlignment`) | `ggml-vulkan.cpp:5125` | ✅ Complete |
| Host pointer import in buffer creation | `ggml-vulkan.cpp:2662-2724` | ✅ Complete |
| `ggml_backend_vk_device_buffer_from_host_ptr()` API | `ggml-vulkan.cpp:15982-15999` | ✅ Complete |
| **MUL_MAT_ID dedicated pipelines** | `ggml-vulkan.cpp:689-695` | ✅ Complete |
| MUL_MAT_ID dispatch (matmul + matvec paths) | `ggml-vulkan.cpp:8835-8846` | ✅ Complete |
| Operation fusion (MUL_MAT_ID + ADD_ID + MUL) | `ggml-vulkan.cpp:14124-14171` | ✅ Complete |
| Command buffer pool with recycling | `ggml-vulkan.cpp:6551-6561` | ✅ Complete |
| Staging buffer for host↔device copies | `ggml-vulkan.cpp:6722-6736` | ✅ Complete |

### 1.3 The Gap

The `VK_EXT_external_memory_host` extension is **not used by default** for model weights. The primary data path allocates device-local memory and uses explicit staging copies via the `sync_staging` buffer. VITRIOL's key innovation — mapping page-locked host RAM directly into GPU address space and driving DMA from a dedicated stream — has no Vulkan equivalent in the current code.

### 1.4 What a Vulkan VITRIOL Requires

```
┌──────────────────────────────────────────────────────┐
│                   VITRIOL Vulkan Layer                │
├──────────────────────────────────────────────────────┤
│ Memory Layer (external memory host)                  │
│   VkBuffer ← ggml_vk_buffer_from_host_ptr(host_ptr)  │
│   VkDeviceMemory ← VK_EXT_external_memory_host       │
│   Zero-copy: GPU reads host RAM directly              │
├──────────────────────────────────────────────────────┤
│ Command Layer (per-token command buffer rebuild)      │
│   vkBeginCommandBuffer → bind MoE experts → dispatch  │
│   vkEndCommandBuffer → submit → vkQueueWaitIdle       │
│   Problem: defeats pre-baked pipeline advantage       │
├──────────────────────────────────────────────────────┤
│ Stream / Sync Layer                                   │
│   Dedicated transfer queue ←→ compute queue           │
│   VkSemaphore timeline sync for LRU cache fills       │
└──────────────────────────────────────────────────────┘
```

### 1.5 The MoE Obstacle

Vulkan excels at **fixed pipeline** workloads. MoE is the worst case:

| Aspect | Dense Model (Vulkan ideal) | MoE Model (Vulkan worst case) |
|--------|---------------------------|-------------------------------|
| Pipeline | Record once, execute N times | Re-record every token |
| Expert selection | N/A | Changes each decode step |
| Memory binding | Fixed set of descriptor sets | Dynamic per-expert binding |
| Driver overhead | ~0 (pre-baked) | Full cost every token |

VITRIOL's predictor (`VITRIOL_PREDICTIVE_PREFETCH`) would need to tell Vulkan *which* experts to bind before each decode — essentially building a new command buffer for every token. This defeats the "pre-baked pipeline" advantage that makes Vulkan fast for generation.

One mitigation: **pre-record command buffers for all possible expert combinations** (impossible for 256 experts × 8 active = C(256,8) ≈ 2.5 × 10¹³ combinations). Practical mitigation: **pre-record per-expert command fragments** and chain them dynamically — a substantial infrastructure change.

### 1.6 Effort Estimate

| Phase | Scope | Lines | Time |
|-------|-------|-------|------|
| 1. Memory layer | Wire `external_memory_host` into VITRIOL buffer type | ~200 | 1-2 weeks |
| 2. Expert LRU cache | Vulkan equivalent of `cuMemcpyHtoDAsync` with timeline semaphores | ~500 | 2-4 weeks |
| 3. Dynamic command buffer | Per-token MoE dispatch with dynamic expert binding | ~800 | 4-8 weeks |
| 4. Predictor integration | Port cross-layer + temporal predictor to Vulkan codepath | ~300 | 1-2 weeks |
| 5. Production hardening | Edge cases, multi-GPU, testing | ~1000 | 4-8 weeks |
| **Total** | | **~2800** | **3-6 months** |

### 1.7 Verdict

Deferred. The effort is substantial, and the primary benefit (AMD/Intel support) is not urgent while VITRIOL targets NVIDIA GPUs. **Revisit if:**
- VITRIOL achieves product-market fit and needs cross-GPU vendor support
- ROCm/HIP remains significantly slower than Vulkan on AMD hardware
- A contributor with Vulkan expertise joins the project

---

## 2. SSM-MoE Hybrids

### 2.1 The Insight

The KV cache grows O(N) with context length. SSM (State Space Model) state is fixed-size regardless of context. Hybrid architectures like Jamba alternate SSM layers (fixed memory) with Attention+MoE layers (model capacity), combining the best of both:

| Architecture | Context Memory | Model Capacity | Expert DMA Needed |
|-------------|---------------|----------------|-------------------|
| Pure Transformer (Qwen3) | O(N) — ~1.6 GB at 32K | High (MoE) | Yes (all layers) |
| Pure SSM (Mamba) | O(1) — ~84 MB | Low (dense) | No |
| **Hybrid (Jamba)** | **O(1) — ~84 MB** | **High (MoE)** | **Yes (attention layers only)** |

### 2.2 The Math

SSMs use eigendecomposed state transition matrices ($A = V \Lambda V^{-1}$) where $\Lambda$ controls per-dimension decay rates:
- **Fast-decay modes** (~1 token half-life): syntactic precision (variable names, braces, indentation)
- **Slow-decay modes** (~100K token half-life): architectural rules (system prompt, coding conventions, project structure)

This is **automatic context compression** — not through explicit summarization but through the mathematical structure of the state update. The fixed-size state (~84 MB for 40 layers) acts as a learned summary of the entire token history.

### 2.3 Supported Architectures in llama.cpp

| Architecture | File | SSM Type | Attention | MoE | GGUF Available? |
|-------------|------|----------|-----------|-----|-----------------|
| Mamba-1 | `src/models/mamba.cpp` | Mamba-1 | No | No | ✅ Common |
| Mamba-2 | `src/models/mamba2.cpp` | Mamba-2 | No | No | ✅ Common |
| **Jamba** | **`src/models/jamba.cpp`** | **Mamba-1** | **Yes** | **Yes (attn layers)** | 🟡 Rare |
| Granite Hybrid | `src/models/granite-hybrid.cpp` | Mamba-2 | Yes | No | 🟡 Rare |
| Nemotron-H | `src/models/nemotron-h.cpp` | Mamba-2 | Yes | No | 🟡 Rare |
| Falcon-H1 | `src/models/falcon-h1.cpp` | Mamba-2 (d_state=256) | Yes | No | 🟡 Rare |
| Plamo-2 | `src/models/plamo2.cpp` | Mamba-1 | Yes | No | 🟡 Rare |

**Jamba is the only architecture combining SSM + MoE.** This makes it the target for VITRIOL integration. Jamba-1.5-Mini (12B params, 900M active) exists but GGUF quantizations are rare due to the hybrid architecture requiring special handling in the quantizer.

### 2.4 VITRIOL Integration for Jamba

The graph construction in `jamba.cpp` (198 lines) alternates:
```
Layer 0:  Mamba-1 (SSM scan, 0 expert fetch, 0 PCIe traffic)
Layer 1:  Mamba-1 (SSM scan, 0 expert fetch, 0 PCIe traffic)
Layer 2:  Attention + MoE → VITRIOL DMA: fetch 8 of 256 experts across PCIe
Layer 3:  Mamba-1 (SSM scan, 0 expert fetch, 0 PCIe traffic)
...
```

The VITRIOL predictor must become **layer-type-aware**:

```cpp
// In vitriol_predictor_prefetch():
if (is_ssm_layer(current_layer)) {
    // No-op: SSM state is already VRAM-resident
    // PCIe bus has N tokens of idle time to prefetch next MoE batch
    return;
}
if (is_moe_layer(current_layer)) {
    // Current VITRIOL prefetch logic
    cross_layer_predict(g_cur_exp, g_prev_exp, current_layer);
    temporal_predict(g_cur_exp, g_prev_exp, current_layer);
    vitriol_lru_prefetch(predicted_experts);
}
```

The advantage: between MoE layers, the GPU executes 1-2 Mamba layers (SSM scan kernels), giving the PCIe bus **more time** to prefetch the next batch of experts. The predictor can look ahead 2-3 layers instead of just 1.

### 2.5 VRAM Impact

With Jamba (32 layers, ~12B params):
- SSM state (fixed): ~84 MB (always VRAM-resident)
- Expert cache: remaining ~7.9 GB VRAM → ~120 expert slots (vs ~16 today with Qwen3 KV cache)
- Pin pool: 8 layers × 3 expert tensors × ~66 MB = ~1.6 GB

**Result:** Far fewer LRU cache misses. Most experts are VRAM-resident. PCIe traffic drops from 8 experts/token to ~0-2.

### 2.6 SSM-Aware Context Compression for VITRIOL

For the current Qwen3 architecture, the SSM layer is already present — Qwen3 uses a hybrid SSM-Attention architecture (full attention at intervals, SSM-based Gated Delta Net in between). This was previously the source of the "erased invalidated context checkpoint" issues because the recurrent state doesn't support partial `seq_rm`.

The insight: **the SSM state IS the compressed context**. The checkpoint mechanism exists precisely because the SSM state is the sole carrier of context between generations. If we improve checkpoint granularity (Approach E), the SSM state is perfectly usable as a "compressed context" mechanism — the 128-dimensional SSM vectors per layer encode the entire conversation history in a fixed footprint.

### 2.7 Effort Estimate

| Phase | Scope | Lines | Time |
|-------|-------|-------|------|
| 1. Jamba model acquisition | Find/download quantized Jamba GGUF | — | 1-2 days |
| 2. VITRIOL layer-type routing | Detect Mamba vs MoE layers, gate DMA | ~100 | 1 day |
| 3. Predictor lookahead | Predict 2-3 layers ahead for SSM gaps | ~80 | 1 day |
| 4. Testing & profiling | Benchmark vs Qwen3 baseline | — | 2-3 days |
| **Total** | | **~180** | **1-2 weeks** |

**Blocking dependency:** Availability of a Jamba GGUF at a useful quantization (Q4_K_M or better).

### 2.8 Verdict

High potential, blocked on model availability. The ~180 line change to make VITRIOL layer-type-aware is small. **Action item:** Search for Jamba GGUF models and test-load into llama.cpp to verify compatibility.

---

## 3. Modal Compressed Cross-Conditioning

### 3.1 Clarification

"Modal" in this context refers to **eigenmodes** (the $\Lambda$ diagonal in eigendecomposed SSM transition matrices), not **multi-modal** (vision/audio/text). The term describes how SSMs decompose their state space into independent decaying-frequency channels, analogous to an audio equalizer's per-band dynamics.

### 3.2 The Mathematical Foundation

The SSM state update: $x_t = A x_{t-1} + B u_t$

Eigendecomposition: $A = V \Lambda V^{-1}$ where $\Lambda$ is a diagonal matrix of eigenvalues.

Each eigenvalue $\lambda_i$ defines the decay rate of mode $i$:
- $|\lambda_i| \approx 0$: Fast decay — remembers ~1 token (syntax, character-level)
- $|\lambda_i| \approx 1$: Slow decay — remembers ~100K tokens (project rules, persona)
- $0 < |\lambda_i| < 1$: The full spectrum in between

The "cross-conditioning" happens when the SSM state ($x_t$) is used as the Key/Value in a cross-attention mechanism, conditioning the output on the compressed state rather than the full token history. This is the architectural basis for hybrid models like Jamba.

### 3.3 Relevance to VITRIOL

The Qwen3 model already has SSM layers (Gated Delta Net). The "cross-conditioning" manifests as the recurrent state checkpoints we've been working with. Understanding this math makes it clear why:

1. **Checkpoints are necessary**: The SSM state at position N is a function of *all* preceding tokens. There's no per-tension KV cell to delete — `seq_rm` is inherently limited.
2. **The state IS the compression**: The 128-dimensional SSM vectors per layer encode the conversation history in ~84 MB regardless of context length. This is the native "modal compression."
3. **VITRIOL's role**: The DMA pipeline ensures the expert weights (which DO change per token) keep up with the fixed-size SSM state.

### 3.4 No Action Required

The modal compressed cross-conditioning is **already how Qwen3 works internally**. VITRIOL doesn't need to implement anything new — it needs to:
1. Keep the SSM state in VRAM (it's small enough)
2. Keep checkpoints at good boundaries (Approach E)
3. Keep experts flowing across PCIe (current VITRIOL DMA)

---

## 4. Implementation Priority & Assessment

| Path | Effort | Lines | VRAM Win | Speed Win | Dependency | Priority |
|------|--------|-------|----------|-----------|------------|----------|
| **Approach E (exact ckpt)** | **~30 lines** | **~30** | **None** | **-117s/turn** | **None** | **🔴 NOW** |
| Jamba SSM-MoE integration | 1-2 weeks | ~180 | ~1.6 GB freed | +2-5 tok/s | Jamba GGUF availability | 🟡 Next |
| SSM predictor lookahead | 1 day | ~80 | None | +0.5-1 tok/s | Jamba path first | 🟡 Next |
| Vulkan VITRIOL backend | 3-6 months | ~2800 | None (port) | Same perf, AMD | Vulkan expertise | ⚪ Deferred |
| Modal compression research | — | 0 | Already in Qwen3 | Already active | — | ⚪ Informational |

### Quick Reference: Key File Locations

| Component | Path | Lines |
|-----------|------|-------|
| Checkpoint creation | `llama.cpp/tools/server/server-context.cpp` | 1863-1886 |
| SSM seq_rm limitation | `llama.cpp/src/llama-memory-recurrent.cpp` | 161-162 |
| SSM scan CUDA kernel | `llama.cpp/ggml/src/ggml-cuda/ssm-scan.cu` | 342 |
| SSM conv CUDA kernel | `llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu` | 175 |
| Jamba model definition | `llama.cpp/src/models/jamba.cpp` | 198 |
| Mamba base layer builder | `llama.cpp/src/models/mamba-base.cpp` | 289 |
| Recurrent memory backend | `llama.cpp/src/llama-memory-recurrent.cpp` | 1199 |
| Vulkan backend | `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | 16939 |
| VK_EXT_external_memory_host detection | `ggml-vulkan.cpp` | 4964 |
| VK_EXT_external_memory_host import | `ggml-vulkan.cpp` | 2662-2724 |
| Vulkan MUL_MAT_ID dispatch | `ggml-vulkan.cpp` | 8835-8846 |
| Vulkan MUL_MAT_ID pipelines | `ggml-vulkan.cpp` | 689-695 |
