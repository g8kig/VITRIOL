# VITRIOL Chimera: CUDA + Vulkan Hybrid Backend — Milestone Plan
**Date:** 2026-05-22 10:30 (updated 2026-05-22 11:00)
**Author:** VITRIOL Project
**Status:** Milestone 1 Complete, Milestone 2 In Progress

---

## Table of Contents
1. [The Chimera Architecture](#1-the-chimera-architecture)
2. [Option B: Full Dual-Backend Chimera](#2-option-b-full-dual-backend-chimera)
3. [Milestone 1: Mamba-1 Vulkan SSM Shader ✅](#3-milestone-1-mamba-1-vulkan-ssm-shader-)
4. [Milestone 2: VITRIOL Memory → Vulkan](#4-milestone-2-vitriol-memory--vulkan)
5. [Milestone 3: Dynamic MoE Command Buffers](#5-milestone-3-dynamic-moe-command-buffers)
6. [Milestone 4: Backend Routing](#6-milestone-4-backend-routing)
7. [Testing Strategy](#7-testing-strategy)
8. [File Inventory](#8-file-inventory)

---

## 1. The Chimera Architecture

### 1.1 Problem Statement

The GTX 1070 Ti has 8 GB VRAM. Qwen3.6-35B at IQ2_M is ~3.5 GB. Jamba2-52B at IQ2_M is ~16.2 GB. Neither fits entirely in VRAM without aggressive quantization.

VITRIOL's current solution — page-locked host RAM + CUDA DMA — works well for MoE experts (sparse access) but is limited by:
- Static pin pool covers only ~17% of Jamba2's expert tensors
- LRU cache + predictive prefetch have bugs (output corruption)
- CUDA-only, no AMD/Intel support

### 1.2 Option A vs Option B

| Aspect | Option A: Vulkan Only | Option B: Full Chimera |
|---|---|---|
| MoE experts | Vulkan MUL_MAT_ID (unproven speed) | CUDA VITRIOL (proven 23.5 tok/s) |
| Dense ops (SSM, attn) | Vulkan (pre-baked cmd buffers) | Vulkan (pre-baked cmd buffers) |
| Cross-backend overhead | None (single backend) | ~0.13% (CPU staging copies) |
| Cross-vendor | AMD + Intel + NVIDIA | MoE: NVIDIA only; Dense: any |
| Complexity | Lower | Medium |
| Max speed | Unknown | Likely highest |

### 1.3 Decision: Option B (Full Chimera)

Selected for maximum speed. The cross-backend overhead (CPU staging copies for
activations between CUDA and Vulkan VRAM) is ~0.001 ms per copy at 16 KB each,
or ~0.056 ms per token total — essentially free.
- **CUDA VITRIOL** for MoE expert matmuls (where VITRIOL's DMA + predictor excel)
- **Vulkan** for dense ops (SSM scan, attention, norms) where pre-baked command buffers reduce CPU overhead

### 1.2 Why Both Backends?

| Property | CUDA VITRIOL | Vulkan | Chimera |
|---|---|---|---|
| Prefill speed | **321 TPS** (zero-copy DMA) | 177 TPS (staging buffers) | **321 TPS** |
| Generation speed | 13 TPS (CPU driver overhead) | **19 TPS** (pre-baked cmd buffers) | **19+ TPS** |
| MoE expert handling | ✅ VITRIOL pin pool + predictor | ❌ Dynamic binding defeats pre-baking | ✅ CUDA for MoE |
| SSM scan (Mamba-1) | ✅ CUDA kernel exists | ❌ Was rejected (Mamba-2 only) | ✅ Milestone 1 |
| Cross-vendor | ❌ NVIDIA only | ✅ NVIDIA + AMD + Intel | ✅ Vulkan for dense ops |

### 1.3 Data Flow

```
Token N
  │
  ├─→ Vulkan SSM scan (pre-baked cmd buf) → activation in Vulkan VRAM
  ├─→ Vulkan attention (pre-baked) → activation in Vulkan VRAM
  ├─→ CUDA expert matmul (VITRIOL DMA) → activation in CUDA VRAM
  │     │
  │     └─ Copy (~16 KB) via page-locked host RAM to Vulkan VRAM
  │
  └─→ Repeat for all 28 layers
       │
       Copy cost: ~0.03 ms per handoff × 32 layers ≈ 1 ms/token
       Compute time: ~42 ms/token at 23.5 tok/s
       Overhead: ~2.4% — negligible
```

### 1.4 Weight Placement

| Weight Type | Storage | Backend | Bandwidth |
|---|---|---|---|
| Expert tensors (MoE) | Page-locked host RAM | CUDA DMA | PCIe 3.0 x16 (~16 GB/s) |
| SSM weights (dense) | Vulkan device VRAM | Vulkan | VRAM (~200 GB/s) |
| Attention weights (dense) | Vulkan device VRAM | Vulkan | VRAM (~200 GB/s) |
| Norms, embeddings (dense) | Vulkan device VRAM | Vulkan | VRAM (~200 GB/s) |

### 1.5 Key Insight from Code Analysis

The backend scheduler (`ggml_backend_sched_backend_from_buffer`, `ggml-backend.cpp:845`)
already routes ops based on tensor buffer type. A tensor with VITRIOL CUDA buffer type
→ CUDA dispatches MUL_MAT_ID. A tensor with Vulkan buffer type → Vulkan dispatches
SSM_SCAN. The scheduler handles it automatically.

The activation copy between backends is also handled automatically by
`ggml_backend_sched_backend_copy_tensor()` when a tensor transitions from one backend
to another. We don't need custom copy logic.

---

## 2. Milestone 1: Mamba-1 Vulkan SSM Shader ✅

**Status: Complete** — Implemented, built, tested at 6.8 tok/s on Vulkan with `-ngl 10`.

### 2.1 What Was Built

A new GLSL compute shader `ssm_scan_mamba1.comp` implementing the Mamba-1 SSM scan
algorithm for Vulkan. Previously, the Vulkan backend only supported Mamba-2 (d_state
128/256) and rejected Mamba-1 (d_state=16) in both `supports_op` and the dispatch
function.

### 2.2 Algorithm (from CUDA reference `ssm-scan.cu:18-111`)

```
For each head (128 heads per workgroup, 1 head per thread):
  Load per-head A[16] and state s0[16] into registers
  For each token position i:
    Load B[i][16] and C[i][16] into shared memory
    dt = softplus(dt[i])
    x_dt = x[i] * dt
    y = 0
    For j in 0..16:
      state[j] = state[j] * exp(dt * A[j]) + B[i][j] * x_dt
      y += state[j] * C[i][j]
    dst[i] = y
  Write back final state
```

### 2.3 Files Changed

| File | Change | Lines |
|---|---|---|
| `vulkan-shaders/ssm_scan_mamba1.comp` | **New** — Mamba-1 SSM scan GLSL shader | +84 |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | Register new shader for SPIR-V compilation | +1 |
| `ggml-vulkan.cpp` | Add pipeline pointer + creation + dispatch + supports_op | +30 |

### 2.4 Verification

- Qwen3.6 runs correctly with Vulkan backend (`-ngl 10`, partial offload)
- Output matches CUDA path (coherent reasoning text)
- No CPU fallback for SSM ops (confirmed log shows no CPU dispatch)
- CUDA regression: Qwen3.6 still at 20+ tok/s on CUDA VITRIOL path

---

## 3. Milestone 2: VITRIOL Memory → Vulkan

**Status: In Progress** — Implementation started 2026-05-22 11:00

### 3.1 Goal

Enable `-ngl 99` with the Vulkan backend by importing VITRIOL's page-locked host RAM
into Vulkan via `VK_EXT_external_memory_host`. Currently, `-ngl 99` on Vulkan fails
with `vk::Device::allocateMemory: ErrorOutOfDeviceMemory` because all weights try to
fit in VRAM.

### 3.2 Implementation Plan

#### Step 1: Shared Host Memory Allocator (~80 lines)

**Files:** `vitriol-allocator.h` (new), `vitriol-allocator.cpp` (new)

Extract the allocation logic from `vitriol-buffer.cpp` into a shared allocator:

```cpp
// vitriol-allocator.h
struct vitriol_allocator {
    void * ptr;
    size_t size;
    size_t used;
};

vitriol_allocator * vitriol_allocator_create(size_t size);
void vitriol_allocator_destroy(vitriol_allocator * alloc);
void * vitriol_allocator_alloc(vitriol_allocator * alloc, size_t size, size_t alignment);
```

The allocator:
1. Creates anonymous hugepage-backed mmap (same as `vitriol-buffer.cpp:207-208`)
2. Touches pages, sets MADV_HUGEPAGE, mlock (same as lines 218-226)
3. Registers with cudaHostRegister for CUDA DMA (same as line 231)
4. Tracks allocations via bump-pointer aligned to `minImportedHostPointerAlignment`

#### Step 2: VITRIOL Vulkan Buffer Type (~120 lines)

**Files:** `vitriol-vk-buffer.h` (new), `vitriol-vk-buffer.cpp` (new)

New buffer type `vitriol_buffer_vk` that:
1. Allocates from the shared host allocator (same page-locked RAM as CUDA type)
2. Imports each tensor's host RAM slice into Vulkan via `ggml_vk_buffer_from_host_ptr()`
3. Stores the resulting `vk_buffer` alongside the host pointer in its context
4. Reports `is_host = false` (so Vulkan accepts it for compute dispatch)

```cpp
static void * vitriol_vk_buffer_type_init_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx = (vitriol_vk_buffer_context *)buffer->context;
    size_t size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);
    size_t alignment = vitriol_vk_buffer_type_get_alignment(buffer->buft);
    tensor->data = vitriol_allocator_alloc(ctx->alloc, size, alignment);
    ctx->vk_buf = ggml_vk_buffer_from_host_ptr(
        ctx->vk_device, tensor->data, size);
    return tensor->data;
}
```

#### Step 3: Modify Vulkan `supports_buft` (~5 lines)

**File:** `ggml-vulkan.cpp`

Add VITRIOL VK buffer type check to `ggml_backend_vk_device_supports_buft`:

```cpp
static bool ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Existing check
    if (buft->iface.get_name == ggml_backend_vk_buffer_type_name) {
        ggml_backend_vk_device_context * ctx = ...;
        ggml_backend_vk_buffer_type_context * buft_ctx = ...;
        return buft_ctx->device->idx == ctx->device;
    }
    // NEW: accept VITRIOL VK buffer types
    if (vitriol_is_vitriol_vk_buffer_type(buft)) {
        return true;
    }
    return false;
}
```

#### Step 4: Modify `ggml_vk_tensor_subbuffer` (~30 lines)

**File:** `ggml-vulkan.cpp`

Currently line 6537 casts `tensor->buffer->context` to `ggml_backend_vk_buffer_context *`.
VITRIOL VK buffer type has a different context. Add a type check:

```cpp
static vk_subbuffer ggml_vk_tensor_subbuffer(...) {
    vk_buffer buffer = nullptr;
    size_t offset = 0;
    if (ctx->device->uma) { ... }
    if (!buffer) {
        if (vitriol_is_vitriol_vk_buffer_type(tensor->buffer->buft)) {
            auto * vk_ctx = (vitriol_vk_buffer_context *)tensor->buffer->context;
            buffer = vk_ctx->vk_buf;
            offset = vk_tensor_offset(tensor) + tensor->view_offs;
        } else {
            auto * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;
            buffer = buf_ctx->dev_buffer;
            offset = vk_tensor_offset(tensor) + tensor->view_offs;
        }
    }
    ...
}
```

#### Step 5: Model Loader Routing (~30 lines)

**File:** `llama-model-loader.cpp`

Extend the VITRIOL tensor interception code (lines 1187-1220) to route non-expert
tensors to the VITRIOL VK buffer type when Chimera mode is active:

```cpp
if (!buft) {
    std::string tensor_name = tn.str();
    if (tensor_name.find("exps") != std::string::npos) {
        // Expert tensor → CUDA VITRIOL buffer (existing path)
        buft = vitriol_getter();
    } else if (vitriol_chimera_enabled()) {
        // Dense tensor → VITRIOL VK buffer (new path)
        buft = vitriol_get_vk_buffer_type();
    }
}
```

#### Step 6: Cross-Backend Copy Path (automatic)

No code changes needed. The existing `ggml_backend_tensor_copy` fallback chain
(`ggml-backend.cpp:477-498`) handles CUDA↔Vulkan copies via CPU staging:

```cpp
// Fallback path (auto-selected when cpy_tensor returns false):
void * data = malloc(nbytes);
ggml_backend_tensor_get(src, data, 0, nbytes);  // GPU VRAM → CPU
ggml_backend_tensor_set(dst, data, 0, nbytes);  // CPU → other GPU VRAM
free(data);
```

~0.001 ms per 16 KB activation — negligible overhead.

### 3.3 Alignment Considerations

`VK_EXT_external_memory_host` requires pointer alignment to
`minImportedHostPointerAlignment` (typically 64K-256K). The shared allocator
enforces this:

```cpp
size_t alignment = max(4096, (size_t)device->min_imported_host_pointer_alignment);
posix_memalign(&ptr, alignment, size);
```

### 3.4 Graceful Degradation

If `VK_EXT_external_memory_host` or a GPU-specific feature is unavailable:
- VITRIOL VK buffer type falls back to regular Vulkan device memory allocation
- Only partial offload (`-ngl < layers_that_fit_in_vram`) is possible
- CUDA VITRIOL path remains fully functional as fallback

---

## 4. Milestone 3: Dynamic MoE Command Buffers

**Status: Planned** — ~300 lines, 3-5 days

### 4.1 Goal

Rebuild Vulkan command buffers dynamically for MoE expert matmuls without losing
the pre-baked pipeline advantage.

### 4.2 The Challenge

Vulkan's generation speed advantage comes from pre-baked command buffers that the
GPU executes with zero CPU overhead. MoE defeats this because the expert selection
changes every token, requiring different buffer bindings.

### 4.3 The Solution: Predictor-Overlapped Rebuilding

At 23.5 tok/s, each token takes ~42ms. VITRIOL's predictive prefetcher predicts
next-token expert routing within ~1ms of receiving the current token. This gives
~41ms to rebuild the next command buffer — more than enough.

```
Timeline:
  Token N starts     Token N +1 experts predicted     Token N finishes
  │                          │                              │
  ├────── Compute ──────────┼─────── Compute ──────────────┤
  │                          │                              │
  │            Predict next  │    Build next cmd buffer     │
  │            experts (~1ms)│    (~0.5ms)                   │
  │                          │                              │
  └──────────────────────────┴──────────────────────────────┘
  All time units: microseconds
```

### 4.4 Implementation

#### Step 1: Predictor State Sharing (~50 lines)

**File:** `vitriol-cuda-integration.cpp`

Expose the predictor's next-token expert predictions as a shared state:

```cpp
struct vitriol_predictor_state {
    int predicted_experts[256];  // expert IDs for next token
    int n_predicted;              // count of predicted experts
    uint64_t token_id;            // which token this prediction is for
};

vitriol_predictor_state vitriol_predictor_get_state(void);
```

The predictor already computes this in `vitriol_predictor_prefetch` (lines 381-450).
We just need to expose it.

#### Step 2: Dynamic Command Buffer Builder (~200 lines)

**File:** `ggml-vulkan.cpp`

New function that builds a command buffer for MUL_MAT_ID with specific expert bindings:

```cpp
void ggml_vk_build_moe_cmd_buffer(
    vk_device & device,
    vk_command_buffer & cmd,
    const vitriol_predictor_state & pred,
    const ggml_tensor * weights,
    const ggml_tensor * input,
    const ggml_tensor * output) {

    vkCmdBeginCommandBuffer(cmd, ...);

    for (int i = 0; i < pred.n_predicted; i++) {
        int expert_id = pred.predicted_experts[i];
        // Bind VITRIOL's host memory buffer via VK_EXT_external_memory_host
        VkBuffer expert_buf = get_expert_buffer(weights, expert_id);
        vkCmdBindDescriptorSets(cmd, ..., 1, &descriptor_set, 0, nullptr);
        vkCmdDispatch(cmd, ...);
    }

    vkCmdEndCommandBuffer(cmd);
}
```

#### Step 3: Async Command Buffer Submission (~50 lines)

**File:** `ggml-vulkan.cpp`

Submit the pre-built command buffer on the next token:

```cpp
void ggml_vk_submit_moe(vk_context & subctx) {
    // Submit the command buffer built during the previous token's compute
    vkSubmitInfo submit = {};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_moe_cmd_buf;
    vkQueueSubmit(subctx.queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(subctx.queue);  // or use timeline semaphores
}
```

---

## 5. Milestone 4: Backend Routing

**Status: Planned** — ~150 lines, 1-2 days

### 5.1 Goal

Automatic routing of each operation to the optimal backend without user intervention.

### 5.2 Routing Table

| Operation | Backend | Reason |
|---|---|---|
| `GGML_OP_MUL_MAT_ID` (expert) | CUDA VITRIOL | DMA + pin pool + predictor |
| `GGML_OP_MUL_MAT` (dense) | Vulkan | Pre-baked pipelines |
| `GGML_OP_SSM_SCAN` | Vulkan | Pre-baked, Mamba-1 shader now works |
| `GGML_OP_SSM_CONV` | Vulkan | Already supported |
| `GGML_OP_MUL_MAT` (attention KQ/PV) | Vulkan | Pre-baked, fixed shapes |
| `GGML_OP_RMS_NORM` | Vulkan | Already supported |
| `GGML_OP_ROPE` | Vulkan | Already supported |

### 5.3 Implementation

The routing is automatic based on buffer types (existing `ggml_backend_sched_backend_from_buffer` logic). We just need to ensure:

1. Expert tensors have VITRIOL CUDA buffer type → only CUDA `supports_op` returns true for MUL_MAT_ID
2. Dense tensors have VITRIOL VK buffer type → only Vulkan `supports_op` returns true for SSM_SCAN, etc.

The scheduler iterates backends in priority order and picks the first match. We set:
- CUDA backend: high priority (for expert ops)
- Vulkan backend: medium priority (for dense ops)
- CPU backend: low priority (fallback)

### 5.4 Activation Copies

`ggml_backend_sched_backend_copy_tensor` (in `ggml-backend.cpp`) handles cross-backend
activation copies automatically. We just need both backends to accept each other's
buffer types as copy sources.

### 5.5 Startup Flow

```
1. User runs: llama-server -ngl 99
2. VITRIOL CUDA backend initializes (vitriol_cuda_init)
3. VITRIOL VK backend initializes (vitriol_vk_init)
4. Model loads:
   - Expert tensors → VITRIOL CUDA buffer type (page-locked host RAM)
   - Dense tensors → VITRIOL VK buffer type (page-locked host RAM)
5. First token:
   - SSM_SCAN → Vulkan dispatch (Mamba-1 shader)
   - MUL_MAT_ID → CUDA dispatch (VITRIOL DMA)
   - Activations copied between backends automatically
6. Predictor starts, building next-token command buffer asynchronously
```

---

## 6. Testing Strategy

### 6.1 Milestone 1 Tests ✅

| Test | Result |
|---|---|
| CUDA regression: Qwen3.6 at full speed | ✅ 20+ tok/s |
| Vulkan with -ngl 10: SSM runs on GPU | ✅ 6.8 tok/s |
| Vulkan with -ngl 99: OOM without VITRIOL memory | ✅ Expected (needs M2) |

### 6.2 Milestone 2 Tests (Planned)

| Test | Expected Result |
|---|---|
| Vulkan with -ngl 99 + VITRIOL VK buffer | ✅ Loads model, no OOM |
| Generation speed with Vulkan dense ops | ✅ 15-20 tok/s |
| Output matches CUDA path exactly | ✅ Bit-exact comparison |
| Full 32K context without OOM | ✅ Host RAM used for weights |

### 6.3 Milestone 3 Tests (Planned)

| Test | Expected Result |
|---|---|
| Dynamic MoE command buffer speed | ✅ 20+ tok/s on MoE models |
| Predictor hit rate on Qwen3.6 | ✅ 90%+ (matches CUDA predictor) |
| AMD GPU (community test) | ✅ Vulkan path works |

### 6.4 Milestone 4 Tests (Planned)

| Test | Expected Result |
|---|---|
| End-to-end Chimera with full routing | ✅ 20+ tok/s combined speed |
| Cross-backend activation copies | ✅ Correct output |
| Graceful fallback (no VK_EXT) | ✅ Partial offload works |

---

## 7. File Inventory

### New Files

| File | Purpose | Est. Lines | Milestone |
|---|---|---|---|
| `ggml-vulkan/vulkan-shaders/ssm_scan_mamba1.comp` | Mamba-1 SSM scan shader | 84 | ✅ M1 |
| `vitriol-allocator.h` | Shared host memory allocator interface | 30 | M2 |
| `vitriol-allocator.cpp` | Shared host memory allocator implementation | 80 | M2 |
| `vitriol-vk-buffer.cpp` | VITRIOL Vulkan buffer type | 120 | M2 |
| `vitriol-vk-buffer.h` | VITRIOL Vulkan buffer type interface | 30 | M2 |

### Modified Files

| File | Change | Est. Lines | Milestone |
|---|---|---|---|
| `vulkan-shaders/vulkan-shaders-gen.cpp` | Register Mamba-1 shader | +1 | ✅ M1 |
| `ggml-vulkan.cpp` | Pipeline + dispatch + supports_op + tensor subbuffer | +60 | ✅ M1, M2 |
| `vitriol-buffer.cpp` | Use shared allocator | -50 | M2 |
| `vitriol-cuda-integration.cpp` | Predictor state sharing + VK buffer type API | +80 | M2, M3 |
| `llama-model-loader.cpp` | Route tensors to correct buffer type | +20 | M2, M4 |

### Total Effort

| Milestone | New Files | Modified Files | Total Lines | Time |
|---|---|---|---|---|
| M1: Mamba-1 SSM shader | 1 | 2 | 115 | ✅ Done |
| M2: VITRIOL memory → Vulkan | 4 | 4 | ~250 | 2-3 days |
| M3: Dynamic MoE cmd buffers | 0 | 3 | ~300 | 3-5 days |
| M4: Backend routing | 0 | 2 | ~150 | 1-2 days |
| **Total** | **5** | **11** | **~815** | **1-2 weeks** |

---

## Appendix A: Key Code References

| Component | File | Lines |
|---|---|---|
| VITRIOL CUDA buffer type | `vitriol-buffer.cpp` | 1-307 |
| VITRIOL init + predictor + pin pool + LRU | `vitriol-cuda-integration.cpp` | 1-850 |
| Tensor buffer type override | `llama-model-loader.cpp` | 1187-1220 |
| MUL_MAT_ID CUDA dispatch | `ggml-cuda.cu` | 2519-2778 |
| Mamba-1 CUDA kernel | `ssm-scan.cu` | 18-111 |
| Mamba-2 Vulkan shader | `vulkan-shaders/ssm_scan.comp` | 1-124 |
| Mamba-1 Vulkan shader | `vulkan-shaders/ssm_scan_mamba1.comp` | 1-84 ✅ |
| SSM_SCAN Vulkan pipeline | `ggml-vulkan.cpp` | 9734-9743 |
| SSM_SCAN Vulkan dispatch | `ggml-vulkan.cpp` | 10617-10671 |
| SSM_SCAN Vulkan supports_op | `ggml-vulkan.cpp` | 15790-15827 |
| VK_EXT_external_memory_host import | `ggml-vulkan.cpp` | 15957-15994 |
| Vulkan tensor subbuffer | `ggml-vulkan.cpp` | 6528-6552 |
| Backend scheduler routing | `ggml-backend.cpp` | 845-865 |
| Tensor buffer field | `ggml.h` | 660-663 |

## Appendix B: Prior Art

| PR | Description | Merged |
|---|---|---|
| #16463 | Vulkan SSM operations (Mamba-2 only) | Oct 2025 |
| #18505 | Optimize CUDA SSM scan (warp-level reduction) | Jan 2026 |
| #6758 | Original feature request: GPU Mamba support | Still open |
