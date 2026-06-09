# Plan: Vulkan Chimera Milestone 3 — Dynamic MoE Command Buffers
**Date:** 2026-06-09
**Status:** Proposed (revised from deferred)

## Objective

Extend the Chimera Vulkan backend to support MoE expert computation on Vulkan. Currently, dense ops (attention, SSM, norms) run on Vulkan via `vitriol-vk-buffer`, but MoE experts still route to CUDA. Completing M3 enables the Vulkan-only path (Milestone 4) and unlocks CPU-driven WC writes on ReBAR hardware.

## Current State

**Files:**
- `vitriol-vk-buffer.cpp` — Page-locked host RAM buffer importable into Vulkan via `VK_EXT_external_memory_host`
- `vitriol-vk-buffer.h` — Header for buffer type
- `ggml-vulkan.cpp` — Main Vulkan backend implementation (upstream llama.cpp, VITRIOL-modified)
- `vulkan-shaders/*.comp` — Shaders for matmul ops

**What's running on Vulkan today:**
- Dense tensor compute (attention QKV projections, SSM scan, layer norms)
- These are "fixed pipeline" — tensor shapes and buffer bindings are known at graph build time

**What's NOT on Vulkan:**
- MoE expert matmuls (gate, up, down projections with expert_id indexing)
- These require dynamic expert selection per token

**The upstream Vulkan backend** (`ggml-vulkan.cpp`) does support MoE via:
- `mul_mat_id` operation that binds expert tensors
- But it assumes fixed tensor bindings known at graph compile time
- VITRIOL's expert buffers are dynamic (expert indices change per token)

## Design

### Approach: Indirect Expert Indirection via Push Constants

Instead of rebuilding the entire command buffer per token (expensive), use **indirect expert selection**:

```
Shaders receive expert indices via push constants.
The expert weight buffer is a single large VkBuffer containing ALL experts.
Shaders index into this buffer using push_constant.expert_id * expert_stride.
```

This avoids command buffer rebuilding entirely — only push constants change per token.

### Buffer Layout

```
[Expert 0 weights] [Expert 1 weights] ... [Expert 255 weights]
         │                  │                       │
         └──────────────────┴───────────────────────┘
         All stored in ONE VkBuffer (host-visible, page-locked RAM)
         Shader computes offset = expert_id * expert_size_bytes
```

### Shader Changes

Current `mul_mat_vec` shaders read weights from a buffer bound via descriptor. For MoE:

```glsl
// In shader (push constant block)
layout(push_constant) uniform PushConstants {
    uint expert_id;
    uint expert_stride;  // bytes per expert tensor
} pc;

// Weight access
uint weight_offset = pc.expert_id * pc.expert_stride + element_index;
float weight = float(weight_buffer[weight_offset / 4]);
```

No per-token command buffer rebuild needed — just `vkCmdPushConstants` before each expert dispatch.

### Pipeline Flow

```
Per token, per MoE layer:

1. Router runs (on either CUDA or Vulkan, whichever is faster for small ops)
2. Get active expert IDs (e.g., [7, 142, 55, 203])
3. For each active expert:
   a. vkCmdBindPipeline(matmul_pipeline)
   b. vkCmdPushConstants(expert_id, expert_stride)
   c. vkCmdBindDescriptorSets(input_activations, output_buffer)
   d. vkCmdDispatch(...)
4. Sum expert outputs (element-wise add with router weights)
```

### New Code

**File 1:** `vitriol-vk-expert.h` / `vitriol-vk-expert.cpp`

```cpp
// Manages Vulkan MoE expert matmul state
struct vitriol_vk_expert_pipeline {
    VkPipeline          gate_pipeline;   // gate_proj matmul
    VkPipeline          up_pipeline;     // up_proj matmul
    VkPipeline          down_pipeline;   // down_proj matmul
    VkPipelineLayout    pipeline_layout;
    VkDescriptorSetLayout descriptor_layout;
    uint32_t            expert_stride;    // bytes per expert
};

// Initialize pipelines (once at model load)
bool vitriol_vk_expert_init(
    VkDevice dev,
    const ggml_tensor *gate_tensor,
    const ggml_tensor *up_tensor,
    const ggml_tensor *down_tensor);

// Execute one expert's matmul
void vitriol_vk_expert_dispatch(
    VkCommandBuffer cmd,
    vitriol_vk_expert_pipeline *pipe,
    int expert_id,
    VkBuffer input_buf,    // activations
    VkBuffer output_buf,   // result
    VkBuffer expert_buf);  // weight buffer (VITRIOL_VK type)
```

**File 2:** Modify `vitriol-vk-buffer.cpp` to support expert tensor storage

The existing buffer type already allocates page-locked host RAM and imports into Vulkan. For expert weights, allocate one large buffer per expert tensor type (gate, up, down) that contains all 256 experts consecutively.

**File 3:** Modify `ggml-vulkan.cpp` to route MoE ops to the expert pipeline

When `VITRIOL_CHIMERA_MODE=vulkan` or `auto` is set and the op is `GGML_OP_MUL_MAT_ID`:

```cpp
// In ggml_vk_can_use_mul_mat_id or similar dispatch function
if (vitriol_chimera_enabled() && vitriol_vk_expert_available()) {
    // Route to VITRIOL Vulkan expert path instead of CUDA
    vitriol_vk_expert_dispatch(cmd, &expert_pipe, expert_id,
                                input_buf, output_buf, weight_buf);
    return;
}
```

## Implementation Steps

### Phase 1: Expert Weight Buffer (days 1-2)

1. In `vitriol-vk-buffer.cpp`, add a new buffer type for expert tensors that stores all 256 experts contiguously:
   ```cpp
   ggml_backend_buffer_type_t vitriol_get_vk_expert_buffer_type(int device_idx) {
       // Same as vitriol_get_vk_buffer_type but marks is_host=false
       // and ensures VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
   }
   ```
2. Modify model loader (`llama-model-loader.cpp`) to use this type for expert tensors when Chimera is active.

### Phase 2: Shader Authoring (days 3-5)

3. Create `vulkan-shaders/moe_gate.comp`, `moe_up.comp`, `moe_down.comp`:
   - Same matmul logic as existing `mul_mat_vec` shaders
   - Add `push_constant` block for `expert_id` and `expert_stride`
   - Weight buffer indexed by `expert_id * expert_stride + offset`

4. Update `vulkan-shaders-gen.cpp` to include new shaders.

### Phase 3: Pipeline Management (days 6-8)

5. Create `vitriol-vk-expert.cpp` with:
   - `vitriol_vk_expert_init()` — create pipelines from compiled shaders
   - `vitriol_vk_expert_dispatch()` — bind + push + dispatch
   - `vitriol_vk_expert_cleanup()` — destroy pipelines

6. Integration point in `vitriol_init()` for CUDA path: when Chimera Vulkan mode is active, call `vitriol_vk_expert_init()` after Vulkan device is created.

### Phase 4: Backend Integration (days 9-12)

7. In `ggml-vulkan.cpp`, modify `ggml_vk_mul_mat_id()` or equivalent:
   - Detect VITRIOL expert buffer on `src0` (weight tensor)
   - Route to `vitriol_vk_expert_dispatch` instead of standard matmul
   - Handle fallback to CUDA if Vulkan expert path not available

8. Handle router output: the router's softmax gating runs on the dense path (Vulkan). The resulting expert IDs are passed to the expert dispatch.

### Phase 5: Cross-Backend Activation Sharing (days 13-14)

9. Currently, the chimera path copies activations between CUDA and Vulkan via CPU staging (0.001 ms per 16 KB). For Vulkan-only MoE, activations stay in Vulkan memory — no cross-backend copy needed.

10. Verify that the Vulkan path doesn't need CUDA for any MoE operation. If it does, add the copy path.

### Phase 6: Testing (days 15-16)

11. Test on GTX 1070 Ti: compare token/s between:
    - CUDA-only (baseline): `VITRIOL_CHIMERA_MODE=cuda`
    - Chimera (dense Vulkan, expert CUDA): current behavior
    - Vulkan-only (all Vulkan): `VITRIOL_CHIMERA_MODE=vulkan`

12. On GTX 1070 Ti, Vulkan-only will likely be slower than CUDA for MoE (no WC fast path). Document performance.

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `vitriol-vk-buffer.h` | Declare `vitriol_get_vk_expert_buffer_type()` | +5 |
| `vitriol-vk-buffer.cpp` | Implement expert buffer type | +60 |
| `vitriol-vk-expert.h` | *New file* — pipeline declarations | +50 |
| `vitriol-vk-expert.cpp` | *New file* — pipeline implementation | +300 |
| `vulkan-shaders/moe_gate.comp` | *New file* | +80 |
| `vulkan-shaders/moe_up.comp` | *New file* | +80 |
| `vulkan-shaders/moe_down.comp` | *New file* | +80 |
| `vulkan-shaders-gen.cpp` | Include new shaders | +10 |
| `ggml-vulkan.cpp` | Route MUL_MAT_ID to expert path | +100 |
| `vitriol-cuda-integration.cpp` | Init expert pipeline when Vulkan mode | +20 |
| `llama-model-loader.cpp` | Use expert buffer type for MoE tensors | +30 |
| **Total** | | **~735 lines** |

## Expected Performance

On **GTX 1070 Ti** (no ReBAR):
- Vulkan host-visible memory = system RAM read over PCIe (same as CUDA direct host read)
- No WC/device-local fast path
- Expect: **≤CUDA performance** (same bandwidth, potentially more driver overhead)
- Value: Cross-platform enablement, not performance

On **ReBAR-capable GPU** (RTX 3060+, AMD RX 6000+):
- Vulkan can use `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`
- CPU writes with WC coalescing → expert weight updates in ~1 µs instead of ~6 µs + PCIe transfer
- Potential for CPU-driven WC writes at GDRCopy-like latency
- **This is where the gain comes from**

On **AMD GPU** (RX 6000+, no CUDA):
- Vulkan-only path is the ONLY option for VITRIOL
- M3 is a prerequisite for AMD support
- Performance depends on PCIe bandwidth + ROCm/Vulkan compute capability

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Vulkan MoE slower than CUDA on GTX 1070 Ti | Negative performance impact | Keep CUDA fallback (Chimera hybrid mode). Vulkan-only is opt-in. |
| Push constant size limits (max 128 bytes) | Can't pass enough expert indices | Pass one expert at a time. Multiple dispatches per layer. |
| Shader compilation time | Slow model load | Pre-compile spirv at build time |
| Vulkan driver bugs on GeForce | Crashes or incorrect results | Extensive testing; fallback to CUDA on failure |

## Relationship to Other Plans

- **Double-buffer DMA (Plan 1)**: CUDA-specific optimization. Vulkan path doesn't benefit from CUDA double-buffer.
- **PCIe coalescing (Plan 2)**: CUDA-specific. Vulkan path gets coalescing naturally (single buffer with all experts).
- **Predictor (Plan 3)**: Architecture-agnostic. Vulkan path uses the same predictor.
- **BAR1 WC (Plan 4)**: Vulkan path on ReBAR hardware IS the BAR1 WC path — through standard Vulkan, not kernel module.

## Success Criteria

1. Vulkan-only path produces correct output (bit-exact with CUDA within quantization tolerance)
2. Vulkan-only path on GTX 1070 Ti achieves ≥90% of CUDA throughput
3. On ReBAR AMD GPU (if tested): CPU-driven WC writes working via Vulkan
4. No regressions in Chimera hybrid mode (dense Vulkan + expert CUDA)
