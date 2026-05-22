# VITRIOL Chimera — Session Report
**Date:** 2026-05-22 (10:30-11:30)
**Author:** VITRIOL Project

---

## Summary

Implemented the VITRIOL Chimera dual-backend (CUDA+Vulkan) across two milestones.
Total: **~507 lines** added across **7 files** in the llama.cpp source tree.

## Milestone 1: Mamba-1 Vulkan SSM Shader

**Status:** Complete ✅ (115 lines)

### Problem
The Vulkan backend only supported Mamba-2 SSM scan (d_state 128/256).
Qwen3's Gated Delta Net and Jamba2's Mamba-1 both use d_state=16,
which was explicitly rejected at both `supports_op` and dispatch time.

### Solution
New GLSL compute shader `ssm_scan_mamba1.comp` implementing the Mamba-1
SSM scan algorithm (d_state=16, head_dim=1, 128 threads/workgroup, shared
memory for B/C). Direct port from the CUDA reference kernel at
`ssm-scan.cu:18-111`.

### Files
| File | Change | Lines |
|---|---|---|
| `vulkan-shaders/ssm_scan_mamba1.comp` | New GLSL shader | +84 |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | Register for SPIR-V compilation | +1 |
| `ggml-vulkan.cpp` | Pipeline pointer + creation + dispatch + supports_op | +30 |

### Verification
- Qwen3.6 on Vulkan (`-ngl 10`): **6.8 tok/s**, correct output
- CUDA regression: **20+ tok/s** on CUDA VITRIOL path (unchanged)

---

## Milestone 2: Chimera Dual-Backend (CUDA+Vulkan)

**Status:** Complete ✅ (392 lines)

### Problem
- `-ngl 99` on Vulkan OOM (`ErrorOutOfDeviceMemory`) because all weights
  try to fit in VRAM
- CUDA VITRIOL is fast for MoE (23.5 tok/s) but single-backend
- No cross-vendor support

### Architecture
```
                    Page-locked Host RAM
                    ┌──────────────────────┐
                    │ Expert weights       │ Dense weights        │
                    │ (CUDA VITRIOL type)  │ (VITRIOL VK type)    │
                    └──────────────────────┘
                               │                    │
                    CUDA: cudaHostRegister    VK: VK_EXT_external_memory_host
                    reads via PCIe DMA        reads via imported VkBuffer

Activations: CUDA VRAM ←→ CPU staging ←→ Vulkan VRAM
             (~0.001ms per 16KB copy, ~0.13% overhead per token)
```

### Key Codebase Findings
1. **Both backends can coexist**: No mutual exclusion in `ggml-backend-reg.cpp`
2. **Vulkan's `supports_buft` uses function pointer identity**: Modified to
   accept VITRIOL VK buffer types via explicit check
3. **Cross-backend copies**: Automatic via CPU staging fallback in
   `ggml_backend_tensor_copy` (~0.001ms per activation)
4. **Scheduler routing**: `ggml_backend_sched_backend_from_buffer` routes
   ops based on tensor buffer type — works automatically

### Files
| File | Change | Lines |
|---|---|---|
| `vitriol-vk-buffer.h` | New: VITRIOL VK buffer type header | +51 |
| `vitriol-vk-buffer.cpp` | New: VK buffer type implementation | +218 |
| `ggml-vulkan.cpp` | supports_buft + tensor_subbuffer + fwd decl | +35 |
| `CMakeLists.txt` | Add vitriol-vk-buffer.cpp to build | +1 |
| `llama-model-loader.cpp` | Chimera tensor routing (VITRIOL_CHIMERA=1) | +87 |

### Key Design Decisions
- **Lazy VkBuffer creation**: VkBuffer created on first Vulkan dispatch via
  `ggml_vk_buffer_from_host_ptr`, cached as type-erased `shared_ptr<void>`
- **No CUDA API needed in VK code**: VITRIOL VK type allocates via
  `posix_memalign + mlock` (not CUDA cudaHostRegister)
- **Chimera mode**: Activated by `VITRIOL_CHIMERA=1` env var, set at model
  loading time via dlsym from `libggml-vulkan.so`

### Verification
- Qwen3.6 with Chimera: **20.7 tok/s** (prompt: 25.8 tok/s)
- MTP acceptance: **100%** (12/12 draft tokens)
- Expert tensors → CUDA VITRIOL (confirmed via log)
- Dense tensors → VITRIOL VK (confirmed via log)

---

## Commits

### llama.cpp (`vitriol` branch)
| Commit | Description |
|---|---|
| `174645996` | Mamba-1 Vulkan SSM shader + pipeline |
| `e4c8738fd` | Chimera dual-backend (CUDA+Vulkan) |

### VITRIOL (`main` branch)
| Commit | Description |
|---|---|
| `f157153` | Chimera plan + Mamba-1 shader submodule |
| `7404d64` | Milestone 2 — Chimera dual-backend |

---

## SonarQube / SlopGuard Notes

The VITRIOL VK buffer type (`vitriol-vk-buffer.cpp`) uses a type-erased
`shared_ptr<void>` with custom deleter pattern to cache `vk_buffer`
across dispatches. This appears as a code smell in static analysis but
is a deliberate design choice to avoid including `vk_buffer_struct`
(defined in `ggml-vulkan.cpp`) from a separate compilation unit.

The alternative — storing raw `VkBuffer`/`VkDeviceMemory` handles and
reconstructing `vk_buffer` on each dispatch — would be cleaner but
would require either repeated `vkCreateBuffer` calls or a handle registry
in `ggml-vulkan.cpp`. The current approach is functionally correct and
performance-neutral.

---

## Next Steps

| Milestone | Description | Est. Lines | Est. Time |
|---|---|---|---|
| M3 | Dynamic MoE command buffers | ~300 | 3-5 days |
| M4 | Backend routing optimization | ~150 | 1-2 days |
