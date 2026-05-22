# VITRIOL Chimera — Milestones 3 & 4 Plan
**Date:** 2026-05-22 11:30

---

## Milestone 3: Dynamic MoE Command Buffers (Deferred)

**Status:** Deferred — not needed for the Option B Chimera.

### Original Idea
Rebuild Vulkan command buffers dynamically per token to handle MoE expert
binding changes, using the VITRIOL predictor to overlap rebuilding with
current-token compute.

### Why Deferred
In the Option B Chimera, MoE experts run on CUDA VITRIOL (not Vulkan).
The dense ops that DO run on Vulkan (SSM scan, attention, norms) have
fully fixed pipelines — no dynamic binding needed. There is no MoE on
Vulkan in the current architecture.

### Possible Future Use
If Vulkan-only inference is desired (e.g., for AMD GPUs without CUDA),
this would be needed to handle MoE efficiently on Vulkan. The approach:
- VITRIOL predictor predicts next-token expert routing (~1ms)
- While current token computes, rebuild Vulkan cmd buffer with predicted
  expert bindings (~0.5ms)
- Submit pre-built cmd buffer for next token
- Estimated effort: ~300 lines, 3-5 days

### Key Technical Challenge
Vulkan's pre-baked pipeline advantage is defeated by MoE's dynamic expert
selection. Each token needs different buffer bindings. The rebuild must
complete before the next token starts computing.

---

## Milestone 4: Chimera Backend Routing (In Progress)

**Date:** 2026-05-22
**Status:** Implementation in progress

### Goal
Replace the manual `VITRIOL_CHIMERA=1` env var with an auto-detect config
that works seamlessly. Users set `chimera.mode` in `~/.vitriol/config`.

### Config Design
```ini
[chimera]
mode = auto       # auto | cuda | vulkan | off
```

| Mode | Effect |
|---|---|
| `auto` (default) | Auto-detect. If `dlsym` finds VK buffer type → Chimera. If not → CUDA-only. |
| `cuda` | Force CUDA-only. Expert CUDA type + default dense type = standard VITRIOL |
| `vulkan` | Force Vulkan-only. All tensors get VK buffer type. |
| `off` | Explicitly disable Chimera. Same as `cuda`. |

### Implementation Plan (~60 lines across 4 files)

| File | Change | Lines |
|---|---|---|
| `~/.vitriol/config` | Add `[chimera]` section | +3 |
| `scripts/vitriol` | Parse chimera.mode → `VITRIOL_CHIMERA_MODE` env var | +5 |
| `vitriol-cuda-integration.cpp` | Read `VITRIOL_CHIMERA_MODE` in cuda_init | +15 |
| `llama-model-loader.cpp` | Replace `VITRIOL_CHIMERA=1` check with mode logic | +35 |

### Auto-Detect Flow
```
Model loader reads VITRIOL_CHIMERA_MODE env var
  → "auto" (unset):
      dlsym("vitriol_get_vk_buffer_type")
      if found → Chimera active (experts→CUDA, dense→VK)
      if not found → CUDA-only (existing behavior)
  → "cuda" → CUDA-only (existing behavior)
  → "vulkan" → Vulkan-only (all tensors→VK type)
  → "off" → CUDA-only (existing behavior)
```

### Testing
| Config | Backends Available | Result |
|---|---|---|
| `auto` | CUDA + Vulkan | Chimera: 20+ tok/s |
| `auto` | CUDA only (no VK) | CUDA-only: 20+ tok/s |
| `cuda` | CUDA + Vulkan | CUDA-only: regression check |
| `vulkan` | CUDA + Vulkan | Vulkan-only: -ngl must fit VRAM |
| `off` | CUDA + Vulkan | CUDA-only: same as default |
