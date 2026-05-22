# VITRIOL Session Report — K/V Cache Quant Split + Chimera M4
**Date:** 2026-05-22 11:30-12:00

---

## Summary

Two tasks completed: K/V cache quantization split in the vitriol config system
and the auto-detect Chimera routing (M4).

---

## 1. K/V Cache Quantization Split

### Problem
The TUI config menu showed `KV Quant [q4_0]` which implied it set both K and V
caches. In reality, only `--cache-type-k` was configured; `--cache-type-v` was
never set, leaving V at f16 always. The note at line 1381 explained that
quantizing V causes garbage output with VITRIOL, but the menu didn't warn users.

### Changes
**Config `~/.vitriol/config`:**
```ini
[kv]
quant_mode = q4_0      # K cache only (was: misleading "KV Quant")
quant_mode_v = f16     # NEW: V cache, with warning before allowing non-f16
```

**TUI Memory Menu (item 5 and 6):**
```
5) K Cache Quant     [q4_0]   ← same as before, no warning needed
6) V Cache Quant     [f16]    ← NEW: warns before allowing q8_0/q4_0
```

The V Cache Quant prompt shows:
```
⚠  WARNING: --cache-type-v causes garbage output with VITRIOL
   expert offloading. Only set to q8_0 or q4_0 if you have
   verified it works correctly with your specific model.
```

Non-f16 selections require typing `YES` to confirm.

### Files Changed
| File | Change |
|---|---|
| `~/.vitriol/config` | Added `quant_mode_v = f16` |
| `scripts/vitriol` (~45 lines) | Defaults, config parsing, TUI menu, config write template, display, server args |

### Server Arg Generation
```bash
# K quant → --cache-type-k (existing)
# V quant → --cache-type-v (new, only if not f16)
```

---

## 2. Chimera M4 — Auto-Detect Backend Routing

### Config
```ini
[chimera]
mode = auto        # auto | cuda | vulkan | off
```

### Modes
| Mode | Effect |
|---|---|
| `auto` (default) | Auto-detect Chimera if both CUDA + Vulkan available |
| `cuda` | CUDA-only (existing behavior) |
| `vulkan` | Vulkan-only (all tensors → VK type) |
| `off` | CUDA-only (same as cuda) |

### Verification
- Qwen3.6 auto-detect: **20.7 tok/s** with 100% MTP
- Dense tensors → VITRIOL VK buffer (confirmed via log)
- Expert tensors → CUDA VITRIOL (confirmed via log)

---

## 3. Qwen3-Coder-Next Compatibility

The 24 GB model runs with file-backed mmap at **2.7 tok/s** (`-ngl 12`).
Native 262k context works on Qwen3.6 at **17.3 tok/s**.

---

## Commits

| Repo | Branch | Ref |
|---|---|---|
| llama.cpp | `vitriol` | `23081ba8f` |
| VITRIOL | `main` | `d03496c` |
