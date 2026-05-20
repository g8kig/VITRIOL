# Plan: Expert Pinning v2 (Approach B — Software-Only Tensor-Level Preload)

**Goal:** Pre-load entire expert weight tensors (all 256 experts) of the first N layers into VRAM, so the MMVQ/MMQ/MMF fast-path kernels read directly from VRAM instead of PCIe-hosted page-locked RAM.

**Estimated gain:** +30-80% on pinned layers (VRAM bandwidth ~200 GB/s vs PCIe Gen3 x16 ~16 GB/s).

**Trade-off:** Consumes ~221 MB per pinned layer. With ~6 GiB free VRAM (KV offloaded), can pin ~20-25 layers. Disables output cache when active (they target different code paths).

---

## Bottleneck Analysis

Current throughput: ~10 t/s = ~100 ms/tok across 40 layers ≈ 2.5 ms/layer.

| Step per layer | Time | Bound by |
|---|---|---|
| IDs copy H2D + sort | ~1 ms | PCIe |
| Expert matmul (8 active × ~350 KB) | ~1.5 ms | Compute (Pascal CC 6.1) |

With DMA overlap + predictor, the PCIe copy starts early but only ~34-49% of predictions are correct → ~half of layers still stall.

**Pinning eliminates PCIe entirely for pinned layers**, reducing per-layer time to compute-only (~1.5 ms). This is the hard floor — IQ2_M matmul on Pascal CC 6.1 cannot be sped up without CUDA architecture changes.

**Hard bottleneck:** The 8-bit integer compute on GTX 1070 Ti (Pascal) peaks at ~21 INT8 TFLOPS for MoE weight shapes. At ~550 INT8 operations per active parameter (8 active × 2048 hidden × 1024 FF), we're compute-bound at roughly 1.5-2.0 ms per matmul. No further optimization can reduce this without a GPU upgrade.

---

## VRAM Budget

Config: `--kv-mode offload`, LRU=0, output cache=OFF

| Component | VRAM (MiB) |
|---|---|
| Model weights (attention/base, -ngl 99) | ~1,331 |
| KV cache (offloaded to host RAM) | 0 |
| CUDA compute buffers + driver overhead | ~450 |
| **Free for pinning** | **~6,075** |

Per-layer costs (IQ2_M, 256 experts, fused gate+up tensor):
- `ffn_gate_up_exps`: **~143 MB** (557 KB/expert × 256)
- `ffn_down_exps`: **~78 MB** (303 KB/expert × 256)
- **Total per layer: ~221 MB**

| Pin count | VRAM consumed | Remaining |
|---|---|---|
| 5 layers | ~1,081 MiB | ~4,994 MiB |
| 10 layers | ~2,162 MiB | ~3,913 MiB |
| 15 layers | ~3,243 MiB | ~2,832 MiB |
| 20 layers | ~4,324 MiB | ~1,751 MiB |
| 25 layers | ~5,405 MiB | ~670 MiB (tight) |

**Recommended default when enabled:** `pin_first_n_layers=10`.

---

## Implementation

### C/C++ Changes

#### `vitriol-cuda-integration.h`

Add to `vitriol_config_t`:
```c
int pin_first_n_layers;  // 0 = off (default)
bool pin_active;         // true after first pin allocated
```

#### `vitriol-cuda-integration.cpp`

- **Pin table**: `std::unordered_map<uintptr_t, CUdeviceptr> g_pin_map` mapping `tensor_base` → VRAM buffer
- **`vitriol_pin_ensure(src0->data, nb02, ne02, stream)`**:
  1. `get_layer_index(tensor_base)` → `layer_idx`
  2. If `layer_idx >= pin_first_n_layers` → return NULL
  3. If `layer_idx` already in `g_pin_map` → return mapped VRAM ptr
  4. Otherwise: `cuMemAlloc` → `cuMemcpyHtoDAsync(full_tensor, expert_size * n_experts)` → store in map → return VRAM ptr
  5. On alloc failure: log warning, return NULL (silent fallback to host)
- **`vitriol_pin_lookup(src0->data)`**: Returns VRAM pointer or NULL
- **`vitriol_pin_active()`**: Returns `g_pin_map.size() > 0`
- **Cleanup**: Free all entries in `vitriol_cuda_cleanup_vram()`
- **Stats**: Print pinned count + MB consumed in `vitriol_cuda_print_stats()`

#### `ggml-cuda.cu` — `ggml_cuda_mul_mat_id`

Hook inserted after `GGML_TENSOR_BINARY_OP_LOCALS` (line 2528), before fast-path checks (line 2532):

```cpp
// ── Expert Pinning: redirect src0 to VRAM for fast-path kernels ──
ggml_tensor pinned_src0_override;
const ggml_tensor *orig_src0 = src0;
bool use_pinned = false;

if (vitriol_pin_active()) {
    CUdeviceptr pin_ptr = vitriol_pin_lookup(src0->data);
    if (pin_ptr) {
        pinned_src0_override = *src0;
        pinned_src0_override.data = (void*)pin_ptr;
        src0 = &pinned_src0_override;
        use_pinned = true;
    }
}
```

This single redirect covers MMVQ (line 2541), MMVF (2548), MMQ (2557), MMF (2564) — they all receive the modified `src0`.

After the fast-path checks (line 2568), if none of the fast paths applied, restore `src0` for the per-expert loop:

```cpp
// Restore original src0 for per-expert loop (LRU + output cache path)
if (use_pinned) {
    src0 = orig_src0;
}
```

This keeps the per-expert loop at lines 2637-2716 completely unchanged — it continues using LRU, predictor, and output cache as before.

#### Key design choices

- **Lazy allocation**: Pin buffers are allocated on first encounter of each layer during prefill. The H2D copy for that layer happens once, then all subsequent passes (including prefill's remaining tokens) read from VRAM.
- **Scoped redirect**: Only the fast-path section sees the VRAM pointer. The per-expert loop always sees the original host pointer, so LRU/predictor/output cache continue working identically.
- **No tensor modification**: We reassign the local pointer variable `src0` (not the original tensor). No `const_cast`, no persistent state changed.

### Shell Script Changes (`scripts/vitriol`)

1. **Default**: `DEFAULT_PIN_FIRST_N_LAYERS=0`
2. **Config key**: `vitriol.pin_first_n_layers` (in `[vitriol]` section)
3. **Config file template** (user-facing):
   ```ini
   # Expert Pinning: pin all experts of first N layers permanently in VRAM
   #  0 = disabled (default)
   #  1-25: number of layers to pin
   #  Auto-disables output cache when active (they target different code paths)
   pin_first_n_layers = 0
   ```
4. **Env var**: `VITRIOL_PIN_FIRST_N_LAYERS`
5. **TUI**: VITRIOL Mode Settings → new option **5**:
   - Display: `"Expert Pin Layers (0=off): [value]"`
   - Input: `"Enter number of layers to pin (0=off, 5-15 suggested): "`
   - On change from 0→positive: show warning: *"Output cache will be disabled (pinning + output cache target different decode paths). OK?"*
   - Auto-sets output_cache=0 when pin_layers > 0 (user can manually re-enable)
6. **Env passthrough** in `run` and `serve` blocks
7. **`config_show()`**: Display pin setting and whether output cache was auto-disabled

### User-Facing Interactions & Warnings

| Setting | Interaction | Message |
|---|---|---|
| `pin_first_n_layers` > 0 | Auto-disables output cache | "Pinning targets the fast path. Output cache has been disabled. You can re-enable it manually, but pinning + output cache won't stack during decode." |
| `pin_first_n_layers` = 0 | No effect | — |
| `pin_first_n_layers` set too high | Weak warning | "Pin count of N may leave only X MB free VRAM. Reduce if you see cuMemAlloc failures." |

---

## Benchmark Plan

### Configurations (all with `--kv-mode offload`, VITRIOL_MODE=stream)

| # | Pinned layers | Output cache | Predictive prefetch | Expected t/s |
|---|---|---|---|---|
| 0 (baseline) | 0 | off | off | ~8.24 |
| 1 | 0 | off | on | ~9.34 |
| 2 | 10 | off | off | ? |
| 3 | 15 | off | on | ? |
| 4 | 20 | off | on | ? |
| 5 | 15 | on | on | ? (pin + cache + predictor) |

### Command template

```bash
VITRIOL_MODE=stream VITRIOL_LRU_MB=0 VITRIOL_KV_MODE=offload \
VITRIOL_PREDICTIVE_PREFETCH=1 VITRIOL_OUTPUT_CACHE=0 \
VITRIOL_PIN_FIRST_N_LAYERS=15 \
./build/bin/llama-bench -m /path/to/model.gguf -p 512 -n 256 -ngl 99 -t 4 \
  -o output-csv 2>&1 | tee benchmark_pin_15.log
```

---

## Expected Performance

| Metric | Without pinning | With 15 pinned layers | Gain |
|---|---|---|---|
| PCIe reads per token | ~40 layers × 2.5 ms = 100 ms | ~25 layers × 2.5 ms = 62.5 ms | ~37.5% |
| VRAM reads per token | 0 ms | ~15 layers × ~0.15 ms = 2.25 ms | negligible |
| Total per token | ~100 ms (10 t/s) | ~64.75 ms (~15.4 t/s) | **~54%** |

Reality check: prediction overlap, kernel launch overhead, and PCIe for the 5 non-pinned layers on each path mean realistic gain is closer to **+20-40%** or **12-14 t/s** in practice.

---

## Files to Modify

| File | Lines | Changes |
|---|---|---|
| `vitriol-cuda-integration.h` | ~33-45 | Add `pin_first_n_layers`, `pin_active` to config struct |
| `vitriol-cuda-integration.cpp` | ~222-275 | Read `VITRIOL_PIN_FIRST_N_LAYERS` env var |
| `vitriol-cuda-integration.cpp` | ~498+ | Add pin table, `vitriol_pin_ensure()`, `vitriol_pin_lookup()` |
| `vitriol-cuda-integration.cpp` | ~641-656 | Cleanup pin allocations in `vitriol_cuda_cleanup_vram()` |
| `vitriol-cuda-integration.cpp` | ~658-681 | Print pin stats in `vitriol_cuda_print_stats()` |
| `ggml-cuda.cu` | ~2532-2568 | Scoped src0 redirect in `ggml_cuda_mul_mat_id` |
| `scripts/vitriol` | multiple | Config key, env var, TUI, warnings, auto-disable output cache |

---

## Post-Benchmark Follow-ups

1. If pinning shows clear gains >15%: merge to main, update EXPERIMENT_LOG.md
2. If pinning helps prefill but not decode: consider auto-enabling only during prefill phase
3. If pinning causes OOM: tighten the default to `pin_first_n_layers=5` and add a VRAM check at startup using `cuDeviceGetAttribute`/`cuMemGetInfo`
