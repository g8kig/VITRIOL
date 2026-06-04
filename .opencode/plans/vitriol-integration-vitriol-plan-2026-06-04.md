# VITRIOL — Brief Integration Plan

**Date:** 2026-06-04 08:14 UTC
**Status:** Design complete — awaiting Brief vectorization syntax and SPIR-V backend
**Parent:** VITRIOL × Brief Master Plan (`./vitriol-brief-integration-master-plan-2026-06-04.md`)

---

## 1. Overview

This document covers all changes to the **VITRIOL** codebase (and the `libvitriol` Rust calibration tool) required for the Brief integration. VITRIOL gains:

1. **VPO loader** — load, verify, and select sections from `.vpo` grimoire files
2. **Brief bridge** — FFI linkage to Brief-compiled `liblut_matmul.so` for CPU LUT matmul
3. **SPIR-V loader** — Vulkan compute dispatch for Brief-compiled GPU kernels
4. **Hybrid dispatch** — route each layer to CPU LUT, GPU SPIR-V/CUDA, or fallback
5. **Self-optimization** — profile hot paths, recommend baking passes
6. **PTX loader** (later) — load Brief-compiled `.ptx` via CUDA driver API

---

## 2. VITRIOL Changes

### Phase 2 — `vitriol bake` Tool (libvitriol Rust)

| Step | Detail | Files affected |
|------|--------|----------------|
| **2.1** | Extend GGUF parser with layer analyzer: enumerate all tensors, identify quant type, shape dimensions, MoE expert routing patterns | `libvitriol/src/gguf.rs` |
| **2.2** | Implement `.vpo` v2 multi-section format: read, write, append section, verify checksums | New `libvitriol/src/vpo.rs` |
| **2.3** | **Bake Pass 1:** For each quantized tensor W with activation bits N, compute `LUT[w_idx][act] = dequant(W[w_idx]) * act` for all act ∈ [0, 2^N). Write as new section | New `libvitriol/src/baker.rs` |
| **2.4** | **Bake Pass 2:** KV projection LUTs — context-window-bounded key/value projection precomputation | `libvitriol/src/baker.rs` |
| **2.5** | **Bake Pass 3:** MLP/GLU activation function LUTs (SiLU, GELU — input-bounded → output table) | `libvitriol/src/baker.rs` |
| **2.6** | Incremental append: `vitriol bake --update model.vpo --passes 3` — new section appended, old sections untouched | `libvitriol/src/baker.rs`, `vpo.rs` |
| **2.7** | Configurable activation bit-width: `--act-bits 4` (default), `--act-bits 8` for higher precision | `libvitriol/src/baker.rs` |
| **2.8** | CLI integration: `vitriol bake` subcommand in main CLI | `libvitriol/src/main.rs` |

### Phase 2B — Folding + Pruning

| Step | Detail | Files affected |
|------|--------|----------------|
| **2B.1** | Structural fold: scan all sections for identical `(quant_type, shape, act_bits)` → group into template index, merge instances | `libvitriol/src/baker.rs` |
| **2B.2** | Numeric dedup: within a template, bytewise LUT comparison; identical rows → single row + instance list | `libvitriol/src/baker.rs` |
| **2B.3** | Hardware filter prune: remove sections tagged for unsupported backends | `libvitriol/src/vpo.rs` |
| **2B.4** | Staleness check: re-hash model GGUF, remove sections with mismatched `model_hash` | `libvitriol/src/vpo.rs` |
| **2B.5** | Compress: convert f32 LUT data to f16 or block-quantized format | `libvitriol/src/vpo.rs` |
| **2B.6** | CLI: `vitriol vpo info`, `vitriol vpo fold`, `vitriol vpo prune` subcommands | `libvitriol/src/main.rs` |

### Phase 3 — VITRIOL Runtime Integration

| Step | Detail | Files affected |
|------|--------|----------------|
| **3.1** | VPO loader: open `.vpo`, verify `model_hash` against loaded GGUF (blake3 comparison), iterate section table, select sections matching `hw_requirement` ∩ requested passes | New `vitriol-vpo-loader.cpp` |
| **3.2** | VPO buffer type: allocate page-locked system RAM for LUT data, registered for potential GPU access | `vitriol-buffer.cpp` (extend) |
| **3.3** | Brief bridge: `dlopen("liblut_matmul.so")`, resolve symbol table (init, eval, output_len, stats), call init with `.vpo` path | New `vitriol-brief-bridge.cpp` |
| **3.4** | SPIR-V loader: load `.spv` binary, create Vulkan compute pipeline, allocate descriptor sets + buffers, manage dispatch | New `vitriol-spirv-loader.cpp` |
| **3.5** | Hybrid dispatch in `ggml-cuda.cu`: add `GGML_OP_MUL_MAT_LUT` and `GGML_OP_SPIRV_KERNEL` op cases | `ggml-cuda.cu` |
| **3.6** | Dispatch decision: before each layer, check `layer_id ∈ vpo_sections && activations quantized` → CPU LUT path. Check `SPIR-V kernel exists for this op` → GPU SPIR-V path. Else → existing CUDA path | `ggml-cuda.cu` |
| **3.7** | Model loader hooks: after GGUF model load, auto-discover `.vpo` (same path with `.vpo` extension), load if present | `llama-model-loader.cpp` |
| **3.8** | Asynchronous pipeline: CPU LUT matmul runs on thread pool (separate from GPU stream), results synchronized via CUDA events or explicit sync | `vitriol-brief-bridge.cpp` |
| **3.9** | CLI: `--vpo <path>` flag and `--vpo auto` default behavior | `scripts/vitriol` |

### Phase 4 — Self-Optimization

| Step | Detail | Files affected |
|------|--------|----------------|
| **4.1** | Per-layer profiling: counters for exec_count, latency_ns, pcie_bytes_transferred | New `vitriol-profiler.cpp` |
| **4.2** | Profile export: on graceful shutdown, write `session.json` with per-layer statistics | `vitriol-profiler.cpp` |
| **4.3** | `vitriol bake --recommend --profile session.json`: parse profile, identify hot layers not yet in `.vpo`, suggest pass config | `libvitriol/src/baker.rs` |
| **4.4** | `vitriol bake --update model.vpo --profile session.json`: run reccomended passes, append new sections | `libvitriol/src/baker.rs` |
| **4.5** | Convergence tracking: across sessions, track fraction of layers on LUT path vs CUDA path; report in logs | `vitriol-profiler.cpp` |

### Phase 5 — PTX Loader (NVIDIA Optimization)

| Step | Detail | Files affected |
|------|--------|----------------|
| **5.1** | PTX loader: load Brief-compiled `.ptx` via `cuModuleLoadData`, extract kernel function handles | New `vitriol-ptx-loader.cpp` |
| **5.2** | GPU arch detection: `cudaGetDeviceProperties` → Pascal vs Turing vs Ampere → select dp4a or mma PTX variant | `vitriol-ptx-loader.cpp` |
| **5.3** | Dispatch integration: PTX kernels used alongside existing CUDA kernels for attention/KV ops | `ggml-cuda.cu` |
| **5.4** | Fallback chain: PTX → SPIR-V → CUDA (try each compiled path, use first that loads successfully) | `vitriol-ptx-loader.cpp` |

---

## 3. Hybrid Execution Flow

```
Model Load:
  1. Load GGUF (existing)
  2. If --vpo or auto-detect: load .vpo → verify hash → select sections
  3. If Brief bridge: dlopen liblut_matmul.so → lut_matmul_init(vpo_path)
  4. If SPIR-V: load .spv modules → create Vulkan compute pipelines

Per Layer:
  decision = route_layer(layer_id, input_quant_type)
  if decision == CPU_LUT:
      thread_pool.enqueue([&] {
          lut_matmul_eval(layer_id, input_acts, output, input_len);
      });
  elif decision == GPU_SPIRV:
      spirv_launch(layer_id, input_buf, output_buf, stream);
  else:
      existing_cuda_kernel(layer_id, input_buf, output_buf, stream);

  Synchronization:
      if CPU and GPU paths ran in parallel:
          cuStreamWaitEvent(gpu_stream, cpu_completion_event);
      Continue to next layer.
```

---

## 4. File Formats

### .vpo (Input to VITRIOL Runtime)

The `.vpo` format is defined in the master plan. VITRIOL's loader reads:
- Header: magic, version, model_hash, section_count
- Section table: section_id, pass_id, hw_requirement, offsets, size
- Per-section: layer index + dense LUT data
- Footer: checksum

### session.json (Output from Profiler)

```json
{
  "session": {
    "started_at": "2026-06-04T08:14:00Z",
    "model_hash": "abcd1234...",
    "total_tokens": 1024
  },
  "layers": [
    {
      "layer_id": 5,
      "name": "blk.5.ffn_gate",
      "in_vpo": false,
      "exec_count": 512,
      "latency_ns": 1250000,
      "pcie_bytes": 16777216
    },
    {
      "layer_id": 5,
      "name": "blk.5.attn_k",
      "in_vpo": true,
      "exec_count": 512,
      "latency_ns": 320000,
      "pcie_bytes": 0
    }
  ],
  "recommendation": {
    "suggested_passes": [1],
    "target_layers": ["blk.5.ffn_gate", "blk.5.ffn_down"]
  }
}
```

---

## 5. Build System Changes

### llama.cpp/ggml/CMakeLists.txt

- Add source files: `vitriol-vpo-loader.cpp`, `vitriol-brief-bridge.cpp`, `vitriol-spirv-loader.cpp`, `vitriol-profiler.cpp`
- Optional Vulkan dependency for SPIR-V loader (`find_package(Vulkan)`)
- Optional CUDA driver API dependency for PTX loader

### libvitriol/Cargo.toml

- New dependencies (if needed): `blake3` (already used), `serde`, `serde_json` (for session.json)
- No new major dependencies

### Runtime Dependencies

| Component | Depends On |
|-----------|------------|
| VPO loader | None (pure C++) |
| Brief bridge | `liblut_matmul.so` (distributed alongside) |
| SPIR-V loader | Vulkan loader (`libvulkan.so`) |
| PTX loader | CUDA driver API (`libcuda.so`) |

Optional dependencies: if Vulkan or `liblut_matmul.so` are not present, the corresponding code paths are silently skipped and VITRIOL falls back to pure CUDA.

---

## 6. Error Handling

| Failure mode | Behavior |
|-------------|----------|
| `.vpo` not found | Log warning, run pure CUDA |
| `.vpo` model_hash mismatch | Log error, disable VPO, run pure CUDA |
| `liblut_matmul.so` not found | Log warning, skip CPU LUT path |
| SPIR-V module fails to load | Log error, fall back to CUDA for that op |
| PTX module fails to load | Log error, fall back to SPIR-V → CUDA |
| LUT lookup OOB (act_val > max_act) | Return error code, caller asserts/recomputes via CUDA |

All failures are non-fatal. The GPU fallback always works.

---

## 7. Related Documents

- Master plan: `./vitriol-brief-integration-master-plan-2026-06-04.md`
- Brief-specific plan: `../brief-compiler/plans/2026-06-04-vitriol-integration-brief-plan.md`

---

## 8. Implementation Order

The VITRIOL changes depend on Brief compiler readiness:

| Order | Phase | Wait for |
|-------|-------|----------|
| 1 | 2 (vitriol bake) | No Brief dependency — can be built immediately |
| 2 | 2B (fold+prune) | Phase 2 |
| 3 | 3.1-3.3 (VPO loader + Brief bridge) | Phase 1 (liblut_matmul.so exists) |
| 4 | 3.4-3.9 (SPIR-V loader + hybrid dispatch) | Phase 0 (Brief→SPIR-V works) |
| 5 | 4 (profiler + self-optimization) | Phase 3 |
| 6 | 5 (PTX loader) | Phase 5 (Brief→NVPTX works) |
