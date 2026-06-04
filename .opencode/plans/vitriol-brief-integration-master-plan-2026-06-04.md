# VITRIOL × Brief Integration — Complete Master Plan

**Date:** 2026-06-04 08:14 UTC
**Status:** Design complete — awaiting Brief vectorization syntax (`@`, `<-`, `...`) implementation
**Target Model:** Qwen3.6-35B IQ2_M (extensible to Mellum2 and others)

---

## 1. The Vision

One `.bv` source → three backends (CPU LUT, SPIR-V GPU, NVPTX GPU) → hybrid inference where cold weight matmuls stay in system RAM as memory lookups and only small activation tensors cross PCIe to the GPU.

The `.vpo` (VITRIOL Precomputed Object) is a **grimoire** — a single file accumulated over time through three operations:

| Operation | Effect | Semantics |
|-----------|--------|-----------|
| **Baking** | Additive passes | Precompute LUTs for quantized layers; each pass appends an immutable section |
| **Folding** | Structural compression | Merge sections with identical LUT templates (same quant type, shape, activation bits) |
| **Pruning** | Selective reduction | Dedup superseded sections, filter by hardware target, compress data format, remove stale sections |

Sections are immutable. Pruning and folding create a new `.vpo` — the original is preserved. This makes the grimoire a **transferable, auditable, incrementally-optimized** artifact.

---

## 2. Architecture

```
                    vitriol bake / fold / prune
  GGUF ───────────────────────────────────────────────► .vpo file
                    (Rust, libvitriol/)                    │
                                                           │ multi-section,
                    ┌──────────────────────────────────────┤ immutable,
                    │                                      │ timestamped
                    ▼                                      ▼
           Brief Compiler (.bv → multiple backends)
                    │
          ┌─────────┴────────────────────┐
          │                              │
          ▼                              ▼
   x86_64 (.so)                   spirv64 (.spv)
   CPU LUT matmul                 GPU compute kernels
   (system RAM, DDR4)             (Vulkan, portable)
                                        │
                                   (later) nvptx64 (.ptx)
                                   GPU compute kernels
                                   (NVIDIA CUDA, optimized)
          │                              │
          └──────────┬───────────────────┘
                     ▼
          VITRIOL Hybrid Runtime
          (ggml-cuda.cu dispatch)

          Dispatch logic:
            if layer_id ∈ .vpo AND activations are quantized:
                → Brief CPU LUT matmul (liblut_matmul.so, DDR4, no PCIe xfer)
            elif SPIR-V kernel exists for this op:
                → Brief GPU kernel (Vulkan compute, VRAM)
            else:
                → existing CUDA path (fallback)
```

---

## 3. The Hyperfold

For a quantized LLM layer computing `output = W × X`:

- **W** is a static quantized weight matrix (loaded from GGUF at runtime)
- **X** is an activation vector bounded by quantization (e.g., ∈ [0, 15] for 4-bit)

The product decomposes elementwise: each weight has N possible products where N = 2^(activation_bits). With 4-bit activations, N = 16. Brief's region analysis (`range.rs` → `region.rs`) recognizes this pattern:

1. `range.rs` proves activation bounds from type constraints or preconditions
2. `region.rs` classifies the weight as `Pure` (loaded once, never changes) and the activation as `Bounded`
3. Chain composition fuses the multiply-and-accumulate loop into a fused LUT lookup region
4. The inner `W[i] × X[j]` becomes a single `load` from a precomputed address — no multiply at runtime

The LUT data is populated at model-load time (by the `.vpo` loader), not at Brief compile time. Brief compiles the **structure** of the LUT lookup, not the values.

---

## 4. .VPO File Format (v2)

```
┌──────────────────────────────────────┐
│ magic: "VPO2"                        │
│ version: u32                         │
│ model_hash: blake3 (32 B)            │
│ section_count: u32                   │
├──────────────────────────────────────┤
│ Section Table (n entries):           │
│   section_id: u32                    │
│   pass_id: u32                       │
│   created_at: u64 (unix ms)          │
│   hw_requirement: u8                 │
│     (0=CPU_LUT, 1=SPIRV, 2=PTX)     │
│   layer_index_offset: u64            │
│   lut_data_offset: u64               │
│   data_size: u64                     │
│   data_format: u8                    │
│     (0=f32, 1=f16, 2=quantized)     │
├──────────────────────────────────────┤
│ Per-section layer index:             │
│   layer_id: u32                      │
│   tensor_name_hash: u32              │
│   quant_type: u8                     │
│   shape: [u32; 4]                    │
│   act_bits: u8                       │
│   lut_template_id: u32               │
│   lut_offset_in_section: u64         │
│   ... (repeated per layer)           │
├──────────────────────────────────────┤
│ Per-section LUT data:                │
│   [floatN; data_size]                │
├──────────────────────────────────────┤
│ (Optional) Template index:           │
│   template_id: u32                   │
│   template_hash: blake3              │
│   instance_count: u32                │
│   instance_layer_ids: [u32]          │
├──────────────────────────────────────┤
│ Footer: blake3 checksum (32 B)       │
└──────────────────────────────────────┘
```

---

## 5. Baking Passes

| Pass | What it precomputes | Configuration |
|------|-------------------|---------------|
| 1 | Quantized FFN matmul LUTs (W×X per weight, for all X in [0, 2^act_bits)) | `--passes 1 --act-bits 4` |
| 2 | KV projection LUTs (context-window-bounded key/value projections) | `--passes 2 --ctx-size 65536` |
| 3 | MLP/GLU activation function LUTs (SiLU, GELU — input bounded → output table) | `--passes 3` |
| N | Cross-layer expert reuse patterns (from profile data) | `--profile session.json` |

Incremental: `vitriol bake --update model.vpo --passes 2` appends a new section without touching existing ones.

---

## 6. Folding Passes

Folding detects structural equivalence between sections and merges them:

| Fold mode | What it does |
|-----------|-------------|
| `--structural` | Scan all sections for identical `(quant_type, shape, act_bits)` triple → group into a single template with N instances |
| `--numeric` | Within a template, bytewise-compare LUT rows; identical rows dedup to one |
| `--all` | Both passes |

CLI: `vitriol vpo fold model.vpo --structural --output model.folded.vpo`

---

## 7. Pruning Passes

Pruning creates a new `.vpo` with a subset of sections:

| Prune mode | What it does |
|-----------|-------------|
| `--dedup` | Remove sections superseded by later passes targeting the same layers |
| `--filter-hardware` | Keep only sections matching current hardware (e.g., CPU-only → strip SPIR-V sections) |
| `--staleness-check --model m.gguf` | Re-hash model, remove sections with mismatched `model_hash` |
| `--compress f16` | Convert f32 LUT data to f16 (50% size reduction) |
| `--compress quantized` | Apply block quantization to LUT data (e.g., IQ2-style grid) |

CLI: `vitriol vpo prune model.vpo --dedup --compress f16 --output model.pruned.vpo`

---

## 8. Files to Create

### Brief Compiler (Rust)

| File | Purpose | Phase |
|------|---------|-------|
| `src/backend/spirv.rs` | SPIR-V backend skeleton + implementation notes for future self-contained emitter | 0 |
| `src/backend/llvm.rs` | SPIR-V/NVPTX target triple selection, address space mapping, kernel emission | 0, 5 |
| `src/analysis/address_space.rs` | GPU address space enum extension (`CrossWorkgroup`, `Function`, `Uniform`, `Private`) | 0 |

### libvitriol (Rust)

| File | Purpose | Phase |
|------|---------|-------|
| `libvitriol/src/vpo.rs` | `.vpo` format: read, write, section append, fold, prune | 2, 2B |
| `libvitriol/src/baker.rs` | Pass engine: Pass 1 (matmul LUTs), Pass 2 (KV), Pass 3 (MLP), recommendation | 2 |
| `libvitriol/src/gguf.rs` | Extended layer analyzer — enumerate tensors, identify quant/shape/MoE routing | 2 |
| `libvitriol/src/main.rs` | `bake`, `vpo info/fold/prune` subcommands | 2, 2B |

### VITRIOL (C++)

| File | Purpose | Phase |
|------|---------|-------|
| `vitriol-vpo-loader.cpp` | Load `.vpo`, verify model hash, select matching sections | 3 |
| `vitriol-brief-bridge.cpp` | FFI to Brief-compiled `liblut_matmul.so` | 3 |
| `vitriol-spirv-loader.cpp` | Load `.spv` modules via Vulkan compute pipeline | 3 |
| `vitriol-ptx-loader.cpp` | Load `.ptx` via CUDA driver API | 5 |
| `vitriol-profiler.cpp` | Per-layer profiling, export `session.json` | 4 |
| `ggml-cuda.cu` | `GGML_OP_MUL_MAT_LUT`, `GGML_OP_SPIRV_KERNEL` dispatch cases | 3 |

### Docs

| File | Purpose | Phase |
|------|---------|-------|
| `AGENTS.md` | SPIR-V build requirements, bake workflow, all integration findings | 0+ |

---

## 9. Phases

| Phase | What | Output | Depends On |
|-------|------|--------|------------|
| **0** | Brief SPIR-V backend: target triple, address spaces, kernel emission | `brief --target spirv64` produces `.spv` | Brief vectorization syntax |
| **1** | Brief CPU LUT matmul: `.bv` → x86_64 `.so` via FFI | `liblut_matmul.so` | Brief vectorization syntax |
| **2** | `vitriol bake` CLI: GGUF analysis, Pass 1-3, `.vpo` output | `vitriol bake` command | Phase 1 (interface) |
| **2B** | Folding + pruning: template detection, dedup, compress | `vitriol vpo fold/prune` commands | Phase 2 |
| **3** | VITRIOL integration: VPO loader, SPIR-V loader, hybrid dispatch | `llama-server --vpo` | Phases 0, 1, 2 |
| **4** | Self-optimization: profiling, `bake --recommend`, `bake --update` | Convergent self-adaptive runtime | Phase 3 |
| **5** | NVPTX optimization path: `brief --target nvptx64`, PTX loader | Brief-compiled PTX kernels | Phase 3 |

### Dependency Graph

```
Phase 0 ──→ Phase 3 ──→ Phase 5
   │            ↑
Phase 1 ───────┘
   │
Phase 2 ──→ Phase 3
   │            │
Phase 2B ───────┘
                │
                └──→ Phase 4
```

---

## 10. Timeline (After Vectorization Syntax)

| Month | Phases | Milestone |
|-------|--------|-----------|
| 1 | 0, 1 | `brief --target spirv64` produces `.spv`; `liblut_matmul.so` compiles |
| 2 | 2, 2B | `vitriol bake` produces `.vpo`; `vpo fold` merges templates |
| 3 | 3 | `llama-server --vpo` runs hybrid: Brief LUT on CPU + CUDA on GPU |
| 4 | 4 | Profile-loop converges to 80%+ layers on LUT path |
| 5+ | 5 | NVPTX path matches CUDA kernel performance from same `.bv` source |

---

## 11. Related Documents

- Brief-specific plan: `../../brief-compiler/plans/2026-06-04-vitriol-integration-brief-plan.md`
- VITRIOL-specific plan: `./vitriol-integration-vitriol-plan-2026-06-04.md`

---

## 12. Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| VPO sections | Additive, immutable, timestamped | Preserves audit trail, enables rollback |
| Activation bit-width | Configurable via `--act-bits` CLI flag | Flexible: 4-bit default, 8-bit with larger LUTs |
| GPU codegen | SPIR-V first (LLVM target), then self-contained, then NVPTX | Most portable path first; document everything for future |
| SPIR-V toolchain | LLVM `spirv64-unknown-unknown` target, with `src/backend/spirv.rs` skeleton documenting self-contained path | Build clean; write down knowledge for later |
| Model target | Qwen3.6-35B IQ2_M initially | Already owned, already tested, IQ2 maximizes hyperfold impact |
| Start trigger | After Brief vectorization syntax (`@`, `<-`, `...`) is fully implemented | The matmul `.bv` source needs these features |
| .vpo location | Sidecar file (`.gguf` → `.vpo`); also supports `--vpo` flag | Flexible, no GGUF format modification needed |
