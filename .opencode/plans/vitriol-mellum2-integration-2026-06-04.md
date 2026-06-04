# VITRIOL — Mellum2 Integration Plan

**Date:** 2026-06-04 09:20 UTC
**Status:** In progress — cherry-picking Mellum2 support onto VITRIOL vitriol branch
**Upstream:** `ggml-org/llama.cpp` PR #23966, merged June 2 2026 (commit `4fb16eccc`)

---

## 1. Background

Mellum2 is a JetBrains 12B-param MoE model (64 experts, 8 active per token, sliding-window attention) with a **native built-in MTP head** — no separate draft model required. Support was merged into upstream `ggml-org/llama.cpp:master` on June 2, 2026.

VITRIOL's `vitriol` branch is based on an older upstream snapshot and needs the Mellum2 model support cherry-picked in. Total change: 20 files, +344 lines. Only 7 files have potential conflicts with VITRIOL's modifications — all are trivial enum/additions.

### Model Architecture

| Property | Value | VITRIOL Impact |
|----------|-------|----------------|
| Total params | 12B | Entire model fits in 64GB DDR4 — no disk offload needed |
| Active params/token | 2.5B | Light compute footprint — ideal for LUT-based offload |
| Expert count | 64 (8 active) | VITRIOL LRU/predictor handles arbitrary counts |
| Q4_K_M size | ~7.5 GB | Fits in DDR4; pin ~0-4 experts in VRAM |
| Native MTP head | Built-in (no separate draft model) | **Major speedup potential for VITRIOL × Brief** |
| Sliding-window attention | Yes | Standard ops — VITRIOL offloading doesn't touch attention |
| GGUF repos | Smoffyy (pure), CodeFault (instruct), bombman (thinking) | Q4_K_M recommended |

---

## 2. Native MTP — Strategic Implications

Mellum2's MTP head is baked into the base architecture during pretraining. At inference time, the head produces speculative drafts from internal attention states with **zero memory overhead** — no separate draft model consuming VRAM or DDR4.

This is a massive multiplier for the VITRIOL × Brief master plan:

### A. Multi-Token LUT Precomputation (Future Baking Pass)
The CPU LUT path can verify M-drafted tokens in a single memory-bandwidth-bound pass instead of sequentially. Brief's `lut_matmul_eval` can accept `batch_size` as a parameter — MTP verification passes multiple draft tokens in one call, amortizing dispatch overhead.

### B. GPU Native Dispatch (`ggml-cuda.cu`)
Track upstream forks (beellama.cpp) for parallel-drafting pathways specific to Mellum2's MTP head. Integrate into VITRIOL's hybrid dispatch: MTP verification ops routed to the same backend (CPU LUT or GPU) for maximal parallelism.

### C. .vpo Baking Pass Extension
Future baking passes can precompute MTP verification LUTs — mapping draft token candidates → verified output in one lookup. The same hyperfold optimization applies to the MTP head as to the FFN layers.

### D. Profile-Driven MTP Baking
Track which draft positions have the highest acceptance rates. `vitriol bake --recommend` targets those layers first for precomputation.

---

## 3. Cherry-Pick Strategy

```bash
cd llama.cpp
git fetch upstream
git checkout -b vitriol-mellum2

# Cherry-pick the Mellum2 merge commit using mainline parent
git cherry-pick -m 1 4fb16eccc
```

The merge commit `4fb16eccc` has parent 1 pointing to upstream master (pre-Mellum2). Cherry-picking with `-m 1` brings in only the Mellum2 changes.

---

## 4. Conflict Resolution (7 files)

| File | Mellum2 Change | VITRIOL Change | Resolution |
|------|---------------|----------------|------------|
| `src/llama-model.cpp` | +10 lines (Mellum arch case) | MTP sync, ggml-ext removal | Add Mellum case after existing arch cases |
| `src/llama-model.h` | +1 `LLM_ARCH_MELLUM` enum value | MTP runtime additions | Add enum value |
| `src/llama-vocab.cpp` | +4 lines (Mellum pretokenizer) | WebUI settings | Keep both |
| `src/llama-vocab.h` | +1 flag | HIP CI fix | Keep both |
| `src/models/models.h` | +12 lines (Mellum feature flags) | MTP sync | Add new `LLM_ARCH_MELLUM` block after existing |
| `gguf-py/gguf/constants.py` | +19 lines (ARCH enum + keys) | MTP sync | Add Mellum entries |
| `pyproject.toml` | transformers version bump | WebUI version bump | Take Mellum's transformers version |

### New Files (Zero Conflict)

| File | Purpose | Lines |
|------|---------|-------|
| `conversion/mellum.py` | Hugging Face → GGUF converter | 61 |
| `src/models/mellum.cpp` | Mellum2 model implementation | 225 |

---

## 5. VITRIOL-Specific Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| `--pin 0` | No expert pinning initially | 64 experts × ~100MB each exceeds 8GB VRAM |
| `--ngl 99` | Max GPU layers | Attention + shared params stay in VRAM |
| `--mtp 0` | Disable MTP initially | Test baseline first, then test native MTP separately |
| `--ctx 32768` | Conservative context | Ramp up after baseline verified |
| Engine | `vitriol-dma` | Page-locked DMA buffers for expert streaming |
| GGUF | Q4_K_M (~7.5 GB) | Community-confirmed working; fits in DDR4 |

---

## 6. Build & Test Sequence

```
1.  git cherry-pick Mellum2 → resolve 7 conflicts
2.  cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)
3.  sudo vitriol setup                                    # re-set CAP_IPC_LOCK
4.  killall -9 llama-server
5.  vitriol run \
      --model Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf \
      --engine-mode vitriol-dma --ngl 99 --pin 0 \
      --ctx 32768 --mtp 0
6.  Verify: model loads, produces coherent output
7.  Baseline benchmark: record t/s without MTP
8.  Test MTP: --mtp 3, --mtp 5 — measure t/s delta
9.  Incremental pin: --pin 4 → --pin 8 → find VRAM ceiling
10. Profile: export session.json for future baking analysis
```

---

## 7. Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| 7 cherry-pick conflicts | Low | All disjoint additions — enum values, new arch blocks |
| Build failure | Low | Only new model files, zero CUDA code touched |
| Model doesn't load | Low | Community confirmed Q4_K_M works on llama.cpp |
| MoE handler expert mismatch | None | VITRIOL is expert-count-agnostic |
| Native MTP not supported yet | Medium | Disable MTP initially; track upstream for MTP pathways |
| MTP interacts with VPO LUT | Unknown | Test after Phase 2 — could be a multiplier for LUT batch |

---

## 8. Future Extensions to VITRIOL × Brief Master Plan

1. **Baking Pass 2B (MTP Verification LUTs):** Precompute LUTs for MTP verification layers. Structurally identical to Pass 1 matmul LUTs, specialized for MTP head output projection.

2. **Multi-token CPU LUT dispatch:** Brief's `lut_matmul_eval` accepts `batch_size` — MTP verification passes multiple draft tokens in one call.

3. **ggml-cuda.cu MTP-aware routing:** When native MTP is active, hybrid dispatch recognizes MTP verification ops and routes all to same backend for maximal parallelism.

4. **Profile-driven MTP baking:** Track draft position acceptance rates → `vitriol bake --recommend` targets highest-rate layers first.
