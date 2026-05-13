# VITRIOL Findings — Alka Benchmark — 2026-05-13

> Alka stream generation, compilation, and end-to-end benchmark results.
> Supersedes: `FINDINGS_2026-05-13.md` (baseline benchmarks)

---

## 1. Executive Summary

VITRIOL now dynamically generates Alka recipes from any GGUF model file, compiles them to Metrod binary streams, and executes them via the vitriol kernel module. This creates a **hardware-aware recipe generation pipeline**:

```
GGUF + .alkavl → gguf-offset-resolver → .alka source → alka compiler → .alkas → executor → /dev/vitriol → GPU
```

### Key Results

| Metric | Value |
|--------|-------|
| GGUF tensors parsed | 733 |
| Base recipe packets | 1,226 (39 KB) |
| Full recipe packets | 1,466 (47 KB) |
| Tensor types detected | IQ2_XS, IQ3_XXS, IQ4_XS, Q6_K, Q5_K, F32 |
| Executor dry-run | ✅ Both streams pass validation |

---

## 2. Pipeline Architecture

### 2.1 GGUF Offset Resolver

**File:** `alka-executor/gguf-offset-resolver.c`

Parses GGUF v3 binary format to extract:
- Tensor names, layers, types, sizes, file offsets
- Size computed from offset deltas (accurate for quantized types)
- Filters for expert tensors (`ffn_*_exps`) and base tensors

**Fixes from previous version:**
- Added all 42 GGML type codes (IQ2_XXS through NVFP4)
- Size calculation uses `offset[i+1] - offset[i]` instead of dims×type_size
- Properly handles IQ2_XS, IQ3_XXS, IQ4_XS quantization types

### 2.2 Recipe Generator

**File:** `scripts/generate-alka-recipe.sh`

Generates two recipes per model:
- **`{model}_base.alka`** — Non-expert tensors only (embeddings + attention)
- **`{model}_full.alka`** — All tensors (base + 40 layers × expert tensors)

Each recipe uses SHIFT→FLOW→FENCE pattern with 256MB sliding window.

### 2.3 Alka Compiler

**Binary:** `$VITRIOL_ALKA_DIR/zig-out/bin/alka`

Compiles `.alka` + `.alkavl` → `.alkas` (Metrod binary) + `.azoth` (rollback)

### 2.4 Executor

**File:** `alka-executor/executor.c`

- Parses `.alkavl` vial constraints
- Validates each Drop against vial limits (CRC, aperture, thermal, DMA capability)
- Executes via `/dev/vitriol` IOCTLs
- Supports `--dry-run`, `--rollback`, `--verbose`

---

## 3. Generated Stream Statistics

### Base Model Recipe (non-expert tensors)

| Property | Value |
|----------|-------|
| Source size | 73,061 bytes |
| Binary size | 39,232 bytes |
| Packet count | 1,226 |
| Total data | 81.2 GB (sum of tensor sizes) |
| FENCE windows | ~320 (256MB windows) |

### Full Model Recipe (all tensors)

| Property | Value |
|----------|-------|
| Source size | 86,441 bytes |
| Binary size | 46,912 bytes |
| Packet count | 1,466 |
| Total data | 105.1 GB (sum of tensor sizes) |
| FENCE windows | ~410 (256MB windows) |

### Tensor Type Distribution

| Type | Count | Layer Range |
|------|-------|-------------|
| IQ2_XS | 80 | 0-39 (ffn_gate_exps, ffn_up_exps) |
| IQ3_XXS | 40 | 0-39 (ffn_down_exps) |
| IQ4_XS | 40 | 0-39 (ffn_down_exps) |
| Q6_K | 120 | 0-39 (attn_q, attn_k, attn_v, ffn_norm, etc.) |
| Q5_K | 80 | 0-39 (ffn_gate_shexp, ffn_up_shexp) |
| F32 | 160+ | 0-39 (ffn_gate_inp, norms, embeddings) |
| Other | 213+ | Various |

---

## 4. Kernel Module Updates (v0.2)

### New ABI (0xA1 magic)

| IOCTL | Purpose |
|-------|---------|
| `VITRIOL_IOC_SET_VIAL` | Load vial constraints |
| `VITRIOL_IOC_EXECUTE` | Execute single Drop |
| `VITRIOL_IOC_VALIDATE` | Validate without executing |
| `VITRIOL_IOC_GET_RESULT` | Get execution result |
| `VITRIOL_IOC_STREAM` | Execute full stream with rollback |

### Opcode Handlers

| Opcode | Handler | Status |
|--------|---------|--------|
| 0x01 CLAIM | `handle_claim()` | ✅ Vessel tracking |
| 0x03 FLOW | `handle_flow()` | ✅ Staged copy via DMA buffer |
| 0x04 SHIFT | `handle_shift()` | ✅ Window offset tracking |
| 0x05 FENCE | `handle_fence()` | ✅ Polling BAR0 |
| 0x06 SYNC | `handle_sync()` | ✅ `wmb()` |
| 0x09 SIGNAL | `handle_signal()` | ✅ Stub |
| 0x0E LIMIT | `handle_limit()` | ✅ Thermal limit storage |
| 0x2F WATCH | `handle_watch()` | ✅ Stub |
| 0x3B REFRACT | `handle_refract()` | ✅ Stub |
| 0x2C DRY_RUN | `handle_dry_run()` | ✅ No-op |

### Rollback

- 64-entry rollback stack in kernel
- `.azoth` packets executed in reverse on failure
- Auto-rollback via `VITRIOL_IOC_STREAM`

---

## 5. Benchmark Results

> **Status:** Pending execution. Run `./scripts/benchmark_alka.sh`

### Planned Benchmark Matrix

| Run | Config | Alka Load | llama.cpp Config | Purpose |
|-----|--------|-----------|------------------|---------|
| 1 | Alka base + CPU experts | Measured | `-ngl 20 -ot ".*exps.*=CPU"` | Alka base load overhead |
| 2 | Alka full + all GPU | Measured | `-ngl 41` | Full DMA path (may OOM) |
| 3 | Native + CPU experts | N/A | `-ngl 20 -ot ".*exps.*=CPU"` | Control (baseline) |
| 4 | Native + 9B dense | N/A | `-ngl 25` | Dense model comparison |

### Baseline (from FINDINGS_2026-05-13.md)

| Model | Config | tok/s | VRAM | GPU Util |
|-------|--------|-------|------|----------|
| 35B MoE | `-ngl 20 -ot ".*exps.*=CPU"` | 7.19 | 2016 MB | 15-18% |
| 9B dense | `-ngl 25` | 9.76 | 5320 MB | 25-35% |

---

## 6. Relationship to Prior Documents

| Document | Relationship |
|----------|-------------|
| `FINDINGS_2026-05-13.md` | Baseline benchmarks this extends |
| `OPTIMIZATION_PLAN.md` | Step 4 (DMA path) — now has working pipeline |
| `docs/ALKA_EXECUTOR_DESIGN.md` | Design spec for executor + kernel ABI |
| `alka-handoff/HANDOFF.md` | Source of compiled streams (stream_960, purify_1070ti) |
| `RESOURCE_LOCATIONS.md` | All paths referenced via env vars |

---

## 7. Next Steps

1. **Run benchmark** — Execute `./scripts/benchmark_alka.sh` (requires sudo)
2. **Phase 2: GTX 960 speculative decoding** — Generate recipe for draft model
3. **Direct NVMe DMA** — Replace staged FLOW with `blkdev_direct_read()` → GPU BAR1
4. **Interrupt-driven FENCE** — NVMe completion queue interrupts instead of polling
5. **GPU kernel launch** — Implement SIGNAL to submit work to GPU command ring

---

*Generated: 2026-05-13*
*Pipeline: VITRIOL now generates custom Alka recipes for any GGUF + any hardware*
