# VITRIOL Performance Optimization Plan
**Date:** 2026-05-22 12:00
**Source:** SonarLint/SonarQube static analysis of llama.cpp/VITRIOL codebase

---

## Overview

~20,000 lines of C/C++ across 8 files analyzed. ~40 actionable findings
organized by performance impact, risk, and effort.

**Strategy:**
1. Tier 1 — Safe refactors, apply immediately (~30 min)
2. Tier 2 — Math changes, needs regression testing (~3 hrs)
3. Tier 3 — Structural/architectural, big wins but bigger diffs (~4 hrs)
4. Tier 4 — Cleanup, marginal gains (~3 hrs)

---

## File Inventory

| File | Path | Lines | Scope |
|---|---|---|---|
| ggml-backend.cpp | `ggml/src/` | 2,364 | Backend scheduler + allocator |
| ggml-backend-meta.cpp | `ggml/src/` | 1,920 | Backend metadata + registration |
| gguf.cpp | `ggml/src/` | 1,556 | GGUF format reader/writer |
| ggml-quants.c | `ggml/src/` | 6,256 | Quantization kernels (AVX/SIMD) |
| ggml.c | `ggml/src/` | 7,776 | Core tensor graph engine |
| ggml-vulkan.cpp | `ggml/src/ggml-vulkan/` | ~17k total | Vulkan backend + device struct |
| vitriol-vk-buffer.cpp | `ggml/src/ggml-vulkan/` | 232 | VITRIOL VK buffer type |
| ggml-threading.cpp | `ggml/src/` | 12 | Threading primitives |

---

## Tier 1: High Impact, Safe (Apply Immediately)

**~30 min total.** Safe refactors. No behavioral change, no math change.
Mostly const-correctness, RAII, and modern C++ idioms.

| Rule | File | Line | Current | Fix | Est. Time |
|---|---|---|---|---|---|
| `S1238` | `ggml-backend.cpp` | 89 | `iface` passed by value | `const &` | 5 min |
| `S6030` | `ggml-backend-meta.cpp` | 240, 366 | `emplace(...)` | `try_emplace(...)` | 5 min |
| `S6009` | `gguf.cpp` | 1072, 1305 | `const std::string&` | `std::string_view` | 5 min |
| `S3230` | `ggml-backend-meta.cpp` | — | Body assignment | Init list | 5 min |
| `S1172` | `vitriol-vk-buffer.cpp` | 120, 124, 128 | Unused `buft` param | Remove or `(void)buft` | 2 min |
| `S5506` | `ggml-threading.cpp` | 7, 11 | Manual `lock()/unlock()` | `std::lock_guard` | 2 min |
| `S2259` | `ggml.c` | 1497, 1524 | Potential null deref | Add null check | 3 min |
| `S2259` | `ggml-backend-meta.cpp` | 453 | Potential null deref | Add null check | 2 min |

### Implementation Plan

```
1. vitriol-vk-buffer.cpp  — remove unused params       [2 min]
2. ggml-threading.cpp     — lock_guard RAII             [2 min]
3. ggml-backend.cpp:89    — pass iface by const&        [5 min]
4. ggml-backend-meta.cpp  — emplace→try_emplace + init  [10 min]
5. gguf.cpp               — string→string_view          [5 min]
6. ggml.c + meta.cpp      — null checks                 [5 min]
--- rebuild + regression test ---
```

---

## Tier 2: High Impact, Needs Testing

**~3 hrs.** Math/quantization changes. Any mistake corrupts model output.
Must be tested against known-good output after each change.

| Rule | File | Line | Current | Fix | Est. Time |
|---|---|---|---|---|---|
| `S5276` | `ggml-quants.c` | Hundreds | Implicit int/float conv | Explicit casts | 2 hrs |
| `S5276` | `ggml-opt.cpp` | 390, 425 | Implicit int/float conv | Explicit casts | 15 min |
| `S5276` (double) | `ggml-quants.c` | 6104 | `0.1` → double pollutes SIMD | `0.1f` | 15 min |
| `S5276` (double) | `ggml-opt.cpp` | 690 | double literal in loop | `f` suffix | 5 min |
| `S3630` | `gguf.cpp` | 202, 1296 | `reinterpret_cast` | `std::bit_cast` or `memcpy` | 15 min |
| `S6022` | `gguf.cpp` | 1296 | `char*` byte access | `std::byte*` | 5 min |

### Testing Protocol (per change)
```
1. Apply change to quant kernel
2. Rebuild libggml-cuda.so
3. Run: Qwen3.6 generate 100 tokens
4. Compare output bit-exact with pre-change run
5. If mismatch → revert (math kernel regression)
```

### Key Risk
`ggml-quants.c` contains the SIMD quant/dequant kernels. These are the
most performance-critical files in the entire codebase. The `S5276` fixes
involve adding explicit casts inside hot loops. While the cast instructions
are cheap, an incorrect cast can silently change behavior.

---

## Tier 3: Medium Impact, Structural

**~4 hrs.** Architectural changes. Larger diffs, conceptually simple.

| Rule | File | Line | Current | Fix | Est. Time |
|---|---|---|---|---|---|
| `S107` | `ggml.c`, `ggml-quants.c` | Many | 8-15 params/func | Wrap in struct | 1 hr |
| `S1820` | `ggml-vulkan.cpp` | ~591 | 312-field struct | Group hot/cold fields | 30 min |
| `S1479` | `ggml-backend-meta.cpp` | 791 | 97-case switch | Function pointer table | 1 hr |
| `S1188` | `ggml-backend-meta.cpp` | 760 | 258-line lambda | Split into named fns | 30 min |
| `S1231` | `ggml-backend.cpp` | 493, 1307, 1749 | `realloc` copies data | `vector::reserve` | 30 min |
| `S5025` | `ggml-opt.cpp` | 98, 550, 640 | Raw `new` | `std::make_unique` | 15 min |
| `S5025` | `gguf.cpp` | 368, 403 | Raw `new` | `std::make_unique` | 10 min |

### Biggest Wins

**S107 (Group params into struct):**
Currently functions like `ggml_mul_mat` take 10-15 individual parameters.
On x86_64, only the first 6 fit in registers; the rest go on the stack.
Wrapping into a single `struct ggml_op_params` pointer uses 1 register.
This is the single biggest CPU-side optimization available.

**S1820 (Struct layout):**
`vk_device_struct` has 312 fields. Hot fields (used every dispatch) and
cold fields (configuration, init-only) are interleaved. Grouping hot
fields at the top keeps them in L1 cache. Cold fields at the bottom
never pollute the cache during inference.

**S1479 (97-case switch → function pointer table):**
```cpp
// Before (jump table, ~97 entries):
switch (op) {
    case GGML_OP_MUL_MAT: ...
    case GGML_OP_MUL_MAT_ID: ...
    // ... 95 more
}

// After (function pointer array, O(1)):
static const op_fn op_table[GGML_OP_COUNT] = {
    [GGML_OP_MUL_MAT] = &ggml_compute_forward_mul_mat,
    [GGML_OP_MUL_MAT_ID] = &ggml_compute_forward_mul_mat_id,
    // ...
};
op_table[op](params);
```

---

## Tier 4: Lower Impact, Cleanup

**~3 hrs.** Marginal performance gains individually, but cumulative benefit.

| Rule | File | Lines | Fix | Est. Time |
|---|---|---|---|---|
| `S5566` | `gguf.cpp`, `ggml-backend.cpp` | 715, 814, 971, 1170 | Range-based loops | 30 min |
| `S6005` | `ggml-backend-meta.cpp` | 240, 366 | Structured bindings | 10 min |
| `S6004` | `ggml-backend.cpp`, `ggml-quants.c` | 207, 1694, 1624 | Declare in `if` | 20 min |
| `S3358` | `ggml-quants.c` | 605, 3783, 5080 | Flatten nested ternaries | 30 min |
| `S1709` | `gguf.cpp`, `ggml-backend-meta.cpp` | 231, 1393, 1431, 253, 1411 | `explicit` constructors | 15 min |
| `S923` | `ggml.c` | 252, 306, 1911 | Replace variadic fns | 20 min |
| `S1301` | `ggml.c` | 6825 | Small switch → `if` | 5 min |
| `S1066` | `ggml-backend.cpp` | 509, 1329, 1889 | Merge nested `if`s | 10 min |
| `S1235` | `gguf.cpp` | 1286 | Virtual destructor | 5 min |
| `S3574` | `ggml-backend-meta.cpp` | 456, 476, 500 | Remove redundant return types | 10 min |
| `S1121` | `ggml-backend-meta.cpp`, `ggml-backend.cpp` | 599, 1740, 1344, 1364 | Extract assignments | 10 min |
| `S1144` | `ggml-alloc.c`, `ggml-impl.h` | 105, 73, 88 | Remove unused fns | 10 min |
| `S5028` | Various | — | Macro → const/enum | 15 min |

---

## DO NOT APPLY

These warnings MUST be ignored. Applying them will degrade performance.

| Rule | Occurrences | Why Dangerous |
|---|---|---|
| **`S1836` (restrict)** | ~150 | **Core to SIMD.** `restrict` promises non-aliasing pointers, enabling AVX vectorization. Removing it drops quant kernel throughput 20-50%. |
| **`S5205` (function ptr → `std::function`)** | ~5 | `std::function` uses heap allocation + indirect call through vtable. In hot dispatch paths, this defeats branch prediction and inlining. |
| **`S5945` (C array → `std::vector`)** | ~5 | `std::vector` adds bounds checking in debug + heap allocation. Stack-allocated C arrays are zero-overhead in tight loops. |

---

## Files That Are Upstream (llama.cpp)

These are contributed to the main llama.cpp repository:

- `ggml/src/ggml-backend.cpp`
- `ggml/src/ggml-backend-meta.cpp`
- `ggml/src/gguf.cpp`
- `ggml/src/ggml-quants.c`
- `ggml/src/ggml.c`
- `ggml/src/ggml-threading.cpp`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

Changes to these files should be submitted upstream when possible.

## Files That Are VITRIOL-Specific

These are our custom additions:

- `ggml/src/ggml-cuda/vitriol-buffer.cpp`
- `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`
- `ggml/src/ggml-vulkan/vitriol-vk-buffer.cpp`

Changes to these are VITRIOL-only, no upstream submission needed.
