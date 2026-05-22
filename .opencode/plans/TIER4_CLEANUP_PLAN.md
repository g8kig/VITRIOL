# VITRIOL Performance — Tier 4 Cleanup Plan
**Date:** 2026-05-22 13:00

---

## Files & Effort

| File | Items | Est. Time |
|---|---|---|
| `ggml-backend.cpp` | 5 fixes | 30 min |
| `ggml-backend-meta.cpp` | 4 fixes | 20 min |
| `ggml-quants.c` | 3 fixes | 15 min |
| `gguf.cpp` | 3 fixes | 15 min |
| `ggml.c` | 2 fixes | 15 min |
| `ggml-alloc.c` + `ggml-impl.h` | 1 fix | 5 min |
| **Total** | **18 fixes** | **~90 min** |

---

## By File

### ggml-backend.cpp (5 fixes, ~30 min)

| Rule | Line | Current | Fix | Impact |
|---|---|---|---|---|
| `S6004` | 207 | `size_t sz = ...; if (sz > 0)` | `if (size_t sz = ...; sz > 0)` | Low — frees register earlier |
| `S6004` | 1694 | var declared outside if | declare inside if | Low |
| `S1066` | 509 | `if (a) { if (b) { } }` | `if (a && b) { }` | Low — fewer branches |
| `S1066` | 1329 | nested if | merge | Low |
| `S1066` | 1889 | nested if | merge | Low |
| `S1121` | 1344, 1364 | `if ((x = get()) > 0)` | `x = get(); if (x > 0)` | Low — cleaner CFG |

### ggml-backend-meta.cpp (4 fixes, ~20 min)

| Rule | Line | Current | Fix | Impact |
|---|---|---|---|---|
| `S3574` | 456, 476, 500 | `[&](...) -> bool { ... }` | `[&](...) { ... }` | Low — enables RVO |
| `S1121` | 599, 1740 | assignment in expression | extract to own line | Low |
| `S1709` | 253, 1411 | `MyClass(...)` | `explicit MyClass(...)` | Medium — stops implicit conversions |

### ggml-quants.c (3 fixes, ~15 min)

| Rule | Line | Current | Fix | Impact |
|---|---|---|---|---|
| `S3358` | 605, 3783, 5080 | `a ? b : c ? d : e` | `if-else` or `fminf`/`fmaxf` | Medium — helps branchless code gen |
| `S6004` | 1624 | var outside if | declare in if | Low |

### gguf.cpp (3 fixes, ~15 min)

| Rule | Line | Current | Fix | Impact |
|---|---|---|---|---|
| `S5566` | 715, 814 | `for (size_t i = 0; ...)` | `for (auto & item : container)` | Medium — helps auto-vectorization |
| `S1709` | 231, 1393, 1431 | constructors without explicit | add explicit | Medium |
| `S1235` | 1286 | virtual fns, non-virtual dtor | virtual destructor | Low — prevents memory leak |

### ggml.c (2 fixes, ~15 min)

| Rule | Line | Current | Fix | Impact |
|---|---|---|---|---|
| `S923` | 252, 306, 1911 | variadic `...` | template or overloaded fns | Low — args stay in registers |
| `S1301` | 6825 | switch with 1-2 cases | `if` statement | Low — smaller code gen |

### ggml-alloc.c + ggml-impl.h (1 fix, ~5 min)

| Rule | Line | Current | Fix | Impact |
|---|---|---|---|---|
| `S1144` | 105, 73, 88 | unused functions | remove | Low — smaller binary |
