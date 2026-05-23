# VITRIOL Prefill Optimization & AST Compaction Plan

**Date**: 2026-05-23
**Author**: VITRIOL AI Agent

## Overview

Two-pronged optimization for Qwen3.6-35B-A3B (IQ2_M) on GTX 1070 Ti (Pascal SM 6.1, 8 GB VRAM):

1. **Checkpoint/Caching fixes** — ensure cache reuse across requests, avoid full reprocess
2. **AST-aware prompt compaction** — strip function bodies from code in prompts to drastically reduce token count

---

## Part 1: Prefill Optimization (Done)

| Change | Status | File | Description |
|--------|--------|------|-------------|
| Checkpoint erasure removal | ✅ Done | `server-context.cpp:2662-2673` | Removed loop that erased valid intermediate checkpoints after restore |
| Checkpoint interval 8192→2048 | ✅ Done | `common/common.h:598` | Finer prefill checkpoints |
| LCP boundary checkpoint | ✅ Done | `server-context.cpp:~2686` | Captures checkpoint at LCP boundary for next request |
| `--kv-unified` flag | ✅ Done | Launch scripts | Unified KV cache shared across all sequences |
| `--cache-idle-slots` flag | ✅ Done | Launch scripts | Save/clear idle slot state for reuse |
| `--ubatch-size 256` | ✅ Done | Launch scripts | Reduced from 512 to lower compute buffer pressure |

**Confirmed working** (from server logs):
- `kv_unified = true`, `n_ubatch = 256`, `cache_idle_slots = true`
- Checkpoints at 2048-token intervals
- No `erased invalidated context checkpoint` messages
- LCP hits working with `sim_best = 0.927` for task 30

---

## Part 2: Hang Analysis (From Logs)

The "10s–1m hang" is the **prefill phase** — server processes tokens silently before streaming starts.

### Two distinct cases:

| Task | Tokens | Prefill Time | Per-token | Root Cause |
|------|--------|-------------|-----------|------------|
| 0 | 868 fresh | 9.0s | 10ms/tok | Empty cache, fast first-prompt |
| 2 | **8420 full-reprocess** | **345.9s (5.7 min)** | 41ms/tok | LCP only 3 tokens, no checkpoint with pos_min < 3 |
| 30 | **663 incremental** | **58.0s** | **87ms/tok** | LCP hit, but attention over ~8800 cached positions |

### Root Cause A: No checkpoint covering small n_past

Checkpoint matching condition (line 2636):
```cpp
return cur.pos_min < pos_min_thold || cur.pos_min == 0;
```

For task 2: `n_past=3`, `pos_min_thold=3`. Nearest checkpoint has `pos_min=607`. Neither `< 3` nor `== 0`. **Full reprocess triggered.**

LCP boundary checkpoint (line 2681) requires `n_past >= 64` — so a 3-token LCP never gets saved. Next request hits the same gap.

### Root Cause B: Slow incremental prefill with large cache

Even with LCP hit, 663 new tokens take 58s at 87ms/tok (vs 10ms/tok for fresh prefill). Each ubatch computes attention over the full cached prefix (~8800 positions). On Pascal SM 6.1 with no Tensor Cores, this is memory-bandwidth-bound.

---

## Part 3: Phase 1 — Fix `n_past >= 64` Guard (Immediate)

**File**: `llama.cpp/tools/server/server-context.cpp:2681`
**Change**: `n_past >= 64` → `n_past >= 1`

Ensures even a 3-token LCP gets a checkpoint at the boundary with `pos_min < n_past`, allowing the next request to match via `cur.pos_min < pos_min_thold`.

**Impact**: Saves the full 8420-token reprocess (~346s) when the next request shares a short prefix. Checkpoint cost: ~1 MiB for a few tokens — negligible.

---

## Part 4: Phase 2–4 — AST-Aware Signature Compaction

### Architecture

Server-side C hook using tree-sitter AST parsing. Compacts code blocks in prompts by stripping function/class/method bodies while keeping signatures.

**Target reduction**: 8420 tok → ~1000–2000 tok for code-heavy prompts.

### Phase 2: Build Integration

**2.1 Vendor tree-sitter core**
- Copy `tree-sitter/lib/include/tree-sitter/tree-sitter.h` → `tools/server/treesitter/include/tree-sitter/tree-sitter.h`
- Core library is header-only for the API; the implementation is compiled via a single translation unit
- The tree-sitter C API is MIT-licensed and compatible

**2.2 Add language grammars**
Grammar C source files fetched from each tree-sitter grammar repository:

| Language | Source | Body Node Type | Definition Nodes |
|----------|--------|----------------|------------------|
| Python | `tree-sitter-python` | `block` | `function_definition`, `class_definition` |
| JavaScript | `tree-sitter-javascript` | `statement_block` | `function_declaration`, `class_declaration` |
| TypeScript | `tree-sitter-typescript` | `statement_block` | Same as JS |
| Rust | `tree-sitter-rust` | `block` | `function_item`, `impl_item` |
| Go | `tree-sitter-go` | `block` | `function_declaration` |
| C | `tree-sitter-c` | `compound_statement` | `function_definition` |
| C++ | `tree-sitter-cpp` | `compound_statement` | `function_definition` |
| Java | `tree-sitter-java` | `block` | `method_declaration` |
| Ruby | `tree-sitter-ruby` | `body_statement` | `method`, `class` |
| PHP | `tree-sitter-php` | `compound_statement` | `function_definition` |
| C# | `tree-sitter-c-sharp` | `block` | `method_declaration` |
| Bash | `tree-sitter-bash` | `compound_statement` | `function_definition` |

Storage: Files vendored directly (not submodules):
```
tools/server/treesitter/
├── CMakeLists.txt
├── include/
│   └── tree-sitter/
│       └── tree-sitter.h
├── grammars/
│   ├── python.c
│   ├── python-scanner.c
│   ├── javascript.c
│   └── ...
├── compact.h
└── compact.cpp
```

**2.3 CMake integration**
Add to `tools/server/CMakeLists.txt`:
```cmake
add_library(treesitter STATIC
    treesitter/compact.cpp
    treesitter/grammars/python.c
    treesitter/grammars/python-scanner.c
    treesitter/grammars/javascript.c
    ...
)
target_include_directories(treesitter PUBLIC treesitter/include)
target_compile_options(treesitter PRIVATE -w)  # suppress grammar warnings
target_link_libraries(server-context PUBLIC treesitter)
```

### Phase 3: Compaction Logic

**3.1 API (compact.h)**
```cpp
namespace compact {

struct CompactConfig {
    bool enabled = false;  // gated by --compact-prompt
};

std::string compact_prompt(const std::string & prompt);

} // namespace compact
```

**3.2 Algorithm (compact.cpp)**

1. **Scan** prompt for fenced code blocks: regex ```` ```(\w*)\n(.*?)``` ```` (DOTALL)

2. **For each block**: detect language → parse with tree-sitter (if grammar available) → walk AST → find function/class definitions → replace body nodes with `{ /* ... */ }` or `: ...`

3. **Fallback** for unsupported languages: brace-counting heuristic (strip between first `{` and matching `}` for lines starting with `def `, `function `, `class `, etc.)

4. **Reconstruct** prompt with compacted blocks

**3.3 Key design decisions**

- **Only fenced code blocks** — inline code (single backtick) is too short to benefit
- **Keep docstrings/comments** — they carry semantic information the model needs
- **Replace body, don't delete** — keep `{ /* ... */ }` placeholder to maintain structure
- **Error tolerance** — if tree-sitter parse fails (e.g., syntax error), keep the block as-is or use heuristic fallback

### Phase 4: Integration into Server

**4.1 New CLI flag**

Add to `common_params` struct (`common/common.h`):
```cpp
bool compact_prompt = false;  // enable AST-aware prompt compaction
```

Add CLI arg (`common/arg.cpp`):
```cpp
{"--compact-prompt", "--no-compact-prompt"},
"compact code blocks in prompts using AST parsing (strips function bodies)",
[](common_params & params, bool value) {
    params.compact_prompt = value;
}
```

**4.2 Hook in handle_completions_impl**

In `server-context.cpp:handle_completions_impl()` at line 3433–3445:

```cpp
const auto & prompt = data.at("prompt");

// AST-aware prompt compaction
std::string compacted_str;
if (params.compact_prompt && prompt.is_string()) {
    compacted_str = compact::compact_prompt(prompt.get<std::string>());
    SRV_INF("compacted prompt: %zu chars -> %zu chars\n",
        prompt.get<std::string>().size(), compacted_str.size());
}

// process prompt
std::vector<server_tokens> inputs;
if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
    inputs.push_back(process_mtmd_prompt(ctx_server.mctx,
        compacted_str.empty() ? prompt.get<std::string>() : compacted_str,
        files));
} else {
    json p = compacted_str.empty() ? prompt : json(compacted_str);
    inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, p, true, true);
}
```

Also hook into the infill endpoint (`post_infill`, line 3964) and the completions endpoint (`post_completions`, line 3988).

**4.3 Launch script update**

Add `--compact-prompt` to `scripts/vitriol` server launch commands.

---

## Part 5: Future Optimizations

| Optimization | Effort | Impact | Notes |
|-------------|--------|--------|-------|
| Reduce `-c` to 65536 | 5 min | Frees ~850 MiB VRAM | May allow larger ubatch |
| Increase `--ubatch-size` to 512 | 1 min | Reduces scheduling overhead | Trade-off with VRAM |
| Flash attention tuning | Medium | O(N²) → O(N log N) | Pascal support uncertain |
| AST compaction caching | Medium | Avoid re-parsing unchanged files | Between requests |
| Client-side compaction | Low | Offload work from server | OpenCode feature |

---

## Status Summary

| Item | Status | Phase |
|------|--------|-------|
| Prefill checkpoint fixes | ✅ Done | Part 1 |
| `n_past >= 64` guard fix | 🔧 In Progress | Part 3, Phase 1 |
| Tree-sitter build integration | ⏳ Not Started | Part 4, Phase 2 |
| Compaction logic (server-compact.cpp) | ⏳ Not Started | Part 4, Phase 3 |
| Integration hooks + `--compact-prompt` | ⏳ Not Started | Part 4, Phase 4 |
| Build & test | ⏳ Not Started | |
