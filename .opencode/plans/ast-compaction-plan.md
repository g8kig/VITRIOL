# AST-Aware Signature Compaction Plan

**Date**: 2026-05-23
**Author**: VITRIOL AI Agent

## Overview

Add server-side prompt compaction via tree-sitter AST parsing. Strips function/class/method bodies from code blocks in prompts, keeping only signatures. This directly reduces prompt token count by 60–90% for code-heavy conversations, dramatically cutting prefill time.

**Target savings**: 8420 tok → ~1000–2000 tok for code prompts (task 2 type). Full reprocess drops from 346s → ~40–80s.

## Phase 1: Fix the `n_past >= 64` Guard (1 line, immediate)

**File**: `llama.cpp/tools/server/server-context.cpp:2681`
**Change**: `n_past >= 64` → `n_past >= 1`

Ensures even a short (3 token) LCP gets a checkpoint at the boundary. Prevents worst-case full reprocess cascades.

## Phase 2: Tree-sitter Build Integration

### 2.1 Vendor tree-sitter core library

- Copy `tree-sitter/lib/src/lib.c` and `tree-sitter/lib/include/tree-sitter/tree-sitter.h` into a new directory: `tools/server/treesitter/`
- Add `tools/server/treesitter/treesitter.c` (the amalgamated core) to `server-context` static library in `tools/server/CMakeLists.txt`

### 2.2 Add language grammars

Each grammar is a ~200KB–2MB C file that exports `tree_sitter_LANGUAGE()`. Start with a focused set, expand later:

| Grammar | C source | Notes |
|---------|----------|-------|
| Python | `tree-sitter-python/src/parser.c` | Most common for AI code tasks |
| JavaScript | `tree-sitter-javascript/src/parser.c` | |
| TypeScript | `tree-sitter-typescript/typescript/src/parser.c` | |
| Rust | `tree-sitter-rust/src/parser.c` | |
| Go | `tree-sitter-go/src/parser.c` | |
| C | `tree-sitter-c/src/parser.c` | |
| C++ | `tree-sitter-cpp/src/parser.c` | |
| Java | `tree-sitter-java/src/parser.c` | |
| Ruby | `tree-sitter-ruby/src/parser.c` | |
| PHP | `tree-sitter-php/src/parser.c` | |
| C# | `tree-sitter-c-sharp/src/parser.c` | |
| Bash | `tree-sitter-bash/src/parser.c` | |

**Storage**: Vendor grammar `.c` files directly (not as submodules) to avoid git dependency chain. Directory layout:
```
tools/server/treesitter/
├── CMakeLists.txt
├── treesitter.c          # amalgamated core
├── include/
│   └── tree-sitter/
│       └── tree-sitter.h
├── grammars/
│   ├── python.c
│   ├── javascript.c
│   ├── typescript.c
│   ├── rust.c
│   ├── go.c
│   └── ...
└── compact.cpp           # wrapper
└── compact.h
```

### 2.3 CMake integration

Add to `tools/server/CMakeLists.txt`:
```cmake
add_library(treesitter STATIC
    treesitter/treesitter.c
    treesitter/grammars/python.c
    treesitter/grammars/javascript.c
    # ... etc
)
target_link_libraries(treesitter PUBLIC)
target_include_directories(treesitter PUBLIC treesitter/include)

target_link_libraries(server-context PUBLIC treesitter)
```

**Build-time concern**: Compiling all grammar `.c` files adds ~30MB of translation units. Use `-Os` for grammar files and mark them with `OBJECT_DEPENDS` to avoid full rebuilds.

## Phase 3: Compaction Logic (C++ Wrapper)

### 3.1 New files: `server-compact.cpp`, `server-compact.h`

#### API:
```cpp
// compact.h
namespace compact {

// Main entry point: compact code blocks in a prompt string.
// Returns compacted prompt with function bodies replaced by "// ..."
// For unsupported languages, falls back to brace-counting heuristic.
std::string compact_prompt(const std::string & prompt);

// Per-code-block compaction
std::string compact_code_block(const std::string & code, const std::string & lang);

// Register tree-sitter language parsers
void init_parsers();

} // namespace compact
```

#### Algorithm:

1. **Scan prompt** for fenced code blocks: regex ```` ```(\w*)\n(.*?)``` ```` (with `s` flag)

2. **For each code block**:
   - Detect language from fence marker
   - If tree-sitter grammar available:
     - Parse with tree-sitter
     - Walk AST, find `function_definition`, `method_definition`, `class_definition`, `arrow_function` nodes
     - Replace each body (up to matching closing brace) with `{ /* ... */ }`
   - If no grammar available:
     - Fallback heuristic: count braces, strip body between first `{` and matching `}` for lines starting with `def `, `function `, `class `, etc.

3. **Reconstruct prompt** with compacted blocks

4. **Return** compacted string

### 3.2 Language Detection Table

Map from code fence markers to tree-sitter parsers:

```cpp
struct LanguageEntry {
    const char * name;       // "python"
    std::vector<const char *> aliases; // {"py", "python3", ...}
    TSLanguage * (*parser)(); // tree_sitter_python
    std::vector<const char *> body_node_types; // {"block", "suite", ...}
};
```

The `body_node_types` vector tells the compactor which AST node types represent "bodies" to strip. This varies by language:
- Python: `block` (suite)
- JS/TS: `statement_block` 
- Rust: `block`
- C/C++: `compound_statement`
- Go: `block`

### 3.3 AST Node Type Mapping (per language)

| Language | Definition Nodes | Body Node | Strip Strategy |
|----------|-----------------|-----------|----------------|
| Python | `function_definition`, `class_definition`, `decorated_definition` | `block` | Replace `block` with `: ...` |
| JavaScript | `function_declaration`, `method_definition`, `arrow_function`, `class_declaration` | `statement_block` | Replace body with `{ ... }` |
| TypeScript | Same as JS | `statement_block` | Same as JS |
| Rust | `function_item`, `struct_item`, `impl_item`, `trait_item` | `block` | Replace body with `{ ... }` |
| Go | `function_declaration`, `method_declaration` | `block` | Replace body with `{ ... }` |
| C/C++ | `function_definition`, `class_specifier`, `struct_specifier` | `compound_statement` | Replace body with `{ ... }` |

### 3.4 Integration into server-context.cpp

**Insertion point**: In `update_slots()` after the prompt is tokenized but before it enters the processing pipeline. Specifically, compact the raw text from `slot.task->params.prompt` before tokenization.

Locate where the prompt string is first available (near line 2460 in `update_slots()`). After the prompt text is obtained, call:

```cpp
if (compact_prompts) {
    slot.task->params.prompt = compact::compact_prompt(slot.task->params.prompt);
    // retokenize after compaction
}
```

Guard with a new server parameter `--compact-prompt` (bool, default false) toggled by a CLI flag.

## Phase 4: Testing & Verification

1. **Unit tests** for the compaction logic:
   - Code block with function → stripped body
   - Code block with class → stripped methods
   - Nested functions → inner bodies also stripped
   - Unsupported language → heuristic fallback
   - No code blocks → no change
   - Malformed code (unbalanced braces) → graceful handling

2. **Integration test**:
   - Start server with `--compact-prompt`
   - Send a code-heavy prompt
   - Verify response time improves
   - Verify output quality (model can still understand signatures)

## Effort Estimate

| Phase | Description | Lines | Effort |
|-------|-------------|-------|--------|
| 1 | `n_past >= 64` → `>= 1` | 1 line | 5 min |
| 2 | tree-sitter build integration | ~100 lines CMake + files | 2-3 hours |
| 3a | Compaction logic core | ~300 lines C++ | 3-4 hours |
| 3b | Integration hook in server | ~20 lines C++ | 30 min |
| 4 | Testing | ~200 lines tests | 2 hours |
| **Total** | | | **~8-10 hours** |

## Future Expansions

- Runtime grammar loading (load `.so` files for grammars instead of compile-time linking)
- Incremental compaction (cache AST between requests for the unchanged prefix)
- Client-side compaction in OpenCode before sending
