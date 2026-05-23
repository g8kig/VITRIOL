# VITRIOL Prefill Optimization Plan

**Date**: 2026-05-23
**Author**: VITRIOL AI Agent (OpenCode)

## Status Overview

| Change | Status | Component | Effort |
|--------|--------|-----------|--------|
| Checkpoint erasure removal | ✅ Done & built | `server-context.cpp` | C++ |
| Checkpoint interval 8192→2048 | ✅ Done & built | `common/common.h` | C++ |
| LCP boundary checkpoint | ✅ Done & built | `server-context.cpp` | C++ |
| `--kv-unified` flag | 🚧 Config update | `scripts/vitriol` | Bash |
| `--cache-idle-slots` flag | 🚧 Config update | `scripts/vitriol` | Bash |
| `--n-ubatch 256` flag | 🚧 Config update | `scripts/vitriol` | Bash |
| AST-aware signature compaction | ⏳ Not started | Orchestrator layer | Rust/Py |

---

## 1. C++ Changes (Completed)

### 1.1 Remove Checkpoint Erasure After Restore

**File**: `tools/server/server-context.cpp:2662-2673`

Removed the loop that erased all checkpoints with `pos_max > pos_next` after restoring a checkpoint. The `n_ctx_checkpoints=32` capacity limit in `create_checkpoint` already handles eviction naturally.

**Why**: The serialized KV cache state is a deterministic function of input tokens. A checkpoint at position X is valid for ANY future request whose first X tokens match, regardless of what happens at later positions. Aggressive erasure destroyed intermediate checkpoints that remained perfectly valid, causing massive re-prefill on the next request.

**Without this fix**: Task 800 fell back 200 tokens behind its LCP (to checkpoint at 7906 instead of 8106) because all intermediate checkpoints had been erased during task 244.

### 1.2 Reduce Checkpoint Granularity

**File**: `common/common.h:598`

```
int32_t checkpoint_every_nt = 2048;  // was 8192
```

**Why**: With 8192-token intervals, an 8k prompt creates only 1-2 intermediate checkpoints. With 2048 intervals, ~5 intermediate checkpoints exist. Each checkpoint is ~73 MiB; 5 = ~365 MiB per 10k prompt — acceptable within the 32-checkpoint max (~2.3 GiB cap).

### 1.3 LCP Boundary Checkpoint

**File**: `tools/server/server-context.cpp:*2686`

Immediately after `slot.prompt.tokens.keep_first(n_past)` truncates the prompt to the LCP boundary, creates a checkpoint at the current position.

**Why**: The LCP boundary is the most likely landing point for the next request. Capturing it ensures near-zero catch-up on the next request.

---

## 2. Config Changes (In Progress)

### 2.1 `--kv-unified`

**Flag**: `--kv-unified`
**Status**: 🔧 Needs addition to `scripts/vitriol`

Enable unified KV cache buffer shared across all sequences. This is **required** for `--cache-idle-slots` to function. Without it, the server logs:
```
srv init: --cache-idle-slots requires --kv-unified, disabling
```

### 2.2 `--cache-idle-slots`

**Flag**: `--cache-idle-slots`
**Status**: 🔧 Needs addition to `scripts/vitriol`

When a slot becomes idle (request completes), its full state — including Recurrent State (RS) buffers for SSM/Gated Delta Net layers — is saved to the prompt cache and the KV cache is freed. On the next matching request, the saved state is restored without a full re-prefill.

**This directly mitigates the "hybrid amnesia" issue** where Qwen3.6's Gated Delta Net recurrent state (62.81 MiB RS buffer) would cause `do_reset = true`, forcing full re-prefill.

### 2.3 `--ubatch-size 256`

**Flag**: `--ubatch-size 256`
**Status**: 🔧 Needs addition to `scripts/vitriol`
**Default**: 512

Reduces the physical batch size for prompt processing from 512 to 256. This lowers peak compute buffer memory pressure, which can allow the ggml scheduler to fuse more graph operations (reducing graph splits).

**Expected effect**: The `graph splits = 22` value may drop to ~12-16, reducing CPU-side scheduling overhead per token.

---

## 3. Future Work: AST-Aware Signature Compaction

### Status: ⏳ Not Started

### Concept

Use tree-sitter (or similar AST parser) to process source files before sending them to the model. For files outside the current edit focus, replace function bodies `{ ... }` with just signatures and docstrings:

**Before** (2000 lines, ~10000 tokens for a large file):
```python
def calculate_entropy(data):
    import math
    total = sum(data)
    if total == 0:
        return 0.0
    entropy = 0.0
    for value in data:
        if value > 0:
            p = value / total
            entropy -= p * math.log2(p)
    return entropy
```

**After** (5 lines, ~25 tokens):
```python
def calculate_entropy(data):
    """{docstring}"""
    # {impl stripped}
```

### Expected Impact

- Typical project: 2000 lines → 150 lines of signatures (~800 tokens)
- Prefill time reduction: **~90%** for context-heavy tasks
- No quality loss for code understanding (the model only needs signatures of non-active functions)

### Implementation Path

1. Add tree-sitter as a dependency (Python `tree_sitter` or Rust `tree-sitter`)
2. Parse the project files on project open
3. Before each API call, run the compaction pass:
   - Keep full text of files in current edit focus
   - Strip function bodies of all other files
   - Only include files referenced by imports
4. Send compacted context to `llama-server`

---

## 4. Optional Improvements

### 4.1 Reduce Context Window

**Flag**: `-c 65536`
**Current**: 136192

Halves KV+RS memory from ~1704 MiB to ~852 MiB. Frees ~852 MiB of VRAM for compute buffers and graph operations. With `--kv-unified`, the shared buffer is also smaller.

**Trade-off**: Reduces maximum prompt+generation length from 136K to 65K tokens. For OpenCode usage, this is likely still ample (most code files are 