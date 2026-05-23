# Checkpoint Cache Miss Investigation & Fix

**Date**: 2026-05-22
**Author**: VITRIOL AI Agent (OpenCode)

## Problem

The server logs show progressively worsening checkpoint reuse across multi-turn requests:

| Task | Tokens | LCP | Restored At | Catch-Up | sim_best |
|------|--------|-----|-------------|----------|----------|
| 0 | 871 | — | — | — | — |
| 2 | 8423 | 3 | none (full reset) | 8423 | — |
| 41 | 9116 | 8492 | 8491 | 624 | 0.932 |
| 186 | 11537 | 9504 | 9504 | 2033 | 0.824 |
| 244 | 13794 | 8419 | 8418 | 5375 | 0.610 |
| 800 | 14804 | 8106 | **7906** | **6897** | 0.548 |

Notice task 800 falls back 200 tokens behind its LCP (8106) because the checkpoint at 8418 has `pos_min(8418) >= n_past(8106)`, and **all intermediate checkpoints** (8491, 8600, 9112, 9504, 11021, 11533, 11684) were erased during task 244's checkpoint restoration.

## Root Causes

### Cause 1: Overly aggressive checkpoint erasure

**File**: `tools/server/server-context.cpp:2662-2673`

After restoring a checkpoint, every checkpoint with `pos_max > pos_next` is erased:

```cpp
if (cur.pos_max > pos_next) {
    it = slot.prompt.checkpoints.erase(it);
}
```

This destroys checkpoints that remain **independently valid** — the serialized KV cache state (`llama_state_seq_get_data_ext`) is a deterministic function of the input tokens. A checkpoint at position X is valid for any future request whose first X tokens match, regardless of what happens at later positions.

The `n_ctx_checkpoints=32` limit in `create_checkpoint` (line 1865) already handles eviction by dropping the oldest checkpoint when capacity is reached — no additional pruning needed.

### Cause 2: Coarse checkpoint granularity

**File**: `common/common.h:598`

```cpp
int32_t checkpoint_every_nt = 8192;
```

With 8192-token intervals, an 8k prompt creates only ~1-2 intermediate checkpoints. The gap between checkpoints can be up to 8192 tokens, causing massive reprocessing when an LCP falls in that gap.

### Cause 3: No checkpoint at LCP boundary

When `slot.prompt.tokens.keep_first(n_past)` truncates the prompt to the LCP boundary (line 2686), no checkpoint is created at this position. The LCP boundary is the most likely landing point for the next request — it should always be checkpointed.

## Changes Made

### Change 1: Remove checkpoint erasure loop

**File**: `tools/server/server-context.cpp`

Removed the loop at lines 2662-2673 that erases checkpoints with `pos_max > pos_next`. Checkpoints now survive restoration and accumulate across requests.

The `n_ctx_checkpoints=32` capacity limit in `create_checkpoint` (line 1865) handles eviction naturally by dropping the oldest checkpoint when full.

### Change 2: Reduce checkpoint_every_nt to 2048

**File**: `common/common.h:598`

Changed default from 8192 to 2048. Creates ~5 intermediate checkpoints per 10k tokens (vs ~1-2). Each checkpoint is ~73 MiB, so 5 = ~365 MiB per 10k prompt — acceptable within the 32-checkpoint max (~2.3 GiB cap).

### Change 3: Checkpoint at LCP boundary after truncation

**File**: `tools/server/server-context.cpp`, after `keep_first(n_past)` (line 2686)

Immediately after truncating the prompt to the LCP boundary, creates a checkpoint at the current position. This ensures the exact restoration point is captured for the next request.

## Expected Improvement

With these changes:
- Checkpoints accumulate across requests (Change 1)
- More intermediate checkpoints exist during prefill (Change 2)
- The LCP boundary is always captured (Change 3)

Task 800 would find a checkpoint much closer to its LCP of 8106 — potentially at exactly 8106 (from Change 3 during task 244) or within 2048 tokens (from Change 2's granularity) — instead of falling back 200+ tokens to 7906.