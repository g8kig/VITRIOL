# Plan: Fix Within-Conversation Cache Tax

**Date:** 2026-05-21
**Status:** Draft / Planned

---

## Problem

Every continuation request in a conversation forces partial cache re-processing due to an inherent limitation of the Qwen3 SSM/recurrent memory backend.

### Current Behavior

```
Request arrives with 8490 tokens
  → LCP finds 8103 tokens match (cached from previous turn)
  → Need to discard tokens 8104..8476 and write new tokens 8104..8490
  → llama_memory_recurrent::seq_rm FAILS
     ("models like Mamba or RWKV can't have a state partially erased")
  → Fallback: search checkpoints for nearest save before position 8103
  → Found checkpoint at position 7903 (gap = 200 tokens)
  → Restore checkpoint → re-process 7904..8490 = 586 tokens
  → Should only need 387 tokens (8104..8490)
  → Waste = 199 tokens (varies; avg = interval/2)
```

### Root Cause

The SSM recurrent state is a cumulative hidden state — there is no per-token storage to delete. The `seq_rm` operation fails for partial tail removal because reconstructing the state at position N-1 requires either:
- A saved snapshot at that position, or
- Full recomputation from scratch

**Default checkpoint interval = 8192 tokens** → average waste = 4096 tokens per continuation turn.

### Impact

| Metric | Value |
|--------|-------|
| Prefill speed | ~35 tokens/sec |
| **Avg wasted time per turn** | **~117 seconds** |
| Worst case (checkpoint just created) | ~234 seconds |
| Best case (LCP lands on checkpoint) | ~0 seconds |

---

## Approach A: Denser Checkpoints

**Effort:** 1 line config change
**Risk:** Low (already-supported CLI flag)

### Implementation

Add to server args:
```
--checkpoint-every-n-tokens 2048
```

Also supported via env var `LLAMA_ARG_CHECKPOINT_EVERY_NT=2048`

### Tradeoff Analysis

| Interval | Avg Waste | Saved/Turn | Ckpts at 32K | Mem/Prompt | Prompts in Cache |
|----------|-----------|------------|-------------|------------|-----------------|
| 8192 (current) | 4096 tok | — | 4 × 75 MiB | 300 MiB | ~27 |
| 4096 | 2048 tok | ~58s | 8 × 75 MiB | 600 MiB | ~13 |
| **2048** | **1024 tok** | **~88s** | **16 × 75 MiB** | **1200 MiB** | **~7** |
| 1024 | 512 tok | ~102s | 32 × 75 MiB | 2400 MiB | ~3 |
| 512 | 256 tok | ~110s | capped at 32 | 2400 MiB | ~3 |

**Recommendation:** 2048. Saves ~88s per turn while keeping ~7 prompts cached.

### Combined with Approach B: Max Checkpoints

Add `--ctx-checkpoints 64` to prevent early-eviction at longer contexts. With 2048 interval and 65K+ contexts, 32 slots fill up.

---

## Approach B: Increase Max Checkpoints

**Effort:** 1 line
**Risk:** None

### Implementation

```
--ctx-checkpoints 64
```

This prevents the 32-ckpt cap from evicting early checkpoints in long-context sessions. Alone it provides no benefit (same 8192 interval, just more checkpoints in the same places). **Only useful combined with Approach A.**

---

## Approach C: `--kv-unified`

**Effort:** 1 line
**Risk:** Medium — changes KV cache layout

### Implementation

Add `--kv-unified` to server args.

### What It Does

Collapses the KV cache from per-sequence streams into a shared pool:
- `n_stream` changes from `n_seq_max` (4) → 1
- All sequences share the same KV cell pool
- Enables `--cache-idle-slots` (save/restore slot state)

### SSM Interaction

The recurrent state is **not affected** by `kv-unified`. It always allocates per-sequence (`rs_size = max(1, n_seq_max)`). The hybrid batch split changes from sequential to non-sequential, which may affect GPU utilization.

### Assessment

Currently `n_parallel = 1` (MTP forces this). With one sequence, unified vs partitioned KV cache is a wash. If MTP is ever disabled and `n_parallel > 1`, revisit.

**Recommendation:** Skip for now. Addresses a different problem (cross-slot cache sharing) than the within-turn waste.

---

## Approach D: Increase SSM Rollback Snapshots (`n_rs_seq`)

**Effort:** ~50 lines (add CLI flag, wire through config)
**Risk:** Low (feature already exists, just not exposed)

### What `n_rs_seq` Does

Allocates extra rows in the recurrent state tensor for historical snapshots. When `seq_rm` is called with a small rollback distance, the SSM can point to a saved snapshot instead of failing.

### VRAM Cost

Each snapshot row = 40 layers × (d_conv × d_inner + d_state × d_inner) × f32
= 40 × (4 × 4096 + 128 × 4096) × 4 bytes
= 40 × 540672 × 4
= **~84 MiB per snapshot**

| n_rs_seq | VRAM Cost | Rollback Capacity | Feasible? |
|----------|-----------|-------------------|-----------|
| 0 (current) | 0 MiB | None | ✅ Always |
| 1 | 84 MiB | 1 token | ✅ Fits |
| 4 | 336 MiB | 4 tokens | ⚠️ Tight (302 MiB free after MTP) |
| 8 | 672 MiB | 8 tokens | ❌ OOM with MTP |
| 64 | 5376 MiB | 64 tokens | ❌ Impossible |

### Assessment

Not practical for this problem. The rollback capacity scales linearly with VRAM cost, and even `n_rs_seq=8` (8-token rollback) exceeds available VRAM when MTP is enabled. The average gap is **4096 tokens** — 500× more than what's feasible.

**Recommendation:** Skip. If VRAM headroom ever increases (e.g., TQ1_0 fits entirely), revisit.

---

## Approach E: Exact-Boundary Checkpointing (The Optimal Fix)

**Effort:** ~30 lines modification in `server-context.cpp`
**Risk:** Medium — changes server checkpoint logic

### Concept

Instead of relying on fixed-interval checkpoints, save a checkpoint at the **exact LCP boundary** at the end of every generation. This guarantees the next turn restores to within 1 token of where it should resume.

### The Actual Fix

The end-of-prompt checkpoint code already creates checkpoints at `4 + n_ubatch` and `4` tokens before the end. The gap occurs because generation adds N new tokens after the last checkpoint, and the next request's LCP boundary falls between the last checkpoint and the new end position.

**Fix:** Always create a final checkpoint at the generation end point:

```cpp
// In server-context.cpp, after generation completes:
if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
    const int64_t n_tokens_cur = slot.prompt.n_tokens();
    const llama_pos pos_min = slot.prompt.n_tokens() - 1;
    const llama_pos pos_max = slot.prompt.n_tokens() - 1;
    create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
}
```

### Impact

| Metric | Before | After |
|--------|--------|-------|
| Avg waste per turn | 4096 tokens | ~0 tokens |
| Saved time per turn | — | **~117 seconds** |
| Extra memory per turn | — | 1 checkpoint × ~75 MiB |
| Extra writes to cache | — | 75 MiB serialization |

### Edge Cases

1. **First request** (no prior generation): No checkpoint needed, full prefill required anyway.
2. **Very short generations** (< 4 tokens): End-of-prompt checkpoints already cover this.
3. **Cache eviction on save**: The 75 MiB extra per turn counts against the 8192 MiB prompt cache. At 110 turns, cache fills.

**Recommendation:** Primary target for implementation. Combined with Approach A (2048 interval) as a safety net for edge cases.

---

## Implementation Priority

| Order | Approach | Time Saved | Effort | Risk | Dependencies |
|-------|----------|-----------|--------|------|-------------|
| 1 | **A** (2048 interval) | ~88s/turn | 1 line | None | None |
| 2 | **B** (64 max ckpts) | 0 alone | 1 line | None | Combine with A |
| 3 | **E** (exact ckpt) | ~117s/turn | ~30 lines | Medium | Needs careful testing |
| 4 | **C** (kv-unified) | 0 alone | 1 line | Medium | Revisit if MTP off + n_parallel > 1 |
| 5 | **D** (n_rs_seq) | Negligible | ~50 lines | Low | Impractical unless VRAM freed |

### Quick Win Path (this session)

1. Add `--checkpoint-every-n-tokens 2048` and `--ctx-checkpoints 64` to server args
2. Restart server
3. Verify improvement in next continuation turn

### Optimal Path (next session)

1. Implement Approach E (exact-boundary checkpoint) in `server-context.cpp`
2. Add Approach A/B as safety net with `--checkpoint-every-n-tokens 4096`
3. Benchmark: measure reprocessed tokens before vs after

---

## Appendix: Key Code References

| Component | File | Line(s) |
|-----------|------|---------|
| Checkpoint interval constant | `common/common.h:598` | `checkpoint_every_nt = 8192` |
| Max checkpoints | `common/common.h:597` | `n_ctx_checkpoints = 32` |
| CLI flag (interval) | `common/arg.cpp:1307-1312` | `--checkpoint-every-n-tokens` |
| CLI flag (max ckpts) | `common/arg.cpp:1299-1305` | `--ctx-checkpoints` |
| Interval decision logic | `server-context.cpp:2863` | `>= params_base.checkpoint_every_nt` |
| End-of-prompt offsets | `server-context.cpp:2810-2816` | `{4 + n_ubatch, 4}` |
| Checkpoint creation | `server-context.cpp:1864-1886` | `create_checkpoint()` |
| SSM seq_rm comment | `llama-memory-recurrent.cpp:161-162` | "can't be partially erased" |
| SSM rollback mechanism | `llama-memory-recurrent.cpp:172-180` | `set_rs_idx` bounded rollback |
| `n_rs_seq` default (0) | `llama-context.cpp:3420` | Zero-initialized |
| `--kv-unified` definition | `arg.cpp:1321-1328` / `common.h:546` | `kv_unified = false` |
