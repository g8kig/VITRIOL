# Plan: Top-K Expert Pruning (Compute Halving)

**Problem:** Each layer computes 8 active experts in the MoE FFN. Many of the bottom-ranked experts contribute negligibly to the output (the router's softmax heavily skews toward the top 1-2 experts).

**Idea:** After the router generates expert IDs, drop the bottom N experts and only compute the top (8-N). For coding/syntax-heavy tasks where most tokens are predictable, this can halve compute with minimal quality impact.

## How It Works

In `ggml_cuda_mul_mat_id`, after the IDs tensor is copied to host and sorted (lines 2600-2617):

```
Current:        8 experts × 2 matmuls per layer = 16 matmuls
VITRIOL_PRUNE=4:  4 experts × 2 matmuls per layer = 8 matmuls  (-50%)
```

Implementation: zero out `tokens_per_expert[i02]` for the bottom-N experts before the main loop.

## Expected Gain

- -50% matmul time per layer: 1.5 ms → 0.75 ms
- Theoretical speed limit: 16.6 t/s → **33 t/s**
- Quality impact should be small for:
  - Code generation (structurally constrained outputs)
  - Syntax completion
  - Short-form responses
- Quality risk for:
  - Creative writing
  - Complex reasoning chains

## Implementation

### C/C++ Changes

**Env var:** `VITRIOL_PRUNE_EXPERTS=N` (0 = off, 1-7 = drop bottom N)

**In `ggml_cuda_mul_mat_id`:**

After `tokens_per_expert[]` is built (line 2617), if PRUNE > 0:

```cpp
int vitriol_prune = g_vitriol_config.prune_experts;
if (vitriol_prune > 0 && vitriol_prune < n_expert_used) {
    // Move tokens from bottom experts to top, zero out bottom
    int keep = n_expert_used - vitriol_prune;
    // IDs are sorted by expert index, but we want to zero the
    // least-used experts. Simpler approach: keep experts that
    // already contribute the most tokens.
    // For single-token decode (ne12=1), each expert has 1 token max,
    // so we just keep the first `keep` experts that have tokens.
    int kept = 0;
    for (int i = 0; i < ne02 && kept < keep; i++) {
        if (tokens_per_expert[i] > 0) kept++;
    }
    for (; i < ne02; i++) {
        tokens_per_expert[i] = 0;  // drop this expert
    }
}
```

No kernel changes needed — just manipulate the `tokens_per_expert` array.

### Shell Script Changes

- New config key: `vitriol.prune_experts` (0-7, default 0)
- Env var: `VITRIOL_PRUNE_EXPERTS`
- CLI flag: `--prune-experts N`
- TUI: Model Settings → option with warning
- Warning: "Experimental: drops bottom N of 8 active experts. May affect quality."

## Benchmark Plan

| # | Prune | Pin layers | t/s | Notes |
|---|---|---|---|---|
| 0 | 0 | 0 | ~8.24 | Baseline |
| 1 | 2 | 0 | ? | -25% compute |
| 2 | 4 | 0 | ? | -50% compute |
| 3 | 4 | 15 | ? | Combined |
| 4 | 2 | 20 | ? | Light prune + heavy pin |

## Status

- [ ] C: Add `prune_experts` to config struct
- [ ] C: Env var reading
- [ ] C: Prune logic in `ggml_cuda_mul_mat_id`
- [ ] Shell: Config key, TUI, env passthrough
- [ ] Benchmark
