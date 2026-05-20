# Plan: Early Exit — Skip Layers via Residual Stagnation

**Problem:** At 10.71 t/s with prune=4 + output cache, we're at ~67% of Pascal's 16 t/s ceiling. The bottleneck is ALU saturation. To go higher, we need to **do less compute**, not make existing compute faster.

**Idea:** Deep layers (21-40) often contribute negligibly to the output — the residual stream barely changes. By detecting this stagnation at runtime, we can skip remaining layers and jump directly to the output head. This can save **50% of compute** (skipping 20 of 40 layers).

---

## How Early Exit Works

In deep transformers, the residual stream evolves rapidly in early layers (building syntax/semantics) and plateaus in later layers (refining vocabulary). The change per layer is measured via:

```
Δ = ||output_layer - input_layer||² / ||input_layer||²  (relative L2 norm)
if Δ < threshold → skip remaining layers
```

Once a layer passes this check for K consecutive layers (e.g., K=3), the residual is forwarded directly to the final norm + lm_head. No matmuls are computed for skipped layers.

### Graph Architecture

```
Input Embedding
  │
  ├─ Layer 0 (compute)
  ├─ Layer 1 (compute)
  ├─ Layer 2 (compute)
  ...
  ├─ Layer N (compute + check Δ)
  │           │
  │           ├─ Δ ≥ threshold → continue to Layer N+1
  │           └─ Δ < threshold → EARLY EXIT → final norm + lm_head
  ...
  └─ Layer 39 (compute, if no early exit)
       │
       └─ final norm + lm_head
```

---

## Implementation Strategy

### Approach: Graph View + Adaptive Exit Point

The key insight: ggml builds a static DAG. If all 40 layers are in the graph, they all execute. To skip layers, we need to **not include them in the executed graph**.

**Phase 1 — Detection:** Add residual delta norm nodes to each layer. Compute Δ on GPU, read result to host. Track stagnation counts.

**Phase 2 — Graph slicing:** After the first full pass through all layers (prefill + first decode token), determine the optimal exit layer. For subsequent tokens, use `ggml_graph_view()` to create a sub-graph that only includes layers 0..N + output head.

**Phase 3 — Adaptation:** If the model's internal state changes (new prompt, context shift), reset the exit point and re-evaluate all layers.

### Implementation Details

#### Detection (in `qwen35moe.cpp` layer loop, line ~199)

```cpp
// After layer output, before assigning to inpL:
ggml_tensor * layer_out = cur;

// Compute residual delta: ||layer_out - inpL||²
ggml_tensor * delta  = ggml_sub(ctx0, layer_out, inpL);
ggml_tensor * delta_norm_sq = ggml_norm(ctx0, delta);  // Σ(delta²) per token
// inpL norm: ||inpL||²
ggml_tensor * inp_norm_sq = ggml_norm(ctx0, inpL);

// Both are [1, n_tokens] — one scalar per token batch
// Need to read back to host for decision
```

At runtime, after each layer:
```
float delta_val = read_tensor(delta_norm_sq);
float inp_val   = read_tensor(inp_norm_sq);
float ratio = delta_val / inp_val;

if (ratio < EARLY_EXIT_THRESHOLD) {
    stagnation_count++;
    if (stagnation_count >= 3) {
        early_exit_layer = il;
        break;
    }
} else {
    stagnation_count = 0;
}
```

#### Graph View (in `graph()` function)

The graph is built once per model load. For early exit:

```cpp
// Build full 40-layer graph as before, but mark layer boundaries
static std::vector<int> s_layer_end_nodes; // node indices where each layer ends

// Store the node index after each layer's residual add
int node_idx = gf->n_nodes - 1; // after final add of layer il
s_layer_end_nodes[il] = node_idx;

// At runtime, compute sub-graph:
int exit_node = s_layer_end_nodes[early_exit_layer];
ggml_cgraph sub_graph = ggml_graph_view(gf, 0, exit_node + 1);
// sub_graph includes:
//   - embeddings
//   - layers 0..early_exit_layer
//   - final norm + lm_head
//   - (but NOT layers early_exit_layer+1..39)
```

This requires that the output head (final norm + lm_head) is connected to the graph OUTSIDE the per-layer loop, so it's included in the view.

#### Backend Scheduler

`ggml_backend_sched_alloc_graph()` and `ggml_backend_sched_graph_compute()` operate on a `ggml_cgraph*`. If `ggml_graph_view()` creates a valid sub-graph reference, the scheduler should process only those nodes. **Risk:** The scheduler may have alignment/identity assumptions about the graph. Requires verification.

### Fallback: Graph Rebuild

If `ggml_graph_view()` doesn't work with the backend scheduler, fall back to **rebuilding the graph** with fewer layers on the fly. The graph builder takes ~1-2ms per call, acceptable if early exit decisions are stable over hundreds of tokens.

```cpp
// In the decode loop:
if (current_batch->exit_layer != last_exit_layer) {
    // Rebuild graph with only (exit_layer + 1) layers
    rebuild_graph_with_n_layers(gf, current_batch->exit_layer + 1);
    last_exit_layer = current_batch->exit_layer;
}
```

---

## Config & Control

| Parameter | Env var | Default | Description |
|---|---|---|---|
| Early exit enable | `VITRIOL_EARLY_EXIT` | 0 (off) | Enable/disable early exit |
| Threshold | `VITRIOL_EARLY_EXIT_THRESHOLD` | 0.001 | Relative L2 norm change below which a layer is "stagnant" |
| Stagnation count | `VITRIOL_EARLY_EXIT_STAGNATION` | 3 | Number of consecutive stagnant layers before exit |
| Min layers | `VITRIOL_EARLY_EXIT_MIN_LAYERS` | 10 | Never exit before this many layers |

---

## Interaction with VITRIOL Features

| Feature | Interaction | Notes |
|---|---|---|
| **Output Cache** (per-expert) | Orthogonal | Cache hits within computed layers still work |
| **Expert Pruning** (4 of 8) | Additive | Prune within each computed layer, + skip entire skipped layers |
| **Expert Pinning** | Neutral | Pinned layers that are skipped — wasted VRAM but no harm |
| **Predictor Prefetch** | Needs update | Skip predictor for early-exited layers |
| **Graph split fix** | Neutral | Same buffer type identity, no change |

Both output cache and pruning operate WITHIN a layer. Early exit operates BETWEEN layers. They stack multiplicatively.

---

## Expected Gain

| Config | Layer count | t/s (estimate) | Notes |
|---|---|---|---|
| Baseline (no opts) | 40 | 9.21 | Full compute |
| Prune 4 + output cache | 40 | 10.71 | Best current |
| + Early exit (exit at layer 25) | 25 | ~13.5 | ~38% fewer layers |
| + Early exit (exit at layer 20) | 20 | ~15.5 | ~50% fewer layers |
| + Early exit (exit at layer 15) | 15 | ~17.5 | ~62% fewer layers |

**Practical estimate:** For code generation tasks, early exit is highly effective (structure-heavy, many repetitive tokens). Expect exit at layer 20-25 → **13-15 t/s** combined with prune+cache.

---

## Implementation Phases

| Phase | Effort | Description |
|---|---|---|
| 1. Residual delta nodes | ~0.5 day | Add `ggml_sub` + `ggml_norm` to graph builder |
| 2. Host-side readback | ~0.5 day | Copy delta scalar from GPU to host, compare |
| 3. Graph slicing | ~1 day | Implement `ggml_graph_view()` or rebuild |
| 4. Scheduler integration | ~1 day | Verify graph views work with CUDA backend |
| 5. Config + TUI | ~0.5 day | Env vars, script options |
| 6. Threshold tuning | ~1 day | Find optimal threshold for coding tasks |
| **Total** | **~4.5 days** | |

---

## Risks

1. **Graph view compatibility** — `ggml_graph_view()` may not work with `ggml_backend_sched`. Fallback: graph rebuild adds ~1-2ms overhead.
2. **Quality degradation** — Early exit changes the model's output distribution. For creative/novel tokens, later layers may be critical. Test with code vs prose.
3. **Threshold sensitivity** — Too aggressive: gibberish. Too conservative: no gain. Needs systematic tuning.
4. **Token type variance** — Function headers need full depth, closing brackets exit early. Adaptive threshold per token type (via tree-sitter AST) is a future improvement.
