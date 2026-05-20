# Plan: Early Exit — Skipping Layers via Residual Stagnation

**Problem:** Not all 40 layers are needed for every token. A comma, closing bracket, or common keyword like `return` often reaches its final mathematical state by layer 15-20. Computing layers 21-40 is wasted ALU time.

**Idea:** After each layer, measure how much the residual stream changed. If the change is below a threshold, skip the remaining layers and output the current result directly.

## How It Works

In deep transformers, the residual stream evolves rapidly in early layers (building syntax/semantics) and plateaus in later layers (refining vocabulary choices). The change can be measured cheaply via cosine similarity:

```
Δ = cosine(residual[N], residual[N-1])
if Δ > 0.99 → skip layers N+1..40
```

The early-exit head is a small learned projection from the residual to logits — but a simpler "copy last layer's output" can also work for a proof of concept.

## Expected Gain

- If 50% of tokens exit early (e.g., at layer 20): ~50% compute reduction
- Theoretical limit: 16.6 t/s → **33 t/s**
- Well-known technique: "DeeBERT", "PABEE", "Early Exit" papers
- Standard in production models for latency-critical deployments

## Implementation Complexity: High

### Approach A: Simple Stagnation Detector

Monitor the L2 norm of the residual change between layers. If the change is below a threshold for 3 consecutive layers, signal early exit. No training needed.

```cpp
// After each layer in llm_graph_build, check residual delta
float delta = residual_diff(current, previous);
if (delta < early_exit_threshold) {
    stagnation_count++;
    if (stagnation_count >= 3) {
        // Signal: skip remaining layers
        break_early = true;
    }
} else {
    stagnation_count = 0;
}
```

### Approach B: Learned Exit Head

Train a small classifier (1-2 linear layers) on top of each layer N. The classifier predicts "exit" or "continue". This is more accurate but requires training data.

### Implementation Location

Early exit operates at the **graph building** level (in `llama-graph.cpp` or `llama-model.cpp`), not at the CUDA kernel level. The graph for remaining layers simply isn't built or is masked.

## Status

- [ ] Research existing early-exit implementations
- [ ] Add residual delta tracking infrastructure
- [ ] Implement stagnation detector (Approach A)
- [ ] Benchmark quality vs speed for coding tasks
- [ ] (Optional) Train exit head (Approach B)

## Risks

- Quality degradation for tokens that genuinely need all 40 layers
- Logit distribution mismatch if logits are read from a mid-layer
- Complex interaction with KV cache (which layers to cache?)
