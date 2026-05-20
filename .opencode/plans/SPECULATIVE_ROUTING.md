# Plan: Speculative Routing (Fate-style Expert Prefetching)

**Goal:** Predict which experts the next layer will activate, then prefetch them via DMA while the GPU is still computing the current layer. Overlaps PCIe latency with GPU math — hard latency hiding.

**Fate's key insight:** The gating network's input for layer N+1 has cosine similarity >0.99 with layer N's input. Meaning: `gate(layer_N_output) ≈ gate(layer_N+1_output)`. So while the GPU computes layer N's FFN, the CPU can run the gate for layer N+1 and prefetch the predicted experts.

**arXiv:** 2502.12224. Tested on **PCIe 3.0 x16** (our bus) with Qwen3-30B-A3B (our model family). Third-party fork: `github.com/ongunm/llama-moe-cache` (~500 lines).

---

## Architecture

```
TIME:  ── layer N attention ──► gate(N) ──► FFN(N) ──► layer N+1 attention ──► ...
                                    │                      ▲
                                    │                      │
                              CPU predicts            Experts already
                              experts for             in VRAM (prefetched
                              layer N+1 ─────────────► while FFN(N) ran)
                                    │
                                    ▼
                              DMA prefetch
                              on background stream
```

---

## Implementation

### Phase 1: Gate State Capture

The gate input is the attention output (hidden state). In `ggml_cuda_mul_mat_id`, this is `src1` (the activation tensor). Capture it at the END of each call:

```cpp
// In vitriol_predictor_update (called at end of ggml_cuda_mul_mat_id):
void vitriol_predictor_update(...) {
    // ... existing code ...
    
    // Capture gate input (hidden state) for next-layer prediction
    // src1 = attention output = gate input for current layer
    // Copy to CPU-accessible buffer
    if (g_vitriol_config.speculative_routing && layer_idx >= 0) {
        cudaMemcpyAsync(g_cpu_gate_input, src1->data,
                        n_embd * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        // (async copy — synced at next layer's prefetch)
    }
}
```

### Phase 2: CPU-Side Gate Prediction

At the START of the next layer's `ggml_cuda_mul_mat_id`:

```cpp
void vitriol_predictor_prefetch(...) {
    // ... existing heuristic fallback ...
    
    if (g_vitriol_config.speculative_routing) {
        // Sync the gate input copy from previous layer
        cudaStreamSynchronize(g_pred_stream);
        
        // Run approximate gate: gate_weights * hidden_state
        // Gate weights are ~100 KB (4096 × 8 logits)
        // Hidden state is ~4096 floats (16 KB)
        // Matmul is trivial (<1 ms on CPU)
        float logits[256];
        matmul_cpu(gate_weights, g_cpu_gate_input, logits,
                   n_experts, n_embd);
        
        // Top-k selection
        int predicted[8];
        top_k(logits, n_experts, 8, predicted);
        
        // Prefetch predicted experts
        for (int i = 0; i < 8; i++) {
            const void *data = (const char*)tensor_base + predicted[i] * expert_size;
            cudaMemcpyAsync(vram_slots[i], data, expert_size,
                          cudaMemcpyHostToDevice, g_dma_stream);
        }
    }
}
```

### Phase 3: Gate Weight Access

The gate weights (`ffn_gate.weight` or `ffn_gate_exps.weight` for the first expert) are tensors loaded by ggml. We need access to them in the VITRIOL layer.

**Option A:** Read during model load. At tensor-load time, find the gate weight tensor and save its host pointer + shape for VITRIOL's use.

**Option B:** Use the existing first-expert weights from the expert tensor. The 0th expert of `ffn_gate_exps.weight` IS the gate weights (or close enough).

**Option C:** Train a lightweight predictor (single linear layer) during calibration. This avoids needing runtime gate weight access entirely.

### Phase 4: Dedicated DMA Stream

Create a dedicated async stream for prefetch operations:

```cpp
static CUstream g_dma_stream = 0;

bool vitriol_ensure_dma_stream() {
    if (g_dma_stream) return true;
    return cuStreamCreate(&g_dma_stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS;
}
```

The prefetch cuMemcpyHtoDAsync goes on `g_dma_stream`. At the point where the GPU needs the data, `cuStreamWaitEvent(compute_stream, g_dma_event, 0)` ensures synchronization.

### Phase 5: Fallback on Prediction Miss

If a predicted expert is wrong, the kernel reads from the host pointer as usual (current behavior). The overhead is zero — same as if no prediction was made. If the prediction is correct, the data is in VRAM and the kernel reads at VRAM bandwidth.

---

## Benefits Over Current Predictor

| Aspect | Current (Heuristic) | Fate-Style (Speculative Routing) |
|--------|--------------------|----------------------------------|
| Prediction basis | Past expert patterns | Actual gating computation |
| Accuracy | ~35-50% | ~97% (per Fate paper) |
| Prefetch timing | Best-effort, in-band | Dedicated async DMA stream |
| PCIe overlap | None (sequential) | Full overlap with GPU compute |
| Requires | Nothing extra | Gate weight access + CPU matmul |

---

## Key Challenges

1. **Gate weight access:** We need the gate weight tensor's data. The model loader assigns expert tensors to the VITRIOL buffer, but the gate weight (a single non-expert tensor) stays in the CPU or GPU buffer. We need to capture its pointer during model loading.

2. **CUDA graph compatibility:** The existing `ggml_cuda_mul_mat_id` fast path uses CUDA graphs. Inserting stream operations (sync, async memcpy) breaks graph capture. This may require graph replay mode detection and fallback.

3. **Hidden state capture:** `src1` in `ggml_cuda_mul_mat_id` is the FFN input (also the gate output). But we need the gate INPUT, which is the attention output. This is `src1` in the CALLER's context — the op that feeds `mul_mat_id`. We'd need to pass it through or track it earlier in the graph.

   **Simplification:** Use the existing cross-layer heuristic (layer N's selected experts ≈ layer N+1's experts). This is what Fate's "adjacent hidden state similarity" reduces to. Our current predictor already does this — it just needs better prefetch plumbing (dedicated DMA stream, async overlap).

---

## Effort & Gain Estimates

| Phase | Effort | Gain |
|-------|--------|------|
| P1: Gate state capture | 2-3 days | Infrastructure |
| P2: CPU gate predictor | 3-5 days | ~97% accuracy |
| P3: Gate weight access | 1-2 days | Enables P2 |
| P4: DMA stream plumbing | 1 day | 0-90% overlap |
| P5: Fallback integration | 1 day | Safety net |

**Total: ~2-3 weeks for full implementation.** The DMA stream plumbing (P4) gives the most benefit even with our current heuristic predictor. A lighter first pass would implement P4 alone:

### Quick Win (3-4 days)

Skip the CPU gate predictor for now. Just add the dedicated DMA stream + async overlap to the existing heuristic predictor. The current predictor submits prefetches on the compute stream inline — moving them to a background stream overlaps PCIe with the IDs copy + sorting at the start of each layer.

**Est. gain:** 5-15% (from overlap of existing prefetch with compute)
**Files changed:** vitriol-cuda-integration.cpp only
