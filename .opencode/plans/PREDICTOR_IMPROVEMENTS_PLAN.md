# Plan: Temporal Predictor Improvements
**Date:** 2026-06-09
**Status:** Proposed

## Objective

Improve expert prefetch prediction accuracy to maximize double-buffer hit rate. Each miss costs a synchronous DMA (~78 µs: 6 µs overhead + 72 µs transfer). At 28 unpinned layers × 8 experts/layer, a 50% hit rate wastes ~12 ms/token on sync DMAs. Targeting ≥90% hit rate.

## Current State

**File:** `llama.cpp/ggml/src/ggml-cuda/vitriol-cuda-integration.cpp:440-603`

Current predictor uses simple union of two heuristics:

1. **Cross-layer** (line 523-526): experts from the previous layer of current token
2. **Temporal** (line 529-530): experts from the same layer of the previous token

Token boundary detection (line 483-493): detects when layer index wraps (39→0).

Per-layer tracking state (lines 452-463):
```cpp
static uintptr_t g_layer_bases[VITRIOL_MAX_LAYERS];   // tensor_base → ordinal
static int       g_n_layers;
static int       g_last_layer;

static int  g_cur_exp[VITRIOL_MAX_LAYERS][256];      // current token's experts per layer
static int  g_cur_cnt[VITRIOL_MAX_LAYERS];
static int  g_prev_exp[VITRIOL_MAX_LAYERS][256];      // previous token's experts per layer
static int  g_prev_cnt[VITRIOL_MAX_LAYERS];
```

**Problem**: The current state only tracks the immediately preceding token. Expert usage patterns in LLMs exhibit longer-range temporal dependencies. For example, if a conversation shifts topic, the expert set changes gradually over 5-10 tokens, not instantly.

## Proposed Improvements

### Improvement 1: Running Frequency Table

Track how often each expert fires per layer across recent N tokens (N=8):

```cpp
struct ExpertFreq {
    uint16_t counts[256];   // hit count, decayed each token
    uint16_t total;          // total decays to compute ratio
};

static ExpertFreq g_expert_freq[VITRIOL_MAX_LAYERS];
static const float DECAY = 0.875f;  // 1/8 decay per token = ~8 token half-life

// In vitriol_predictor_update():
for (int i = 0; i < n_experts; i++) {
    g_expert_freq[layer_idx].counts[expert_ids[i]]++;
}
g_expert_freq[layer_idx].total += n_experts;

// At token boundary (detect_token_boundary):
for (int l = 0; l < g_n_layers; l++) {
    for (int e = 0; e < 256; e++) {
        g_expert_freq[l].counts[e] *= DECAY;
    }
    g_expert_freq[l].total *= DECAY;
}
```

**Prediction**: Include top-K experts by frequency as additional candidates.

### Improvement 2: Expert Affinity Matrix

Track which experts tend to co-occur within the same token. If expert A and B frequently fire together, seeing A suggests B will follow.

```cpp
// Co-occurrence matrix: [256][256] of uint8
// Upper triangular only (symmetric)
static uint8_t g_affinity[256][256];

// In vitriol_predictor_update():
for (int i = 0; i < n_experts; i++) {
    for (int j = i+1; j < n_experts; j++) {
        int a = expert_ids[i], b = expert_ids[j];
        if (g_affinity[a][b] < 255) g_affinity[a][b]++;
    }
}

// Decay at token boundary:
for (int a = 0; a < 256; a++)
    for (int b = a+1; b < 256; b++)
        g_affinity[a][b] = (uint8_t)(g_affinity[a][b] * DECAY);
```

**Prediction**: For each active expert in the current token, add top-3 most-affiliated experts for the next layer.

### Improvement 3: Sequence-Level Hot Set

Some experts are consistently used throughout an entire conversation (they represent common knowledge or frequent patterns). Track a global "hot set" across all tokens:

```cpp
static uint16_t g_global_freq[256];     // total usage across all layers
static uint16_t g_global_total;

// In vitriol_predictor_update():
for (int i = 0; i < n_experts; i++)
    g_global_freq[expert_ids[i]]++;
g_global_total += n_experts;

// At token boundary: decay
for (int e = 0; e < 256; e++)
    g_global_freq[e] *= DECAY;
g_global_total *= DECAY;
```

**Prediction**: Include top-K globally frequent experts as candidates for any layer.

### Improvement 4: Confidence-Weighted Prefetch

Not all predictions are equally reliable. Add a confidence score:

```cpp
struct Prediction {
    int   expert_id;
    float confidence;   // 0.0-1.0
};

// Compute confidence based on:
// - Frequency ratio (count/total) — higher = more reliable
// - Affinity strength (co-occurrence count) — higher = stronger signal
// - Recent accruacy (was this expert correctly predicted last time?)
```

Use confidence to decide:
- **High confidence** (≥0.8): Prefetch aggressively (include in double-buffer)
- **Medium confidence** (0.5-0.8): Prefetch but also prepare fallback
- **Low confidence** (<0.5): Don't prefetch (avoid displacing high-confidence predictions)

### Implementation Plan

| Step | File | Change |
|------|------|--------|
| 1 | `vitriol-cuda-integration.cpp` | Add `ExpertFreq` struct + `g_expert_freq[VITRIOL_MAX_LAYERS]` |
| 2 | same | In `vitriol_predictor_update()`, update frequency table per expert |
| 3 | same | In `detect_token_boundary()`, decay all frequency tables |
| 4 | same | Add `g_affinity[256][256]` matrix + update in `vitriol_predictor_update()` |
| 5 | same | Add `g_global_freq[256]` for sequence-level hot set |
| 6 | same | Extend `vitriol_predictor_prefetch()` to include top-K frequency + affinity candidates |
| 7 | same | Add confidence scoring + filter |
| 8 | `vitriol-cuda-integration.h` | Export stats (hit rate, confidence distribution) |

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `VITRIOL_PREDICT_FREQ_TOPK` | 4 | Number of frequency-based candidates to add |
| `VITRIOL_PREDICT_AFFINITY_TOPK` | 2 | Number of affinity-based candidates per active expert |
| `VITRIOL_PREDICT_GLOBAL_TOPK` | 2 | Number of global hot-set candidates |
| `VITRIOL_PREDICT_CONFIDENCE_THRESHOLD` | 0.6 | Minimum confidence to include prefetch |

### Memory Cost

| Structure | Size |
|-----------|------|
| `g_expert_freq[128][256]` (uint16_t) | 128 × 256 × 2 = 65 KB |
| `g_affinity[256][256]` (uint8_t) | 256 × 256 = 65 KB |
| `g_global_freq[256]` (uint16_t) | 512 B |
| **Total** | **~131 KB** |

Negligible memory cost.

### Expected Impact

On current 12.25 tok/s baseline (direct reads, pin=12), the predictor doesn't matter because there's no LRU. The predictor is only meaningful **with double-buffer DMA** (Plan 1), where it directly determines hit rate.

Estimated hit rate improvement:
| Predictor | Hit Rate* |
|-----------|-----------|
| Current (cross-layer + temporal) | ~60-70% |
| + Frequency table | ~75-85% |
| + Affinity | ~80-90% |
| + Global hot set | ~85-92% |
| + Confidence filtering | ~88-95% |

*Estimates for Qwen3.6-35B; actual hit rate depends on model architecture and task.

A 10-point hit rate improvement saves ~1.2 ms per token in sync-DMA fallbacks.

### Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Frequency decay too fast/slow | Suboptimal hit rate | Make `DECAY` configurable; auto-tune with recent accuracy |
| Affinity matrix O(n²) per token | CPU overhead at high n | Limit to top-256; use uint8 for compactness |
| Too many candidates | DMA queue grows, displaces compute | Confidence threshold caps prefetch count |
