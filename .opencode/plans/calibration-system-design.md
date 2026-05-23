# Calibration System Design

**Date:** 2026-05-23  
**Status:** Implementation in progress (Step 1)

## Decision Record

1. **Python** for sweep controller (Step 2+), bash for hardware/model probe (Step 1)
2. **`--quick` first** for immediate value, sweep builds on same data
3. **`POST /props` C++ patch deferred** until sweep controller is proven
4. **3 prompts × 150 tokens** per config for sweep benchmarking

## Overview

```
vitriol calibrate [--quick] [--model <path>] [--profile <name>]
```

Phases:
1. **Hardware probe** — detect GPU, PCIe, CPU → `hardware.json`
2. **Model analysis** — parse GGUF metadata → `<model_hash>/model.json`
3. **VRAM estimator** — math-only bounds → `<model_hash>/bounds.json`
4. **Benchmark sweep** (Python) — auto restart + benchmark → `<model_hash>/sweep.json`
5. **Recommendation** — parse results, save profile → `<model_hash>/best.json`

Files go in `~/.vitriol/calibration/`.

---

## Phase 1: Hardware Probe

**Implementation:** bash functions in `scripts/vitriol`.  
**One-time.** Results cached in `~/.vitriol/calibration/hardware.json`.

### `calibrate_probe_hardware()`

Reads from `nvidia-smi`, `/proc/cpuinfo`, `getcap`.

```json
{
  "probed_at": "2026-05-23 14:00",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce GTX 1070 Ti",
      "vram_mib": 8112,
      "compute_cap": "6.1",
      "pcie_gen": 3,
      "pcie_width": 16,
      "pcie_link_mt_s": 0
    }
  ],
  "cpu": "Intel(R) Core(TM) i7-4790K",
  "has_avx2": false,
  "ram_mib": 32178,
  "has_ipc_lock": true,
  "note": "CC 6.1: max 2^31-1 shared mem per block, no async DMA engine"
}
```

`pcie_link_mt_s` is measured via a quick `cudaMemcpy` test (compile and run a tiny CUDA program once, cache the binary). On error, fills as `0` (unknown).

### Edge cases

- **No NVIDIA GPU**: `vitriol calibrate` warns "no CUDA device found" and suggests `VITRIOL_ENGINE_MODE=native` (CPU). Sweep not possible.
- **CAP_IPC_LOCK missing**: warns "run `vitriol setup` first", skips sweep.
- **Multiple GPUs**: lists all, uses first by default, `--gpu N` flag.
- **No nvidia-smi**: graceful fallback, limited probe.

---

## Phase 2: Model Analysis

**Implementation:** bash function in `scripts/vitriol`.  
**One-time per model** (keyed by SHA256 of first 1 MiB of GGUF).

### `calibrate_analyze_model()`

Parses GGUF metadata via `llama-gguf-tool --print-info` if available, or falls back to `od`/`xxd` hex parsing of the GGUF header (the header is simple: magic + version + tensor_count + metadata_kv). We already did this manually in earlier sessions.

```json
{
  "model_hash": "a1b2c3d4...",
  "model_path": "/home/randozart/Desktop/Projects/Qwen3.6-35B-A3B-UD-IQ2_M.gguf",
  "file_size_gb": 11,
  "architecture": "qwen3",
  "vocab_size": 152064,
  "context_length": 131072,
  "embedding_length": 2048,
  "block_count": 40,
  "head_count": 16,
  "head_dim": 128,
  "expert_count": 256,
  "expert_per_layer": 8,
  "has_mtp": true,
  "tensors": {
    "count": 753,
    "partial_load": true,
    "per_layer_mib_attn": 11,
    "per_layer_mib_ffn_experts": 120,
    "total_model_mib": 10240
  }
}
```

### Tensor → VRAM mapping

Each transformer layer has:
- Attention weights: ~11 MiB (QKV + output projection)
- MLP (MoE): ~120 MiB (up + gate + down for 8 of 256 experts, IQ2_M quantized)
- Total per layer: ~131 MiB

With VITRIOL, attention weights are loaded to GPU always (`-ngl 99`). Expert weights are in page-locked host RAM, fetched via PCIe DMA on demand. Pin-first-N-layers copies expert weights for layers 0..N-1 into VRAM at load time.

Per-layer pin cost: ~120 MiB (expert weights only; attention already in VRAM).

40 layers × 120 MiB = 4,800 MiB to pin all layers. With 8 GiB VRAM, max pin = ~20 before KV cache runs out.

---

## Phase 3: VRAM Estimator

**Implementation:** bash function in `scripts/vitriol`.  
**Pure math, no server needed.**

### `calibrate_estimate_vram()`

```python
# Constants (from Phase 1 + 2)
VRAM_TOTAL = 8112           # MiB
PER_LAYER_EXPERT = 120      # MiB (pinned expert weights per layer)
KV_PER_TOKEN_K = 0.3125     # MiB per token for K cache (q4_0: 4 bytes per element × n_heads × head_dim × n_layers)
KV_PER_TOKEN_V = 1.25       # MiB per token for V cache (f16: 2 bytes per element)
COMPUTE_BASE = 246           # MiB (base compute buffer at ubatch=256)
COMPUTE_PER_UBATCH = -60    # MiB saved per 128 reduction in ubatch
MTP_HEAD_COST = 2           # MiB per MTP head (negligible)

# Model buffer (always in VRAM via -ngl 99)
MODEL_VRAM = 1334           # MiB (from load_tensors output)

# Pinning cost
pin_cost(N) = N * PER_LAYER_EXPERT

# KV cache cost
kv_cost(ctx, k_quant, v_quant):
    k_factor = quant_factor(k_quant)  # q4_0 → 0.5, f16 → 1.0, q8_0 → 0.5
    v_factor = quant_factor(v_quant)  # f16 → 1.0
    return ctx * KV_PER_TOKEN_K * k_factor + ctx * KV_PER_TOKEN_V * v_factor

# Compute buffer cost
compute_cost(ubatch, mtp_n):
    base = COMPUTE_BASE + (-60 * ((256 - ubatch) / 128))
    mtp_overhead = mtp_n * 2  # small extra for MTP verification graph
    return base + mtp_overhead

# Total VRAM
vram_total(N_pin, ctx, ubatch, mtp_n, k_quant, v_quant):
    return MODEL_VRAM + pin_cost(N_pin) + kv_cost(ctx, k_quant, v_quant) + compute_cost(ubatch, mtp_n)

# Find optimal config subject to vram_total ≤ VRAM_TOTAL * 0.9 (10% safety margin)
```

### Output: `bounds.json`

```json
{
  "model_vram_mib": 1334,
  "vram_total_mib": 8112,
  "vram_safety_margin": 0.9,
  "vram_usable_mib": 7301,
  "per_layer_pin_cost_mib": 120,
  "per_token_kv_cost_mib": 1.5625,
  "estimates": {
    "max_pin_for_ctx_65536_ubatch_128": 14,
    "max_ctx_for_pin_12_ubatch_128": 72090,
    "optimal": {
      "pin_first_n_layers": 12,
      "context": 65536,
      "ubatch_size": 128,
      "draft_n_max": 5,
      "k_quant": "q4_0",
      "v_quant": "f16"
    }
  }
}
```

### Reasoning for optimal recommendation

**Pin layers = min(VRAM_available / per_layer_cost , recommended_max)**

Given 8 GiB VRAM:
- Model: 1,334 MiB
- KV cache at 65K ctx q4_0 K + f16 V: ~1,016 MiB
- Compute buffer (ubatch 128): ~186 MiB
- Remaining: 8,112 - 1,334 - 1,016 - 186 = 5,576 MiB
- Max pin layers: 5,576 / 120 = ~46 — but that'd pin all 40 layers. So VRAM math says full pin is possible... but we proved experimentally that pin12 + MTP5 + ubatch128 = 5,931 MiB (73% used). So the estimator should also account for:
  - CUDA driver overhead (~200 MiB)
  - LRU pool default overhead
  - Chimera model buffer

**The estimator should calibrate against our real-world measurement:**
- vram_total(12, 65536, 128, 5, q4_0, f16) = ? + overhead = 5,931 MiB → derive `vram_overhead = 5,931 - MODEL_VRAM - pin_cost(12) - kv_cost(65536, q4_0, f16) - compute_cost(128, 5)`
- Then use the overhead constant for future predictions.

This makes the estimator self-correcting.

---

## Phase 4: Benchmark Sweep

**Implementation:** Python script `libvitriol/vitriol_calibrate.py`.  
Integrates with `scripts/vitriol` via `vitriol calibrate` (which detects a GGUF model, runs Phases 1-3, then invokes Python for the sweep).

### Config sweep space (from bounds.json)

Pruned to fit within VRAM. Each config is a tuple:

```
(pin=0-16 in steps of 4, MTP=2-6, ubatch=64/128/256, ctx=32768/65536)
```

VRAM estimator filters out combos that exceed `VRAM_TOTAL * 0.95`. Pin values beyond available VRAM are skipped.

Estimated: **20-30 configs** to test.

### Sweep algorithm

```python
def calibrate_sweep(model_path, profile_name="calibrated"):
    configs = generate_config_space(bounds_json)
    results = []
    
    resume_file = calibration_dir / "sweep_progress.json"
    if resume and resume_file.exists():
        results = json.loads(resume_file.read_text())
        configs = [c for c in configs if not already_tested(c, results)]
    
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Testing: pin={config.pin} "
              f"MTP={config.mtp} ubatch={config.ubatch} ctx={config.ctx}")
        
        # Apply config
        subprocess.run(["vitriol", "config", "set", "vitriol.pin_first_n_layers", str(config.pin)])
        subprocess.run(["vitriol", "config", "set", "spec.draft_n_max", str(config.mtp)])
        subprocess.run(["vitriol", "config", "set", "kv.ubatch_size", str(config.ubatch)])
        subprocess.run(["vitriol", "config", "set", "model.context", str(config.ctx)])
        
        # Start server
        subprocess.run(["vitriol", "serve", "--detach"], timeout=30)
        wait_for_server("http://0.0.0.0:8279/health", timeout=120)
        
        # Benchmark
        metrics = benchmark_sequence()
        
        # Record
        row = {**config, **metrics}
        results.append(row)
        json.dump(results, resume_file.open("w"), indent=2)
        
        # Stop server
        subprocess.run(["killall", "-9", "llama-server"])
        sleep(2)
    
    # Select best
    best = max(results, key=lambda r: r["predicted_per_second"])
    save_best(best, profile_name)
    report(results)
```

### Benchmark sequence

```python
BENCHMARKS = [
    {"name": "short", "prompt": "Write a haiku.", "max_tokens": 20},
    {"name": "medium", "prompt": "Briefly explain what a GPU does.", "max_tokens": 150},
    {"name": "long", "prompt": "Write a detailed essay on parallel computing history.", "max_tokens": 512},
]

def benchmark_one(prompt, max_tokens):
    resp = requests.post("http://0.0.0.0:8279/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens}, timeout=120)
    data = resp.json()
    timing = data["timings"]
    
    # nvidia-smi delta for VRAM
    vram_before = get_vram_usage()
    vram_after = get_vram_usage()  # or peak tracking
    vram_used = max(vram_after - vram_before, 0)
    
    return {
        "predicted_per_second": timing["predicted_per_second"],
        "prompt_per_second": timing["prompt_per_second"],
        "predicted_n": timing["predicted_n"],
        "prompt_n": timing["prompt_n"],
        "draft_n": timing["draft_n"],
        "draft_n_accepted": timing["draft_n_accepted"],
        "draft_acceptance_pct": round(timing["draft_n_accepted"] / max(timing["draft_n"], 1) * 100, 1),
        "vram_mib": vram_used,
        "finish_reason": data["choices"][0]["finish_reason"],
    }

def benchmark_sequence():
    results = {}
    for bm in BENCHMARKS:
        results[bm["name"]] = benchmark_one(bm["prompt"], bm["max_tokens"])
    return {
        "predicted_per_second": results["medium"]["predicted_per_second"],
        "prompt_per_second": results["medium"]["prompt_per_second"],
        "draft_acceptance_pct": results["medium"]["draft_acceptance_pct"],
        "vram_peak_mib": max(r["vram_mib"] for r in results.values()),
    }
```

### Selection criteria

Primary: `predicted_per_second` on medium prompt (150 tokens).
Tiebreaker: `draft_acceptance_pct` (higher → better draft model utilization).
Penalty: `vram_peak_mib > 0.95 * VRAM_TOTAL` → disqualify (too close to OOM).

### Edge cases

- **Server crash on certain config**: catch `subprocess.CalledProcessError` or timeout, mark as "FAILED", continue. Report failures at end.
- **Garbage output** (like q8_0 V cache): check `finish_reason` — "stop" = clean, "length" = fine, generate and check if output is mostly `?` characters → mark as "GARBLED".
- **OOM during load**: server process exits immediately after start. Wait timeout marks as "OOM". Reduce sweep range.

---

## Phase 5: C++ Runtime Patches

### 5a: `POST /props` → MTP N

**File:** `llama.cpp/tools/server/server-context.cpp`

```cpp
// In post_props handler (~line 3924)
this->post_props = [this](const server_http_req &req) {
    auto res = create_response();
    if (!params.endpoint_props) {
        return res->error(...);
    }
    
    // Parse request body
    auto body = json::parse(req.body);
    
    if (body.contains("spec_draft_n_max")) {
        int new_n = body["spec_draft_n_max"];
        if (new_n < 1 || new_n > 16) {
            return res->error("spec_draft_n_max must be 1-16");
        }
        // Lock speculative state
        params_base.speculative.draft.n_max = new_n;
        // Recalculate per-slot bounds
        for (auto &slot : slots) {
            slot.n_draft_max = std::min(new_n, n_ctx - 2);
        }
    }
    
    return res->ok({{"success", true}});
};
```

**Locking:** The slot loop is already serialized by the task queue. `spec->dparams` may need a mutex if another thread is reading it during decode. Use `params_base.speculative.draft.n_max` as the source of truth and let `get_n_draft_max()` read from it.

**Calibration impact:** Without this patch: ~60 min for 30 configs (90s per config: 60s load + 30s benchmark). With this patch: ~5 min (one load, then 30 API calls).

### 5b: `POST /props` → VITRIOL config

**File:** `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`

```cpp
extern "C" void vitriol_update_config(vitriol_config_t new_config) {
    pthread_mutex_lock(&g_vitriol_config_mutex);
    
    if (new_config.pin_first_n_layers != g_vitriol_config.pin_first_n_layers) {
        vitriol_update_pin_layers(new_config.pin_first_n_layers);
    }
    if (new_config.lru_mb != g_vitriol_config.lru_mb) {
        vitriol_resize_lru(new_config.lru_mb);
    }
    if (new_config.async_prefetch != g_vitriol_config.async_prefetch) {
        vitriol_toggle_prefetch(new_config.async_prefetch);
    }
    if (new_config.prune_experts != g_vitriol_config.prune_experts) {
        vitriol_update_expert_pruning(new_config.prune_experts);
    }
    
    g_vitriol_config = new_config;
    pthread_mutex_unlock(&g_vitriol_config_mutex);
}
```

Each sub-update function invalidates relevant caches so the next decode picks up the change.

---

## Phase 6: Novel Configs (Research)

### 6a: Adaptive pin by layer depth

Pin first N layers fully, next M layers only attention weights. Experts for partially-pinned layers still go through PCIe DMA.

**Why it might help:** Earlier layers see the most diverse expert routing — pinning them avoids the most PCIe traffic. Later layers have more predictable routing; LRU cache handles them.

**VITRIOL change:** Add `pin_first_n_layers_attn_only` config field.

### 6b: Phase-aware ubatch

Use larger ubatch during prompt prefill (256) and smaller during MTP verification (128). Prefill benefits from larger batches, MTP from smaller compute buffers.

**ggml-backend change:** Dynamic ubatch switch between context processing and decode. Complex scheduling change.

### 6c: Hybrid MTP + ngram

Run ngram cache for `spec_draft_n_min` tokens, then MTP for the rest. Catches common patterns cheaply, uses MTP for novel continuations.

**Speculative chain change:** Add `ngram → mtp` as a speculative implementation chain in `common_speculative_init()`. This is a configuration change to the `common_params_speculative.types` ordering.

### 6d: Expert locality pinning

During a profiling run, track which experts the router selects per layer. Pin experts that are selected >5% of the time. This maximizes PCIe savings by pinning only the most-used experts.

**VITRIOL change:** Add runtime expert access profiler, then `vitriol_update_config` to re-pin based on profile data.

---

## File Structure Changes

```
scripts/vitriol                          # + calibrate subcommand, bash probe functions
libvitriol/vitriol_calibrate.py          # Python sweep controller (Phase 4)
libvitriol/calibrate_pcie.cu             # Optional: PCIe bandwidth probe CUDA program
~/.vitriol/
  calibration/
    hardware.json                         # Phase 1 output
    <model_hash>/
      model.json                          # Phase 2 output
      bounds.json                         # Phase 3 output
      sweep.json                          # Phase 4 output (full results)
      best.json                           # Phase 4 output (best config)
    sweep_progress.json                   # Phase 4 resume state
```

---

## Timeline

| Step | What | Effort | Depends on |
|------|------|--------|------------|
| 1a | `calibrate` subcommand + hardware probe | ~80 lines bash | nothing |
| 1b | Model analyzer (GGUF parse) | ~60 lines bash | 1a |
| 1c | VRAM estimator | ~50 lines bash | 1a, 1b |
| 2a | Python sweep controller skeleton | ~150 lines Python | 1a, 1b, 1c |
| 2b | API polling, result recording | ~80 lines Python | 2a |
| 2c | Selection algorithm + profile save | ~50 lines Python | 2b |
| 3 | `POST /props` MTP endpoint | ~50 lines C++ | 2c (optional speedup) |
| 4 | VITRIOL config runtime update | ~150 lines C++ | 3 |
| 5 | Novel configs | varies | proven measurement |

---

## Appendix: VRAM Model from Real-World Data

Icarus v1 (pin12, MTP5, ubatch128, q4_0 K, f16 V, 65K ctx):

```
nvidia-smi:  5,931 MiB
Breakdown:
  Model weights (GPU):           1,334 MiB  (from load_tensors: CUDA0 model buffer)
  Pinned experts (12 layers):    1,440 MiB  (12 × 120 MiB)
  KV cache (65K, q4_0 K, f16 V): 1,016 MiB  (estimated)
  Compute buffers (ubatch 128):    186 MiB  (estimated)
  Chimera/overhead:                 ??
  Total:                          5,931 MiB
  Remaining:                      2,181 MiB
```

Overhead constant: `5,931 - (1,334 + 1,440 + 1,016 + 186) = 1,955 MiB` — this includes Chimera, driver state, MTP context, LRU pool (even at `lru_mb=0` there's baseline overhead), and any inaccuracies in our per-component estimates.

For the estimator, we'll use a fudge factor of `1.3×` on the math estimate until we can refine it.
