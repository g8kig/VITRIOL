## 6. Display Output Format

The Rust binary prints directly to stdout:

```
⚡ Probing hardware...
  → NVIDIA GeForce GTX 1070 Ti, 8192 MiB

📦 Analyzing model...
  → qwen35moe, 41 layers, 256 experts, n_embd=2048, n_head=16, n_kv_head=4

🧮 Estimating VRAM bounds...
  → Optimal: pin14 ctx65536 ubatch128 MTP5 — 6951 MiB (84.9%)

  Derivation:
    Base model (attention + shared):   1557 MiB  (computed from tensor sizes)
    Pinned expert layers (14 x 211):   2954 MiB
    KV cache (65536 ctx @ q4_0 K + f16 V):    87 MiB
    Compute scratch:                     1 MiB
    CUDA overhead (Pascal Gen6):       1800 MiB
    ────────────────────────────────────────────
    Total:                             7023 MiB  (usable: 7373 MiB @ 90%)

💾 Saving as profile 'calibrated'...
✓ Profile 'calibrated' saved
  Description: Calibrated: pin14 ctx65536 ubatch128 MTP5

══════════════════════════════════════════
  Calibration complete!
  Run: vitriol config load calibrated
  Then: vitriol stop && vitriol serve --detach
══════════════════════════════════════════
```

All display print via stdout/stderr. The bash script reads nothing from stdout for calibration (Rust handles all output).