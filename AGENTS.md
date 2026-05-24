# VITRIOL AI Agent Guidelines

## Testing Protocol

1. **Always use VITRIOL DMA offloading** for model tests. Never test with `-ngl 0` (CPU-only) unless explicitly requested. VITRIOL is designed to make large models fit on small GPUs — use `VITRIOL_ENGINE_MODE=vitriol-dma` and `-ngl 99`.

2. **After any build** (`cmake --build`), ask the user to run `sudo vitriol setup` before testing. This sets `CAP_IPC_LOCK` on the server binary, required for page-locked DMA buffers.

3. **Always kill stale servers** with `killall -9 llama-server` before starting a new one.

## Documentation

1. **Write all findings in `.md` reports** with ISO 8601 timestamps (YYYY-MM-DD HH:MM).
2. Place reports in `.opencode/plans/` for integration plans and research findings.
3. Place session logs in `SESSION_LOG_*` and experiment logs in `EXPERIMENT_LOG.md`.
4. Include exact command output, tensor shapes, and error messages in reports.
5. Update the anchored summary in each session with progress, blockers, and decisions.

## Code Conventions

- This is a fork of `ggml-org/llama.cpp` with VITRIOL modifications. The `vitriol` branch contains our changes.
- The VITRIOL predictor is in `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`.
- The server context checkpoint logic is in `tools/server/server-context.cpp`.
- All VITRIOL env vars are prefixed with `VITRIOL_`.

## Calibration Tool (Rust)

- **`libvitriol/`** — Rust binary for `vitriol calibrate --quick`.
- Build with `cargo build --release` in `libvitriol/`.
- Source files: `gguf.rs` (GGUF v3 parser), `probe.rs` (hardware), `estimator.rs` (VRAM model), `main.rs` (CLI).
- The Rust binary is called by `scripts/vitriol` if built; falls back to Python `libvitriol/gguf_reader.py`.
- **No hardcoded model constants** — all VRAM values computed from GGUF tensor data.
- Self-computing formula: `VRAM = base_model + pin * per_layer_expert + ctx * kv_per_token + scratch + overhead`.
- Overhead heuristic: Pascal=1800, Turing=2200, Ampere=2800, Ada=3200 MiB.
- KV cache computed from model dims: `(embd_len / head_count) * head_count_kv * 2.5 / 1M`.
- Per-layer expert cost from tensor name analysis (`ffn_*_exps` patterns).
