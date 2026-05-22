# CLAUDE.md — VITRIOL Project

## Build & Test

- **Build server:** `cmake --build build --target llama-server -- -j$(nproc)` (from `llama.cpp/`)
- **After every build**, ask user to run **`sudo vitriol setup`** before any test
- **Kill stale servers:** `killall -9 llama-server` before starting a new one
- **Never test with `-ngl 0`** (CPU-only) unless explicitly asked. Always use VITRIOL DMA path.

## Test Commands

- **Start server with VITRIOL:** Use env vars `VITRIOL_ENGINE_MODE=vitriol-dma`, `-ngl 99`, `--no-mmap`
- **Check server health:** `curl http://127.0.0.1:8279/health`
- **Quick inference test:** `curl http://127.0.0.1:8279/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'`
- **Model paths:** Qwen3.6 at `~/Downloads/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`, Jamba2-Mini at `~/Desktop/Projects/ai21labs_AI21-Jamba2-Mini-IQ2_M.gguf`

## Documentation Standards

- Write all findings in `.md` reports with **ISO 8601 timestamps** (`YYYY-MM-DD HH:MM`)
- Place integration plans in `.opencode/plans/`
- Update `ANCHORED_SUMMARY.md` with progress, blockers, and decisions each session
- Include exact commands, tensor shapes, and error messages in reports

## Key Files

- VITRIOL predictor: `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`
- Server checkpoints: `tools/server/server-context.cpp`
- Jamba model: `src/models/jamba.cpp`
- GTX 1070 Ti (8 GB VRAM, CC 6.1, PCIe 3.0 x16)
