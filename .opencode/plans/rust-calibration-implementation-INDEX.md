# VITRIOL Calibration — Rust Implementation Plan (Index)

**Date:** 2026-05-24 | **Status:** Design phase (not yet implemented)
**Objective:** Rewrite `vitriol calibrate --quick` in Rust with self-computing VRAM

## Files (read in order)

| # | File | Content |
|---|------|---------|
| 1 | `rust-calibration-implementation-pt1.md` | Why Rust, file structure, dependencies |
| 2 | `rust-calibration-implementation-pt2.md` | gguf.rs — GGUF v3 parser design |
| 3 | `rust-plan-probe.md` | probe.rs — nvidia-smi, /proc, getcap |
| 4 | `rust-plan-estimator.md` | estimator.rs — data structures |
| 5 | `rust-plan-estimator-formulas.md` | estimator.rs — VRAM formulas |
| 6 | `rust-plan-search-algo.md` | estimator.rs — optimal config search |
| 7 | `rust-plan-config.md` | config.rs — INI config writer |
| 8 | `rust-plan-calibrate.md` | calibrate.rs — orchestration flow |
| 9 | `rust-plan-main.md` | main.rs — CLI entry with clap |
| 10 | `rust-plan-integration.md` | bash integration, what moves vs stays |
| 11 | `rust-plan-display.md` | CLI output format |
| 12 | `rust-plan-edgecases-and-verify.md` | Edge cases + verification |
| 13 | `rust-plan-summary.md` | What is/isn't hardcoded, next steps |

## Key design decisions

- **No hardcoded model constants** — all VRAM values computed from GGUF tensor data
- **GPU-gen overhead heuristic** — calibrated per compute capability, not per model
- **No pin cap** — recommends max feasible pin; true optimum found by sweep
- **MTP heuristic** — `min(5, block_count/8)`, clearly labeled as architecture rule
- **Rust binary called by bash** — minimally disruptive, bash retains profile/config/server mgmt