# VITRIOL Calibration Tool — Rust Implementation Plan

**Date:** 2026-05-24
**Status:** Design phase (not yet implemented)
**Objective:** Rewrite `vitriol calibrate --quick` in Rust with a self-computing VRAM model (no hardcoded model-specific constants)

---

## 1. Why Rust

The project already has Rust infrastructure (`vitriol-daemon/Cargo.toml`, `src/main.rs`). Rust provides:

- Same language, same build system (Cargo) — no new toolchain to maintain
- `serde_json` for JSON I/O (hardware, model, bounds data)
- `clap` for CLI argument parsing
- `std::process::Command` for nvidia-smi probing
- `sha2` for model hash computation
- `anyhow` for error handling (already used by daemon)

---

## 2. File Structure

```
libvitriol/
├── Cargo.toml
└── src/
    ├── main.rs          # CLI entry — clap subcommands
    ├── gguf.rs          # GGUF v3 binary parser
    ├── probe.rs         # Hardware probing (nvidia-smi, /proc, getcap)
    ├── estimator.rs     # Self-computing VRAM model + optimal config search
    ├── config.rs        # INI config writer (profile format)
    └── calibrate.rs     # calibrate --quick orchestration
```

---

## 3. Dependencies (Cargo.toml)

```
[package]
name = "vitriol-calibrate"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
sha2 = "0.10"
log = "0.4"
env_logger = "0.11"
```

No HTTP deps yet — those come in Step 2 (sweep controller).