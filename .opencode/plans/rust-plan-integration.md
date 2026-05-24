## 5. Integration with Bash

The bash `vitriol` script calls the Rust binary:
```bash
calibrate_quick() {
    vitriol-calibrate calibrate --quick \
        --model "$1" \
        --profile "${2:-calibrated}"
}
```

### What moves to Rust

- Hardware probing (nvidia-smi, /proc, getcap)
- GGUF model parsing
- VRAM estimation + optimal search
- Profile INI writing
- All JSON output to ~/.vitriol/calibration/

### What stays in bash

- Profile listing/loading/deleting (integrated with config system)
- Config management (load/apply/reload)
- Server start/stop/run
- Other subcommands

### Bash fallback

The Rust binary returns exit code 0 on success, non-zero on error. Bash checks this and can fall back to the Python implementation if needed (during transition).