## 4.4 config.rs — INI Config Writer

Writes the VITRIOL config INI file matching the format from bash's `write_config`:

```ini
[model]
path = /path/to/model.gguf
context = 65536
threads = 4
ngl = 99

[vitriol]
mode = stream
pin_first_n_layers = 14
lru_mb = 0

[kv]
ubatch_size = 128
quant_mode_v = f16

[spec]
draft_n_max = 5

[engine]
mode = vitriol-dma
```

Function signature:
```rust
pub fn write_profile_config(
    profile_dir: &Path,
    model_path: &str,
    optimal: &OptimalConfig
) -> Result<()>;
```
