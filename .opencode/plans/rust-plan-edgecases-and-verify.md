## 7. Edge Cases

| Condition | Behavior |
|-----------|----------|
| Dense model (expert_count == 0) | Warn: no MoE experts, VITRIOL DMA not applicable |
| GPU not found | Error with actionable message |
| VRAM too small for min config | Suggest smaller context or quantization |
| Unknown GPU generation | Use overhead = 2000 MiB (conservative) |
| Missing metadata (n_head, n_kv_head) | Fallback defaults: n_head=32, n_kv_head=8 |
| GGUF parse error | Report which part failed, use partial data |
| Model hash collision | Not handled (SHA256 of 1 MiB, astronomically unlikely) |
| Profile name collision | Overwrite with confirmation |

## 8. Verification

After Rust implementation, verify with:
```bash
cd libvitriol && cargo build

# Test on known model
./target/debug/vitriol-calibrate calibrate --quick \
    --model /path/to/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

# Verify: model info matches Python GGUF reader output
# Verify: VRAM estimate within 10% of actual measurement
# Verify: optimal config close to experimental icarus v1 result

# Test edge cases
./target/debug/vitriol-calibrate calibrate --quick \
    --model /path/to/dense_model.gguf  # expect MoE warning
./target/debug/vitriol-calibrate calibrate --quick \
    --model /nonexistent.gguf          # expect graceful error
```