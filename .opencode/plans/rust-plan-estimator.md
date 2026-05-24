## 4.3 estimator.rs — Self-Computing VRAM Model

Zero hardcoded model constants. Everything from ModelInfo + HardwareInfo.

```rust
pub struct VramEstimate {
    pub base_model_mib: f64,
    pub per_layer_pin_cost_mib: f64,
    pub kv_mib_per_token: f64,
    pub scratch_mib: f64,
    pub overhead_mib: f64,
    pub vram_total_mib: f64,
    pub vram_safety_margin: f64,
    pub vram_usable_mib: f64,
    pub estimates: Vec
    pub is_feasible: bool,
}
```
