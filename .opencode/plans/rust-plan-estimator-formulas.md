### VRAM formula (no hardcoded model constants)

```
1. base_model_mib = total_size_mib - per_layer_experts_mib * block_count
2. head_dim = embedding_length / head_count
3. kv_dim = head_dim * head_count_kv
4. kv_mib_per_token = kv_dim * (0.5 + 2.0) / 1_048_576
   (0.5 = q4_0 K cache, 2.0 = f16 V cache)
5. scratch_mib = ubatch * embedding_length * 4 / 1_048_576
6. overhead_mib = f(compute_cap)  // per GPU generation
7. safety_margin = 0.9
8. usable = vram_total * safety
9. total = base + pin * per_layer_expert + ctx * kv_per_token + scratch + overhead
```

### Overhead by GPU generation

| Compute Cap | Generation | Overhead (MiB) |
|-------------|-----------|-----------------|
| 6.x | Pascal | 1800 |
| 7.x | Volta/Turing | 2200 |
| 8.x | Ampere | 2800 |
| 9.x | Ada | 3200 |
| unknown | fallback | 2000 |

These are calibrated per generation, not per model. Refinable by Step 2 measurement.