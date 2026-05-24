## 9. Summary of What Is NOT Hardcoded

| Item | Source | Status |
|------|--------|--------|
| Base model VRAM | total - expert_total (from tensor sizes) | Computed ✅ |
| Per-layer pin cost | avg expert bytes per layer (from tensor names) | Computed ✅ |
| KV cache per token | n_embd / n_head * n_kv_head * (k_sz + v_sz) | Computed ✅ |
| Compute scratch | ubatch * n_embd * 4 / 1M | Computed ✅ |
| Overhead | GPU generation heuristic (Pascal=1800, Turing=2200, ...) | GPU-gen heuristic |
| Optimal pin | Max feasible from VRAM search | Discovered ✅ |
| Optimal context | User default capped by VRAM | Discovered ✅ |
| Optimal ubatch | 128 (experimentally proven) | Universal |
| draft_n_max | min(5, block_count / 8) | Architecture rule |
| Safety margin | 90% | Configurable default |

### What requires Step 2 (benchmark sweep) to truly optimize

- MTP acceptance rate (can't estimate from VRAM)
- Optimal pin past the DMA throughput plateau (can't estimate from VRAM alone)
- ubatch=128 vs 256 performance difference (small, but model-dependent)

The VRAM estimator bounds the search space. The sweep controller (Step 2) measures the actual throughput landscape.

---

**End of plan.**