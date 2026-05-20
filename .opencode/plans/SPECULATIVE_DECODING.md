# Speculative Decoding Plans

**Status:** MTP N=2 implemented and tested. Speculative Routing is the planned next step.

## Implemented: MTP N=2

| Param | Config key | CLI flag |
|-------|------------|----------|
| Enable | `vitriol.spec_type` | `--spec-type mtp` |
| Draft depth | `vitriol.spec_draft_n_max` | `--spec-draft-n-max 2` |

MTP is auto-detected for MTP-capable models (reads `nextn_predict_layers` from GGUF header). N=2 is optimal (acceptance rate = exactly 1/N).

**Benchmark:** 10.96 t/s with MTP alone. Does NOT stack with prune+cache (both target the compute bottleneck).

## Planned: Speculative Routing

**Idea:** Use the FFN gate's input activation from the previous layer to predict which experts will be needed in the current layer. This enables earlier prefetch starts and higher prediction accuracy.

See the full plan at [`SPECULATIVE_ROUTING.md`](SPECULATIVE_ROUTING.md).

## Original Documents

- [Speculative Routing Plan](SPECULATIVE_ROUTING.md) — Full design doc with tensor analysis
