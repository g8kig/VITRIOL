# TUI Dashboard: Implementation Plan

## Goal
A terminal UI for VITRIOL that shows live server stats, throughput, VRAM usage, bottleneck analysis, and config management. Uses **Textual** (Python) for a responsive, OpenCode-like interface.

## Layout

### Large window (>80 cols)
```
┌─────────────── VITRIOL ───────────────┐
│ ┌──── logo ────┐ ┌───── Status ─────┐│
│ │              │ │ ⚡ 17.62 t/s     ││
│ │              │ │ 🧠 MTP 98.5%     ││
│ │              │ │ 💾 3.5/8.0 GiB   ││
│ └──────────────┘ │ 📐 25K/131K ctx  ││
│                  └─────────────────┘│
│ ┌───── Bottlenecks ────────────────┐│
│ │ Layer │ FFN ██  │ Attn █ │ PCIe █││
│ │  0-10 │ ██████ │ ███   │ █████  ││
│ │ 11-20 │ ██████ │ ███   │ ██     ││
│ │ 21-30 │ ██████ │ ███   │ ░░ pin ││
│ │ 31-39 │ ██████ │ ███   │ ░░ pin ││
│ └──────────────────────────────────┘│
│ ┌─ Config ───────┐ ┌─── Log ──────┐│
│ │ [1] Model      │ │ 12:34 ready  ││
│ │ [2] GPU        │ │ 12:35 req    ││
│ │ [3] VITRIOL    │ │ 12:36 17.6t  ││
│ └────────────────┘ └──────────────┘│
└─────────────────────────────────────┘
```

### Small window (≤80 cols)
Single column: Status → Bottlenecks → Config → Log. Logo is compacted to a single-line banner.

## Data Sources

| Widget | Data | Source | Interval |
|--------|------|--------|----------|
| t/s | tokens/sec | Server log (parse slot print_timing) | Per-timing line |
| VRAM | used/total | nvidia-smi --query-gpu=memory.used,memory.total | 1s |
| MTP acceptance | rate | Server log (draft acceptance) | Per-timing line |
| Context usage | tokens | Server log (n_tokens, prompt) | Per-request |
| Layer timing | ms per section | Server log or /v1/vitriol/stats | TBD |
| Active requests | count | Server health endpoint | Polling |

## Files

| File | Role |
|------|------|
| `libvitriol/vitriol-tui.py` | Python TUI daemon using Textual |
| `assets/ansi-logo.txt` | VITRIOL ANSI logo |
| `scripts/vitriol` | Add `vitriol tui` subcommand |

## Implementation Phases

### Phase 1: TUI skeleton + log parsing (this session)
- Create `libvitriol/vitriol-tui.py` with Textual
- Parse server log for `slot print_timing` lines
- Display live t/s, VRAM (nvidia-smi), context usage
- Responsive layout (stack vs columns)
- `vitriol tui` subcommand that launches alongside server

### Phase 2: Stats endpoint
- Add `/v1/vitriol/stats` to llama-server
- Returns JSON with per-layer timing, cache stats, VRAM breakdown

### Phase 3: Config editor
- In-TUI config management (model, GPU, VITRIOL settings)
- Save to `~/.vitriol/config`

### Phase 4: Polish
- Bottleneck visualization
- Historical graphs (t/s over time)
- Hot-reload config without restart

## Dependencies
- Python 3.10+
- Textual (`pip install textual`)
- `nvidia-smi` (included with NVIDIA drivers)
