# VITRIOL Architecture — Direct NVMe→VRAM DMA

> **Date**: 2026-05-14
> **Status**: Phase 1 complete, Phase 2 in progress (cooperative nvidia P2P)
> **See also**: `COOPERATIVE_DMA.md`, `FINDINGS_2026-05-13-alka-benchmark.md`

## The Stack

### Layer 1: Data Path
```
GGUF file on NVMe SSD
  │
  │ kernel_read() — goes through VFS → ext4 → block layer → NVMe driver
  │   Chunked in 1MB blocks via dma_alloc_coherent or vmalloc'd buffer
  │
  ▼
Staging buffer (kernel memory)
  │
  │ memcpy_toio() — over PCIe to GPU BAR1 (if vitriol owns the GPU)
  │   OR  nvidia_p2p_get_pages + ioremap + memcpy_toio (cooperative, preferred)
  │   OR  staging buffer only (simulation mode, for testing)
  │
  ▼
GPU VRAM
```

### Layer 2: Kernel Module (`vitriol.ko`)

| IOCTL | Purpose |
|-------|---------|
| `EXECUTE` | Dispatch a single Drop packet (32-byte Alka instruction) |
| `SET_VIAL` | Pass hardware constraints (aperture size, thermal limits, cooperative flag) |
| `SET_SOURCE` | Receive GGUF file descriptor from userspace via `fget()` |
| `GET_RESULT` | Read back execution stats (bytes transferred, cycles, errors) |
| `BIND_DEVICE` | Stub — returns ENOTSUPP (userspace `--bind` handles PCI rebinding) |

Key Opcodes:
- **CLAIM** — Register a hardware vessel (GPU, NVMe, etc.)
- **SHIFT** — Track BAR1 window offset
- **FLOW** — `kernel_read()` from GGUF → destination (BAR1 / P2P pages / staging)
- **FENCE** — Wait for completion (poll BAR0 or simulated)
- **SYNC** — Memory barrier (`wmb()`)
- **SIGNAL** — Trigger event (stub)
- **LIMIT** — Set thermal limits
- **REFRACT** — Slice operation (stub)
- **DRY_RUN** — Validate without executing

FLOW has three paths selected automatically:
1. **Cooperative P2P** (preferred) — uses `nvidia_p2p_get_pages()` resolved via kprobe
2. **BAR1 fallback** — writes directly to mapped PCI BAR1 (when vitriol claims the GPU)
3. **Simulated** — reads data into staging buffer only (for testing, no GPU needed)

### Layer 3: Executor (`alka-executor`)

```
alka-executor stream.alkas vial.alkavl \
  --source model.gguf          # Open GGUF, pass fd to kernel
  --cooperative                # Use nvidia P2P cooperative DMA
  --gpu-va 0x7f00000000       # CUDA-allocated GPU VA for DMA target
  --bind 0000:02:00.0          # Userspace PCI rebind (fork-safe, 5s timeout)
  --rollback stream.azoth      # Rollback on failure
  --dry-run                    # Validate without touching hardware
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Stream dispatch | 1,226 drops in 0.017s | ~14μs per Drop, IOCTL overhead negligible |
| Total data described | 81.2 GB | Sum of all tensor sizes in the recipe |
| Real DMA throughput | TBD (Phase 2) | Bounded by PCIe 3.0 x16 (~10 GB/s) |
| kernel_read throughput | ~2-3 GB/s | From NVMe SSD via VFS/filesystem |
| GPU idle waiting for weights | 82-85% | Baseline from OPTIMIZATION_PLAN.md |

## Safety

- **No kernel BIND workqueue** — removed; userspace `--bind` uses fork-based timeout
- **rmmod always works** — no D-state kworkers holding module ref
- **D state orphans cleanly** — if `--bind` child blocks, it's orphaning-safe
- **Cooperative P2P** — doesn't touch PCI subsystem at all, can't block

## Why kernel_read() Instead of Direct NVMe DMA

1. **Portability** — works on any filesystem (ext4, btrfs, XFS, etc.)
2. **Simplicity** — standard Linux kernel API, no NVMe-specific code
3. **Adequate** — 2-3 GB/s is faster than current CPU-bound loading bottleneck
4. **Upgrade path** — replace with `blkdev_direct_read()` for zero-copy later

## KV Cache Tiering Strategy

The KV cache grows at ~5-10 MB per token for a 35B MoE with 40 layers. At 2048 tokens that's 10-20 GB — far exceeding the 8GB VRAM available. VITRIOL handles this with a three-tier hierarchy:

```
VRAM (2-4 GB reserved):  HOT  — recent ~512 tokens  (~2.5-5 GB)
    ↑↓ VITRIOL DMA (nvidia_p2p_get_pages)
System RAM (4-8 GB):      WARM — ~1024 older tokens  (~5-10 GB)
    ↑↓ VITRIOL DMA (nvidia_p2p_get_pages + krealloc + DMA map)
SSD (unlimited):          COLD — overflow + context beyond active window
```

### Tier Rules

| Tier | Location | Size | Latency | Refill Strategy |
|------|----------|------|---------|-----------------|
| **Hot** | VRAM | ~512 tokens (~2.5-5 GB) | ~0.010ms (already resident) | LRU eviction to Warm on full |
| **Warm** | System RAM | ~1024 tokens (~5-10 GB) | ~3-5ms (PCIe DMA) | LRU eviction to Cold on full |
| **Cold** | SSD | Unlimited | ~8-15ms (NVMe read + RAM DMA) | Loaded on-demand via kernel_read |

### KV Cache FLOW

The `FLOW` instruction handles all three tiers because VITRIOL controls both source and destination:

```
# Tier promotion: SSD → RAM (read into DMA buffer, copy to krealloc'd RAM)
FLOW NVME_BOOT[0x1a2b000] -> RAM_BUF[0x0000] 0x10000;

# Tier promotion: RAM → VRAM (nvidia_p2p_get_pages on destination GPU VA)
FLOW RAM_BUF[0x0000] -> GPU_MAIN.KV_CACHE[0x5000] 0x10000;

# Tier demotion: VRAM → RAM (reverse: source is GPU VA, destination is RAM phys addr)
FLOW GPU_MAIN.KV_CACHE[0x5000] -> RAM_BUF[0x0000] 0x10000;
```

### Implementation Plan

| Step | Description |
|------|-------------|
| 1 | Reserve a CUDA buffer for KV cache in VRAM (decoupled from weights) |
| 2 | Add a RAM-backed staging pool (kmalloc'd, DMA-mappable) for Warm tier |
| 3 | Extend `handle_flow` to detect RAM source addresses vs file offsets |
| 4 | Implement LRU eviction: Hot→Warm→Cold when the token window advances |
| 5 | Prefetch next 4 tokens while computing current one (hide SSD→Warm latency) |

### Impact

On a full 2048-token context pass:
- ~95% of attention steps hit the Hot tier (recent tokens drawn from VRAM)
- ~4% hit the Warm tier (need a RAM→VRAM DMA, ~3-5ms added)
- ~1% hit the Cold tier (need SSD→RAM→VRAM, ~15ms added)
- Total latency impact: <5% over a pure VRAM baseline

## Key Files

| File | Lines | Role |
|------|-------|------|
| `vitriol-daemon/vitriol.c` | ~1210 | Kernel module: IOCTL dispatch, opcode handlers, P2P coop, kprobe resolution |
| `vitriol-daemon/vitriol_alka_kernel.h` | ~150 | Kernel ABI: structs, ioctl defs, opcodes |
| `alka-executor/executor.c` | ~835 | Userspace: stream loader, vial parser, PCI bind, IOCTL gateway |
| `alka-executor/vitriol_alka_user.h` | ~155 | Userspace ABI mirror |
| `docs/COOPERATIVE_DMA.md` | — | Implementation plan for cooperative P2P DMA |
| `FINDINGS_2026-05-13-alka-benchmark.md` | — | Full benchmark results and obstacle log |
