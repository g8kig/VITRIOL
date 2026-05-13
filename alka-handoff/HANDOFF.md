# VITRIOL Stream Handoff — Alka → VITRIOL

## What's in this folder

### GTX 960 2GB Stream
| File | Size | Description |
|------|------|-------------|
| `stream_960.alka.alkas` | 224 bytes | VITRIOL stream — 7 Drop packets (32 bytes each) |
| `stream_960.alka.azoth` | 224 bytes | Rollback stream — 7 Azoth packets for undo |
| `gtx960_2gb.alkavl` | 628 bytes | Hardware Vial (contract) for GTX 960 2GB |

### GTX 1070 Ti 8GB Stream
| File | Size | Description |
|------|------|-------------|
| `purify_1070ti.alka.alkas` | 544 bytes | VITRIOL stream — 17 Drop packets (LLM weight loading) |
| `purify_1070ti.alka.azoth` | 544 bytes | Rollback stream — 17 Azoth packets for undo |
| `ivyb_pascal.alkavl` | — | Hardware Vial for Ivy Bridge + GTX 1070 Ti |

### Common
| File | Description |
|------|-------------|
| `vitriol_alka.h` | C ABI header: Drop struct, opcodes, IOCTL commands |
| `HANDOFF.md` | This document |

## How It Works

### The Pipeline

```
Alka Source (.alka) + Vial (.alkavl)
         │
         ▼
  ┌──────────────┐
  │   Parser     │  Tokenize → AST → Instruction list
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Compiler   │  Resolve vessels, eval operands, validate tools
  │   (alkac)    │  SPARK tools validated at compile-time
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Codegen    │  Emit 32-byte Drop packets (.alkas)
  │              │  Emit rollback packets (.azoth)
  └──────┬───────┘
         │
         ▼
  VITRIOL Kernel Executor (vitriol.ko)
         │
         ▼
  GPU Hardware (PCIe BAR1, DMA, VRAM)
```

### The Drop Packet (32 bytes)

```c
struct vitriol_drop {
    uint8_t  op_code;      /* 0x01–0x3F */
    uint8_t  flags;
    uint16_t vessel_id;
    uint64_t src_addr;     /* Physical address */
    uint64_t dst_addr;     /* Physical address */
    uint32_t size;
    uint32_t reserved;
    uint32_t crc;
};
```

Each instruction in the Alka source becomes one or more Drop packets.
The executor reads packets sequentially, validates each against the Vial
constraints, executes via DMA/ioctl, and writes results back.

### The Vial Contract

The `.alkavl` file defines hardware limits that every Drop must respect:

```
Vessel GTX_960 {
    PCI_ID: 10de:1401;
    BAR1_Size: 256MB;
    VRAM: 2048MB;
    Aperture {
        Size: 256MB;
        Max_Window: 256MB;
    }
    Thermal {
        HALT_AT: 95C;
        THROTTLE_AT: 85C;
    }
}
```

The executor **must** re-validate every Drop against these constraints
before execution. The compiler's validation is a compile-time check only.

## GTX 960 Stream — 7 Packets

| # | Opcode | Name | Operands | Purpose |
|---|--------|------|----------|---------|
| 1 | 0x01 | CLAIM | GPU_MAIN | Stake the GPU node |
| 2 | 0x0E | LIMIT | 95C | Set thermal halt limit |
| 3 | 0x3B | REFRACT | 0x0 → 0x20000000, 256MB | Sub-tensor slice for micro-paging |
| 4 | 0x06 | SYNC | — | Memory barrier |
| 5 | 0x05 | FENCE | GTX_960.METAPAGE, 1 | Wait for metapage ready |
| 6 | 0x2F | WATCH | — | Monitor hardware state |
| 7 | 0x09 | SIGNAL | STREAM_COMPLETE | Trigger compute |

Total: 224 bytes (7 × 32 bytes)

## GTX 1070 Ti Stream — 17 Packets

**Goal:** Move LLM weights from NVMe to VRAM via sliding window (Purify pattern)

| # | Opcode | Name | Operands | Purpose |
|---|--------|------|----------|---------|
| 1 | 0x01 | CLAIM | GPU_MAIN | Stake the GPU node |
| 2 | 0x01 | CLAIM | NVME_BOOT | Stake the NVMe device |
| 3 | 0x0E | LIMIT | GPU_MAIN.THERMAL 85C | Set thermal halt limit |
| 4 | 0x04 | SHIFT | GPU_MAIN.DATA_PLANE @ 0 | Map BAR1 window to offset 0 |
| 5 | 0x03 | FLOW | NVME_BOOT[OFFSET_1] → DATA_PLANE[0] 256MB | DMA first chunk |
| 6 | 0x05 | FENCE | GPU_MAIN.METAPAGE == 1 | Wait for chunk 1 ready |
| 7 | 0x04 | SHIFT | GPU_MAIN.DATA_PLANE @ 256MB | Remap window to next 256MB |
| 8 | 0x03 | FLOW | NVME_BOOT[OFFSET_2] → DATA_PLANE[0] 256MB | DMA second chunk |
| 9 | 0x05 | FENCE | GPU_MAIN.METAPAGE == 2 | Wait for chunk 2 ready |
| 10 | 0x04 | SHIFT | GPU_MAIN.DATA_PLANE @ 512MB | Remap window to 512MB |
| 11 | 0x03 | FLOW | NVME_BOOT[OFFSET_3] → DATA_PLANE[0] 256MB | DMA third chunk |
| 12 | 0x05 | FENCE | GPU_MAIN.METAPAGE == 3 | Wait for chunk 3 ready |
| 13 | 0x04 | SHIFT | GPU_MAIN.DATA_PLANE @ 768MB | Remap window to 768MB |
| 14 | 0x03 | FLOW | NVME_BOOT[OFFSET_4] → DATA_PLANE[0] 144MB | DMA final chunk |
| 15 | 0x05 | FENCE | GPU_MAIN.METAPAGE == 4 | Wait for chunk 4 ready |
| 16 | 0x06 | SYNC | L3 | Memory barrier |
| 17 | 0x09 | SIGNAL | INFERENCE_COMPLETE | Trigger inference |

Total: 544 bytes (17 × 32 bytes)

**Pattern:** SHIFT → FLOW → FENCE repeated 4 times with sliding 256MB window.
This loads 896MB of LLM weights (256+256+256+144) through the 256MB BAR1 aperture.

## System Specs (Source Machine)

| Component | Value |
|-----------|-------|
| CPU | Intel Core i7-3770 @ 3.40GHz (Ivy Bridge) |
| RAM | 16 GB DDR3 |
| Kernel | 6.17.0-20-generic |
| NVIDIA Driver | 535.288.01 |

### GTX 1070 Ti
| Property | Value |
|----------|-------|
| PCI ID | 10de:1b82 |
| Bus | 01:00.0 |
| VRAM | 8192 MB (8 GB) |
| BAR0 (MMIO) | f6000000, 16MB non-prefetchable |
| BAR1 (prefetchable) | e0000000, 256MB |
| BAR2 (prefetchable) | f0000000, 32MB |
| Power Limit | 180W |
| Idle Temp | 55°C |
| Thermal Halt | 85°C |
| Thermal Throttle | 80°C |

### GTX 960
| Property | Value |
|----------|-------|
| PCI ID | 10de:1401 |
| Bus | 02:00.0 |
| VRAM | 2048 MB (2 GB) |
| BAR0 (MMIO) | f4000000, 16MB non-prefetchable |
| BAR1 (prefetchable) | c0000000, 256MB |
| BAR2 (prefetchable) | d0000000, 32MB |
| Power Limit | 120W |
| Idle Temp | 33°C |
| Thermal Halt | 95°C |
| Thermal Throttle | 85°C |

## What Was Tested

### Compiler Tests (zig build test)
- Parse CLAIM, FLOW, REQUIRE directives
- Parse Vial with Vessel definitions
- Codegen emits 32-byte packets
- Drop has correct opcodes

### SPARK Integration Tests (zig build test-spark)
- SHIFT: page-aligned offset passes, non-aligned fails, exceeds aperture fails
- FLOW: valid DMA passes, zero size fails, non-DMA fails
- FENCE: valid timeout passes, zero timeout fails
- SIGNAL: valid ID passes, zero ID fails, 64-bit ID fails
- All 12 tests pass — C ABI bridge verified

### Tool Harness Tests (zig build test-harness)
- **All 43 tools** tested against:
  - Empty inputs (zero operands)
  - Boundary values (max u64, zero, page-aligned)
  - Zero operands
  - Zero aperture context
  - Thermal extreme context (150°C, halt=0)
- SPARK-specific validation:
  - SHIFT rejects non-page-aligned
  - FLOW rejects zero size
  - FENCE rejects zero timeout
  - SIGNAL rejects zero ID
  - REFRACT rejects zero total (dst_addr)
  - REFRACT accepts valid chunk within aperture
  - SHIFT rejects exceeding aperture
- Dispatch table completeness verified
- Info array matches dispatch table
- No null names or descriptions
- **14/15 pass** (REFRACT zero-size uses Max_Aperture default per SPARK spec — by design)

### SPARK Formal Verification (gnatprove)
- 5 SPARK tools formally verified: FLOW, SHIFT, FENCE, SIGNAL, REFRACT
- 30 checks proved, 0 errors
- Loop invariants, pre/post conditions verified
- No runtime exceptions possible

### Recipe Compilation
- `stream_960.alka` + `gtx960_2gb.alkavl` → `.alkas` + `.azoth`
  - 7 instructions → 224 bytes (7 packets)
- `purify_1070ti.alka` + `ivyb_pascal.alkavl` → `.alkas` + `.azoth`
  - 17 instructions → 544 bytes (17 packets)
- All SPARK validations pass at compile-time

## Executor Requirements

1. **Read `.alkas`** as a sequence of `struct vitriol_drop` packets
2. **Load `.alkavl`** Vial constraints before execution
3. **Validate each Drop** against Vial:
   - Aperture bounds (src/dst within BAR1 window)
   - Thermal limits (halt/throttle temperatures)
   - DMA capability (FLOW requires dma_capable=1)
   - Page alignment (SHIFT requires 4K alignment)
4. **Execute via DMA/ioctl** using physical addresses
5. **Rollback on failure** using `.azoth` packets
6. **Never call `device_release_driver()`** — use `ioremap()` on known BAR addresses

## Safety Notes

- **PAT conflict is non-fatal**: DMA uses physical addresses regardless of kernel virtual mapping
- **SPARK validation is mandatory**: Every Drop must pass SPARK validation before execution
- **Thermal limits are hard**: If current_temp >= thermal_halt, abort immediately
- **Rollback is required**: On any failure, execute `.azoth` packets in reverse order
- **Sliding window pattern**: The 1070 Ti stream uses SHIFT→FLOW→FENCE loops. The executor must complete each FENCE before proceeding to the next SHIFT.

## License

Apache 2.0 with Runtime Exception.
The `.alkas` and `.azoth` streams are generated output — not subject to license terms.
The compiler source and SPARK tools remain under Apache 2.0.
