# Plan: Expert Pinning (PopFetcher/HOBBIT-style)

**Goal:** Pre-load frequently-used experts into VRAM at startup, eliminating their PCIe transfer cost during inference. MoE expert usage follows a power law — a tiny fraction handles most tokens.

**Current bottleneck:** The MMV fast path (MMVQ/MMQ/MMF) reads ALL experts from the host memory tensor pointer via a single fused kernel launch. There is no per-expert VRAM redirection. Without kernel changes, pinning only helps the slow per-expert loop (2-3× slower, not worth it).

---

## The Core Problem

```
            ┌─ Fast path (single kernel):
            │   src0->data = host pointer
            │   All 8 experts read from host over PCIe in one launch
            │
ggml_cuda_  ┤   To add VRAM redirection: need per-expert slot_table[]
mul_mat_id  │   in the kernel argument. Requires CUDA kernel change.
            │
            └─ Slow path (per-expert loop):
                Each expert: check LRU, DMA to VRAM if needed, matmul
                2-3× slower than fast path even with cache hits
```

Two approaches to bypass this:

---

## Approach A: Per-Expert VRAM Side-Table (1-2 weeks)

Modify MMV/MMQ/MMF kernels to accept an optional `vitriol_expert_slot` table:

```c
struct vitriol_expert_slot {
    void *vram_ptr;   // VRAM address, or NULL = read from host
};
```

The kernel reads `slot_table[i02].vram_ptr` — if non-NULL, use it; otherwise read from `src0->data + i02 * nb02` (current behavior). This is a one-line indirection inside the kernel.

**Steps:**
1. Allocate device-side table (`cudaMalloc`)
2. Fill table entries for pinned experts at load time via `vitriol_pin_load()`
3. Pass table pointer as kernel argument in `ggml_cuda_mul_mat_vec_q` etc.
4. Kernel uses `vram_ptr` when available, falls through to host read otherwise

**Effort:** 1-2 weeks (CUDA kernel changes, testing across quant types)
**Gain:** Enables all future optimization (pinning, speculative routing, LRU at generation time)

---

## Approach B: Software-Only Tensor-Level Preload (~3-4 days)

No kernel changes. Instead:
1. At each layer, BEFORE the FFN, DMA the ENTIRE expert weight tensor to VRAM
2. Temporarily redirect `src0->data` to VRAM pointer for that layer's FFN
3. After FFN, restore the host pointer

The MMV kernel reads from `.data` — it doesn't care whether it's host or device memory. If the entire tensor is in VRAM, all 8 experts read from VRAM.

**Cost:** ~128 MB per tensor × 3 tensors (gate/up/down) = ~384 MB per layer. With ~4.9 GB free VRAM, can keep 10-12 layers' tensors resident.

**Trade-off:** Higher VRAM usage per layer (entire tensor vs individual experts), but zero CUDA kernel changes. Works right now.

---

## Calibration (Both Approaches)

Script that runs inference with `VITRIOL_PIN_CALIBRATE=1` to log all `(tensor_name, expert_idx, count)` tuples. Outputs sorted pin list for the top-N most-frequently-used expert slots.

```bash
scripts/calibrate_pins.sh --model model.gguf --dataset ~/calib.txt --output ~/.vitriol/pins.txt
```

**Config keys added to scripts/vitriol:**
```
vitriol.pin_file = ~/.vitriol/pins.txt
vitriol.pin_count = 20
```

---

## Recommendation

**Start with Approach B (no kernel changes).** It's achievable in 3-4 days and gives an immediate 5-10% gain with pre-loaded layers. If that works well, Approach A is the long-term investment for deeper integration.
