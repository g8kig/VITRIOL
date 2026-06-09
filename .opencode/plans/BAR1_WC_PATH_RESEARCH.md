# Plan: BAR1 Write-Combining Path (Research & Feasibility)
**Date:** 2026-06-09
**Status:** Research / Future

## Objective

Evaluate whether the existing kernel-level BAR1 WC mapping (`vitriol-daemon/vitriol.c`) can be used for CPU-driven expert weight writes to VRAM, bypassing CUDA's DMA engine entirely — inspired by GDRCopy.

## Current Infrastructure

**File:** `vitriol-daemon/vitriol.c`

| Feature | Location | Status |
|---------|----------|--------|
| `pci_iomap_wc(pdev, 1, BAR_1_SIZE)` | Line 623 | Kernel-maps BAR1 with WC attribute |
| `pgprot_writecombine(vma->vm_page_prot)` | Line 697 | Userspace mmap with WC |
| `io_remap_pfn_range(vma, start, pfn, size, prot)` | Line 698 | Creates userspace mapping of BAR1 physical pages |
| `vitriol_mmap` | Lines 687-712 | Exposes BAR1 as a WC mmap region to userspace |

This infrastructure was built for the Alka NVMe→GPU direct DMA path and is operational for that purpose. It has never been used for CUDA-managed memory transfers.

## The GDRCopy Model

GDRCopy works as follows:
1. `cudaMalloc` allocates GPU memory (VRAM)
2. `gdr_pin_buffer(devptr, size, ...)` pins the GPU pages in the BAR1 aperture → GPU physical pages become visible on PCIe
3. `gdr_map(pin_handle, ...)` creates a CPU virtual mapping of the pinned BAR1 pages
4. CPU reads/writes via `memcpy` on the mapped pointer

The key enabling API: **`gdr_pin_buffer`** — which uses GPUDirect RDMA infrastructure to expose CUDA device pointers through BAR1. This requires:
- GPUDirect RDMA support in the NVIDIA driver
- The GPU to support BAR1 remapping (Tesla/Quadro; NOT GeForce)
- The CUDA driver to cooperate with the BAR1 mapping

## Why GTX 1070 Ti Likely Can't Do This

| Requirement | GTX 1070 Ti (GP104) | Required by GDRCopy |
|-------------|---------------------|---------------------|
| GPUDirect RDMA | **No** (GeForce) | Yes |
| BAR1 remapping | Limited (256 MB, no RDMA) | Full GPU VA→BAR1 mapping |
| `cudaMalloc` in BAR1 | No (can't pin arbitrary allocs) | Yes (`gdr_pin_buffer`) |
| NVIDIA driver flavor | Proprietary only | Proprietary or open |

**Technical barrier**: On GeForce cards, the NVIDIA driver does not expose the BAR1 page table to third-party kernel modules. The GPU's MMU (memory management unit) maps GPU virtual addresses to physical VRAM pages, and the BAR1 aperture is a fixed-size window into this space. Without driver cooperation, a kernel module cannot translate a `cudaMalloc`'d address to a BAR1 offset.

## What the Current `vitriol.c` BAR1 Mapping Actually Sees

The current BAR1 mmap exposes the raw PCIe BAR1 region. On GTX 1070 Ti, this is a 256 MB aperture into VRAM — but **which VRAM** depends on the GPU's internal BAR1 page table, which is managed by the NVIDIA driver. The kernel module's `pci_iomap_wc` maps this aperture, but the kernel cannot control which VRAM pages are visible through it.

The NVIDIA driver may use BAR1 for:
- Display scanout buffer
- Driver-internal allocations
- GART (Graphics Address Remapping Table) entries

Without GPUDirect RDMA, the CUDA driver won't route `cudaMalloc`'d memory through BAR1. So the BAR1 visible to `vitriol.c` likely contains display buffers and driver internals — not expert weights.

## What Would Need to Happen (Theoretical)

1. **NVIDIA driver modification** (impossible — closed source): Expose a BAR1 page table manipulation ioctl so a kernel module can map arbitrary CUDA allocations into BAR1.

2. **Use `nvidia-peermem`** (NVIDIA's GPUDirect RDMA peer memory driver): This module enables RDMA on GeForce cards in some configurations. But:
   - Only for Tesla/Quadro officially
   - Community workarounds for GeForce exist but are fragile
   - Requires kernel version and driver version matching
   - Not intended for CPU→GPU memory access; designed for NIC→GPU RDMA

3. **Abandon CUDA entirely**: Use the Vulkan path (Plan 5) with `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` on ReBAR-enabled hardware. The BAR1 WC mapping would be through Vulkan, not the custom kernel module.

## Recommendation: Do Not Pursue for GTX 1070 Ti

The BAR1 WC path is **not viable** on the current target hardware (GTX 1070 Ti). The infrastructure in `vitriol.c` is valuable for the Alka NVMe→GPU path and should stay, but it cannot be used for CUDA-managed memory writes on GeForce.

## Future Path: ReBAR + Vulkan

When targeting newer hardware with Resizable BAR (ReBAR) support:

| Hardware | ReBAR Support | Vulkan WC path available? |
|----------|---------------|--------------------------|
| GTX 1070 Ti (GP104) | No (BIOS hack possible) | No (not ReBAR-enabled) |
| RTX 3060+ (Ampere) | Yes | Yes |
| RX 6000+ (RDNA2) | Yes (native) | Yes |
| RX 7000+ (RDNA3) | Yes (native) | Yes |

On ReBAR-enabled GPUs, Vulkan exposes `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` memory types with WC attributes. This gives the same CPU-driven write capability as GDRCopy, but through standard Vulkan APIs — no kernel module needed.

The Chimera Vulkan path (Milestone 3) is the correct vehicle for this optimization on future hardware.

## What to Keep in `vitriol.c` BAR1 Code

| Code | Keep? | Reason |
|------|-------|--------|
| `vitriol_map_bar1()` (line 615) | Yes | Used by Alka DMA path |
| `vitriol_mmap()` WC (line 687) | Yes | Alka userspace needs it |
| `pci_iomap_wc` (line 623) | Yes | Alka kernel path |
| `pgprot_writecombine` (line 697) | Yes | Alka userspace path |

No changes to `vitriol.c` needed. Add a comment noting the GeForce limitation.

## Research Steps (if pursuing anyway)

1. **Verify BAR1 content on GTX 1070 Ti**: Write a test program that mmaps the vitriol device, writes a known pattern to BAR1, then reads it back from a CUDA kernel at a matching address. If the GPU reads the pattern, some VRAM pages are BAR1-accessible.

2. **Test `nvidia-peermem`**: Install the NVIDIA peer memory kernel module and check if `gdr_pin_buffer` can be called (will likely fail on GeForce).

3. **Benchmark current `cuMemcpyHtoDAsync`**: Confirm the ~6 µs overhead is real on this hardware. If it's actually lower (e.g., 2-3 µs), the BAR1 path's benefit diminishes.

4. **Test on RTX 3060+**: If access to a ReBAR-capable GPU is available, test the Vulkan WC path as a GDRCopy alternative. This is the most productive research direction.

## Conclusion

| Question | Answer |
|----------|--------|
| Can BAR1 WC writes replace CUDA DMA on GTX 1070 Ti? | **No** — hardware/driver limitations prevent it |
| Is the `vitriol.c` BAR1 code useful? | **Yes** — for the Alka NVMe DMA path |
| Best path to CPU-driven WC writes? | **Vulkan Chimera M3** on ReBAR hardware |
| Should time be spent on this now? | **No** — focus on double-buffer + predictor for 1070 Ti |
