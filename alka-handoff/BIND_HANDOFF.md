# VITRIOL BIND — PCIe Device Seizure

> **Updated**: 2026-05-13 — v5.1 BIND/BIND! implementation
> **See also**: `HANDOFF.md` for core stream format and executor requirements
> **Dependency**: This document describes what VITRIOL must implement to support Alka's `BIND` instruction

---

## Overview

`BIND` (opcode 0x3A) seizes a PCIe device from its current driver and hands it to VITRIOL/Alka. `BIND!` (force flag) uses hot-remove/rescan for devices that resist unbind (notably NVIDIA's proprietary driver).

Alka provides a **userspace fallback** (`vitriol-bind`) that works on any Linux 4.x+ via sysfs. VITRIOL's kernel module should implement the same operations internally for lower latency.

## Three Implementation Levels

### Level 1: Userspace Sysfs (Fallback)

Alka ships `vitriol-bind` — a standalone CLI that performs the sysfs dance:

```
vitriol-bind 0000:01:00.0                # Normal: unbind → vfio-pci
vitriol-bind 0000:01:00.0 --force        # Force: hot-remove → rescan
vitriol-bind 0000:01:00.0 --status       # Query current driver
vitriol-bind 0000:01:00.0 --restore nvidia  # Restore original driver
```

It can also be invoked from the executor:
```
alka --execute --bind 0000:01:00.0 stream.alkas
alka --execute --bind-force 0000:01:00.0 stream.alkas
```

**Implementation** (Zig, `src/bind/binder.zig`):
| Operation | Sysfs Path | Action |
|-----------|------------|--------|
| Unbind | `/sys/bus/pci/drivers/{name}/unbind` | Write BDF |
| Bind | `/sys/bus/pci/drivers/vfio-pci/bind` | Write BDF |
| New ID | `/sys/bus/pci/drivers/vfio-pci/new_id` | Write `0x0000 0x0000` |
| Hot-remove | `/sys/bus/pci/devices/{BDF}/remove` | Write `1` |
| Rescan | `/sys/bus/pci/rescan` | Write `1` |

**Important**: All sysfs operations require root (CAP_SYS_ADMIN). The `--force` path (hot-remove) works even against NVIDIA's driver which refuses `device_release_driver()` due to internal refcounts.

### Level 2: Kernel Module (vitriol.ko) — Required for Real-Time

VITRIOL's kernel module should implement BIND as a direct IOCTL to avoid userspace context switches:

```c
// IOCTL: BIND_DEVICE
struct vitriol_bind_request {
    char bdf[13];           // "0000:01:00.0\0"
    uint8_t force;          // 0 = normal, 1 = force
    char prev_driver[64];   // OUT: previous driver name (for rollback)
};

// Normal bind:
//   1. pci_get_domain_bus_and_slot() → struct pci_dev*
//   2. device_release_driver(dev)
//   3. Bind to vfio-pci stub internally
//   4. Return previous driver name

// Force bind:
//   1. pci_get_domain_bus_and_slot() → struct pci_dev*
//   2. pci_stop_and_remove_bus_device(dev)
//   3. Notify userspace to rescan (or rescan internally)
```

**NVIDIA-specific**: The NVIDIA driver (`nvidia.ko`) uses internal refcounts that cause `device_release_driver()` to return -EBUSY. The kernel module must use the hot-remove path for NVIDIA devices:
```c
if (force || dev->driver == &nvidia_driver) {
    pci_stop_and_remove_bus_device(dev);
    // Schedule rescan via workqueue
    schedule_work(&rescan_work);
}
```

### Level 3: Cooperative Mode (GPUDirect RDMA) — For CUDA Workloads

When NVIDIA's driver must stay loaded (for CUDA compute), BIND is not used. Instead, VITRIOL uses NVIDIA's P2P API for cooperative memory access:

```c
// Step 1: Request VRAM pages from NVIDIA driver
struct nvidia_p2p_page **pages;
nvidia_p2p_get_pages(dev, 0, vram_size, &pages);

// Step 2: DMA from NVMe to VRAM using physical addresses from pages
for (/* each page */) {
    // Use page->physical_address as DMA destination
    dma_engine.transfer(nvme_addr, page->physical_address, page_size);
}

// Step 3: Release pages
nvidia_p2p_put_pages(dev, pages);
```

**Detection**: Check `/sys/bus/pci/drivers/nvidia/` for presence. If NVIDIA driver is bound AND the recipe uses FLOW without BIND, use cooperative mode.

## Packet Format

The BIND packet in `.alkas` files uses the standard 32-byte Drop format:

```
Byte 0:    0x3A (opcode)
Byte 1:    0x01 (force flag in bit 0)
Bytes 2-3: vessel_id
Bytes 4-11: src_addr (PCI vendor/device: device<<16 | vendor)
Bytes 12-31: reserved / crc
```

## Rollback (Azoth)

The Azoth counterpart of BIND is `CLAIM` (or restore to original driver):

| Forward | Azoth Counterpart | Action |
|---------|-------------------|--------|
| `BIND` | Restore original driver | Write BDF to original driver's `bind` |
| `BIND!` | Rescan | Device already re-appears after rescan |
| `BIND` (cooperative) | `nvidia_p2p_put_pages()` | Release pinned pages |

## Error Handling

| Scenario | Behavior | Message |
|----------|----------|---------|
| BDF not found | Error | "Device not found at BDF" |
| Unbind refused (NVIDIA) | Fall back to hot-remove | "Unbind failed, trying hot-remove" |
| vfio-pci not loaded | `modprobe vfio-pci` | "Loaded vfio-pci module" |
| Permission denied | Error | "Need root (CAP_SYS_ADMIN)" |
| `BIND!` already unbound | Warning then rescan | "Device already unbound, rescanning" |

## Testing Without Hardware

Use the dry-run mode to verify BIND sequences without touching real devices:

```
alka --mock stream_960.alka.alkas
```

The mock executor logs each packet without executing. For BIND specifically, mock returns success without touching sysfs.

## VITRIOL Integration Checklist

- [ ] **Level 1 (userspace fallback)**: Already implemented in `vitriol-bind`. Test via `sudo vitriol-bind 0000:01:00.0 --status`
- [ ] **Level 2 (kernel module)**: Add `BIND_DEVICE` IOCTL to `vitriol.ko`:
  - [ ] `pci_get_domain_bus_and_slot()` / `device_release_driver()` path
  - [ ] `pci_stop_and_remove_bus_device()` hot-remove path
  - [ ] Rescan workqueue
  - [ ] Return previous driver name for rollback
- [ ] **Level 3 (cooperative)**: Add `nvidia_p2p_get_pages()` integration:
  - [ ] Export symbol lookup (NVIDIA driver is proprietary, use `kallsyms_lookup_name`)
  - [ ] Page pinning and physical address extraction
  - [ ] Page release on completion/error
- [ ] **Packet handling**: Parse `flags` byte for force bit (0x01)
- [ ] **Azoth rollback**: Implement `BIND_RESTORE` IOCTL for driver restoration
- [ ] **Integration test**: Stream from NVMe to VRAM using each BIND level

## Source Files in Alka Repo

| File | Purpose |
|------|---------|
| `src/bind/binder.zig` | Userspace sysfs helper (driver unbind/bind, hot-remove, rescan) |
| `src/vitriol_bind.zig` | `vitriol-bind` CLI binary |
| `src/main.zig` | `--execute --bind` integration |
| `src/tools/substrate/bind.zig` | BIND tool (validation + mock execute) |
| `pharmacopia.json` | BIND chain metadata (post_state: `device_bound`, `vessel_claimed`) |
