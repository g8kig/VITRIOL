## 4.2 probe.rs — Hardware Probing

nvidia-smi fields: name, memory.total, compute_cap, pcie.link.gen.current, pcie.link.width.current

CPU: /proc/cpuinfo "model name" + AVX2 flag
RAM: /proc/meminfo MemTotal
IPC lock: getcap on server binary

Structs:

```rust
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vram_mib: u64,
    pub compute_cap: String,
    pub pcie_gen: u32,
    pub pcie_width: u32,
}

pub struct HardwareInfo {
    pub probed_at: String,
    pub gpus: Vec\u003cGpuInfo\u003e,
    pub cpu: String,
    pub has_avx2: bool,
    pub ram_mib: u64,
    pub gpu_count: u32,
    pub has_ipc_lock: bool,
}
```
