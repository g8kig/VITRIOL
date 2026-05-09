# VITRIOL - Spatial Transformer for Infinite VRAM
<img src="vitriol_logo.svg" alt="VITRIOL" width="200"/>

*"Visita Interiora Terrae Rectificando Invenies Occultum Lapidem"*

(Visit the Interior of the Earth, by Rectifying you will find the Hidden Stone)

## Current Status

**Phase 1: Foundation (Complete)**
- [x] Stub kernel module (test mode) - PASSED
- [x] Socket API design
- [x] Python client library
- [x] Daemon skeleton
- [x] KoboldCPP integration via context rectifier shim
- [x] End-to-end inference test - OPERATIONAL

## Quick Start

### Phase 1: KoboldCPP + VITRIOL Shim (Stable)

```bash
# 1. Launch the VITRIOL stack
./launch_vitriol.sh

# 2. Test the integration
python3 test_shim.py

# 3. Point your agent to VITRIOL
# Configure OpenCode/agent to use: http://localhost:5010/v1/chat/completions
```

### Phase 2+: Kernel Module (Advanced)

```bash
# 1. Build kernel module (with stubs for safe testing)
cd ../brief-compiler && cargo build
cd ../linux-pipe-module
rm -f vitriol_new_ffi.c
./brief-compiler c vitriol_new_ffi.bv --target linux_kernel --test-mode
make

# 2. Load module (stub mode - safe, no real hardware)
sudo insmod vitriol_new_ffi.ko test_mode=1

# 3. Check output
sudo dmesg | grep VITRIOL

# 4. Unload
sudo rmmod vitriol_new_ffi
```

### Phase 2+: Kernel Module (Advanced)

```bash
# 1. Build kernel module (with stubs for safe testing)
cd ../brief-compiler && cargo build
cd ../linux-pipe-module
rm -f vitriol_new_ffi.c
./brief-compiler c vitriol_new_ffi.bv --target linux_kernel --test-mode
make

# 2. Load module (stub mode - safe, no real hardware)
sudo insmod vitriol_new_ffi.ko test_mode=1

# 3. Check output
sudo dmesg | grep VITRIOL

# 4. Unload
sudo rmmod vitriol_new_ffi
```

## Architecture

### Phase 1: Context Rectification (Operational)

```
┌──────────────────────────────────────────────────────────────┐
│              OpenCode / Agent Clients                        │
│         (sending large contexts)                             │
└─────────────────────────┬────────────────────────────────────┘
                          │ HTTP POST /v1/chat/completions
                          ▼
┌──────────────────────────────────────────────────────────────┐
│         VITRIOL Shim (port 5010, Python)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. Thermal Poll: nvidia-smi → halt if >85°C           │  │
│  │  2. Calcination: Drop middle messages, keep last 4     │  │
│  │  3. Sublimation: Strip <reasoning>, tool_results       │  │
│  │  4. Coagulation: Enforce 7k token cap                  │  │
│  └────────────────────────────────────────────────────────┘  │
│          Output: Rectified context (7k tokens max)           │
└─────────────────────────┬────────────────────────────────────┘
                          │ Forward rectified request
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              KoboldCPP (port 5001, CUDA)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Qwen3.5 9B Q4_K_M (5.5GB)                             │  │
│  │  GPU Layers: 25-30                                     │  │
│  │  Context: 4096-8192 tokens                             │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Phase 2+: Hardware Acceleration (Planned)

```
┌──────────────────────────────────────────────────────────┐
│                    Python Agent Clients                  │
│            (LangChain, CrewAI, Custom Agents)            │
└─────────────────────────┬────────────────────────────────┘
                          │ Unix Socket (/var/run/vitriol.sock)
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    vitriol-daemon (Userspace)            │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐│
│  │   Socket    │  │   llama.cpp │  │   Layer Manager    ││
│  │   Server    │  │  Inference  │ │   (LRU, Streaming) ││
│  └─────────────┘  └─────────────┘  └────────────────────┘│
└─────────────────────────┬────────────────────────────────┘
                          │ ioctl / Character Device
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    vitriol.ko (Kernel Module)            │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐│
│  │   Safety    │  │     DRM     │  │   DMA Engine       ││
│  │   Layer     │  │  (nvidia)   │ │  (dmaengine API)   ││
│  └─────────────┘  └─────────────┘  └────────────────────┘│
└─────────────────────────┬────────────────────────────────┘
                          │ PCIe
                          ▼
┌────────────────┐     ┌────────────────┐
│      SSD       │────▶│   GPU VRAM     │
│   (Storage)    │ DMA │  (8GB GTX)     │
└────────────────┘     └────────────────┘
```

```
┌──────────────────────────────────────────────────────────────┐
│              OpenCode / Agent Clients                        │
│         (sending large contexts)                             │
└─────────────────────────┬────────────────────────────────────┘
                          │ HTTP POST /v1/chat/completions
                          ▼
┌──────────────────────────────────────────────────────────────┐
│         VITRIOL Shim (port 5010, Python)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  1. Thermal Poll: nvidia-smi → halt if >85°C           │  │
│  │  2. Calcination: Drop middle messages, keep last 4     │  │
│  │  3. Sublimation: Strip <reasoning>, tool_results       │  │
│  │  4. Coagulation: Enforce 7k token cap                  │  │
│  └────────────────────────────────────────────────────────┘  │
│          Output: Rectified context (7k tokens max)           │
└─────────────────────────┬────────────────────────────────────┘
                          │ Forward rectified request
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              KoboldCPP (port 5001, CUDA)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Qwen3.5 9B Q4_K_M (5.5GB)                             │  │
│  │  GPU Layers: 25-30                                     │  │
│  │  Context: 4096-8192 tokens                             │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Phase 2+: Hardware Acceleration (Planned)

```
┌──────────────────────────────────────────────────────────┐
│                    Python Agent Clients                  │
│            (LangChain, CrewAI, Custom Agents)            │
└─────────────────────────┬────────────────────────────────┘
                          │ Unix Socket (/var/run/vitriol.sock)
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    vitriol-daemon (Userspace)            │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐│
│  │   Socket    │  │   llama.cpp │  │   Layer Manager    ││
│  │   Server    │  │  Inference  │ │   (LRU, Streaming) ││
│  └─────────────┘  └─────────────┘  └────────────────────┘│
└─────────────────────────┬────────────────────────────────┘
                          │ ioctl / Character Device
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    vitriol.ko (Kernel Module)            │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐│
│  │   Safety    │  │     DRM     │  │   DMA Engine       ││
│  │   Layer     │  │  (nvidia)   │ │  (dmaengine API)   ││
│  └─────────────┘  └─────────────┘  └────────────────────┘│
└─────────────────────────┬────────────────────────────────┘
                          │ PCIe
                          ▼
┌────────────────┐     ┌────────────────┐
│      SSD       │────▶│   GPU VRAM     │
│   (Storage)    │ DMA │  (8GB GTX)     │
└────────────────┘     └────────────────┘
```

## Safety Levels

| Level | Operations | Risk |
|-------|-------------|------|
| `safety_level=1` | Read-only GPU queries, DRM copy | **LOW** |
| `safety_level=2` | + DMA writes (opt-in) | **MEDIUM** |
| `safety_level=3` | + Raw PCI access | **HIGH** |

**Default**: `safety_level=1` (read-only, safe)

## Socket API

Connect to `/var/run/vitriol.sock`:

```bash
# Status check
python -c "
from libvitriol import VitriolClient
with VitriolClient() as c:
    print(c.get_status())
"
```

## Testing

**IMPORTANT**: Always test in stub mode first:

```bash
# Safe stub test
sudo insmod vitriol_new_ffi.ko test_mode=1
sudo dmesg | grep VITRIOL
sudo rmmod vitriol_new_ffi
```

## Files

| File | Purpose |
|------|---------|
| `vitriol_shim.py` | **Phase 1**: KoboldCPP context rectifier proxy |
| `launch_vitriol.sh` | **Phase 1**: Unified launch script |
| `vitriol_new_ffi.bv` | **Phase 2+**: Brief source (kernel module) |
| `vitriol-daemon/` | **Phase 2+**: Rust daemon (socket server) |
| `libvitriol/` | Python client library |
| `test_shim.py` | **Phase 1**: Integration test suite |
| `test_vitriol.sh` | **Phase 2+**: Kernel module test harness |
| `VITRIOL_IMPLEMENTATION_PLAN.md` | Complete 4-phase implementation plan |
| `PHASE1_COMPLETE.md` | Phase 1 operational status |

## Dependencies

### Phase 1 (KoboldCPP Shim)
- Python 3.8+
- Flask (`pip3 install flask requests`)
- KoboldCPP
- CUDA-capable GPU (NVIDIA with 8GB+ VRAM recommended)

### Phase 2+ (Kernel Module)
- Linux kernel headers (6.x)
- Rust (for daemon)
- Brief compiler (../brief-compiler)
- CUDA/llama.cpp (for inference)

## Model

**Recommended:** Qwen3.5 9B Q4_K_M (~5.5GB)

Download to a location of your choice:
```bash
huggingface-cli download Qwen/Qwen3.5-9B-Instruct-GGUF \
  --include "qwen3.5-9b-instruct-q4_k_m.gguf" \
  --local-dir ~/models/
```

Update paths in:
- `launch_vitriol.sh` (KoboldCPP `--model` parameter)
- `run_qwen.sh` (if using separate launch script)

## Safety Notes

### Phase 1 (Shim) - SAFE
- Pure Python userspace code
- No hardware access
- Thermal monitoring via `nvidia-smi` (read-only)
- Can be safely tested on any system with KoboldCPP

### Phase 2+ (Kernel Module) - ADVANCED
- This machine is primary driver - HIGH CAUTION
- Never use `safety_level=3` without backup
- Test incrementally: each phase before advancing
- Check `dmesg` after each operation
- Always start with `test_mode=1` (stub mode)
- Use iGPU for display when testing hardware access
- Always start with `test_mode=1` (stub mode)
- Use iGPU for display when testing hardware access
