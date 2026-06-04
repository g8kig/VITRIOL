# VITRIOL × Brief Integration — Pre-Chewed Implementation Cookbook

**Date:** 2026-06-04 08:14 UTC
**Parent:** `vitriol-brief-integration-master-plan-2026-06-04.md`
**Status:** Design reference — ready for implementation

---

This document provides the concrete binary layouts, algorithms, API signatures, and code patterns needed to implement the VITRIOL × Brief integration. It bridges from the high-level phase plan to actual code.

---

## 1. .VPO Binary Layout (Exact Byte Specification)

All integers are little-endian. All structs are packed with no padding.

```c
#define VPO_MAGIC   0x32504F56  // "VPO2"
#define VPO_VERSION 2

// ── File Header (56 bytes) ──

struct vpo_header {
    uint32_t magic;              // offset 0x00 — 0x32504F56
    uint32_t version;            // offset 0x04 — 2
    uint8_t  model_hash[32];     // offset 0x08 — blake3 of source GGUF
    uint32_t section_count;      // offset 0x28 — number of sections
    uint32_t template_count;     // offset 0x2C — 0 if not folded
    uint64_t total_lut_bytes;    // offset 0x30 — sum of all LUT data
};                               // total: 56 bytes

// ── Section Table Entry (48 bytes each) ──
// Follows immediately after header. Exactly section_count entries.

struct vpo_section_entry {
    uint32_t section_id;         // unique, monotonic, append-only
    uint32_t pass_id;            // 1=matmul, 2=KV, 3=MLP, 0xFF=profile
    uint64_t created_at_ms;      // unix timestamp milliseconds
    uint8_t  hw_requirement;     // 0=CPU_LUT, 1=SPIRV, 2=PTX, 0xFF=any
    uint8_t  data_format;        // 0=f32, 1=f16, 2=block_quant
    uint16_t _reserved;
    uint32_t layer_count;        // layers described in this section
    uint64_t layer_index_offset; // file offset to vpo_layer_entry[]
    uint64_t lut_data_offset;    // file offset to raw LUT data
    uint64_t lut_data_size;      // bytes of LUT data
};

// ── Layer Index Entry (40 bytes each) ──
// At layer_index_offset. Exactly layer_count entries.

struct vpo_layer_entry {
    uint32_t layer_id;           // logical layer index in model
    uint32_t tensor_name_hash;   // fnv1a-32("blk.N.ffn_gate.weight")
    uint8_t  quant_type;         // GGML_IQ2_XXS=39, etc.
    uint8_t  act_bits;           // 4 for 4-bit activations
    uint16_t _reserved;
    uint32_t shape[4];           // rows, cols, (0, 0) for 2D
    uint32_t template_id;        // index into template table, 0xFFFFFFFF if none
    uint64_t lut_offset;         // byte offset within section's lut_data
    uint64_t lut_entry_size;     // bytes per (neuron × max_act) block
};

// ── Template Table Entry (variable size) ──
// At section_table_offset + section_count * sizeof(entry).
// Exactly template_count entries.

struct vpo_template_entry {
    uint32_t template_id;
    uint8_t  quant_type;
    uint8_t  act_bits;
    uint16_t _reserved;
    uint32_t shape[4];
    uint32_t instance_count;
    uint32_t instance_layer_ids[];  // variable-length
};

// ── Footer (32 bytes) ──
// Last 32 bytes of file.

struct vpo_footer {
    uint8_t checksum[32];  // blake3 of all preceding bytes
};
```

### File Layout (Diagram)

```
Offset  │ Content
────────┼──────────────────────────────────────────────
0x0000  │ vpo_header (56 bytes)
0x0038  │ vpo_section_entry[0] (48 bytes)
0x0068  │ vpo_section_entry[1] (48 bytes)
        │ ...
        │ (section_count × 48 bytes)
        │
        │ vpo_template_entry[0] (32 + instance_count*4 bytes)
        │ vpo_template_entry[1]
        │ ...
        │
        │ ── Section 0 ──
   ┌──  │ vpo_layer_entry[0] (40 bytes)
   │    │ vpo_layer_entry[1]
   │    │ ...
   │    │ (section->layer_count × 40 bytes)
   │    │
   │    │ LUT data (section->lut_data_size bytes)
   │    │     raw float/f16/quantized array
   └──  │
        │ ── Section 1 ──
        │ ...
        │
(end-32)│ vpo_footer.blake3 checksum (32 bytes)
```

### LUT Data Layout (Per Layer)

For a weight matrix W with shape `[output_neuron_count, input_channel_count]` and activation bit-width N:

```
For each output neuron o (0..output_neuron_count):
  For each possible activation value a (0..2^N-1):
    For each input channel i (0..input_channel_count):
      LUT[o][a][i] = dequant(W[o][i]) * a   // precomputed product
    (Accumulation across input channels happens at runtime:
     output[o] = sum over i of LUT[o][activation[i]][i])

Total bytes per layer: output_neuron_count × 2^N × input_channel_count × sizeof(element)
```

For IQ2 (2-bit weights) with 4-bit activations on a 14336×2048 expert:
- `14336 × 16 × 2048 × 4` bytes (f32) = 1.8 GB per expert — too large for all experts
- With f16: 0.9 GB per expert
- With block quantization (int8): 0.45 GB per expert

This means for large experts, the LUT is proportionally large. The baking pass should store the LUT in the most compact format available (configurable via `--compress`). For very wide layers, consider:

1. **Per-block LUT**: subdivide the input channel dimension into blocks (e.g., 32 channels each), compute per-block LUTs, sum blocks at runtime. Reduces LUT size by a factor of `input_channel_count / block_size`.

2. **Hybrid**: LUT for the first few bits, arithmetic for the remaining bits (e.g., 4-bit → 2-bit LUT + 2-bit multiply).

The baking tool selects the optimal strategy per layer based on a size × speed heuristic.

#### Cache Line Alignment

The inner loop at runtime is:
```
for i in 0..input_channel_count:
    sum += LUT[neuron][activation[i]][i]
```

For CPU LUT lookups, the critical performance factor is **L1 cache utilization**. The memory layout must ensure that `i` (the input channel) is the **innermost striding dimension**, so sequential iterations access adjacent memory addresses:

```
Memory order: LUT[0][0][0], LUT[0][0][1], LUT[0][0][2], ..., LUT[0][0][N-1],
              LUT[0][1][0], LUT[0][1][1], ...
```

This allows the CPU hardware prefetcher to stay ahead of the loop. Additionally, each neuron's LUT block should be aligned to **64-byte cache lines**:

```c
// In the baking tool, pad LUT blocks to 64-byte boundary:
size_t block_size = (2^act_bits) * input_channel_count * sizeof(element);
size_t aligned_block = ((block_size + 63) / 64) * 64;

// The runtime assumes:
//   LUT[layer][neuron] is at layer_base + neuron * aligned_block
//   All elements within a block are sequential in memory
```

In VITRIOL's VPO loader, the mmap'd LUT data inherits the OS page alignment (4 KB). For fine-grained cache alignment at the neuron level, the baking tool computes `aligned_block` and writes padding bytes between neuron blocks. The `lut_entry_size` in the layer index stores the aligned block size (including padding).

#### Hybrid Fallback Path (GPU DMA from Mapped LUTs)

The `cudaHostRegisterMapped` + `cudaHostGetDevicePointer` path gives a graceful degradation option: if the CPU LUT matmul is too slow for a particular layer (e.g., very wide FFN), the GPU can read the exact same LUT data directly from system RAM via DMA, treating it as a device-accessible SSBO.

This means:
- CPU LUT path: `brief_lut_matmul` reads LUT data from `lut_data` (host pointer, system RAM)
- GPU fallback path: existing CUDA kernels read LUT data from `gpu_lut_ptr` (mapped device pointer, same system RAM)
- **No data copy** — the same physical memory pages serve both paths
- Selection per-layer at runtime: if CPU LUT latency exceeds threshold, redirect that layer to GPU path

The VPO loader stores `gpu_lut_ptr` in `loaded_section_t` for this purpose.

---

## 2. VPO Loader Algorithm — C Pseudocode

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <cuda_runtime.h>

// ── Device capabilities ──
typedef struct {
    bool has_cuda;
    int  cuda_cc_major;         // 6=Pascal, 7=Volta, 8=Turing, 9=Ampere
    int  cuda_cc_minor;
    bool has_vulkan;
    bool prefer_cpu_lut;        // true when GPU VRAM is small
    char device_name[256];
} device_caps_t;

device_caps_t g_device_caps;

void detect_device_caps() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    g_device_caps.has_cuda = true;
    g_device_caps.cuda_cc_major = prop.major;
    g_device_caps.cuda_cc_minor = prop.minor;
    g_device_caps.prefer_cpu_lut = (prop.totalGlobalMem < 12ULL * 1024 * 1024 * 1024);
    strncpy(g_device_caps.device_name, prop.name, 255);
    // Vulkan detection via vkEnumeratePhysicalDevices (simplified)
}

// ── Section selection ──
bool section_matches_hardware(vpo_section_entry* s, device_caps_t* caps) {
    uint8_t req = s->hw_requirement;
    if (req == 0)   return true;                 // CPU_LUT always works
    if (req == 1)   return caps->has_vulkan;     // SPIRV needs Vulkan
    if (req == 2)   return caps->has_cuda;       // PTX needs CUDA
    return true;                                   // unknown = load anyway
}

// ── Main load ──
typedef struct {
    uint32_t    layer_count;
    vpo_layer_entry* layers;     // owned copy
    void*       lut_data;        // mmap'd region
    uint64_t    lut_data_size;
    int         data_format;     // 0=f32, 1=f16, 2=block_quant
} loaded_section_t;

typedef struct {
    uint8_t  model_hash[32];
    uint32_t section_count;
    loaded_section_t* sections;
    int      fd;                  // kept open for mmap lifetime
} vpo_handle_t;

vpo_handle_t* vpo_load(const char* vpo_path, const uint8_t model_hash[32]) {
    FILE* f = fopen(vpo_path, "rb");
    if (!f) return NULL;
    int fd = fileno(f);

    // Read header
    vpo_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }
    if (hdr.magic != VPO_MAGIC)               { fclose(f); return NULL; }
    if (hdr.version != VPO_VERSION)           { fclose(f); return NULL; }

    // Verify model hash
    if (memcmp(hdr.model_hash, model_hash, 32) != 0) { fclose(f); return NULL; }

    // Read section table
    size_t table_bytes = hdr.section_count * sizeof(vpo_section_entry);
    vpo_section_entry* entries = (vpo_section_entry*)malloc(table_bytes);
    fread(entries, sizeof(vpo_section_entry), hdr.section_count, f);

    // Count matching sections
    int match_count = 0;
    int match_ids[256]; // fixed max for simplicity
    for (uint32_t i = 0; i < hdr.section_count; i++) {
        if (section_matches_hardware(&entries[i], &g_device_caps))
            match_ids[match_count++] = i;
    }

    // Allocate handle
    vpo_handle_t* vpo = (vpo_handle_t*)calloc(1, sizeof(vpo_handle_t));
    memcpy(vpo->model_hash, hdr.model_hash, 32);
    vpo->section_count = match_count;
    vpo->sections = (loaded_section_t*)calloc(match_count, sizeof(loaded_section_t));
    vpo->fd = fd;

    // Load each matching section
    for (int m = 0; m < match_count; m++) {
        int si = match_ids[m];
        vpo_section_entry* se = &entries[si];

        loaded_section_t* ls = &vpo->sections[m];
        ls->data_format = se->data_format;
        ls->lut_data_size = se->lut_data_size;
        ls->layer_count = se->layer_count;

        // Read layer index
        ls->layers = (vpo_layer_entry*)malloc(
            se->layer_count * sizeof(vpo_layer_entry));
        fseek(f, se->layer_index_offset, SEEK_SET);
        fread(ls->layers, sizeof(vpo_layer_entry), se->layer_count, f);

        // mmap LUT data
        ls->lut_data = mmap(NULL, se->lut_data_size,
                            PROT_READ,
                            MAP_PRIVATE | MAP_POPULATE,
                            fd, se->lut_data_offset);

        // Register for GPU DMA access
        cudaError_t err = cudaHostRegister(
            ls->lut_data, se->lut_data_size,
            cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            // Non-fatal: CPU LUT path still works without DMA
            fprintf(stderr, "VPO: cudaHostRegister failed: %s\n",
                    cudaGetErrorString(err));
        }
    }

    // Verify footer checksum
    uint8_t footer_hash[32];
    fseek(f, -32, SEEK_END);
    fread(footer_hash, 32, 1, f);
    // blake3 recomputation over file minus last 32 bytes
    // (omitted for brevity — use blake3_hasher in practice)

    free(entries);
    fclose(f);
    return vpo;
}

const float* vpo_lookup_lut(vpo_handle_t* vpo, uint32_t layer_id,
                            uint64_t* out_entry_size) {
    for (uint32_t s = 0; s < vpo->section_count; s++) {
        loaded_section_t* ls = &vpo->sections[s];
        for (uint32_t i = 0; i < ls->layer_count; i++) {
            if (ls->layers[i].layer_id == layer_id) {
                if (out_entry_size)
                    *out_entry_size = ls->layers[i].lut_entry_size;
                return (const float*)((const uint8_t*)ls->lut_data
                                      + ls->layers[i].lut_offset);
            }
        }
    }
    return NULL;
}

void vpo_unload(vpo_handle_t* vpo) {
    for (uint32_t s = 0; s < vpo->section_count; s++) {
        loaded_section_t* ls = &vpo->sections[s];
        cudaHostUnregister(ls->lut_data);
        munmap(ls->lut_data, ls->lut_data_size);
        free(ls->layers);
    }
    free(vpo->sections);
    close(vpo->fd);
    free(vpo);
}
```

---

## 3. Template Folding Algorithm

The folding pass identifies layers with identical structural properties and groups them into templates.

```python
# Pseudocode (Python-like) for vitriol vpo fold --structural

def fold_sections(sections):
    """Group layers by structural identity, produce template index."""

    # Template key: (quant_type, tuple(shape), act_bits)
    template_map = {}  # key -> { template_id, layers[] }

    next_template_id = 0
    for section in sections:
        for layer in section.layers:
            key = (layer.quant_type,
                   tuple(layer.shape),
                   layer.act_bits)

            if key not in template_map:
                template_map[key] = {
                    'template_id': next_template_id,
                    'layers': [],
                    'quant_type': layer.quant_type,
                    'shape': layer.shape,
                    'act_bits': layer.act_bits,
                }
                next_template_id += 1

            template_map[key]['layers'].append(layer)

    # Build output: template table + rewritten layer entries
    templates = []
    for key, tmpl in template_map.items():
        unique_luts = {}  # hash(lut_data) -> instance indices
        for i, layer in enumerate(tmpl['layers']):
            lut_hash = hash_lut_data(layer.lut_data, layer.lut_entry_size)
            if lut_hash not in unique_luts:
                unique_luts[lut_hash] = {
                    'instance_indices': [],
                    'template_lut': layer.lut_data,
                }
            unique_luts[lut_hash]['instance_indices'].append(
                layer.layer_id)

        template_entry = {
            'template_id': tmpl['template_id'],
            'quant_type': tmpl['quant_type'],
            'shape': tmpl['shape'],
            'act_bits': tmpl['act_bits'],
            'instances': [],  # (layer_id, template_lut_offset)
        }
        for lut_hash, group in unique_luts.items():
            for layer_id in group['instance_indices']:
                template_entry['instances'].append({
                    'layer_id': layer_id,
                    'lut_data': group['template_lut'],
                })
        templates.append(template_entry)

    return templates

def hash_lut_data(data, size):
    """FNV-1a 64-bit hash of LUT data for dedup detection."""
    import struct
    h = 0xCBF29CE484222325
    for byte in data[:size]:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h
```

---

## 4. Hybrid Dispatch Decision Tree (Complete)

```
At model load:
  Load .vpo → verify model_hash → select matching sections
  Open liblut_matmul.so → resolve symbols
  Init Vulkan → load SPIR-V modules

At each layer evaluation (in ggml_cuda_compute_forward):

  ┌────────────────────────────────────────────────────┐
  │  Is this op MUL_MAT or MUL_MAT_ID?                 │
  │  AND is src1 quantized?                            │
  │  AND is the VPO loaded?                            │
  │  AND does VPO have this layer_id?                   │
  ├── YES ─────────────────────────────────────────────┤
  │    ┌────────────────────────────────────────────┐  │
  │    │ Is brief_lut_matmul.so loaded?             │  │
  │    ├── YES ─────────────────────────────────────┤  │
  │    │  Route to CPU LUT:                        │  │
  │    │  1. brief_lut_matmul(layer_id, activations, │  │
  │    │                         output, n)         │  │
  │    │  2. Record in profiler as CPU_LUT          │  │
  │    │  return                                    │  │
  │    └────────────────────────────────────────────┘  │
  └────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────┐
  │  Is there a SPIR-V kernel for this op?             │
  │  AND is the SPIR-V loader available?               │
  ├── YES ─────────────────────────────────────────────┤
  │  Route to GPU SPIR-V:                             │
  │  1. spirv_dispatch(kernel_name, buffers, args)    │
  │  2. Record in profiler as GPU_SPIRV               │
  │  return                                            │
  └────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────┐
  │  Fallback: route to existing CUDA kernel           │
  │  (MMQ, MMVQ, cuBLAS, etc.)                        │
  │  Record in profiler as GPU_CUDA                    │
  │  return                                            │
  └────────────────────────────────────────────────────┘
```

---

## 5. Data Flow: Per-Token Sizes (Qwen3.6-35B)

| Stage | Data | Size | Location |
|-------|------|------|----------|
| Input token | 1 × 5120 f32 embeddings | 20 KB | GPU VRAM |
| Per MoE layer | 8 expert IDs × 4 bytes | 32 bytes | CPU → GPU |
| Expert FFN input | 8 × 2048 i8 (IQ2 activations) | 16 KB | CPU produces via LUT |
| Expert FFN output | 8 × 14336 f32 | 448 KB | CPU → GPU (activation only!) |
| Attention (GPU) | Q/K/V projections, scores | ~200 MB | GPU VRAM (stays on GPU) |
| KV cache / token | 5120 × 2.5 bytes × 40 layers | ~500 KB/token | GPU VRAM |

**Key insight:** Without VPO, each MoE layer transfers ~200 MB of weights over PCIe. With VPO, the CPU LUT path eliminates the weight transfer entirely — only the 16 KB input and 448 KB output cross PCIe. That is a **~100× reduction** in PCIe traffic per MoE layer.

---

## 6. Self-Optimization Feedback Loop

```
Session N execution:
  ┌────────────────────┐
  │ Profiler records   │  ← per-layer: exec_count, latency, path, PCIe bytes
  │ per-token stats    │
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ On shutdown:       │
  │ export session.json│  ← includes recommendations
  └─────────┬──────────┘
            ▼
  ┌─────────────────────────────────────┐
  │ vitriol bake --update model.vpo     │
  │   --profile session.json            │
  │                                     │
  │ 1. Parse profile                    │
  │ 2. Identify hottest layers NOT in   │
  │    VPO (highest latency × count)    │
  │ 3. Run passes for those layers      │
  │ 4. Append new sections              │
  └─────────┬───────────────────────────┘
            ▼
Session N+1 execution:
  More layers in VPO → more CPU LUT hits → less PCIe traffic → faster tokens
```

### `session.json` Format

```json
{
  "session": {
    "started_at": "2026-06-04T08:14:00Z",
    "ended_at": "2026-06-04T09:14:00Z",
    "model_hash": "abcd1234...",
    "total_tokens": 4096,
    "device": "GeForce GTX 1070 Ti"
  },
  "vpo_status": {
    "path": "/models/qwen.vpo",
    "sections_loaded": 2,
    "layers_in_vpo": 120,
    "total_layers": 160
  },
  "summary": {
    "tokens_per_second": 9.8,
    "avg_pcie_bytes_per_token": 16500000,
    "cpu_lut_fraction": 0.45,
    "gpu_cuda_fraction": 0.55
  },
  "layers": [
    {
      "layer_id": 5,
      "name": "blk.5.ffn_gate",
      "in_vpo": false,
      "exec_count": 4096,
      "avg_latency_ns": 1250000,
      "total_pcie_bytes": 68719476736,
      "recommendation": {
        "bake_pass": 1,
        "estimated_speedup": 3.2
      }
    }
  ],
  "recommendations": [
    {
      "pass_id": 1,
      "target_layers": ["blk.5.ffn_gate", "blk.5.ffn_down", "blk.5.ffn_up"],
      "estimated_savings_gb_per_token": 0.6,
      "estimated_ts_improvement": "+2.1 t/s"
    }
  ]
}
```

---

## 7. Key Implementation Rules

1. **Never modify a .vpo section.** Baking appends. Folding and pruning create new files. The original grimoire is always recoverable.

2. **Always fall back gracefully.** If `.vpo` is missing, hash doesn't match, `liblut_matmul.so` is absent, or Vulkan is not available — VITRIOL runs pure CUDA as if nothing happened. Log a warning, keep going.

3. **cudaHostRegister on mmap'd LUT data.** The LUT data is mmap'd from the `.vpo` file, then registered for GPU DMA via `cudaHostRegister` with `cudaHostRegisterMapped` flag. This allows the GPU to read the LUT data if needed (for hybrid GPU+LUT paths), but the primary LUT path is CPU-only.

4. **Profiling is always on.** The profiler records per-layer statistics in a lock-free ring buffer. Overhead is <0.1%. Export happens on `SIGINT`/`SIGTERM` via the `atexit` handler.

5. **Thread safety.** `liblut_matmul.so` must be reentrant. Multiple CPU threads can call `brief_lut_matmul` concurrently for different layers. The VPO data is read-only after load, so no locking is needed for LUT data.
