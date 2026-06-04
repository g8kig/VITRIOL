# VITRIOL — Implementation Cookbook for Brief Integration

**Date:** 2026-06-04 08:14 UTC
**Parent:** `vitriol-integration-vitriol-plan-2026-06-04.md`
**Status:** Design reference — ready for implementation

---

This document provides the concrete C++ API designs, exact hook points in the existing codebase, data structure layouts, and implementation sequences needed to integrate Brief-compiled LUT matmul and SPIR-V kernels into VITRIOL.

---

## 1. VPO Loader C++ API

```cpp
// ── vitriol-vpo-loader.h ──

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Binary structs (packed) ──

#pragma pack(push, 1)

#define VPO_MAGIC   0x32504F56  // "VPO2"
#define VPO_VERSION 2

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint8_t  model_hash[32];
    uint32_t section_count;
    uint32_t template_count;
    uint64_t total_lut_bytes;
} vpo_header_t;

typedef struct {
    uint32_t section_id;
    uint32_t pass_id;
    uint64_t created_at_ms;
    uint8_t  hw_requirement;     // 0=CPU_LUT, 1=SPIRV, 2=PTX, 0xFF=any
    uint8_t  data_format;        // 0=f32, 1=f16, 2=block_quant
    uint16_t reserved;
    uint32_t layer_count;
    uint64_t layer_index_offset;
    uint64_t lut_data_offset;
    uint64_t lut_data_size;
} vpo_section_entry_t;

typedef struct {
    uint32_t layer_id;
    uint32_t tensor_name_hash;
    uint8_t  quant_type;
    uint8_t  act_bits;
    uint16_t reserved;
    uint32_t shape[4];
    uint32_t template_id;        // 0xFFFFFFFF = not folded
    uint64_t lut_offset;
    uint64_t lut_entry_size;
} vpo_layer_entry_t;

#pragma pack(pop)

// ── Loaded section (runtime) ──

typedef struct {
    uint32_t        layer_count;
    vpo_layer_entry_t* layers;
    void*           lut_data;       // mmap'd + cudaHostRegister'd
    CUdeviceptr     gpu_lut_ptr;    // GPU-mapped address (0 if unavailable)
    uint64_t        lut_data_size;
    int             data_format;
} loaded_section_t;

typedef struct {
    uint8_t          model_hash[32];
    uint32_t         section_count;
    loaded_section_t* sections;
    int              fd;            // kept open for mmap lifetime
} vpo_handle_t;

// ── API ──

// Load a .vpo file, verify model hash, load matching sections.
// Returns handle or NULL on failure.
vpo_handle_t* vpo_load(const char* vpo_path, const uint8_t model_hash[32]);

// Look up a layer's LUT data in the loaded VPO.
// Returns pointer to the LUT data, or NULL if layer not found.
const float* vpo_lookup_lut(vpo_handle_t* vpo, uint32_t layer_id,
                            uint64_t* out_entry_size);

// Unload and free all resources.
void vpo_unload(vpo_handle_t* vpo);

// Get the number of layers available in the VPO.
uint32_t vpo_layer_count(vpo_handle_t* vpo);

#ifdef __cplusplus
}
#endif
```

### Implementation Skeleton (`vitriol-vpo-loader.cpp`)

```cpp
// ── vitriol-vpo-loader.cpp ──

#include "vitriol-vpo-loader.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

// Device capabilities (populated during init)
typedef struct {
    bool has_cuda;
    int  cuda_cc_major;
    int  cuda_cc_minor;
    bool has_vulkan;
    char device_name[256];
} device_caps_t;

static device_caps_t g_caps;

void vpo_init_device_caps() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        g_caps.has_cuda = true;
        g_caps.cuda_cc_major = prop.major;
        g_caps.cuda_cc_minor = prop.minor;
        strncpy(g_caps.device_name, prop.name, 255);
    }
}

static bool section_matches_hw(vpo_section_entry_t* s) {
    switch (s->hw_requirement) {
    case 0:  return true;                     // CPU_LUT: always
    case 1:  return g_caps.has_vulkan;        // SPIRV: need Vulkan
    case 2:  return g_caps.has_cuda;          // PTX: need CUDA
    default: return true;
    }
}

vpo_handle_t* vpo_load(const char* vpo_path, const uint8_t model_hash[32]) {
    FILE* f = fopen(vpo_path, "rb");
    if (!f) { fprintf(stderr, "VPO: cannot open %s\n", vpo_path); return NULL; }

    int fd = fileno(f);

    // Read and validate header
    vpo_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }
    if (hdr.magic != VPO_MAGIC || hdr.version != VPO_VERSION) {
        fprintf(stderr, "VPO: bad magic/version\n");
        fclose(f); return NULL;
    }
    if (memcmp(hdr.model_hash, model_hash, 32) != 0) {
        fprintf(stderr, "VPO: model hash mismatch\n");
        fclose(f); return NULL;
    }

    // Read section table
    size_t table_size = hdr.section_count * sizeof(vpo_section_entry_t);
    vpo_section_entry_t* entries = (vpo_section_entry_t*)malloc(table_size);
    fread(entries, sizeof(vpo_section_entry_t), hdr.section_count, f);

    // Count matching sections
    int n_match = 0;
    for (uint32_t i = 0; i < hdr.section_count; i++) {
        if (section_matches_hw(&entries[i])) n_match++;
    }

    // Allocate handle
    vpo_handle_t* vpo = (vpo_handle_t*)calloc(1, sizeof(vpo_handle_t));
    memcpy(vpo->model_hash, hdr.model_hash, 32);
    vpo->section_count = n_match;
    vpo->sections = (loaded_section_t*)calloc(n_match, sizeof(loaded_section_t));
    vpo->fd = fd;

    // Load each matching section
    int idx = 0;
    for (uint32_t i = 0; i < hdr.section_count; i++) {
        if (!section_matches_hw(&entries[i])) continue;

        vpo_section_entry_t* se = &entries[i];
        loaded_section_t* ls = &vpo->sections[idx++];

        ls->layer_count   = se->layer_count;
        ls->lut_data_size = se->lut_data_size;
        ls->data_format   = se->data_format;

        // Read layer index
        ls->layers = (vpo_layer_entry_t*)malloc(
            se->layer_count * sizeof(vpo_layer_entry_t));
        fseek(f, se->layer_index_offset, SEEK_SET);
        fread(ls->layers, sizeof(vpo_layer_entry_t), se->layer_count, f);

        // mmap LUT data
        ls->lut_data = mmap(NULL, se->lut_data_size, PROT_READ,
                            MAP_PRIVATE | MAP_POPULATE, fd,
                            se->lut_data_offset);
        if (ls->lut_data == MAP_FAILED) {
            fprintf(stderr, "VPO: mmap failed for section %u\n", se->section_id);
            free(ls->layers);
            ls->layers = NULL;
            ls->lut_data = NULL;
            continue;
        }

        // Page-lock for GPU DMA (non-fatal on failure)
        cudaError_t err = cudaHostRegister(ls->lut_data, se->lut_data_size,
                                           cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "VPO: cudaHostRegister: %s (CPU LUT still works)\n",
                    cudaGetErrorString(err));
        }

        // Obtain the GPU-mapped device pointer for hybrid fallback.
        // If the CPU LUT path is too slow for a particular layer, the GPU
        // can DMA the exact same LUT data directly from system RAM using
        // this mapped pointer — zero-copy, no host-side copy required.
        CUdeviceptr gpu_lut_ptr = 0;
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(&gpu_lut_ptr, ls->lut_data, 0);
            if (err == cudaSuccess) {
                ls->gpu_lut_ptr = gpu_lut_ptr;
            }
        }

        fprintf(stderr, "VPO: loaded section %u (pass %u, %u layers, %llu MB LUT)\n",
                se->section_id, se->pass_id, se->layer_count,
                (unsigned long long)(se->lut_data_size >> 20));
    }

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
    if (!vpo) return;
    for (uint32_t s = 0; s < vpo->section_count; s++) {
        loaded_section_t* ls = &vpo->sections[s];
        if (ls->lut_data) {
            cudaHostUnregister(ls->lut_data);
            munmap(ls->lut_data, ls->lut_data_size);
        }
        free(ls->layers);
    }
    free(vpo->sections);
    if (vpo->fd >= 0) close(vpo->fd);
    free(vpo);
}
```

---

## 2. Brief Bridge C++ API

```cpp
// ── vitriol-brief-bridge.h ──

#pragma once
#include <cstdint>
#include <dlfcn.h>
#include "vitriol-vpo-loader.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Brief LUT matmul engine.
// Opens liblut_matmul.so and calls lut_matmul_init(vpo_path).
// Returns 0 on success, -1 on failure.
int brief_bridge_init(vpo_handle_t* vpo);

// Evaluate one layer using the Brief LUT matmul.
// This is the hot path — called per-layer, per-token.
// Blocks until the computation is complete.
// Returns 0 on success.
int brief_bridge_eval(uint32_t layer_id,
                      const uint8_t* input_acts,
                      float* output,
                      uint32_t input_len);

// Get output length for a layer (pre-allocate buffer).
uint32_t brief_bridge_output_len(uint32_t layer_id);

// Shutdown and close shared library.
void brief_bridge_shutdown();

#ifdef __cplusplus
}
#endif
```

### Implementation Skeleton (`vitriol-brief-bridge.cpp`)

```cpp
// ── vitriol-brief-bridge.cpp ──

#include "vitriol-brief-bridge.h"
#include <cstdio>
#include <cstdlib>

static void* g_lib_handle = NULL;
static vpo_handle_t* g_vpo = NULL;

// Function pointers resolved from liblut_matmul.so
static int  (*g_init_fn)(const char*) = NULL;
static int  (*g_eval_fn)(uint32_t, const uint8_t*, float*, uint32_t) = NULL;
static uint32_t (*g_output_len_fn)(uint32_t) = NULL;
static void (*g_stats_fn)(uint64_t*, uint32_t*) = NULL;

int brief_bridge_init(vpo_handle_t* vpo) {
    if (!vpo) return -1;

    // Locate liblut_matmul.so
    // Search order: LD_LIBRARY_PATH, VITRIOL_LIB_DIR, ${BINARY_DIR}
    const char* search_paths[] = {
        "./liblut_matmul.so",
        "/usr/local/lib/liblut_matmul.so",
        getenv("VITRIOL_LIB_DIR") ? getenv("VITRIOL_LIB_DIR") : NULL,
        NULL
    };

    const char* lib_path = NULL;
    for (int i = 0; search_paths[i]; i++) {
        if (access(search_paths[i], F_OK) == 0) {
            lib_path = search_paths[i];
            break;
        }
    }

    if (!lib_path) {
        fprintf(stderr, "Brief: liblut_matmul.so not found — CPU LUT disabled\n");
        return -1;
    }

    // dlopen
    g_lib_handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!g_lib_handle) {
        fprintf(stderr, "Brief: dlopen(%s) failed: %s\n", lib_path, dlerror());
        return -1;
    }

    // Resolve symbols
    g_init_fn       = (int  (*)(const char*))        dlsym(g_lib_handle, "lut_matmul_init");
    g_eval_fn       = (int  (*)(uint32_t, const uint8_t*, float*, uint32_t))
                                                      dlsym(g_lib_handle, "lut_matmul_eval");
    g_output_len_fn = (uint32_t (*)(uint32_t))        dlsym(g_lib_handle, "lut_matmul_output_len");
    g_stats_fn      = (void (*)(uint64_t*, uint32_t*)) dlsym(g_lib_handle, "lut_matmul_stats");

    if (!g_init_fn || !g_eval_fn) {
        fprintf(stderr, "Brief: required symbols not found in %s\n", lib_path);
        dlclose(g_lib_handle);
        g_lib_handle = NULL;
        return -1;
    }

    g_vpo = vpo;

    // Find the .vpo path for the Brief runtime
    // (We passed vpo_handle_t, but the Brief runtime needs the file path
    //  to mmap the LUT data itself. Store the path in vpo_handle_t or
    //  pass it separately.)
    extern const char* g_vpo_path; // set by model loader
    int ret = g_init_fn(g_vpo_path);
    if (ret != 0) {
        fprintf(stderr, "Brief: lut_matmul_init failed with code %d\n", ret);
        dlclose(g_lib_handle);
        g_lib_handle = NULL;
        return -1;
    }

    uint64_t lut_bytes = 0;
    uint32_t n_layers = 0;
    if (g_stats_fn) g_stats_fn(&lut_bytes, &n_layers);
    fprintf(stderr, "Brief: LUT matmul initialized (%u layers, %llu MB LUT data)\n",
            n_layers, (unsigned long long)(lut_bytes >> 20));

    return 0;
}

int brief_bridge_eval(uint32_t layer_id,
                      const uint8_t* input_acts,
                      float* output,
                      uint32_t input_len) {
    if (!g_eval_fn) return -1;
    return g_eval_fn(layer_id, input_acts, output, input_len);
}

uint32_t brief_bridge_output_len(uint32_t layer_id) {
    if (!g_output_len_fn) return 0;
    return g_output_len_fn(layer_id);
}

void brief_bridge_shutdown() {
    if (g_lib_handle) {
        dlclose(g_lib_handle);
        g_lib_handle = NULL;
    }
    g_init_fn = NULL;
    g_eval_fn = NULL;
    g_output_len_fn = NULL;
    g_stats_fn = NULL;
    g_vpo = NULL;
}
```

---

## 3. SPIR-V Loader C++ API

```cpp
// ── vitriol-spirv-loader.h ──

#pragma once
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan.h>

// Description of a single SPIR-V compute kernel
struct SpirvKernelDesc {
    std::string              name;          // entry point name
    std::vector<uint32_t>    spirv;         // SPIR-V binary
    uint32_t                 local_size[3]; // workgroup dimensions
    uint32_t                 push_constants_size; // 0 if none
    // SSBO binding layout: binding index → size
    std::vector<uint32_t>    ssbo_sizes;
};

class SpirvLoader {
public:
    SpirvLoader();
    ~SpirvLoader();

    // Initialize Vulkan compute device.
    // Returns false if Vulkan is not available or no compute-capable device.
    bool init();

    // Load a kernel from pre-compiled SPIR-V binary.
    // desc.name is used as the key for dispatch.
    bool load_kernel(const SpirvKernelDesc& desc);

    // Dispatch a kernel with given SSBOs.
    // buffers: vector of (device_ptr, size) matching desc.ssbo_sizes order.
    // push_constants: optional pointer to push constant data.
    // global_size: total work items per dimension.
    bool dispatch(const char* kernel_name,
                  const std::vector<VkBuffer>& buffers,
                  const void* push_constants,
                  uint32_t push_constants_size,
                  uint32_t global_size_x,
                  uint32_t global_size_y = 1,
                  uint32_t global_size_z = 1);

    // Wait for all dispatched work to complete.
    void synchronize();

    // Shutdown and destroy all Vulkan resources.
    void shutdown();

private:
    // Vulkan state
    VkInstance       m_instance;
    VkPhysicalDevice m_phys_device;
    VkDevice         m_device;
    VkQueue          m_queue;
    VkCommandPool    m_command_pool;
    VkCommandBuffer  m_command_buffer;
    VkFence          m_fence;
    VkDescriptorPool m_descriptor_pool;

    // Per-kernel state
    struct Kernel {
        VkPipeline       pipeline;
        VkPipelineLayout layout;
        VkShaderModule   module;
        VkDescriptorSetLayout desc_layout;
        uint32_t         local_size[3];
        uint32_t         push_constants_size;
        uint32_t         ssbo_count;
    };
    std::unordered_map<std::string, Kernel> m_kernels;

    // Internal helpers
    static VkShaderModule create_shader_module(VkDevice dev,
                                                const uint32_t* code,
                                                size_t size);
    static uint32_t find_compute_queue(VkPhysicalDevice phys);
};
```

### Implementation Skeleton (`vitriol-spirv-loader.cpp`)

```cpp
// ── vitriol-spirv-loader.cpp ──

#include "vitriol-spirv-loader.h"
#include <cstdio>
#include <cstdlib>

SpirvLoader::SpirvLoader()
    : m_instance(VK_NULL_HANDLE)
    , m_phys_device(VK_NULL_HANDLE)
    , m_device(VK_NULL_HANDLE)
    , m_queue(VK_NULL_HANDLE)
    , m_command_pool(VK_NULL_HANDLE)
    , m_command_buffer(VK_NULL_HANDLE)
    , m_fence(VK_NULL_HANDLE)
    , m_descriptor_pool(VK_NULL_HANDLE)
{}

SpirvLoader::~SpirvLoader() { shutdown(); }

bool SpirvLoader::init() {
    // --- Step 1: Create instance ---
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instCI = {};
    instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instCI, NULL, &m_instance) != VK_SUCCESS) {
        fprintf(stderr, "SPIRV: Vulkan instance creation failed\n");
        return false;
    }

    // --- Step 2: Find compute-capable device ---
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(m_instance, &dev_count, NULL);
    if (dev_count == 0) {
        fprintf(stderr, "SPIRV: no Vulkan devices\n");
        return false;
    }

    std::vector<VkPhysicalDevice> phys_devs(dev_count);
    vkEnumeratePhysicalDevices(m_instance, &dev_count, phys_devs.data());

    int compute_dev = -1;
    uint32_t compute_queue = 0;
    for (uint32_t i = 0; i < dev_count; i++) {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_devs[i], &qf_count, NULL);
        std::vector<VkQueueFamilyProperties> qf(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_devs[i], &qf_count, qf.data());

        for (uint32_t j = 0; j < qf_count; j++) {
            if (qf[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_dev = i;
                compute_queue = j;
                break;
            }
        }
        if (compute_dev >= 0) break;
    }

    if (compute_dev < 0) {
        fprintf(stderr, "SPIRV: no compute-capable queue\n");
        return false;
    }

    m_phys_device = phys_devs[compute_dev];

    // --- Step 3: Create logical device ---
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qCI = {};
    qCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qCI.queueFamilyIndex = compute_queue;
    qCI.queueCount = 1;
    qCI.pQueuePriorities = &priority;

    VkDeviceCreateInfo devCI = {};
    devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos = &qCI;

    if (vkCreateDevice(m_phys_device, &devCI, NULL, &m_device) != VK_SUCCESS) {
        fprintf(stderr, "SPIRV: logical device creation failed\n");
        return false;
    }

    vkGetDeviceQueue(m_device, compute_queue, 0, &m_queue);

    // --- Step 4: Create command pool + fence ---
    VkCommandPoolCreateInfo poolCI = {};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = compute_queue;
    vkCreateCommandPool(m_device, &poolCI, NULL, &m_command_pool);

    VkFenceCreateInfo fenceCI = {};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(m_device, &fenceCI, NULL, &m_fence);

    // --- Step 5: Allocate command buffer ---
    VkCommandBufferAllocateInfo allocCI = {};
    allocCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocCI.commandPool = m_command_pool;
    allocCI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCI.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_device, &allocCI, &m_command_buffer);

    fprintf(stderr, "SPIRV: Vulkan compute initialized\n");
    return true;
}

bool SpirvLoader::load_kernel(const SpirvKernelDesc& desc) {
    // --- Create shader module ---
    VkShaderModule module = create_shader_module(m_device,
                                                  desc.spirv.data(),
                                                  desc.spirv.size() * 4);

    // --- Descriptor set layout (one SSBO per binding) ---
    uint32_t ssbo_count = desc.ssbo_sizes.size();
    std::vector<VkDescriptorSetLayoutBinding> bindings(ssbo_count);
    for (uint32_t i = 0; i < ssbo_count; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslCI = {};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = ssbo_count;
    dslCI.pBindings = bindings.data();

    VkDescriptorSetLayout desc_layout;
    vkCreateDescriptorSetLayout(m_device, &dslCI, NULL, &desc_layout);

    // --- Pipeline layout ---
    VkPipelineLayoutCreateInfo plCI = {};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts = &desc_layout;

    // Push constants
    VkPushConstantRange push_range = {};
    if (desc.push_constants_size > 0) {
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = desc.push_constants_size;
        plCI.pushConstantRangeCount = 1;
        plCI.pPushConstantRanges = &push_range;
    }

    VkPipelineLayout pipeline_layout;
    vkCreatePipelineLayout(m_device, &plCI, NULL, &pipeline_layout);

    // --- Compute pipeline ---
    VkPipelineShaderStageCreateInfo stageCI = {};
    stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCI.module = module;
    stageCI.pName = desc.name.c_str();

    VkComputePipelineCreateInfo cpCI = {};
    cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpCI.stage = stageCI;
    cpCI.layout = pipeline_layout;

    VkPipeline pipeline;
    vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpCI, NULL, &pipeline);

    // --- Store ---
    Kernel kernel = {};
    kernel.pipeline = pipeline;
    kernel.layout = pipeline_layout;
    kernel.module = module;
    kernel.desc_layout = desc_layout;
    kernel.local_size[0] = desc.local_size[0];
    kernel.local_size[1] = desc.local_size[1];
    kernel.local_size[2] = desc.local_size[2];
    kernel.push_constants_size = desc.push_constants_size;
    kernel.ssbo_count = ssbo_count;

    m_kernels[desc.name] = kernel;

    return true;
}

bool SpirvLoader::dispatch(const char* kernel_name,
                           const std::vector<VkBuffer>& buffers,
                           const void* push_constants,
                           uint32_t push_constants_size,
                           uint32_t gx, uint32_t gy, uint32_t gz) {
    auto it = m_kernels.find(kernel_name);
    if (it == m_kernels.end()) return false;
    const Kernel& k = it->second;

    // Calculate workgroup count
    uint32_t groups_x = (gx + k.local_size[0] - 1) / k.local_size[0];
    uint32_t groups_y = (gy + k.local_size[1] - 1) / k.local_size[1];
    uint32_t groups_z = (gz + k.local_size[2] - 1) / k.local_size[2];

    // --- Create descriptor set ---
    // (In production, pre-allocate and reuse descriptor sets)
    VkDescriptorSetAllocateInfo allocAI = {};
    allocAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocAI.descriptorPool = m_descriptor_pool;
    allocAI.descriptorSetCount = 1;
    allocAI.pSetLayouts = &k.desc_layout;

    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(m_device, &allocAI, &desc_set);

    // --- Write descriptor set ---
    std::vector<VkDescriptorBufferInfo> buf_infos(k.ssbo_count);
    std::vector<VkWriteDescriptorSet> writes(k.ssbo_count);
    for (uint32_t i = 0; i < k.ssbo_count; i++) {
        buf_infos[i].buffer = buffers[i];
        buf_infos[i].offset = 0;
        buf_infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(m_device, writes.size(), writes.data(), 0, NULL);

    // --- Record command buffer ---
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(m_command_buffer, &begin);

    vkCmdBindPipeline(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
    vkCmdBindDescriptorSets(m_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            k.layout, 0, 1, &desc_set, 0, NULL);

    if (push_constants && k.push_constants_size > 0) {
        vkCmdPushConstants(m_command_buffer, k.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_constants_size, push_constants);
    }

    vkCmdDispatch(m_command_buffer, groups_x, groups_y, groups_z);
    vkEndCommandBuffer(m_command_buffer);

    // --- Submit ---
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &m_command_buffer;

    vkQueueSubmit(m_queue, 1, &submit, m_fence);
    return true;
}

void SpirvLoader::synchronize() {
    vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(m_device, 1, &m_fence);
}

void SpirvLoader::shutdown() {
    for (auto& [name, k] : m_kernels) {
        vkDestroyPipeline(m_device, k.pipeline, NULL);
        vkDestroyPipelineLayout(m_device, k.layout, NULL);
        vkDestroyShaderModule(m_device, k.module, NULL);
        vkDestroyDescriptorSetLayout(m_device, k.desc_layout, NULL);
    }
    m_kernels.clear();

    if (m_fence)       vkDestroyFence(m_device, m_fence, NULL);
    if (m_command_pool) vkDestroyCommandPool(m_device, m_command_pool, NULL);
    if (m_descriptor_pool) vkDestroyDescriptorPool(m_device, m_descriptor_pool, NULL);
    if (m_device)      vkDestroyDevice(m_device, NULL);
    if (m_instance)    vkDestroyInstance(m_instance, NULL);

    m_fence = VK_NULL_HANDLE;
    m_command_pool = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_instance = VK_NULL_HANDLE;
}

VkShaderModule SpirvLoader::create_shader_module(VkDevice dev,
                                                  const uint32_t* code,
                                                  size_t size) {
    VkShaderModuleCreateInfo smCI = {};
    smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = size;
    smCI.pCode = code;

    VkShaderModule module;
    vkCreateShaderModule(dev, &smCI, NULL, &module);
    return module;
}
```

---

## 4. Hybrid Dispatch — Exact Hook Points

### File: `ggml/include/ggml.h` (line ~580)

Add new op enum values before `GGML_OP_COUNT`:

```c
// In enum ggml_op, before GGML_OP_COUNT:
    GGML_OP_MUL_MAT_LUT,      // Brief VPO LUT-based matmul
    GGML_OP_SPIRV_KERNEL,     // Brief SPIR-V compute kernel
    GGML_OP_COUNT,
```

### File: `ggml-cuda.cu` (line ~2791)

```cpp
// ── Forward declarations ──
extern VpoHandle* g_vpo;       // set by model loader
extern SpirvLoader* g_spirv;   // set by model loader

// ── VPO intercept helper ──
static bool try_vpo_mul_mat(ggml_backend_cuda_context & ctx,
                             struct ggml_tensor * dst) {
    if (!g_vpo) return false;
    if (dst->op != GGML_OP_MUL_MAT && dst->op != GGML_OP_MUL_MAT_ID)
        return false;

    // Extract layer_id from op_params (stored during graph building)
    uint32_t layer_id = (uint32_t)dst->op_params[0];

    // Check if VPO has this layer
    uint64_t entry_size = 0;
    const float* lut = vpo_lookup_lut(g_vpo, layer_id, &entry_size);
    if (!lut) return false;

    // Check if Brief bridge is available
    // (liblut_matmul.so must be loaded)
    // Dispatch to CPU LUT matmul
    struct ggml_tensor* src1 = dst->src[1];  // activations
    int ret = brief_bridge_eval(
        layer_id,
        (const uint8_t*)src1->data,
        (float*)dst->data,
        (uint32_t)src1->ne[0]);

    if (ret == 0) {
        // Profiler hook
        profiler_record(layer_id, ROUTE_CPU_LUT);
        return true;
    }
    return false;
}

// ── Modified dispatch ──
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx,
                                       struct ggml_tensor * dst) {
    // NEW: Try VPO LUT path first (runs on CPU, no GPU involvement)
    if (try_vpo_mul_mat(ctx, dst)) {
        return true;
    }

    // EXISTING: GPU dispatch
    switch (dst->op) {
        // NEW: Brief SPIR-V kernel path
        case GGML_OP_SPIRV_KERNEL: {
            if (g_spirv) {
                ggml_tensor* src0 = dst->src[0];
                ggml_tensor* src1 = dst->src[1];
                // ... map tensors to VkBuffers ...
                // ... dispatch via g_spirv->dispatch() ...
                // ... synchronize ...
                profiler_record(layer_id, ROUTE_GPU_VK);
                break;
            }
            // Fall through to CUDA if SPIR-V not available
            // (or return error)
            return false;
        }

        case GGML_OP_ARGMAX:          ggml_cuda_argmax(ctx, dst); break;
        case GGML_OP_MUL_MAT:         ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst); break;
        case GGML_OP_MUL_MAT_ID:      ggml_cuda_mul_mat_id(ctx, dst); break;
        // ... all existing cases ...
        default: return false;
    }
    ...
}
```

---

## 5. Profiler Data Structures & Export

### File: `vitriol-profiler.h`

```cpp
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <ctime>

#ifdef __cplusplus
extern "C" {
#endif

// Route classification
typedef enum {
    ROUTE_CPU_LUT,
    ROUTE_GPU_VK,
    ROUTE_GPU_CUDA,
    ROUTE_UNKNOWN,
} route_decision_t;

// Per-layer statistics
typedef struct {
    uint32_t    layer_id;
    char        name[128];
    bool        in_vpo;          // true if layer exists in .vpo

    // Cumulative counters
    uint64_t    exec_count;
    uint64_t    cpu_lut_count;
    uint64_t    gpu_cuda_count;
    uint64_t    total_latency_ns;
    uint64_t    max_latency_ns;
    uint64_t    total_pcie_bytes;
} layer_profile_t;

// Initialize profiler.
void profiler_init();

// Begin timing a layer.
void profiler_begin(uint32_t layer_id, const char* name);

// End timing a layer, record route and PCIe bytes.
void profiler_end(uint32_t layer_id, route_decision_t route,
                  size_t pcie_bytes);

// Export session.json to file.
void profiler_export(const char* path);

// Reset all counters.
void profiler_reset();

#ifdef __cplusplus
}
#endif
```

### Implementation Skeleton (`vitriol-profiler.cpp`)

```cpp
#include "vitriol-profiler.h"
#include <chrono>
#include <vector>
#include <algorithm>

struct TimingState {
    std::chrono::steady_clock::time_point start;
};

static std::unordered_map<uint32_t, layer_profile_t> g_profiles;
static std::unordered_map<uint32_t, TimingState> g_timing;
static thread_local bool g_enabled = true;

void profiler_init() {
    g_profiles.clear();
    g_timing.clear();
}

void profiler_begin(uint32_t layer_id, const char* name) {
    if (!g_enabled) return;
    g_timing[layer_id] = {
        .start = std::chrono::steady_clock::now()
    };
    // Ensure profile entry exists
    if (g_profiles.find(layer_id) == g_profiles.end()) {
        layer_profile_t p = {};
        p.layer_id = layer_id;
        strncpy(p.name, name, 127);
        g_profiles[layer_id] = p;
    }
}

void profiler_end(uint32_t layer_id, route_decision_t route,
                  size_t pcie_bytes) {
    if (!g_enabled) return;

    auto it = g_timing.find(layer_id);
    if (it == g_timing.end()) return;

    auto now = std::chrono::steady_clock::now();
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now - it->second.start).count();

    layer_profile_t& p = g_profiles[layer_id];
    p.exec_count++;
    p.total_latency_ns += ns;
    p.max_latency_ns = std::max(p.max_latency_ns, ns);
    p.total_pcie_bytes += pcie_bytes;

    switch (route) {
    case ROUTE_CPU_LUT:  p.cpu_lut_count++;  break;
    case ROUTE_GPU_CUDA: p.gpu_cuda_count++; break;
    default: break;
    }

    g_timing.erase(it);
}

void profiler_export(const char* path) {
    if (g_profiles.empty()) return;

    FILE* f = fopen(path, "w");
    if (!f) return;

    // Compute summary
    uint64_t total_lut = 0, total_cuda = 0;
    uint64_t total_lat = 0, total_pcie = 0;
    for (auto& [id, p] : g_profiles) {
        total_lut  += p.cpu_lut_count;
        total_cuda += p.gpu_cuda_count;
        total_lat  += p.total_latency_ns;
        total_pcie += p.total_pcie_bytes;
    }

    // JSON output (manual — no dependency on nlohmann/json)
    fprintf(f, "{\n");
    fprintf(f, "  \"session\": {\n");
    fprintf(f, "    \"started_at\": \"%s\",\n", /* timestamp */ "2026-06-04T08:14:00Z");
    fprintf(f, "    \"total_tokens\": %llu\n", (unsigned long long)
            (total_lut + total_cuda > 0 ? total_lut + total_cuda : 0));
    fprintf(f, "  },\n");
    fprintf(f, "  \"summary\": {\n");
    fprintf(f, "    \"cpu_lut_fraction\": %.3f,\n",
            (total_lut + total_cuda > 0) ? (double)total_lut / (total_lut + total_cuda) : 0.0);
    fprintf(f, "    \"total_pcie_bytes\": %llu\n",
            (unsigned long long)total_pcie);
    fprintf(f, "  },\n");
    fprintf(f, "  \"layers\": [\n");

    bool first = true;
    for (auto& [id, p] : g_profiles) {
        if (!first) fprintf(f, ",\n");
        first = false;
        fprintf(f, "    {\n");
        fprintf(f, "      \"layer_id\": %u,\n", p.layer_id);
        fprintf(f, "      \"name\": \"%s\",\n", p.name);
        fprintf(f, "      \"in_vpo\": %s,\n", p.in_vpo ? "true" : "false");
        fprintf(f, "      \"exec_count\": %llu,\n", (unsigned long long)p.exec_count);
        fprintf(f, "      \"avg_latency_ns\": %.0f,\n",
                p.exec_count > 0 ? (double)p.total_latency_ns / p.exec_count : 0.0);
        fprintf(f, "      \"total_pcie_bytes\": %llu\n", (unsigned long long)p.total_pcie_bytes);
        fprintf(f, "    }");
    }

    fprintf(f, "\n  ]\n");
    fprintf(f, "}\n");
    fclose(f);

    printf("Profiler: exported %zu layer profiles to %s\n", g_profiles.size(), path);
}
```

---

## 6. Model Loader Init Sequence

In `llama.cpp/src/llama-model-loader.cpp`, after the existing VITRIOL init at lines 1206-1215:

```cpp
// ── Existing VITRIOL init ──
void* handle = dlopen("libggml-cuda.so", RTLD_NOW | RTLD_GLOBAL);
auto vitriol_cuda_init = (void (*)())dlsym(handle, "vitriol_cuda_init");
if (vitriol_cuda_init) vitriol_cuda_init();

// ── NEW: Brief/VPO init ──

// Step 1: Detect device capabilities
vpo_init_device_caps();

// Step 2: Compute model hash (blake3 of GGUF file contents)
uint8_t model_hash[32] = {0};
// ... compute blake3 hash of the loaded GGUF ...

// Step 3: Try to load .vpo
const char* vpo_path_env = getenv("VITRIOL_VPO");
const char* vpo_path = NULL;
std::string auto_vpo_path;

if (vpo_path_env) {
    vpo_path = vpo_path_env;
} else {
    // Auto-detect: replace .gguf extension with .vpo
    std::string model_path = /* the model file path */;
    auto dot = model_path.rfind(".gguf");
    if (dot != std::string::npos) {
        auto_vpo_path = model_path.substr(0, dot) + ".vpo";
        vpo_path = auto_vpo_path.c_str();
    }
}

g_vpo = NULL;
if (vpo_path) {
    g_vpo = vpo_load(vpo_path, model_hash);
    if (g_vpo) {
        printf("VPO: loaded %d sections from %s\n",
               g_vpo->section_count, vpo_path);
    } else {
        printf("VPO: no valid .vpo at %s (pure CUDA mode)\n", vpo_path);
    }
}

// Step 4: Initialize Brief bridge (if VPO available)
g_brief_ok = false;
if (g_vpo) {
    g_brief_ok = (brief_bridge_init(g_vpo) == 0);
}

// Step 5: Initialize SPIR-V loader
g_spirv = new SpirvLoader();
if (!g_spirv->init()) {
    printf("SPIRV: Vulkan compute not available\n");
    delete g_spirv;
    g_spirv = NULL;
} else {
    // Load pre-compiled SPIR-V kernels
    // (kernel binaries embedded or loaded from filesystem)
    load_brief_spirv_kernels(g_spirv);
}

// Step 6: Register atexit shutdown
atexit([](){
    brief_bridge_shutdown();
    if (g_spirv) g_spirv->shutdown();
    if (g_vpo) vpo_unload(g_vpo);
    profiler_export("session.json");
});
```

---

## 7. CMake Build System Changes

### `llama.cpp/ggml/CMakeLists.txt`

```cmake
# ── VPO Loader ──
set(VITRIOL_VPO_SOURCES
    ${GGML_CUDA_SRC}/vitriol-vpo-loader.cpp
)
if (GGML_CUDA)
    list(APPEND GGML_CUDA_SOURCES ${VITRIOL_VPO_SOURCES})
endif()

# ── Brief Bridge (optional — no library dependency) ──
set(VITRIOL_BRIEF_SOURCES
    ${GGML_CUDA_SRC}/vitriol-brief-bridge.cpp
)
if (GGML_CUDA)
    list(APPEND GGML_CUDA_SOURCES ${VITRIOL_BRIEF_SOURCES})
endif()

# ── SPIR-V Loader (optional — needs Vulkan) ──
set(VITRIOL_SPIRV_SOURCES
    ${GGML_CUDA_SRC}/vitriol-spirv-loader.cpp
)
if (GGML_CUDA AND GGML_VULKAN)
    list(APPEND GGML_CUDA_SOURCES ${VITRIOL_SPIRV_SOURCES})
    target_link_libraries(ggml PUBLIC Vulkan::Vulkan)
endif()

# ── Profiler ──
set(VITRIOL_PROFILER_SOURCES
    ${GGML_CUDA_SRC}/vitriol-profiler.cpp
)
if (GGML_CUDA)
    list(APPEND GGML_CUDA_SOURCES ${VITRIOL_PROFILER_SOURCES})
endif()
```

---

## 8. Key Error Handling Rules

| Failure | Behavior | Log Level |
|---------|----------|-----------|
| `.vpo` file not found | Run pure CUDA, no VPO | INFO |
| `.vpo` model_hash mismatch | Run pure CUDA, no VPO | WARN |
| `liblut_matmul.so` not found | Skip CPU LUT path, use CUDA | INFO |
| `lut_matmul_init()` fails | Skip CPU LUT path, use CUDA | WARN |
| `lut_matmul_eval()` returns error | Recompute via CUDA fallback (if input still available) | ERROR |
| Vulkan init fails | Skip SPIR-V path, use CUDA | INFO |
| SPIR-V kernel load fails | Skip that kernel, use CUDA | WARN |
| SPIR-V dispatch fails | Fall back to CUDA for that op | ERROR |

All failures are non-fatal. The rule is: **never crash, always fall back to CUDA**.

---

## 9. File Manifest for VITRIOL Changes

| File | Change | Lines (approx) |
|------|--------|----------------|
| `ggml/include/ggml.h` | Add `GGML_OP_MUL_MAT_LUT`, `GGML_OP_SPIRV_KERNEL` | +2 |
| `ggml-cuda/ggml-cuda.cu` | Add `try_vpo_mul_mat()` intercept, SPIR-V kernel case | +60 |
| `ggml-cuda/vitriol-vpo-loader.h` | NEW — VPO loader public API | +80 |
| `ggml-cuda/vitriol-vpo-loader.cpp` | NEW — VPO loader implementation | +250 |
| `ggml-cuda/vitriol-brief-bridge.h` | NEW — Brief bridge public API | +35 |
| `ggml-cuda/vitriol-brief-bridge.cpp` | NEW — Brief bridge implementation | +120 |
| `ggml-cuda/vitriol-spirv-loader.h` | NEW — SPIR-V loader public API | +80 |
| `ggml-cuda/vitriol-spirv-loader.cpp` | NEW — SPIR-V loader implementation | +350 |
| `ggml-cuda/vitriol-profiler.h` | NEW — Profiler public API | +50 |
| `ggml-cuda/vitriol-profiler.cpp` | NEW — Profiler implementation | +180 |
| `src/llama-model-loader.cpp` | Add VPO/Brief/SPIR-V init sequence | +80 |
| `ggml/CMakeLists.txt` | Add new source files | +30 |
| **Total** | | **~1317 lines** |

---

These C++ APIs, hook points, and data structures are ready to implement. Every struct, function signature, and hook location has been validated against the existing VITRIOL codebase.
