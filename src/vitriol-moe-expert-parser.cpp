// VITRIOL MoE Expert Parser
// Parses GGUF files to extract expert tensor offsets for on-demand loading

#include "vitriol-moe-expert-parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <algorithm>
#include <iostream>

// Forward declare GGUF functions from llama.cpp
// We'll dynamically load or link against ggml

struct vitriol_moe_context {
    char model_path[1024];
    vitriol_model_meta_t meta;
    
    std::vector<vitriol_expert_tensor_info_t> tensors;
    bool initialized;
    
    // Map: layer -> expert_idx -> tensor index
    std::map<int, std::map<int, int>> layer_expert_map_gate;
    std::map<int, std::map<int, int>> layer_expert_map_up;
    std::map<int, std::map<int, int>> layer_expert_map_down;
};

extern "C" {

vitriol_moe_context_t* vitriol_moe_init(const char* gguf_path) {
    if (!gguf_path) {
        fprintf(stderr, "vitriol_moe_init: NULL path\n");
        return NULL;
    }
    
    vitriol_moe_context* ctx = new vitriol_moe_context();
    strncpy(ctx->model_path, gguf_path, sizeof(ctx->model_path) - 1);
    ctx->initialized = false;
    
    // Try to use llama.cpp's GGUF parser
    // For now, we'll do a simple scan of the GGUF file structure
    
    FILE* f = fopen(gguf_path, "rb");
    if (!f) {
        fprintf(stderr, "vitriol_moe_init: failed to open %s: %s\n", gguf_path, strerror(errno));
        delete ctx;
        return NULL;
    }
    
    // Read GGUF header to get version and metadata offset
    uint32_t magic = 0;
    fread(&magic, sizeof(magic), 1, f);
    
    // GGUF magic: "GGUF" in little endian
    if (magic != 0x46475547) {  // "GGUF" reversed
        fprintf(stderr, "vitriol_moe_init: not a GGUF file (magic: 0x%x)\n", magic);
        fclose(f);
        delete ctx;
        return NULL;
    }
    
    uint32_t version = 0;
    fread(&version, sizeof(version), 1, f);
    printf("vitriol_moe_init: GGUF version %u\n", version);
    
    // For now, we'll use a simpler approach: examine known MoE tensor patterns
    // The real implementation would use gguf_init_from_file()
    
    // Set default metadata (will be updated with real values)
    ctx->meta.n_expert = 256;    // Qwen3.6-35B-A3B has 256 experts
    ctx->meta.n_expert_used = 8; // Uses 8 per token
    ctx->meta.n_layer = 40;      // 40 layers
    ctx->meta.n_embd = 2048;     // Embedding dim
    ctx->meta.n_ff_exp = 512;    // Expert FFN dim
    
    ctx->initialized = true;
    fclose(f);
    
    printf("vitriol_moe_init: Initialized MoE parser for %s\n", gguf_path);
    printf("  Experts: %d total, %d used per token\n", ctx->meta.n_expert, ctx->meta.n_expert_used);
    printf("  Layers: %d, Embedding: %d, Expert FFN: %d\n", 
           ctx->meta.n_layer, ctx->meta.n_embd, ctx->meta.n_ff_exp);
    
    return (vitriol_moe_context_t*)ctx;
}

void vitriol_moe_free(vitriol_moe_context_t* ctx_raw) {
    if (!ctx_raw) return;
    
    vitriol_moe_context* ctx = (vitriol_moe_context*)ctx_raw;
    
    // Free all tensor names
    for (auto& t : ctx->tensors) {
        free(t.tensor_name);
        free(t.expert_type);
    }
    ctx->tensors.clear();
    
    delete ctx;
}

const vitriol_expert_tensor_info_t* vitriol_moe_get_tensor(
    vitriol_moe_context_t* ctx_raw,
    int layer,
    int expert_idx,
    const char* expert_type
) {
    if (!ctx_raw || !expert_type) return NULL;
    
    vitriol_moe_context* ctx = (vitriol_moe_context*)ctx_raw;
    
    // Find matching tensor
    for (const auto& t : ctx->tensors) {
        if (t.layer == layer && 
            strstr(t.expert_type, expert_type)) {
            // Check if this is the right expert index
            // Tensor names are like "model.layers.0.mlp.experts.7.weight"
            // We need to check if it's the Nth expert in this layer/type
            // For now, return the first match (simplified)
            return &t;
        }
    }
    
    return NULL;
}

int* vitriol_moe_get_layer_tensors(
    vitriol_moe_context_t* ctx_raw,
    int layer,
    size_t* n_results
) {
    if (!ctx_raw || !n_results) return NULL;
    
    vitriol_moe_context* ctx = (vitriol_moe_context*)ctx_raw;
    
    std::vector<int> indices;
    for (size_t i = 0; i < ctx->tensors.size(); i++) {
        if (ctx->tensors[i].layer == layer) {
            indices.push_back(i);
        }
    }
    
    if (indices.empty()) {
        *n_results = 0;
        return NULL;
    }
    
    int* result = (int*)malloc(indices.size() * sizeof(int));
    for (size_t i = 0; i < indices.size(); i++) {
        result[i] = indices[i];
    }
    *n_results = indices.size();
    
    return result;
}

void vitriol_moe_print_info(vitriol_moe_context_t* ctx_raw) {
    if (!ctx_raw) return;
    
    vitriol_moe_context* ctx = (vitriol_moe_context*)ctx_raw;
    
    printf("=== VITRIOL MoE Model Info ===\n");
    printf("Model: %s\n", ctx->model_path);
    printf("Expert count: %d\n", ctx->meta.n_expert);
    printf("Experts used per token: %d\n", ctx->meta.n_expert_used);
    printf("Layers: %d\n", ctx->meta.n_layer);
    printf("Embedding dimension: %d\n", ctx->meta.n_embd);
    printf("Expert FFN dimension: %d\n", ctx->meta.n_ff_exp);
    printf("Total tensors: %zu\n", ctx->tensors.size());
    printf("=============================\n");
}

bool vitriol_moe_is_moe(vitriol_moe_context_t* ctx_raw) {
    if (!ctx_raw) return false;
    vitriol_moe_context* ctx = (vitriol_moe_context*)ctx_raw;
    return ctx->meta.n_expert > 1;
}

size_t vitriol_moe_read_tensor_data(
    vitriol_moe_context_t* ctx_raw,
    const vitriol_expert_tensor_info_t* tensor_info,
    void* buffer,
    size_t buffer_size
) {
    if (!ctx_raw || !tensor_info || !buffer) return 0;
    
    vitriol_moe_context* ctx = (vitriol_moe_context*)ctx_raw;
    
    if (buffer_size < tensor_info->tensor_size) {
        fprintf(stderr, "vitriol_moe_read_tensor_data: buffer too small\n");
        return 0;
    }
    
    FILE* f = fopen(ctx->model_path, "rb");
    if (!f) return 0;
    
    if (fseek(f, tensor_info->file_offset, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    
    size_t read = fread(buffer, 1, tensor_info->tensor_size, f);
    fclose(f);
    
    return read;
}

uint64_t vitriol_moe_total_expert_size(vitriol_moe_context_t* ctx_raw) {
    if (!ctx_raw) return 0;
    
    vitriol_moe_context* ctx = (vitriol_moe_context*)ctx_raw;
    
    // Estimate: n_layer * n_expert * 3 (gate/up/down) * expert_size
    // Expert size = n_ff_exp * n_embd * 2 bytes (Q2_K quant)
    uint64_t expert_size = (uint64_t)ctx->meta.n_ff_exp * ctx->meta.n_embd * 2;
    uint64_t total = (uint64_t)ctx->meta.n_layer * ctx->meta.n_expert * 3 * expert_size;
    
    return total;
}

uint64_t vitriol_moe_estimate_expert_size(vitriol_moe_context_t* ctx_raw) {
    if (!ctx_raw) return 0;
    
    vitriol_moe_context* ctx = (vitriol_moe_context*)ctx_raw;
    
    // Single expert size (gate + up + down projections)
    // gate: n_embd * n_ff_exp * 2 bytes
    // up: n_embd * n_ff_exp * 2 bytes  
    // down: n_ff_exp * n_embd * 2 bytes
    uint64_t expert_size = (uint64_t)ctx->meta.n_embd * ctx->meta.n_ff_exp * 2 * 3;
    
    return expert_size;
}

} // extern "C"