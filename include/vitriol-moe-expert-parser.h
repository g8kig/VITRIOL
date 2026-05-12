#ifndef VITRIOL_MOE_EXPERT_PARSER_H
#define VITRIOL_MOE_EXPERT_PARSER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string>
#include <vector>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

// Expert information extracted from GGUF
typedef struct {
    char* tensor_name;           // e.g., "model.layers.0.mlp.experts.0.weight"
    uint64_t file_offset;         // Offset in GGUF file
    size_t tensor_size;           // Size in bytes
    int32_t n_dims;              // Number of dimensions
    int64_t dims[4];             // Dimensions [n_ff_exp, n_embd, n_expert]
    int32_t layer;               // Layer index (0-39 for 40 layers)
    char* expert_type;           // "ffn_gate", "ffn_up", "ffn_down"
} vitriol_expert_tensor_info_t;

// Model metadata
typedef struct {
    int32_t n_expert;            // Total number of experts (e.g., 256)
    int32_t n_expert_used;       // Number of experts used per token (e.g., 8)
    int32_t n_layer;             // Number of layers
    int32_t n_embd;              // Embedding dimension
    int32_t n_ff_exp;            // Expert FFN dimension
} vitriol_model_meta_t;

// Main parser structure
typedef struct {
    char* model_path;
    vitriol_model_meta_t meta;
    
    // All expert tensors (256 experts * 3 types * 40 layers = lots!)
    vitriol_expert_tensor_info_t* tensors;
    size_t n_tensors;
    
    // Expert index by layer and type
    // layer_experts[layer][expert_idx] -> tensor index
    std::map<int, std::map<int, int>> layer_expert_map;
    
    bool initialized;
} vitriol_moe_context_t;

// Initialize parser with GGUF file
vitriol_moe_context_t* vitriol_moe_init(const char* gguf_path);

// Free parser
void vitriol_moe_free(vitriol_moe_context_t* ctx);

// Get tensor info by layer and expert index
// Returns NULL if not found
const vitriol_expert_tensor_info_t* vitriol_moe_get_tensor(
    vitriol_moe_context_t* ctx,
    int layer,
    int expert_idx,
    const char* expert_type  // "gate", "up", "down"
);

// Get all tensors for a specific layer
// Returns array of tensor indices, count in *n_results
int* vitriol_moe_get_layer_tensors(
    vitriol_moe_context_t* ctx,
    int layer,
    size_t* n_results
);

// Print model and expert info
void vitriol_moe_print_info(vitriol_moe_context_t* ctx);

// Check if model is MoE
bool vitriol_moe_is_moe(vitriol_moe_context_t* ctx);

// Read raw tensor data from GGUF file
// Returns number of bytes read, 0 on error
size_t vitriol_moe_read_tensor_data(
    vitriol_moe_context_t* ctx,
    const vitriol_expert_tensor_info_t* tensor_info,
    void* buffer,
    size_t buffer_size
);

// Get total expert memory (all experts)
uint64_t vitriol_moe_total_expert_size(vitriol_moe_context_t* ctx);

// Estimate single expert size
uint64_t vitriol_moe_estimate_expert_size(vitriol_moe_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // VITRIOL_MOE_EXPERT_PARSER_H