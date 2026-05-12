#ifndef VITRIOL_EXPERT_CACHE_H
#define VITRIOL_EXPERT_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Cache entry for a single expert
typedef struct {
    int expert_idx;           // Expert index (0-255)
    int layer;               // Layer (0-39)
    char expert_type[16];    // "gate", "up", "down"
    void* gpu_buffer;        // GPU memory pointer
    size_t size;             // Size in bytes
    bool loaded;             // Is currently loaded
    uint64_t last_access;    // Timestamp for LRU
} vitriol_expert_cache_entry_t;

// Cache configuration
typedef struct {
    int max_cached_experts;  // Max experts in VRAM at once (default: 8)
    size_t max_vram_mb;      // Max VRAM for expert cache (default: 512MB)
    bool use_lru;           // Use LRU eviction (default: true)
    bool async_load;        // Async loading from SSD (default: true)
} vitriol_expert_cache_config_t;

// Expert cache manager
typedef struct vitriol_expert_cache vitriol_expert_cache_t;

// Initialize expert cache
vitriol_expert_cache_t* vitriol_expert_cache_init(
    vitriol_expert_cache_config_t* config
);

// Free expert cache
void vitriol_expert_cache_free(vitriol_expert_cache_t* cache);

// Request an expert - loads from SSD if not cached
// Returns 0 on success, -1 on error
int vitriol_expert_cache_get(
    vitriol_expert_cache_t* cache,
    int layer,
    int expert_idx,
    const char* expert_type,
    void** gpu_buffer_out,
    size_t* size_out
);

// Release an expert (mark as available for eviction)
void vitriol_expert_cache_release(
    vitriol_expert_cache_t* cache,
    int layer,
    int expert_idx
);

// Preload experts for a layer (prepare ahead of time)
int vitriol_expert_cache_preload_layer(
    vitriol_expert_cache_t* cache,
    int layer,
    int* expert_indices,
    int n_experts
);

// Get cache statistics
typedef struct {
    size_t total_vram_used_mb;
    int n_cached_experts;
    int n_cache_hits;
    int n_cache_misses;
    int n_evictions;
} vitriol_expert_cache_stats_t;

void vitriol_expert_cache_get_stats(
    vitriol_expert_cache_t* cache,
    vitriol_expert_cache_stats_t* stats
);

// Print cache status
void vitriol_expert_cache_print_status(vitriol_expert_cache_t* cache);

// Clear all cached experts
void vitriol_expert_cache_clear(vitriol_expert_cache_t* cache);

// Set the model path for loading
void vitriol_expert_cache_set_model_path(
    vitriol_expert_cache_t* cache,
    const char* gguf_path
);

#ifdef __cplusplus
}
#endif

#endif // VITRIOL_EXPERT_CACHE_H