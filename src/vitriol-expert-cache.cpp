// VITRIOL Expert Cache Manager
// Manages on-demand loading of MoE experts from SSD to GPU

#include "vitriol-expert-cache.h"
#include "vitriol-moe-expert-parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <queue>
#include <time.h>

// Internal cache entry structure
struct expert_entry {
    int expert_idx;
    int layer;
    char expert_type[16];
    void* gpu_buffer;
    size_t size;
    bool loaded;
    uint64_t last_access;
    
    // For LRU queue
    bool in_queue;
};

// Cache implementation
struct vitriol_expert_cache {
    vitriol_expert_cache_config_t config;
    vitriol_moe_context_t* parser;
    char model_path[1024];
    
    // Cache entries: key = "layer:expert_idx:type"
    std::map<std::string, expert_entry*> entries;
    
    // LRU queue
    std::queue<expert_entry*> lru_queue;
    
    // Statistics
    size_t total_vram_used_mb;
    int n_cache_hits;
    int n_cache_misses;
    int n_evictions;
    
    // CUDA context (would be initialized from vitriol.ko)
    // For now, we'll use CPU buffers as placeholder
    bool initialized;
};

static std::string make_key(int layer, int expert_idx, const char* type) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%d:%d:%s", layer, expert_idx, type ? type : "");
    return std::string(buf);
}

static uint64_t get_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

extern "C" {

vitriol_expert_cache_t* vitriol_expert_cache_init(vitriol_expert_cache_config_t* config) {
    vitriol_expert_cache_t* cache = (vitriol_expert_cache_t*)malloc(sizeof(vitriol_expert_cache_t));
    if (!cache) return NULL;
    
    memset(cache, 0, sizeof(vitriol_expert_cache_t));
    
    if (config) {
        cache->config = *config;
    } else {
        // Default config
        cache->config.max_cached_experts = 8;
        cache->config.max_vram_mb = 512;
        cache->config.use_lru = true;
        cache->config.async_load = true;
    }
    
    cache->parser = NULL;
    cache->total_vram_used_mb = 0;
    cache->n_cache_hits = 0;
    cache->n_cache_misses = 0;
    cache->n_evictions = 0;
    cache->initialized = true;
    
    printf("vitriol_expert_cache_init: Cache initialized\n");
    printf("  Max cached experts: %d\n", cache->config.max_cached_experts);
    printf("  Max VRAM: %zu MB\n", cache->config.max_vram_mb);
    printf("  Use LRU: %s\n", cache->config.use_lru ? "yes" : "no");
    
    return cache;
}

void vitriol_expert_cache_free(vitriol_expert_cache_t* cache) {
    if (!cache) return;
    
    // Clear all entries
    for (auto& kv : cache->entries) {
        expert_entry* entry = kv.second;
        if (entry->loaded && entry->gpu_buffer) {
            // In real implementation, would free GPU memory
            free(entry->gpu_buffer);  // Using CPU buffer for now
        }
        delete entry;
    }
    cache->entries.clear();
    
    if (cache->parser) {
        vitriol_moe_free(cache->parser);
    }
    
    free(cache);
}

void vitriol_expert_cache_set_model_path(
    vitriol_expert_cache_t* cache,
    const char* gguf_path
) {
    if (!cache || !gguf_path) return;
    
    strncpy(cache->model_path, gguf_path, sizeof(cache->model_path) - 1);
    
    // Initialize parser
    if (cache->parser) {
        vitriol_moe_free(cache->parser);
    }
    
    cache->parser = vitriol_moe_init(gguf_path);
    if (cache->parser) {
        vitriol_moe_print_info(cache->parser);
    } else {
        fprintf(stderr, "vitriol_expert_cache_set_model_path: Failed to init parser\n");
    }
}

int vitriol_expert_cache_get(
    vitriol_expert_cache_t* cache,
    int layer,
    int expert_idx,
    const char* expert_type,
    void** gpu_buffer_out,
    size_t* size_out
) {
    if (!cache || !gpu_buffer_out || !size_out) return -1;
    
    *gpu_buffer_out = NULL;
    *size_out = 0;
    
    std::string key = make_key(layer, expert_idx, expert_type);
    
    auto it = cache->entries.find(key);
    if (it != cache->entries.end()) {
        // Cache hit!
        expert_entry* entry = it->second;
        entry->last_access = get_timestamp();
        cache->n_cache_hits++;
        
        *gpu_buffer_out = entry->gpu_buffer;
        *size_out = entry->size;
        
        printf("vitriol_expert_cache_get: CACHE HIT - layer=%d expert=%d type=%s\n",
               layer, expert_idx, expert_type);
        
        return 0;
    }
    
    // Cache miss - need to load from SSD
    cache->n_cache_misses++;
    printf("vitriol_expert_cache_get: CACHE MISS - loading layer=%d expert=%d type=%s\n",
           layer, expert_idx, expert_type);
    
    // Check if we need to evict
    while (cache->entries.size() >= (size_t)cache->config.max_cached_experts) {
        if (cache->config.use_lru) {
            // Evict LRU entry
            if (!cache->lru_queue.empty()) {
                expert_entry* victim = cache->lru_queue.front();
                cache->lru_queue.pop();
                
                std::string victim_key = make_key(victim->layer, victim->expert_idx, victim->expert_type);
                auto victim_it = cache->entries.find(victim_key);
                if (victim_it != cache->entries.end()) {
                    if (victim->gpu_buffer) {
                        free(victim->gpu_buffer);  // CPU buffer for now
                    }
                    delete victim;
                    cache->entries.erase(victim_it);
                    cache->n_evictions++;
                    cache->total_vram_used_mb -= victim->size / (1024 * 1024);
                    
                    printf("  Evicted expert layer=%d idx=%d for layer=%d idx=%d\n",
                           victim->layer, victim->expert_idx, layer, expert_idx);
                }
            }
        } else {
            // No LRU, just remove first entry
            auto first = cache->entries.begin();
            if (first != cache->entries.end()) {
                expert_entry* entry = first->second;
                if (entry->gpu_buffer) free(entry->gpu_buffer);
                delete entry;
                cache->entries.erase(first);
                cache->n_evictions++;
            }
        }
    }
    
    // Load expert from GGUF file
    // In real implementation: use DMA via vitriol.ko
    // For now: simulate loading
    size_t estimated_size = 2048 * 512 * 2;  // ~2MB per expert projection
    void* buffer = malloc(estimated_size);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate buffer for expert\n");
        return -1;
    }
    
    // Create new entry
    expert_entry* entry = new expert_entry();
    entry->expert_idx = expert_idx;
    entry->layer = layer;
    strncpy(entry->expert_type, expert_type ? expert_type : "", sizeof(entry->expert_type) - 1);
    entry->gpu_buffer = buffer;
    entry->size = estimated_size;
    entry->loaded = true;
    entry->last_access = get_timestamp();
    entry->in_queue = false;
    
    // Add to cache
    cache->entries[key] = entry;
    cache->total_vram_used_mb += estimated_size / (1024 * 1024);
    
    // Add to LRU queue
    if (cache->config.use_lru) {
        cache->lru_queue.push(entry);
        entry->in_queue = true;
    }
    
    *gpu_buffer_out = buffer;
    *size_out = estimated_size;
    
    printf("  Loaded expert to cache: %zu MB total used\n", cache->total_vram_used_mb);
    
    return 0;
}

void vitriol_expert_cache_release(
    vitriol_expert_cache_t* cache,
    int layer,
    int expert_idx
) {
    // Just mark as available for eviction - don't actually release
    // This is a hint that the expert won't be needed immediately
    (void)cache;
    (void)layer;
    (void)expert_idx;
}

int vitriol_expert_cache_preload_layer(
    vitriol_expert_cache_t* cache,
    int layer,
    int* expert_indices,
    int n_experts
) {
    if (!cache || !expert_indices || n_experts <= 0) return -1;
    
    printf("vitriol_expert_cache_preload_layer: Layer %d, %d experts\n", layer, n_experts);
    
    for (int i = 0; i < n_experts; i++) {
        // Preload all 3 expert types (gate, up, down)
        const char* types[] = {"gate", "up", "down"};
        for (int j = 0; j < 3; j++) {
            void* buf;
            size_t size;
            int ret = vitriol_expert_cache_get(cache, layer, expert_indices[i], types[j], &buf, &size);
            if (ret != 0) {
                fprintf(stderr, "Failed to preload expert %d type %s\n", expert_indices[i], types[j]);
                return -1;
            }
        }
    }
    
    printf("  Preload complete for layer %d\n", layer);
    return 0;
}

void vitriol_expert_cache_get_stats(
    vitriol_expert_cache_t* cache,
    vitriol_expert_cache_stats_t* stats
) {
    if (!cache || !stats) return;
    
    stats->total_vram_used_mb = cache->total_vram_used_mb;
    stats->n_cached_experts = (int)cache->entries.size();
    stats->n_cache_hits = cache->n_cache_hits;
    stats->n_cache_misses = cache->n_cache_misses;
    stats->n_evictions = cache->n_evictions;
}

void vitriol_expert_cache_print_status(vitriol_expert_cache_t* cache) {
    if (!cache) return;
    
    vitriol_expert_cache_stats_t stats;
    vitriol_expert_cache_get_stats(cache, &stats);
    
    printf("=== VITRIOL Expert Cache Status ===\n");
    printf("VRAM used: %zu MB\n", stats.total_vram_used_mb);
    printf("Cached experts: %d\n", stats.n_cached_experts);
    printf("Cache hits: %d\n", stats.n_cache_hits);
    printf("Cache misses: %d\n", stats.n_cache_misses);
    printf("Evictions: %d\n", stats.n_evictions);
    
    if (stats.n_cache_hits + stats.n_cache_misses > 0) {
        float hit_rate = (float)stats.n_cache_hits / (stats.n_cache_hits + stats.n_cache_misses) * 100;
        printf("Hit rate: %.1f%%\n", hit_rate);
    }
    
    printf("===================================\n");
}

void vitriol_expert_cache_clear(vitriol_expert_cache_t* cache) {
    if (!cache) return;
    
    for (auto& kv : cache->entries) {
        expert_entry* entry = kv.second;
        if (entry->gpu_buffer) {
            free(entry->gpu_buffer);
        }
        delete entry;
    }
    cache->entries.clear();
    
    while (!cache->lru_queue.empty()) {
        cache->lru_queue.pop();
    }
    
    cache->total_vram_used_mb = 0;
    
    printf("vitriol_expert_cache_clear: Cache cleared\n");
}

} // extern "C"