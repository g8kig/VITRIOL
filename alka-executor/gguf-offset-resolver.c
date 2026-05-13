/*
 * gguf-offset-resolver — Extract expert tensor offsets from GGUF files
 *
 * Copyright 2026 Randy Smits-Schreuder Goedheijt
 * Licensed under Apache 2.0 with Runtime Exception.
 *
 * Parses GGUF v3 binary format to find MoE expert tensor offsets.
 * Outputs a JSON-like report that can be used to generate Alka streams.
 *
 * Usage: gguf-offset-resolver <model.gguf> [--layer N] [--expert N]
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <getopt.h>

/* ── GGML Type Constants ─────────────────────────────────────────── */

#define GGML_TYPE_F32      0
#define GGML_TYPE_F16      1
#define GGML_TYPE_Q4_0     2
#define GGML_TYPE_Q4_1     3
#define GGML_TYPE_Q5_0     6
#define GGML_TYPE_Q5_1     7
#define GGML_TYPE_Q8_0     8
#define GGML_TYPE_Q8_1     9
#define GGML_TYPE_Q2_K     10
#define GGML_TYPE_Q3_K     11
#define GGML_TYPE_Q4_K     12
#define GGML_TYPE_Q5_K     13
#define GGML_TYPE_Q6_K     14
#define GGML_TYPE_Q8_K     15
#define GGML_TYPE_IQ2_XXS  16
#define GGML_TYPE_IQ2_XS   17
#define GGML_TYPE_IQ3_XXS  18
#define GGML_TYPE_IQ1_S    19
#define GGML_TYPE_IQ4_NL   20
#define GGML_TYPE_IQ3_S    21
#define GGML_TYPE_IQ2_S    22
#define GGML_TYPE_IQ4_XS   23
#define GGML_TYPE_I8       24
#define GGML_TYPE_I16      25
#define GGML_TYPE_I32      26
#define GGML_TYPE_I64      27
#define GGML_TYPE_F64      28
#define GGML_TYPE_IQ1_M    29
#define GGML_TYPE_BF16     30
#define GGML_TYPE_TQ1_0    34
#define GGML_TYPE_TQ2_0    35
#define GGML_TYPE_MXFP4    39
#define GGML_TYPE_NVFP4    40

/* ── GGUF Format Constants ─────────────────────────────────────── */

#define GGUF_MAGIC "GGUF"
#define GGUF_VERSION 3

/* GGUF value types */
#define GGUF_TYPE_UINT8    0
#define GGUF_TYPE_INT8     1
#define GGUF_TYPE_UINT16   2
#define GGUF_TYPE_INT16    3
#define GGUF_TYPE_UINT32   4
#define GGUF_TYPE_INT32    5
#define GGUF_TYPE_FLOAT32  6
#define GGUF_TYPE_BOOL     7
#define GGUF_TYPE_STRING   8
#define GGUF_TYPE_ARRAY    9
#define GGUF_TYPE_UINT64   10
#define GGUF_TYPE_INT64    11
#define GGUF_TYPE_FLOAT64  12

/* ── Data Structures ───────────────────────────────────────────── */

struct gguf_string {
    uint64_t len;
    char *data;
};

struct gguf_tensor {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;  /* Offset within the tensor data section */
};

struct expert_offset {
    int layer;
    int expert_idx;
    char tensor_name[256];
    uint64_t file_offset;  /* Absolute offset in GGUF file */
    uint64_t tensor_size;
    uint32_t tensor_type;
};

/* ── GGUF Reader ───────────────────────────────────────────────── */

static uint32_t read_u32(FILE *f)
{
    uint32_t v;
    fread(&v, 4, 1, f);
    return v;
}

static uint64_t read_u64(FILE *f)
{
    uint64_t v;
    fread(&v, 8, 1, f);
    return v;
}

static int32_t read_i32(FILE *f)
{
    int32_t v;
    fread(&v, 4, 1, f);
    return v;
}

static int64_t read_i64(FILE *f)
{
    int64_t v;
    fread(&v, 8, 1, f);
    return v;
}

static float read_f32(FILE *f)
{
    float v;
    fread(&v, 4, 1, f);
    return v;
}

static struct gguf_string read_string(FILE *f)
{
    struct gguf_string s;
    s.len = read_u64(f);
    s.data = malloc(s.len + 1);
    fread(s.data, 1, s.len, f);
    s.data[s.len] = '\0';
    return s;
}

static void free_string(struct gguf_string s)
{
    free(s.data);
}

static void skip_value(FILE *f, uint32_t type)
{
    switch (type) {
    case GGUF_TYPE_UINT8:   fread(&(uint8_t){0}, 1, 1, f); break;
    case GGUF_TYPE_INT8:    fread(&(int8_t){0}, 1, 1, f); break;
    case GGUF_TYPE_UINT16:  fread(&(uint16_t){0}, 2, 1, f); break;
    case GGUF_TYPE_INT16:   fread(&(int16_t){0}, 2, 1, f); break;
    case GGUF_TYPE_UINT32:  read_u32(f); break;
    case GGUF_TYPE_INT32:   read_i32(f); break;
    case GGUF_TYPE_FLOAT32: read_f32(f); break;
    case GGUF_TYPE_BOOL:    fread(&(uint8_t){0}, 1, 1, f); break;
    case GGUF_TYPE_STRING: { struct gguf_string s = read_string(f); free_string(s); break; }
    case GGUF_TYPE_UINT64:  read_u64(f); break;
    case GGUF_TYPE_INT64:   read_i64(f); break;
    case GGUF_TYPE_FLOAT64: { double v; fread(&v, 8, 1, f); break; }
    case GGUF_TYPE_ARRAY: {
        uint32_t atype = read_u32(f);
        uint64_t count = read_u64(f);
        for (uint64_t i = 0; i < count; i++)
            skip_value(f, atype);
        break;
    }
    }
}

static const char *tensor_type_name(uint32_t type)
{
    switch (type) {
    case GGML_TYPE_F32:    return "F32";
    case GGML_TYPE_F16:    return "F16";
    case GGML_TYPE_Q4_0:   return "Q4_0";
    case GGML_TYPE_Q4_1:   return "Q4_1";
    case GGML_TYPE_Q5_0:   return "Q5_0";
    case GGML_TYPE_Q5_1:   return "Q5_1";
    case GGML_TYPE_Q8_0:   return "Q8_0";
    case GGML_TYPE_Q8_1:   return "Q8_1";
    case GGML_TYPE_Q2_K:   return "Q2_K";
    case GGML_TYPE_Q3_K:   return "Q3_K";
    case GGML_TYPE_Q4_K:   return "Q4_K";
    case GGML_TYPE_Q5_K:   return "Q5_K";
    case GGML_TYPE_Q6_K:   return "Q6_K";
    case GGML_TYPE_Q8_K:   return "Q8_K";
    case GGML_TYPE_IQ2_XXS:return "IQ2_XXS";
    case GGML_TYPE_IQ2_XS: return "IQ2_XS";
    case GGML_TYPE_IQ3_XXS:return "IQ3_XXS";
    case GGML_TYPE_IQ1_S:  return "IQ1_S";
    case GGML_TYPE_IQ4_NL: return "IQ4_NL";
    case GGML_TYPE_IQ3_S:  return "IQ3_S";
    case GGML_TYPE_IQ2_S:  return "IQ2_S";
    case GGML_TYPE_IQ4_XS: return "IQ4_XS";
    case GGML_TYPE_IQ1_M:  return "IQ1_M";
    case GGML_TYPE_BF16:   return "BF16";
    case GGML_TYPE_TQ1_0:  return "TQ1_0";
    case GGML_TYPE_TQ2_0:  return "TQ2_0";
    case GGML_TYPE_MXFP4:  return "MXFP4";
    case GGML_TYPE_NVFP4:  return "NVFP4";
    default:               return "UNKNOWN";
    }
}

static uint64_t tensor_type_size(uint32_t type)
{
    switch (type) {
    case GGML_TYPE_F32:    return 4;
    case GGML_TYPE_F16:    return 2;
    case GGML_TYPE_BF16:   return 2;
    case GGML_TYPE_F64:    return 8;
    case GGML_TYPE_I8:     return 1;
    case GGML_TYPE_I16:    return 2;
    case GGML_TYPE_I32:    return 4;
    case GGML_TYPE_I64:    return 8;
    case GGML_TYPE_Q4_0:   return 1;  /* 4 bits + block header */
    case GGML_TYPE_Q4_1:   return 1;
    case GGML_TYPE_Q5_0:   return 1;
    case GGML_TYPE_Q5_1:   return 1;
    case GGML_TYPE_Q8_0:   return 1;
    case GGML_TYPE_Q8_1:   return 1;
    case GGML_TYPE_Q2_K:   return 1;
    case GGML_TYPE_Q3_K:   return 1;
    case GGML_TYPE_Q4_K:   return 1;
    case GGML_TYPE_Q5_K:   return 1;
    case GGML_TYPE_Q6_K:   return 1;
    case GGML_TYPE_Q8_K:   return 1;
    case GGML_TYPE_IQ2_XXS:return 1;
    case GGML_TYPE_IQ2_XS: return 1;
    case GGML_TYPE_IQ3_XXS:return 1;
    case GGML_TYPE_IQ1_S:  return 1;
    case GGML_TYPE_IQ4_NL: return 1;
    case GGML_TYPE_IQ3_S:  return 1;
    case GGML_TYPE_IQ2_S:  return 1;
    case GGML_TYPE_IQ4_XS: return 1;
    case GGML_TYPE_IQ1_M:  return 1;
    case GGML_TYPE_TQ1_0:  return 1;
    case GGML_TYPE_TQ2_0:  return 1;
    case GGML_TYPE_MXFP4:  return 1;
    case GGML_TYPE_NVFP4:  return 1;
    default:               return 4;
    }
}

/* ── Main Parser ───────────────────────────────────────────────── */

static int is_expert_tensor(const char *name)
{
    return (strstr(name, "ffn_gate_exps") != NULL ||
            strstr(name, "ffn_up_exps") != NULL ||
            strstr(name, "ffn_down_exps") != NULL);
}

static int parse_gguf(const char *path, int target_layer, int target_expert,
                      struct expert_offset **out_offsets, int *out_count)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    /* Read and verify magic */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4) {
        fprintf(stderr, "Error: Cannot read GGUF magic\n");
        fclose(f);
        return -1;
    }
    if (memcmp(magic, GGUF_MAGIC, 4) != 0) {
        fprintf(stderr, "Error: Not a GGUF file (magic: %.4s)\n", magic);
        fclose(f);
        return -1;
    }

    uint32_t version = read_u32(f);
    if (version != GGUF_VERSION) {
        fprintf(stderr, "Warning: GGUF version %u (expected %u)\n", version, GGUF_VERSION);
    }

    uint64_t tensor_count = read_u64(f);
    uint64_t metadata_kv_count = read_u64(f);

    printf("GGUF v%u: %" PRIu64 " tensors, %" PRIu64 " metadata entries\n",
           version, tensor_count, metadata_kv_count);

    /* Skip metadata */
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        struct gguf_string key = read_string(f);
        uint32_t type = read_u32(f);
        skip_value(f, type);
        free_string(key);
    }

    /* Read tensor info — store offsets and names first */
    struct {
        char *name;
        uint64_t offset;
        uint32_t type;
        uint32_t n_dims;
        uint64_t dims[4];
    } *tensors = calloc(tensor_count, sizeof(*tensors));

    for (uint64_t i = 0; i < tensor_count; i++) {
        struct gguf_string name = read_string(f);
        uint32_t n_dims = read_u32(f);

        uint64_t dims[4] = {1, 1, 1, 1};
        for (uint32_t d = 0; d < n_dims; d++)
            dims[d] = read_u64(f);

        uint32_t type = read_u32(f);
        uint64_t offset = read_u64(f);

        tensors[i].name = strdup(name.data);
        tensors[i].offset = offset;
        tensors[i].type = type;
        tensors[i].n_dims = n_dims;
        for (uint32_t d = 0; d < n_dims; d++)
            tensors[i].dims[d] = dims[d];

        free_string(name);
    }

    /* Tensor data section starts here */
    long tensor_data_start = ftell(f);

    /* Compute sizes from offset deltas */
    struct expert_offset *offsets = calloc(tensor_count, sizeof(*offsets));
    int offset_count = 0;

    for (uint64_t i = 0; i < tensor_count; i++) {
        if (!is_expert_tensor(tensors[i].name) && target_layer < 0) {
            /* Not filtering — include all tensors */
        } else if (!is_expert_tensor(tensors[i].name)) {
            continue;
        }

        /* Size = next tensor offset - this tensor offset, or estimate from dims */
        uint64_t size;
        if (i + 1 < tensor_count) {
            size = tensors[i + 1].offset - tensors[i].offset;
        } else {
            /* Last tensor — estimate from dims and type */
            uint64_t num_weights = 1;
            for (uint32_t d = 0; d < tensors[i].n_dims; d++)
                num_weights *= tensors[i].dims[d];
            size = num_weights * tensor_type_size(tensors[i].type);
            /* Round up to 32-byte alignment */
            size = (size + 31) & ~(uint64_t)31;
        }

        /* Parse layer from name */
        int layer = -1;
        char *blk = strstr(tensors[i].name, "blk.");
        if (blk) layer = atoi(blk + 4);

        if (target_layer >= 0 && layer != target_layer) continue;

        struct expert_offset *eo = &offsets[offset_count];
        eo->layer = layer;
        eo->expert_idx = -1;  /* Would need to parse from dims for expert index */
        strncpy(eo->tensor_name, tensors[i].name, sizeof(eo->tensor_name) - 1);
        eo->file_offset = tensors[i].offset + tensor_data_start;
        eo->tensor_size = size;
        eo->tensor_type = tensors[i].type;
        offset_count++;

        if (target_layer >= 0) {
            printf("  [%s] layer=%d type=%s dims=[",
                   tensors[i].name, layer, tensor_type_name(tensors[i].type));
            for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
                if (d > 0) printf(", ");
                printf("%" PRIu64, tensors[i].dims[d]);
            }
            printf("] offset=0x%" PRIx64 " size=%" PRIu64 "\n",
                   tensors[i].offset, size);
        }
    }

    *out_offsets = offsets;
    *out_count = offset_count;

    /* Cleanup */
    for (uint64_t i = 0; i < tensor_count; i++)
        free(tensors[i].name);
    free(tensors);

    fclose(f);
    return 0;
}

/* ── Alka Stream Generation ────────────────────────────────────── */

static void generate_alka_source(struct expert_offset *offsets, int count,
                                 int max_experts, const char *vessel_name)
{
    printf("\n/* Auto-generated Alka source for expert streaming */\n");
    printf("REQUIRE vitriol_rig.alkavl;\n\n");
    printf("CLAIM %s;\n", vessel_name);
    printf("CLAIM NVME_BOOT;\n\n");
    printf("LIMIT %s.THERMAL MAX 85000;\n\n", vessel_name);

    /* Generate SHIFT→FLOW→FENCE pattern for each expert chunk */
    int chunk = 0;
    uint64_t window_offset = 0;
    uint64_t aperture_size = 256ULL * 1024 * 1024;  /* 256MB BAR1 window */

    for (int i = 0; i < count; i++) {
        if (max_experts > 0 && offsets[i].expert_idx >= max_experts)
            continue;

        /* Check if we need a new window */
        if (window_offset + offsets[i].tensor_size > aperture_size) {
            /* FENCE before shifting */
            printf("FENCE %s.METAPAGE == %d;\n", vessel_name, chunk + 1);
            window_offset = 0;
        }

        printf("\n/* Expert %d, layer %d */\n", offsets[i].expert_idx, offsets[i].layer);
        printf("SHIFT %s.DATA_PLANE @ 0x%" PRIx64 ";\n", vessel_name, window_offset);
        printf("FLOW NVME_BOOT[0x%" PRIx64 "] -> %s.DATA_PLANE[0x%" PRIx64 "] %u;\n",
               offsets[i].file_offset, vessel_name, window_offset,
               (uint32_t)offsets[i].tensor_size);

        window_offset += offsets[i].tensor_size;
        chunk++;
    }

    if (chunk > 0) {
        printf("FENCE %s.METAPAGE == %d;\n", vessel_name, chunk);
    }

    printf("\nSYNC L3;\n");
    printf("SIGNAL INFERENCE_COMPLETE;\n");
}

/* ── Main ──────────────────────────────────────────────────────── */

static void usage(const char *prog)
{
    printf("Usage: %s <model.gguf> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --layer N        Show only layer N\n");
    printf("  --expert N       Show only expert N\n");
    printf("  --generate       Generate Alka source code\n");
    printf("  --vessel NAME    Vessel name for generated source (default: GPU_MAIN)\n");
    printf("  --help           Show this help\n");
}

int main(int argc, char *argv[])
{
    const char *gguf_path = NULL;
    int target_layer = -1;
    int target_expert = -1;
    int generate = 0;
    const char *vessel_name = "GPU_MAIN";

    static struct option long_options[] = {
        {"layer",   required_argument, 0, 'l'},
        {"expert",  required_argument, 0, 'e'},
        {"generate",no_argument,       0, 'g'},
        {"vessel",  required_argument, 0, 'v'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "l:e:gv:h", long_options, NULL)) != -1) {
        switch (opt) {
        case 'l': target_layer = atoi(optarg); break;
        case 'e': target_expert = atoi(optarg); break;
        case 'g': generate = 1; break;
        case 'v': vessel_name = optarg; break;
        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Error: Missing GGUF file path\n");
        usage(argv[0]);
        return 1;
    }
    gguf_path = argv[optind];

    struct expert_offset *offsets = NULL;
    int count = 0;

    if (parse_gguf(gguf_path, target_layer, target_expert, &offsets, &count) != 0)
        return 1;

    printf("\nFound %d expert tensors\n", count);

    if (count > 0) {
        printf("\n%-50s  layer  type     size       file_offset\n", "TENSOR");
        printf("%-50s  -----  -------  ---------  ------------\n", "------");
        for (int i = 0; i < count; i++) {
            printf("%-50s  %5d  %-7s  %9" PRIu64 "  0x%016" PRIx64 "\n",
                   offsets[i].tensor_name, offsets[i].layer,
                   tensor_type_name(offsets[i].tensor_type),
                   offsets[i].tensor_size, offsets[i].file_offset);
        }
    }

    if (generate) {
        generate_alka_source(offsets, count, target_expert, vessel_name);
    }

    free(offsets);
    return 0;
}
