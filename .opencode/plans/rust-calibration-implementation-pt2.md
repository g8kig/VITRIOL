# Part 2 — Module Designs

## 4.1 `gguf.rs` — GGUF v3 Parser

Binary layout (from `ggml/include/gguf.h` + `ggml/src/gguf.cpp`):

```
[4 bytes]  magic = "GGUF" (0x46554747 LE)
[4 bytes]  version (u32) — currently 3
[8 bytes]  tensor_count (i64)
[8 bytes]  kv_count (i64)
[...]      KV pairs (key=string, type=i32, value=type-dep)
[...]      Tensor infos (name=string, n_dims=u32, dims=[i64;n_dims], type=i32, offset=u64)
[...]      Data blob (aligned, not read by calibrator)
```

### GGUF value types (KV pair parsing)

| Enum | Name    | Wire size                     |
|------|---------|-------------------------------|
| 0    | uint8   | 1 byte                        |
| 1    | int8    | 1 byte                        |
| 2    | uint16  | 2 bytes                       |
| 3    | int16   | 2 bytes                       |
| 4    | uint32  | 4 bytes                       |
| 5    | int32   | 4 bytes                       |
| 6    | float32 | 4 bytes                       |
| 7    | bool    | 1 byte (i8)                   |
| 8    | string  | u64 len + bytes               |
| 9    | array   | i32 elem_type + u64 count + N elements |
| 10   | uint64  | 8 bytes                       |
| 11   | int64   | 8 bytes                       |
| 12   | float64 | 8 bytes                       |

### GGML tensor types (tensor size calc)

Table of `(enum, name, blck_size, type_size, bytes_per_elem)`.

**Core types:**
- 0=f32: blck=1, type_sz=4, bpe=4.0
- 1=f16: blck=1, type_sz=2, bpe=2.0
- 2=q4_0: blck=32, type_sz=18, bpe=0.5625
- 8=q8_0: blck=32, type_sz=34, bpe=1.0625
- 10=q2_K: blck=256, type_sz=40, bpe=0.15625
- 12=q4_K: blck=256, type_sz=144, bpe=0.5625
- 14=q6_K: blck=256, type_sz=210, bpe=0.82031
- 16=iq2_xxs: blck=256, type_sz=66, bpe=0.25781
- 17=iq2_xs: blck=256, type_sz=74, bpe=0.28906
- 18=iq3_xxs: blck=256, type_sz=98, bpe=0.38281
- 19=iq1_s: blck=256, type_sz=50, bpe=0.19531
- 20=iq4_nl: blck=32, type_sz=18, bpe=0.5625
- 21=iq3_s: blck=256, type_sz=116, bpe=0.45312
- 22=iq2_s: blck=256, type_sz=82, bpe=0.32031
- 23=iq4_xs: blck=256, type_sz=148, bpe=0.57812

Formula: `bytes = type_sz * ne[0] / blck * ne[1] * ne[2] * ne[3]`

### Data structures

```rust
pub struct ModelInfo {
    pub architecture: String,
    pub context_length: u64,
    pub block_count: u64,
    pub expert_count: u64,
    pub expert_used_count: u64,
    pub embedding_length: u64,
    pub head_count: u64,
    pub head_count_kv: u64,
    pub has_mtp: bool,
    pub total_size_bytes: u64,
    pub tensor_count: usize,
    pub per_layer_attn_bytes: u64,     // average per layer
    pub per_layer_experts_bytes: u64,  // average per layer
}
```

### Architecture-aware key resolution

GGUF KV keys use `{architecture}.{key}` prefix. E.g. `qwen35moe.block_count`, `llama.attention.head_count`. Resolution order:
1. `{arch}.{key}` — primary
2. `llama.{key}` — fallback for older models
3. `{arch}_{key}` — secondary fallback

Key names to extract:
- `{arch}.block_count`
- `{arch}.embedding_length`
- `{arch}.context_length`
- `{arch}.expert_count`
- `{arch}.expert_used_count`
- `{arch}.attention.head_count`
- `{arch}.attention.head_count_kv`

### Tensor name categorization

Parse with regex `blk\.(\d+)\.(.+)`:
- `attn_qkv`, `attn_output` → attention layer counter
- `ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps` → expert layer counter
- `mtp.*` → has_mtp flag (skip in per-layer averages)

Non-layer tensors (`token_embd.weight`, `output.weight`, `blk.N.attn_norm`, etc.) contribute to `total_size_bytes` only.

Average per layer = `total_category_bytes / unique_layer_ids_seen`.

### Implementation approach

Trait-based `Read + Seek` generic reader:
```rust
pub fn read_gguf(path: &Path) -> Result