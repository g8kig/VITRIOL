#!/usr/bin/env python3
"""Read GGUF metadata and tensor info. Called by vitriol calibrate."""

import struct
import sys
import json
import re

GGUF_TYPES = {
    0: ("uint8",   1, "B"),
    1: ("int8",    1, "b"),
    2: ("uint16",  2, "<H"),
    3: ("int16",   2, "<h"),
    4: ("uint32",  4, "<I"),
    5: ("int32",   4, "<i"),
    6: ("float32", 4, "<f"),
    7: ("bool",    1, "B"),
    8: ("string",  0, None),
    9: ("array",   0, None),
    10: ("uint64",  8, "<Q"),
    11: ("int64",   8, "<q"),
    12: ("float64", 8, "<d"),
    13: ("bfloat16", 2, "<H"),
}

def read_string(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8', errors='replace')

def skip_kv_value(f, kt):
    if kt == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif kt == 9:
        arr_type, arr_len = struct.unpack('<IQ', f.read(12))
        info = GGUF_TYPES.get(arr_type)
        if info:
            f.read(arr_len * info[1])
        else:
            f.read(arr_len * 4)
    else:
        info = GGUF_TYPES.get(kt)
        if info:
            f.read(info[1])

def read_kv_value(f, kt):
    if kt == 8:
        return read_string(f)
    elif kt == 9:
        arr_type, arr_len = struct.unpack('<IQ', f.read(12))
        info = GGUF_TYPES.get(arr_type)
        if not info:
            f.read(arr_len * 4)
            return None
        items = []
        for _ in range(arr_len):
            if info[2]:
                items.append(struct.unpack(info[2], f.read(info[1]))[0])
            else:
                skip_kv_value(f, arr_type)
        return items
    elif kt == 7:
        return bool(struct.unpack('B', f.read(1))[0])
    else:
        info = GGUF_TYPES.get(kt)
        if info and info[2]:
            return struct.unpack(info[2], f.read(info[1]))[0]
        else:
            skip_kv_value(f, kt)
            return None

def read_kv(f, kv_count):
    meta = {}
    for _ in range(kv_count):
        klen = struct.unpack('<Q', f.read(8))[0]
        key = f.read(klen).decode('utf-8', errors='replace')
        kt = struct.unpack('<I', f.read(4))[0]
        val = read_kv_value(f, kt)
        if val is not None:
            meta[key] = val
    return meta

def read_tensors(f, tensor_count):
    tensors = []
    # Block size and type size for common GGML types (bytes per element)
    # Derived from ggml_type_type_size / ggml_blck_size
    type_sizes = {
        0: 4.0,    # f32
        1: 2.0,    # f16
        2: 0.5,    # q4_0  (2.5B block: 18 bytes / 32)
        3: 0.5,    # q4_1  (2.5B block: 20 bytes / 32)
        6: 0.625,  # q5_0  (2.5B block: 22 bytes / 32)
        7: 0.625,  # q5_1  (2.5B block: 24 bytes / 32)
        8: 1.0,    # q8_0  (2.5B block: 34 bytes / 32)
        10: 0.25,  # q2_K  (2.5B block: 32 bytes / 128)
        11: 0.375, # q3_K  (2.5B block: 48 bytes / 128)
        12: 0.5,   # q4_K  (2.5B block: 64 bytes / 128)
        13: 0.625, # q5_K  (2.5B block: 80 bytes / 128)
        14: 0.75,  # q6_K  (2.5B block: 96 bytes / 128)
        16: 0.2578125,  # IQ2_XXS (2.5B: 33 bytes / 128)
        17: 0.2890625,  # IQ2_XS  (2.5B: 37 bytes / 128)
        18: 0.28125,    # IQ2_S   (2.5B: 72 bytes / 256)
        19: 0.125,      # IQ1_S   (2.5B: 32 bytes / 256)
        22: 0.5625,     # IQ4_NL  (2.5B: 72 bytes / 128)
        23: 0.4375,     # IQ3_S   (2.5B: 112 bytes / 256)
    }
    for _ in range(tensor_count):
        tname = read_string(f)
        ndim = struct.unpack('<I', f.read(4))[0]
        dims = list(struct.unpack('<' + 'q' * ndim, f.read(8 * ndim)))
        dtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        esize = type_sizes.get(dtype, 2)
        elements = 1
        for d in dims:
            elements *= d
        tsize = int(elements * esize)
        tensors.append({
            "name": tname,
            "shape": list(dims),
            "size_bytes": tsize,
            "offset": offset,
        })
    return tensors


def analyze_model(model_path):
    with open(model_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            return {"error": "not a GGUF file"}

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]

        meta = read_kv(f, kv_count)
        tensors = read_tensors(f, tensor_count)

    # Extract key metadata (architecture-aware key prefixes)
    arch = meta.get('general.architecture', 'unknown')
    arch_keys = [f'{arch}.', 'llama.', f'{arch}_']
    def get_meta(key_suffix, default=0):
        for prefix in arch_keys:
            val = meta.get(f'{prefix}{key_suffix}')
            if val is not None:
                return val
        return default

    ctx_len = get_meta('context_length')
    block_count = get_meta('block_count')
    expert_count = get_meta('expert_count')
    embd_len = get_meta('embedding_length')

    # Per-layer tensor sizes
    per_layer_attn = {}
    per_layer_exp = {}
    total_size = 0
    has_mtp = False

    for t in tensors:
        total_size += t["size_bytes"]
        m = re.match(r'blk\.(\d+)\.(.*)', t["name"])
        if m:
            lidx = int(m.group(1))
            name = m.group(2)
            if 'attn_qkv' in name or 'attn_output' in name:
                per_layer_attn[lidx] = per_layer_attn.get(lidx, 0) + t["size_bytes"]
            elif 'ffn_down_exps' in name or 'ffn_gate_exps' in name or 'ffn_up_exps' in name:
                per_layer_exp[lidx] = per_layer_exp.get(lidx, 0) + t["size_bytes"]
        if 'mtp' in t["name"].lower():
            has_mtp = True

    # Also check metadata for MTP
    if not has_mtp:
        for k in meta:
            if 'nextn_predict_layers' in k or 'mtp' in k.lower():
                val = meta[k]
                if isinstance(val, (int, float)) and val > 0:
                    has_mtp = True
                    break

    avg_attn = int(sum(per_layer_attn.values()) / len(per_layer_attn) / 1048576) if per_layer_attn else 0
    avg_exp = int(sum(per_layer_exp.values()) / len(per_layer_exp) / 1048576) if per_layer_exp else 0

    result = {
        "architecture": arch,
        "context_length": ctx_len,
        "block_count": block_count,
        "expert_count": expert_count,
        "embedding_length": embd_len,
        "total_size_mib": int(total_size / 1048576),
        "tensor_count": len(tensors),
        "has_mtp": has_mtp,
        "per_layer_attn_mib": max(avg_attn, 1),
        "per_layer_experts_mib": max(avg_exp, 1),
    }
    return result


def estimate_vram(hardware_path, model_data):
    with open(hardware_path) as f:
        hw = json.load(f)

    vram_total = hw['gpus'][0]['vram_mib']
    block_count = model_data['block_count'] or 40
    per_layer_exp = model_data['per_layer_experts_mib'] or 120
    model_gpu = 1334  # actual measured from load_tensors

    kv_per_token = 1.5625  # q4_0 K + f16 V at 65K
    compute_256 = 246
    compute_128 = 186
    overhead = 2871  # calibrated from icarus v1
    safety = 0.9
    usable = vram_total * safety

    def total_vram(pin, ctx, ubatch):
        pin_cost = pin * per_layer_exp
        kv_cost = ctx * kv_per_token / 1024
        compute = compute_128 if ubatch <= 128 else compute_256
        return model_gpu + pin_cost + kv_cost + compute + overhead

    estimates = {}
    for ctx in [32768, 65536, 131072]:
        for ubatch in [128, 256]:
            max_pin = 0
            for pin in range(0, min(block_count, 24), 2):
                if total_vram(pin, ctx, ubatch) <= usable:
                    max_pin = pin
            v = total_vram(max_pin, ctx, ubatch)
            estimates[f'ctx{ctx}_ub{ubatch}'] = {
                'max_pin': max_pin,
                'vram_mib': round(v, 0),
                'vram_pct': round(v / vram_total * 100, 1)
            }

    best_pin = 12
    best_ctx = 65536
    best_ubatch = 128
    best_vram = total_vram(best_pin, best_ctx, best_ubatch)

    candidates = [
        (12, 65536, 128),
        (16, 65536, 128),
        (12, 65536, 256),
        (8, 65536, 128),
        (12, 32768, 128),
        (16, 65536, 256),
        (0, 65536, 128),
        (12, 131072, 128),
    ]
    for pin, ctx, ubatch in candidates:
        v = total_vram(pin, ctx, ubatch)
        key = f'pin{pin}_ctx{ctx}_ub{ubatch}'
        if key not in estimates:
            estimates[key] = {}
        estimates[key].update({
            'vram_mib': round(v, 0),
            'vram_pct': round(v / vram_total * 100, 1),
            'feasible': v <= usable and pin <= block_count
        })

    return {
        "model_vram_mib": model_gpu,
        "vram_total_mib": vram_total,
        "vram_safety_margin": safety,
        "vram_usable_mib": round(usable, 0),
        "per_layer_pin_cost_mib": per_layer_exp,
        "overhead_constant_mib": overhead,
        "estimates": estimates,
        "optimal": {
            "pin_first_n_layers": best_pin,
            "context": best_ctx,
            "ubatch_size": best_ubatch,
            "draft_n_max": 5,
            "k_quant": "q4_0",
            "v_quant": "f16",
            "estimated_vram_mib": round(best_vram, 0),
            "estimated_vram_pct": round(best_vram / vram_total * 100, 1),
        }
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: gguf_reader.py <model> [hardware.json]"}))
        sys.exit(1)

    model_path = sys.argv[1]
    model_data = analyze_model(model_path)

    if len(sys.argv) >= 3:
        hardware_path = sys.argv[2]
        bounds = estimate_vram(hardware_path, model_data)
        print(json.dumps({"model": model_data, "bounds": bounds}, indent=2))
    else:
        print(json.dumps(model_data, indent=2))
