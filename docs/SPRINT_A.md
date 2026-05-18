# Sprint A: TQ1_0 + Quick Wins

**Created:** 2026-05-18
**Status:** In Progress

**References:**
- `docs/VITRIOL_MASTER_PLAN.md` — Full optimization catalog
- `docs/CONFIG_REFERENCE.md` — Flag documentation
- `docs/TEST_REPORT_2026-05-17.md` — Previous test results
- `docs/PHASE3_OPTIMIZATIONS.md` — Phase 3 prior art
- `libvitriol/vitriol_shim.py` — Shim (for small bugfixes)

## Steps

### Step 1: Test TQ1_0 model load (10 sec)
```bash
./vitriol run --dry-run -m /home/randozart/Desktop/Projects/qwen3.6-35b-a3b-instruct-TQ1_0.gguf
```

### Step 2: Set parallel to 1 slot (10 sec)
Edit `~/.vitriol/config`: `parallel = 1`

### Step 3A: TQ1_0 with native CUDA (if load succeeds)
```bash
vitriol serve --detach -m /path/to/TQ1_0.gguf --engine-mode native --kv-quant q4_0 -c 24576
```

### Step 3B: TQ1_0 with CPU MoE fallback (if CUDA kernels missing)
```bash
vitriol serve --detach -m /path/to/TQ1_0.gguf --engine-mode native --kv-quant q4_0 --cpu-moe -t 2
```

### Step 4: Fix shim bugs (if memory mode needed later)
- Request context crash in `_store_turn`
- Tool call format normalization (missing `type: "function"`)

### Step 5: Test with OpenCode directly (no shim)

---

## Results

| Step | Status | Notes |
|------|--------|-------|
| 1 | ⏳ | |
| 2 | ⏳ | |
| 3 | ⏳ | |
| 4 | ⏳ | |
| 5 | ⏳ | |
