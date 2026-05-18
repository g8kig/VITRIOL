# Sprint B: Switch to vitriol-tq — Max Speed, Max Context

**Created:** 2026-05-18
**Status:** In Progress

**Goal:** Replace the `vitriol` branch with `vitriol-tq` (which has all fixes + TQ CUDA kernels + TurboQuant KV), maximize LRU cache to fill free VRAM, test Q4_0 KV for 96K context, test TQ3_0 KV for 254K+ context.

**References:**
- `docs/VITRIOL_MASTER_PLAN.md`
- `docs/SPRINT_A.md`
- `vitriol-tq` branch on Randozart/llama.cpp
- `~/.config/opencode/opencode.jsonc`

---

## Steps

### Step 1: Replace `vitriol` with `vitriol-tq` on the fork
Push `vitriol-tq` branch to `vitriol` on Randozart/llama.cpp.

### Step 2: Update submodule & rebuild
- Commit the submodule pointer update
- Build llama-server with CUDA

### Step 3: Run `vitriol setup`
Re-grant CAP_IPC_LOCK for the new binary.

### Step 4: Test Q2_K_XL with 6 GB LRU + Q4_0 KV
Benchmark speed and context.

### Step 5: Test TQ1_0 with 6 GB LRU
Check if speed improves from CPU-only 2.0 tok/s.

### Step 6: Test TurboQuant KV (`tq3_0`)
If available, benchmark 254K+ context.

### Step 7: Update OpenCode config
- Set `context: 200000`
- Set `output: 32768`
- Re-enable auto compaction

---

## Results

| Step | Status | Result |
|------|--------|--------|
| 1 | ⏳ | |
| 2 | ⏳ | |
| 3 | ⏳ | |
| 4 | ⏳ | |
| 5 | ⏳ | |
| 6 | ⏳ | |
| 7 | ⏳ | |
