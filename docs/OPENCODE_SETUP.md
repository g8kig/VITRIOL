# Using VITRIOL with OpenCode — Setup Guide

This guide walks through configuring VITRIOL as a local inference provider for
[OpenCode](https://opencode.ai), including why `vitriol setup` is required, how
to verify everything works, and which flags to use for different workflows.

---

## 1. Install Prerequisites

```bash
# VITRIOL itself (you already have this repo)
cd /path/to/vitriol

# Build llama.cpp with CUDA support
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
cd ..

# Python deps for memory-mode shim
pip install flask requests  # always needed
pip install sentence-transformers numpy  # optional, for --semantic-mode on
```

---

## 2. Why `vitriol setup` Must Run Once

VITRIOL's RAM Shot mode stores expert weights in **page-locked host memory** so
the GPU can read them over PCIe DMA. This requires three kernel operations:

1. `mmap` — reserve 10 GB of virtual address space
2. `mlock` — pin the pages to RAM so they are never swapped to disk
3. `cudaHostRegister` — notify the CUDA driver that these pages are DMA-eligible

Step 2 (`mlock`) requires the `CAP_IPC_LOCK` capability. Without it, `mlock`
fails silently and the pages may be swapped out during inference, causing page
faults that stall the GPU for **seconds** at a time (dropping throughput from
6 tok/s to <1 tok/s).

`vitriol setup` grants this capability to the llama-server binary:

```bash
./vitriol setup
# → Setting CAP_IPC_LOCK on .../llama-server
# → ✓ Done
```

This is a **one-time operation** — the capability persists across reboots and
binary updates, unless the binary is replaced by a different file. It requires
`sudo` because capability setting is a privileged operation.

**What happens if you skip it:** The server starts, model loads, inference runs
— but every few tokens a page fault stalls the GPU for 500-2000ms. Average
throughput drops from ~6 tok/s to <1 tok/s. The log will show no errors because
page faults are silent at the application level.

---

## 3. OpenCode Provider Configuration

Edit `~/.config/opencode/opencode.jsonc` (or your project's `opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "VITRIOL",
      "options": {
        "baseURL": "http://127.0.0.1:8279/v1",
        "apiKey": "VITRIOL-LOCAL",
        "headers": {
          "X-Project-Id": "vitriol-project",
          "X-Session-Id": "${workspaceRootHash}"
        }
      },
      "models": {
        "secret-stone": {
          "name": "Lapis Occultus",
          "limit": {
            "context": 16384,
            "output": 8192
          }
        }
      }
    }
  }
}
```

**Key details:**

| Field | Value | Why |
|-------|-------|-----|
| `baseURL` | `http://127.0.0.1:8279/v1` | VITRIOL's OpenAI-compatible endpoint |
| `apiKey` | Any non-empty string | llama-server ignores auth locally; a value is required by the OpenAI SDK |
| `headers.X-Project-Id` | Your project name | Routes memory DB to a per-project SQLite file (only used in `--memory-mode on`) |
| `headers.X-Session-Id` | `${workspaceRootHash}` | Groups conversation turns into a session (only used in `--memory-mode on`) |
| `model id` | Any string | Must match the model ID exposed by the server — you can set it to anything |
| `limit.context` | ≤ server's `-c` value | OpenCode's compaction triggers at 50% of this value. 16384 is safe with `--kv-mode offload` |
| `limit.output` | ≤ server's `-n` default | Maximum tokens per response |

---

## 4. Starting the Server

### Basic (max throughput, no memory)

```bash
vitriol serve --detach --kv-mode offload
# 6.21 tok/s baseline, KV cache in host RAM
```

### With persistent memory (recommended for multi-session projects)

```bash
vitriol serve --detach --memory-mode on --kv-mode offload --frozen-prompt on
# 5.03 tok/s, cross-session recall, prefill caching
```

### Everything enabled

```bash
VITRIOL_PREDICTIVE_PREFETCH=1 vitriol serve --detach \
  --memory-mode on --kv-mode offload --frozen-prompt on --semantic-mode on
# ~4.5-5.0 tok/s, max capability
```

The model takes ~60 seconds to load on first start. The server prints progress
to `~/.vitriol/server.log` (or `llama-server.log` in memory mode).

---

## 5. Workflow

### Starting work

```bash
# Terminal 1: start the server
vitriol serve --detach --memory-mode on --kv-mode offload

# Terminal 2: start OpenCode
cd /path/to/your/project
opencode
```

### Stopping work

```bash
vitriol stop
```

Kills the server and shim processes. The memory DB persists in
`~/.vitriol/<project-id>/memory.db` automatically — next session picks up where
you left off.

### Switching projects

VITRIOL uses `X-Project-Id` to isolate memory databases per project. If your
OpenCode config uses `${workspaceRootHash}` as the session ID (as shown above),
each git repository automatically gets its own memory namespace with no
configuration changes.

---

## 6. Verifying It Works

### Check the server is running

```bash
curl http://127.0.0.1:8279/health
# → {"status":"ok","shim":"running","memory_mode":true,...}
```

### Send a test request

```bash
curl http://127.0.0.1:8279/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Project-Id: test" \
  -H "X-Session-Id: test" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":300}'
```

### Check memory stats (if memory mode is on)

```bash
curl http://127.0.0.1:8279/memory/stats \
  -H "X-Project-Id: your-project" \
  -H "X-Session-Id: your-session"
```

### Verify CAP_IPC_LOCK

```bash
getcap /path/to/vitriol/llama.cpp/build/bin/llama-server
# → llama-server = cap_ipc_lock+ep
```

If empty, re-run `vitriol setup`.

---

## 7. Recommended Configs by Workflow

| Workload | Command | tok/s | Context | Memory |
|----------|---------|-------|---------|--------|
| Quick coding session | `vitriol serve --detach` | 6.21 | 3-4K (VRAM) | No |
| Long session, one project | `vitriol serve --detach --kv-mode offload` | 5.80 | 20K+ | No |
| Multi-session project | `vitriol serve --detach --memory-mode on` | 5.03 | 3-4K (VRAM) | Yes |
| Multi-session + long context | `vitriol serve --detach --memory-mode on --kv-mode offload` | ~4.8 | 20K+ | Yes |
| Everything | All flags + `VITRIOL_PREDICTIVE_PREFETCH=1` | ~4.5 | 20K+ | Yes |

For detailed trade-offs between every flag, see [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md).

For measured test results, see [`TEST_REPORT_2026-05-17.md`](TEST_REPORT_2026-05-17.md).

---

*Last updated: 2026-05-17*
