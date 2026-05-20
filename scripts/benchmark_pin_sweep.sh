#!/bin/bash
# Benchmark: Expert Pinning sweep — focused
set -e
MODEL="/home/randozart/Downloads/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="${PROJECT_DIR}/llama.cpp"
BENCH="${LLAMA_DIR}/build/bin/llama-bench"
LD_PATH="${LLAMA_DIR}/build/bin"

echo "═══ Expert Pinning Sweep ═══"
echo "Model: Qwen3.6-35B-A3B IQ2_M"
echo "Date: $(date)"
echo ""

run_bench() {
    local label="$1" pin="$2"
    shift 2
    echo "─── $label ───"
    env VITRIOL_MODE=stream \
        VITRIOL_PIN_FIRST_N_LAYERS="$pin" \
        VITRIOL_LRU_MB=0 \
        VITRIOL_OUTPUT_CACHE=0 \
        "$@" \
        LD_LIBRARY_PATH="$LD_PATH" \
        "$BENCH" -m "$MODEL" -p 64 -n 100 -ngl 99 -t 4 -r 3 2>&1
    echo ""
}

# Pin-only sweep
run_bench "Baseline (pin=0)" 0
run_bench "Pin 5 layers" 5
run_bench "Pin 10 layers" 10
run_bench "Pin 15 layers" 15
run_bench "Pin 20 layers" 20

# Pinning + predictive prefetch
run_bench "Pin 15 + prefetch" 15 VITRIOL_PREDICTIVE_PREFETCH=1
run_bench "Pin 20 + prefetch" 20 VITRIOL_PREDICTIVE_PREFETCH=1

echo "═══ Done ═══"
