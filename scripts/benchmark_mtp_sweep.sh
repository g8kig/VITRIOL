#!/bin/bash
# Benchmark MTP draft-n-max sweep
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VITRIOL="${SCRIPT_DIR}/vitriol"
CONFIG_DIR="${HOME}/.vitriol"
IQ2M_MODEL="/home/randozart/Downloads/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
HOST="0.0.0.0"
PORT=8279

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_server() {
    for i in $(seq 1 60); do
        if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    log "ERROR: Server did not start in 120 seconds"
    return 1
}

bench_at_n() {
    local n="$1"
    curl -s \
        "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-Project-Id: benchmark" \
        -H "X-Session-Id: mtp-sweep" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a short story about a robot learning to paint.\"}],\"max_tokens\":$n}" \
        >/dev/null
    sleep 1.5
}

get_line_count() { wc -l < "$1" 2>/dev/null || echo 0; }

extract_from_log() {
    local logfile="$1" after_line="$2"
    local gen="" prompt_eval="" draft_rate="" draft_accepted="" draft_total=""

    local section
    section=$(sed -n "${after_line},\$p" "$logfile" 2>/dev/null)

    # prompt eval speed (tok/s)
    local prompt_line
    prompt_line=$(echo "$section" | grep -m1 "^prompt eval time =")
    if [[ -n "$prompt_line" ]]; then
        prompt_eval=$(echo "$prompt_line" | grep -oP '[\d.]+(?= tokens per second)')
    fi

    # generation eval speed (tok/s) — line starts with spaces then "eval time"
    local eval_line
    eval_line=$(echo "$section" | grep -m1 "^       eval time =")
    if [[ -n "$eval_line" ]]; then
        gen=$(echo "$eval_line" | grep -oP '[\d.]+(?= tokens per second)')
    fi

    # draft acceptance
    local draft_line
    draft_line=$(echo "$section" | grep -m1 "draft acceptance rate")
    if [[ -n "$draft_line" ]]; then
        draft_accepted=$(echo "$draft_line" | grep -oP '\d+(?= accepted)')
        draft_total=$(echo "$draft_line" | grep -oP '\d+(?= generated)')
        draft_rate=$(echo "$draft_line" | grep -oP '[\d.]+(?= \()')
    fi

    echo "$gen|$prompt_eval|$draft_rate|$draft_accepted|$draft_total"
}

N_VALUES="2 3 4 6 8 12"
RESULTS_FILE="/tmp/mtp_sweep_results.txt"

echo ""
echo "============================================="
echo "  MTP Draft-N-Max Sweep"
echo "  Date: $(date -I)"
echo "  Model: $(basename "$IQ2M_MODEL")"
echo "============================================="
echo ""

printf "%-6s | %-8s | %-8s | %-12s | %-8s | %-8s | %-12s\n" \
    "DraftN" "N50 gen" "N50 p.eval" "N50 accept" "N100 gen" "N100 p.eval" "N100 accept"
printf "%-6s-+-%-8s-+-%-8s-+-%-12s-+-%-8s-+-%-8s-+-%-12s\n" \
    "------" "--------" "--------" "------------" "--------" "--------" "------------"

echo "# MTP Sweep Results - $(date -Iseconds)" > "$RESULTS_FILE"
echo "# draft_n_max | n=50 gen | n=50 prompt_eval | n=50 accept/total | n=100 gen | n=100 prompt_eval | n=100 accept/total" >> "$RESULTS_FILE"

for N in $N_VALUES; do
    echo ""
    log "=== N = $N ==="

    log "Stopping existing server..."
    "$VITRIOL" stop 2>/dev/null || true
    sleep 2

    rm -f "${CONFIG_DIR}/server.log" "${CONFIG_DIR}/llama-server.log"

    log "Starting server..."
    "$VITRIOL" serve --detach \
        -m "$IQ2M_MODEL" \
        --spec-type mtp \
        --spec-draft-n-max "$N" \
        --kv-quant q4_0 \
        -c 256000

    wait_for_server || { log "FAILED for N=$N"; continue; }

    # Warm up
    log "Warm up (n=10)..."
    bench_at_n 10

    # Test n=50
    LC1=$(get_line_count "${CONFIG_DIR}/server.log")
    log "Benchmark n=50..."
    bench_at_n 50
    r50=$(extract_from_log "${CONFIG_DIR}/server.log" $((LC1+1)))
    IFS='|' read -r gen50 prompt50 dr50 da50 dt50 <<< "$r50"

    # Test n=100
    LC2=$(get_line_count "${CONFIG_DIR}/server.log")
    log "Benchmark n=100..."
    bench_at_n 100
    r100=$(extract_from_log "${CONFIG_DIR}/server.log" $((LC2+1)))
    IFS='|' read -r gen100 prompt100 dr100 da100 dt100 <<< "$r100"

    # Display
    if [[ -n "$da50" ]] && [[ -n "$dt50" ]] && [[ "$dt50" != "0" ]]; then
        a50pct=$(echo "scale=1; 100 * $da50 / $dt50" | bc 2>/dev/null || echo "?")
        acc50="${da50}/${dt50} (${a50pct}%)"
    else
        acc50="--/--"
    fi
    if [[ -n "$da100" ]] && [[ -n "$dt100" ]] && [[ "$dt100" != "0" ]]; then
        a100pct=$(echo "scale=1; 100 * $da100 / $dt100" | bc 2>/dev/null || echo "?")
        acc100="${da100}/${dt100} (${a100pct}%)"
    else
        acc100="--/--"
    fi

    printf "%-6s | %-8s | %-8s | %-12s | %-8s | %-8s | %-12s\n" \
        "$N" "${gen50:-err}" "${prompt50:-err}" "$acc50" \
        "${gen100:-err}" "${prompt100:-err}" "$acc100"

    echo "$N | ${gen50:-err} | ${prompt50:-err} | ${da50:-0}/${dt50:-0} | ${gen100:-err} | ${prompt100:-err} | ${da100:-0}/${dt100:-0}" >> "$RESULTS_FILE"

    log "Stopping server..."
    "$VITRIOL" stop 2>/dev/null || true
    sleep 2
done

echo ""
log "=== Sweep Complete ==="
echo ""

echo "Final Results:"
cat "$RESULTS_FILE"
