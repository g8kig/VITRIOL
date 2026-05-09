#!/bin/bash
# launch_vitriol.sh - Simple launch script for VITRIOL stack

set -e

echo "=== VITRIOL Stack Launch ==="
echo ""

# Kill any existing instances
pkill -9 -f koboldcpp 2>/dev/null || true
pkill -9 -f vitriol_shim.py 2>/dev/null || true
sleep 2

# Start KoboldCPP with stable settings
echo "1. Starting KoboldCPP on port 5001..."
cd ~/Downloads/koboldCPP
nohup ./koboldcpp \
    --model Qwen_Qwen3.5-9B-Q4_K_M.gguf \
    --usecuda \
    --gpulayers 25 \
    --contextsize 4096 \
    --lowvram \
    --usemmap \
    --port 5001 > /tmp/kobold.log 2>&1 &
KOBOLD_PID=$!
echo "   KoboldCPP started (PID: $KOBOLD_PID)"

# Wait for Kobold to load
echo "   Waiting for model to load (30-60 seconds)..."
for i in {1..30}; do
    if curl -s http://localhost:5001/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":5}' > /dev/null 2>&1; then
        echo "   ✓ KoboldCPP is ready!"
        break
    fi
    sleep 2
done

# Check if Kobold is running
if ! curl -s http://localhost:5001/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":5}' > /dev/null 2>&1; then
    echo "   ✗ KoboldCPP failed to start. Check /tmp/kobold.log"
    tail -20 /tmp/kobold.log
    exit 1
fi

# Start VITRIOL Shim
echo ""
echo "2. Starting VITRIOL Context Rectifier on port 5010..."
cd ~/Desktop/Projects/linux-pipe-module
nohup python3 vitriol_shim.py > /tmp/vitriol_shim.log 2>&1 &
SHIM_PID=$!
echo "   VITRIOL shim started (PID: $SHIM_PID)"
sleep 3

# Test VITRIOL
if curl -s http://localhost:5010/health > /dev/null 2>&1; then
    echo "   ✓ VITRIOL shim is ready!"
else
    echo "   ✗ VITRIOL shim failed to start. Check /tmp/vitriol_shim.log"
    exit 1
fi

echo ""
echo "=== VITRIOL Stack Ready ==="
echo ""
echo "Services:"
echo "  KoboldCPP: http://localhost:5001 (PID: $KOBOLD_PID)"
echo "  VITRIOL:   http://localhost:5010 (PID: $SHIM_PID)"
echo ""
echo "Test commands:"
echo "  curl http://localhost:5010/health"
echo "  python3 test_shim.py"
echo ""
echo "Point OpenCode to: http://localhost:5010/v1/chat/completions"
echo ""
echo "To stop: pkill -f koboldcpp; pkill -f vitriol_shim.py"
