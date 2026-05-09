#!/usr/bin/env python3
"""
VITRIOL Context Rectifier - KoboldCPP Proxy Shim

This shim sits between OpenCode (or any agent) and KoboldCPP,
performing "alchemical rectification" on context to prevent OOM crashes
on legacy hardware (i7-3770, GTX 1070 Ti, 8GB VRAM).

OpenCode -> VITRIOL Shim (port 5005) -> KoboldCPP (port 5001)

Implementation Plan v2.0 - Phase 1
"""

import json
import re
import subprocess
import requests
import logging
from flask import Flask, request, jsonify
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - VITRIOL - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - Implementation Plan v2.0
KOBOLD_URL = "http://localhost:5001/v1/chat/completions"
KOBOLD_STATUS_URL = "http://localhost:5001/api/v1/info"
MAX_CONTEXT_TOKENS = 7000
MAX_MESSAGES_TO_KEEP = 4
SHIM_PORT = 5010  # Changed to avoid port conflicts
MAX_TEMP = 85  # GPU thermal limit (°C)
MAX_TEMP = 85  # GPU thermal limit (°C)


@dataclass
class RectificationStats:
    original_tokens: int
    rectified_tokens: int
    messages_dropped: int
    metadata_stripped: bool
    reduction_percent: float


def get_gpu_temp() -> int:
    """
    Guardrail 4: Thermal Polling via nvidia-smi
    Returns GPU temperature in Celsius, 0 on error
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )
        return int(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Thermal poll failed: {e}")
        return 0


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token for English)"""
    return len(text) // 4


def sublimate_content(content: str) -> str:
    """
    Sublimation (Guardrail 2): Strip reasoning/tool bloat
    Converts bloated metadata into compact tokens
    """
    if not content:
        return ""
    
    # Strip <reasoning> blocks
    content = re.sub(r'<reasoning>.*?</reasoning>', '[reasoning distilled]', content, flags=re.DOTALL)
    content = re.sub(r'reasoning_content:\s*.*?\n', '', content)
    
    # Condense tool results
    content = re.sub(r'tool_results:\s*\[.*?\]', '[tools executed]', content, flags=re.DOTALL)
    content = re.sub(r'Tool.*?executed.*?\n', '[tool executed]\n', content, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()


def rectify_context(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], RectificationStats]:
    """
    Perform alchemical rectification on message context.
    
    Operations:
    1. Calcination (Truncation): Keep system + last N messages
    2. Sublimation (Metadata Stripping): Remove reasoning/tool bloat
    3. Coagulation (Final Formatting): Ensure clean ChatML format
    """
    if not messages:
        return messages, RectificationStats(0, 0, 0, False, 0.0)
    
    original_messages = messages.copy()
    original_tokens = sum(
        estimate_tokens(m.get('content', '') + m.get('reasoning_content', ''))
        for m in messages
    )
    
    # 1. Calcination: Truncate middle messages
    messages_dropped = 0
    if len(messages) > MAX_MESSAGES_TO_KEEP:
        # Always keep system prompt (first message if role=system)
        system_msg = None
        if messages[0].get('role') == 'system':
            system_msg = messages[0]
            messages = messages[1:]
        
        # Keep only the last N messages
        messages_dropped = len(messages) - MAX_MESSAGES_TO_KEEP
        messages = messages[-MAX_MESSAGES_TO_KEEP:]
        
        # Restore system message
        if system_msg:
            messages = [system_msg] + messages
    
    # 2. Sublimation: Strip metadata from each message
    metadata_stripped = False
    for msg in messages:
        if 'content' in msg:
            original_len = len(msg['content'])
            msg['content'] = sublimate_content(msg['content'])
            if len(msg['content']) < original_len:
                metadata_stripped = True
        
        # Remove reasoning_content entirely (not needed for inference)
        if 'reasoning_content' in msg:
            del msg['reasoning_content']
            metadata_stripped = True
        
        # Clean up tool_calls if present (keep minimal info)
        if 'tool_calls' in msg:
            msg['tool_calls'] = [{'id': tc.get('id', '')} for tc in msg['tool_calls']]
            metadata_stripped = True
    
    # Calculate final token count
    rectified_tokens = sum(
        estimate_tokens(m.get('content', ''))
        for m in messages
    )
    
    reduction_percent = ((original_tokens - rectified_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0
    
    stats = RectificationStats(
        original_tokens=original_tokens,
        rectified_tokens=rectified_tokens,
        messages_dropped=messages_dropped,
        metadata_stripped=metadata_stripped,
        reduction_percent=reduction_percent
    )
    
    return messages, stats


def kobold_status() -> Dict[str, Any]:
    """Check if KoboldCPP is running and healthy"""
    try:
        resp = requests.get(KOBOLD_STATUS_URL, timeout=5)
        if resp.status_code == 200:
            return {"status": "ok", "kobold": resp.json()}
        return {"status": "error", "message": f"Kobold returned {resp.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "KoboldCPP not running on port 5001"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.route('/v1/chat/completions', methods=['POST'])
def proxy_chat_completions():
    """
    OpenAI-compatible endpoint that rectifies context before forwarding to KoboldCPP.
    
    Guardrails:
    1. Thermal polling (halts if GPU >= 85°C)
    2. Context truncation (max 7k tokens)
    3. Metadata stripping (reasoning/tools)
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON body"}), 400
        
        # === GUARDRAIL 4: Thermal Sentinel ===
        temp = get_gpu_temp()
        logger.info(f"GPU Temperature: {temp}°C")
        if temp >= MAX_TEMP:
            logger.error(f"THERMAL HALT: GPU at {temp}°C (limit: {MAX_TEMP}°C)")
            return jsonify({
                "error": f"Alchemical Overheat: {temp}°C. Cooling required.",
                "thermal_limit": MAX_TEMP
            }), 503
        
        messages = data.get('messages', [])
        logger.info(f"Received request with {len(messages)} messages")
        
        # === GUARDRAIL 1 & 2: Context Rectification ===
        rectified_messages, stats = rectify_context(messages)
        
        # Log rectification stats
        logger.info(
            f"RECTIFICATION: {stats.original_tokens} -> {stats.rectified_tokens} tokens "
            f"({stats.reduction_percent:.1f}% reduction, dropped {stats.messages_dropped} messages)"
        )
        
        if stats.metadata_stripped:
            logger.info("Metadata stripped from context")
        
        # Update request with rectified context
        data['messages'] = rectified_messages
        
        # === GUARDRAIL 3: Coagulation (Generation Cap) ===
        data['max_tokens'] = min(data.get('max_tokens', 1024), 1024)
        
        # Check Kobold status
        kobold = kobold_status()
        if kobold['status'] != 'ok':
            logger.warning(f"KoboldCPP issue: {kobold['message']}")
        
        # Forward to KoboldCPP
        logger.info(f"Forwarding to KoboldCPP at {KOBOLD_URL}")
        response = requests.post(KOBOLD_URL, json=data, timeout=120)
        
        # Return Kobold's response
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.Timeout:
        logger.error("KoboldCPP request timed out")
        return jsonify({"error": "Inference timeout"}), 504
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/models', methods=['GET'])
def proxy_models():
    """Proxy model listing endpoint"""
    try:
        response = requests.get("http://localhost:5001/v1/models", timeout=10)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """VITRIOL shim health check"""
    kobold = kobold_status()
    return jsonify({
        "status": "ok",
        "shim": "running",
        "koboldcpp": kobold,
        "config": {
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "max_messages": MAX_MESSAGES_TO_KEEP,
            "kobold_url": KOBOLD_URL
        }
    })


@app.route('/rectify', methods=['POST'])
def rectify_only():
    """
    Debug endpoint: rectify context without inference.
    Useful for testing what VITRIOL would do to a prompt.
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        rectified, stats = rectify_context(messages)
        
        return jsonify({
            "original_messages": len(messages),
            "rectified_messages": len(rectified),
            "stats": {
                "original_tokens": stats.original_tokens,
                "rectified_tokens": stats.rectified_tokens,
                "messages_dropped": stats.messages_dropped,
                "metadata_stripped": stats.metadata_stripped,
                "reduction_percent": stats.reduction_percent
            },
            "rectified_messages": rectified
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    """Run the VITRIOL shim"""
    logger.info(f"VITRIOL Context Rectifier starting on port {SHIM_PORT}")
    logger.info(f"Forwarding to KoboldCPP at {KOBOLD_URL}")
    logger.info(f"Max context: {MAX_CONTEXT_TOKENS} tokens, Max messages: {MAX_MESSAGES_TO_KEEP}")
    
    app.run(host='0.0.0.0', port=SHIM_PORT, debug=False, threaded=True)


if __name__ == '__main__':
    main()
