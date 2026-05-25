#!/usr/bin/env bash
# Solo test launch for Qwen3.6-35B-A3B NVFP4-MTP on node 0 (192.168.1.236).
# Tests the local checkpoint before HuggingFace upload.
#
# IMPORTANT: Needs ~66 GB unified memory (util 0.55 × 121 GB). Stop the unified
# stack (or at minimum node 0's vLLM services) before launching:
#   sparkrun stop <27B-recipe>   # or ./launch-unified-stack.sh stop
#
# The MTP patch (patches/qwen3_5_mtp.py) is bind-mounted to fix the fnmatch
# bug in fullcg's is_layer_excluded — without it the MTP weights fail to load.
#
# Usage: ./scripts/launch-nvfp4-35b-solo-test.sh [start|stop|logs|status]
set -euo pipefail

REPO_DIR="/home/mebell/code/emosaru_sparkrunner_recipes"
MODEL_DIR="${REPO_DIR}/quantize/output/qwen36-35b-a3b-nvfp4-mtp"
PATCHES_DIR="${REPO_DIR}/patches"
IMAGE="sparkrun-eugr-vllm:latest"
CONTAINER="nvfp4-35b-solo-test"
SERVE_PORT="8004"

HF_CACHE="$HOME/.cache/huggingface"
VLLM_MODELS_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

start() {
    echo "==> Stopping any existing ${CONTAINER} container..."
    docker rm -f "${CONTAINER}" 2>/dev/null || true

    echo "==> Starting 35B-A3B NVFP4-MTP solo test (port ${SERVE_PORT})..."
    docker run -d --name "${CONTAINER}" \
        --privileged \
        --gpus all \
        --ipc host \
        --shm-size 25gb \
        --network host \
        -v "${HF_CACHE}:/cache/huggingface" \
        -v "${MODEL_DIR}:/model:ro" \
        -v "${PATCHES_DIR}/qwen3_5_mtp.py:${VLLM_MODELS_DIR}/qwen3_5_mtp.py:ro" \
        -v "/home/mebell/.cache/vllm:/tmp/.cache/vllm" \
        -v "/home/mebell/.cache/flashinfer:/tmp/.cache/flashinfer" \
        -e ENABLE_NVFP4_SM100=0 \
        -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
        -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
        -e HF_HOME=/cache/huggingface \
        -e OMP_NUM_THREADS=4 \
        "${IMAGE}" \
        vllm serve /model \
            --host 0.0.0.0 \
            --port "${SERVE_PORT}" \
            --max-model-len 262144 \
            --max-num-batched-tokens 16384 \
            --trust-remote-code \
            --quantization modelopt \
            --gpu-memory-utilization 0.55 \
            --kv-cache-dtype fp8 \
            --load-format instanttensor \
            --attention-backend flashinfer \
            --enable-prefix-caching \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_xml \
            --reasoning-parser qwen3 \
            --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
            -tp 1

    echo ""
    echo "Container launched. Watch logs:"
    echo "  docker logs -f ${CONTAINER}"
    echo ""
    echo "Wait for 'Uvicorn running on http://0.0.0.0:${SERVE_PORT}' to confirm load."
    echo "Quick smoke test:"
    echo "  curl -s http://localhost:${SERVE_PORT}/v1/models | python3 -m json.tool"
}

stop() {
    docker rm -f "${CONTAINER}" 2>/dev/null && echo "Stopped ${CONTAINER}." || echo "${CONTAINER} not running."
}

logs() {
    docker logs -f "${CONTAINER}"
}

status() {
    docker ps --filter "name=${CONTAINER}" --format "{{.Status}}" 2>/dev/null || echo "not running"
}

case "${1:-start}" in
    start)  start ;;
    stop)   stop ;;
    logs)   logs ;;
    status) status ;;
    *)      echo "Usage: $0 [start|stop|logs|status]"; exit 1 ;;
esac
