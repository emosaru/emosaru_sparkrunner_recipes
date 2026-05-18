#!/usr/bin/env bash
# Launch Qwen3.6-27B NVFP4-MTP across 2 Spark nodes (TP=2) without sparkrun's
# HF-distribution step, which doesn't support local model paths.
# The model directory is bind-mounted into each container at /model.
#
# Usage:
#   ./scripts/launch-nvfp4-cluster.sh [start|stop|logs|status]
set -euo pipefail

MODEL_DIR="/home/mebell/code/emosaru_sparkrunner_recipes/quantize/output/qwen36-27b-nvfp4-mtp"
PATCHES_DIR="/home/mebell/code/emosaru_sparkrunner_recipes/patches"
IMAGE="sparkrun-eugr-vllm:latest"
CONTAINER="nvfp4-cluster"

NODE1="192.168.1.236"
NODE2="192.168.1.237"
MASTER_ADDR="$NODE1"
MASTER_PORT="25000"
SERVE_PORT="8003"

HF_CACHE="$HOME/.cache/huggingface"

VLLM_MODELS_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

COMMON_DOCKER_FLAGS=(
    --privileged
    --gpus all
    --ipc host
    --shm-size 25gb
    --network host
    -v "${HF_CACHE}:/cache/huggingface"
    -v "${MODEL_DIR}:/model:ro"
    -v "${PATCHES_DIR}/qwen3_5_mtp.py:${VLLM_MODELS_DIR}/qwen3_5_mtp.py:ro"
    -e ENABLE_NVFP4_SM100=0
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1
    -e NCCL_CUMEM_ENABLE=1
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0
    -e OMP_NUM_THREADS=4
    -e HF_HOME=/cache/huggingface
)

# Array form avoids all single-quote/JSON quoting issues for the local docker run.
VLLM_BASE_ARGS=(
    vllm serve /model
    --host 0.0.0.0
    --port "${SERVE_PORT}"
    --max-model-len 262144
    --max-num-batched-tokens 32768
    --trust-remote-code
    --quantization modelopt
    --gpu-memory-utilization 0.32
    --kv-cache-dtype fp8
    --load-format safetensors
    --attention-backend flashinfer
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
    --reasoning-parser qwen3
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    -tp 2
    --nnodes 2
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
)

start() {
    echo "==> Syncing patches to node 2..."
    ssh "${NODE2}" "mkdir -p ${PATCHES_DIR}"
    rsync -q "${PATCHES_DIR}/qwen3_5_mtp.py" "${NODE2}:${PATCHES_DIR}/qwen3_5_mtp.py"

    echo "==> Stopping any existing ${CONTAINER} containers on both nodes..."
    docker rm -f "${CONTAINER}" 2>/dev/null || true
    ssh "${NODE2}" "docker rm -f '${CONTAINER}' 2>/dev/null || true"

    echo "==> Starting worker (node 2: ${NODE2}) ..."
    # Pipe a heredoc over SSH so single quotes and JSON are not re-interpreted locally.
    ssh "${NODE2}" bash -s << EOF
docker run -d --name ${CONTAINER} \
    --privileged --gpus all --ipc host --shm-size 25gb --network host \
    -v ${HF_CACHE}:/cache/huggingface \
    -v ${MODEL_DIR}:/model:ro \
    -v ${PATCHES_DIR}/qwen3_5_mtp.py:${VLLM_MODELS_DIR}/qwen3_5_mtp.py:ro \
    -e ENABLE_NVFP4_SM100=0 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -e NCCL_CUMEM_ENABLE=1 \
    -e NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0 \
    -e OMP_NUM_THREADS=4 \
    -e HF_HOME=/cache/huggingface \
    ${IMAGE} \
    vllm serve /model \
    --host 0.0.0.0 --port ${SERVE_PORT} \
    --max-model-len 262144 --max-num-batched-tokens 32768 \
    --trust-remote-code --quantization modelopt \
    --gpu-memory-utilization 0.32 --kv-cache-dtype fp8 \
    --load-format safetensors --attention-backend flashinfer \
    --enable-prefix-caching --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    -tp 2 --nnodes 2 \
    --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
    --node-rank 1 --headless
EOF

    echo "==> Starting head (node 1: ${NODE1}) ..."
    docker run -d --name "${CONTAINER}" \
        "${COMMON_DOCKER_FLAGS[@]}" \
        "${IMAGE}" \
        "${VLLM_BASE_ARGS[@]}" --node-rank 0

    echo ""
    echo "Containers launched. Watch logs:"
    echo "  Head:   docker logs -f ${CONTAINER}"
    echo "  Worker: ssh ${NODE2} 'docker logs -f ${CONTAINER}'"
    echo ""
    echo "Wait for 'Uvicorn running on http://0.0.0.0:${SERVE_PORT}' in head logs."
}

stop() {
    echo "==> Stopping ${CONTAINER} on both nodes..."
    docker rm -f "${CONTAINER}" 2>/dev/null && echo "  node1: stopped" || echo "  node1: not running"
    ssh "${NODE2}" "docker rm -f '${CONTAINER}' 2>/dev/null && echo '  node2: stopped' || echo '  node2: not running'"
}

logs() {
    echo "=== Head (${NODE1}) ==="
    docker logs "${CONTAINER}" 2>&1 | tail -50
    echo ""
    echo "=== Worker (${NODE2}) ==="
    ssh "${NODE2}" "docker logs '${CONTAINER}' 2>&1 | tail -20"
}

status() {
    echo "=== Head (${NODE1}) ==="
    docker ps --filter "name=${CONTAINER}" --format "{{.Status}}" 2>/dev/null || echo "not running"
    echo "=== Worker (${NODE2}) ==="
    ssh "${NODE2}" "docker ps --filter 'name=${CONTAINER}' --format '{{.Status}}' 2>/dev/null || echo 'not running'"
}

case "${1:-start}" in
    start)  start ;;
    stop)   stop ;;
    logs)   logs ;;
    status) status ;;
    *)      echo "Usage: $0 [start|stop|logs|status]"; exit 1 ;;
esac
