#!/usr/bin/env bash
# Launch the Qwen3-TTS voice-clone server on the worker node (.237), port 8020.
#
# PREREQUISITE: Flash-Next must be running at GMU <= 0.77 (the recipe default since
# 2026-10-14). At 0.80 the worker has only ~12 GiB available and the 1.7B model peaks at
# ~7.5 GiB, which lands next to earlyoom's ~2.5 GB floor — and earlyoom prefers killing
# vLLM, i.e. the TP worker, which takes down the whole :8000 endpoint.
#
#   tts/launch-qwen3-tts.sh             # build if needed, start, wait for /health
#   REBUILD=1 tts/launch-qwen3-tts.sh   # rebuild the image after editing server.py
#   docker -H ssh://192.168.1.237 rm -f qwen3-tts   # stop
set -euo pipefail

NODE="${NODE:-192.168.1.237}"
PORT="${PORT:-8020}"
MODEL="${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-11}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

avail_gib() { ssh "$NODE" "awk '/MemAvailable/ {printf \"%d\", \$2/1048576}' /proc/meminfo"; }

AVAIL=$(avail_gib)
if (( AVAIL < MIN_AVAIL_GIB )); then
  echo "✗ $NODE has ${AVAIL} GiB available; need >= ${MIN_AVAIL_GIB} (TTS peak ~7.5 + earlyoom floor + margin)." >&2
  echo "  Is Flash-Next still on GMU 0.80? Relaunch it at 0.77 first. Override: MIN_AVAIL_GIB=..." >&2
  exit 1
fi

if [[ -n "${REBUILD:-}" ]] || ! ssh "$NODE" docker image inspect qwen3-tts:latest >/dev/null 2>&1; then
  ssh "$NODE" mkdir -p qwen3-tts/voices
  scp -q "$DIR/Dockerfile" "$DIR/server.py" "$NODE:qwen3-tts/"
  ssh "$NODE" "cd qwen3-tts && docker build -q -t qwen3-tts:latest ."
fi

# qwen-tts calls the Hub API even for a cached repo id, which fails under HF_HUB_OFFLINE.
# Hand it the local snapshot directory instead, so the server never needs the network.
SNAP=$(ssh "$NODE" "ls -d \$HOME/.cache/huggingface/hub/models--${MODEL//\//--}/snapshots/* 2>/dev/null | head -1")
[[ -n "$SNAP" ]] || { echo "✗ $MODEL not in $NODE's HF cache; download it first." >&2; exit 1; }
SNAP_IN="/root/.cache/huggingface/${SNAP#*/.cache/huggingface/}"

ssh "$NODE" "docker rm -f qwen3-tts >/dev/null 2>&1 || true; docker run -d --name qwen3-tts \
  --gpus all --ipc=host --restart unless-stopped -p $PORT:8020 \
  -e TTS_MODEL='$SNAP_IN' -e HF_HUB_OFFLINE=1 \
  -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
  -v \$HOME/qwen3-tts/voices:/voices \
  qwen3-tts:latest" >/dev/null

echo "▸ waiting for http://$NODE:$PORT/health ..."
for _ in $(seq 1 60); do
  if curl -fsS -m 5 "http://$NODE:$PORT/health" 2>/dev/null; then
    echo; echo "✓ up. $NODE now has $(avail_gib) GiB available."; exit 0
  fi
  sleep 5
done
echo "✗ not healthy after 5 min:" >&2
ssh "$NODE" docker logs --tail 40 qwen3-tts >&2
exit 1
