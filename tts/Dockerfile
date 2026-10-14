# Qwen3-TTS voice-clone server for DGX Spark (sm_121, CUDA 13).
#
# Built FROM the local eugr vLLM image on purpose: it already carries torch/torchaudio 2.11
# built for cu130/aarch64, which PyPI does not ship. That makes this a ~1 GB layer instead of
# a fresh 20-35 GB image on a disk that was 96% full.
#
# qwen-tts pins transformers==4.57.3 (it breaks on >=5), which conflicts with the vLLM in the
# base image. That is fine: this container never imports vLLM. gradio is a qwen-tts dep only
# for its demo UI, so it is skipped via --no-deps.
FROM eugr/spark-vllm:latest

RUN apt-get update && apt-get install -y --no-install-recommends sox ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --no-deps qwen-tts==0.1.1 \
    && pip install --no-cache-dir transformers==4.57.3 accelerate==1.12.0 \
       librosa soundfile onnxruntime sox python-multipart

COPY server.py /app/server.py
WORKDIR /app
ENTRYPOINT ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0"]
CMD ["--port", "8020"]
