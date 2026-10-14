"""Qwen3-TTS voice-clone server (OpenAI-style /v1/audio/speech).

Voices are registered once from a reference clip and cached as clone prompts, so each
request only pays for generation, not for re-encoding the reference audio.

  POST /v1/voices          multipart: name, audio (wav/mp3/flac), ref_text (transcript; optional
                           but strongly recommended: without it only the speaker embedding is used)
  GET  /v1/voices
  DELETE /v1/voices/{name}
  POST /v1/audio/speech    JSON: {"input": "...", "voice": "<name>", "language": "English",
                           "response_format": "wav"|"flac"|"mp3"|"pcm"}
  GET  /health

Voices persist as <name>.wav + <name>.txt under $VOICES_DIR (mount it as a volume).
"""
import io
import logging
import os
import re
import subprocess
import threading

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

MODEL_ID = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICES_DIR = os.environ.get("VOICES_DIR", "/voices")
DEFAULT_LANGUAGE = os.environ.get("TTS_DEFAULT_LANGUAGE", "English")
MAX_INPUT_CHARS = int(os.environ.get("TTS_MAX_INPUT_CHARS", "4000"))

log = logging.getLogger("tts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

os.makedirs(VOICES_DIR, exist_ok=True)
model = Qwen3TTSModel.from_pretrained(
    MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
)
log.info("loaded %s; languages=%s", MODEL_ID, model.get_supported_languages())

# One GPU shared with a TP rank of the main LLM: serialise generation so concurrent
# requests cannot stack activation peaks on top of each other.
gen_lock = threading.Lock()
prompts: dict = {}
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _load_voice(name: str) -> None:
    wav = os.path.join(VOICES_DIR, f"{name}.wav")
    txt = os.path.join(VOICES_DIR, f"{name}.txt")
    ref_text = open(txt).read().strip() if os.path.exists(txt) else None
    with gen_lock:
        prompts[name] = model.create_voice_clone_prompt(
            ref_audio=wav, ref_text=ref_text, x_vector_only_mode=ref_text is None
        )
    log.info("voice %r ready (%s)", name, "ICL" if ref_text else "x-vector only")


for f in sorted(os.listdir(VOICES_DIR)):
    if f.endswith(".wav"):
        try:
            _load_voice(f[:-4])
        except Exception:
            log.exception("failed to load voice %s", f)

app = FastAPI(title="Qwen3-TTS")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "voices": sorted(prompts)}


@app.get("/v1/voices")
def list_voices():
    return {"voices": sorted(prompts)}


@app.post("/v1/voices")
async def add_voice(name: str = Form(...), audio: UploadFile = File(...), ref_text: str = Form(None)):
    if not NAME_RE.match(name):
        raise HTTPException(400, "name must match [A-Za-z0-9_-]{1,64}")
    try:
        data, sr = sf.read(io.BytesIO(await audio.read()), always_2d=False)
    except Exception as e:
        raise HTTPException(400, f"could not decode audio (wav/flac/ogg): {e}")
    sf.write(os.path.join(VOICES_DIR, f"{name}.wav"), data, sr)
    txt = os.path.join(VOICES_DIR, f"{name}.txt")
    if ref_text and ref_text.strip():
        open(txt, "w").write(ref_text.strip())
    elif os.path.exists(txt):
        os.remove(txt)
    _load_voice(name)
    return {"voice": name, "seconds": round(len(data) / sr, 2), "mode": "icl" if ref_text else "x_vector_only"}


@app.delete("/v1/voices/{name}")
def delete_voice(name: str):
    if name not in prompts:
        raise HTTPException(404, f"unknown voice {name!r}")
    prompts.pop(name)
    for ext in ("wav", "txt"):
        p = os.path.join(VOICES_DIR, f"{name}.{ext}")
        if os.path.exists(p):
            os.remove(p)
    return {"deleted": name}


class SpeechRequest(BaseModel):
    input: str
    voice: str
    model: str | None = None  # accepted for OpenAI-client compatibility, ignored
    language: str | None = None
    response_format: str = "wav"


def _encode(wav: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    if fmt == "pcm":
        return (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes(), "audio/pcm"
    if fmt in ("wav", "flac"):
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format=fmt.upper())
        return buf.getvalue(), f"audio/{fmt}"
    if fmt == "mp3":
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        out = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "-b:a", "192k", "pipe:1"],
            input=buf.getvalue(), capture_output=True, check=True,
        )
        return out.stdout, "audio/mpeg"
    raise HTTPException(400, "response_format must be wav, flac, mp3 or pcm")


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if req.voice not in prompts:
        raise HTTPException(404, f"unknown voice {req.voice!r}; register it via POST /v1/voices")
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"input exceeds {MAX_INPUT_CHARS} chars; split it")
    with gen_lock:
        wavs, sr = model.generate_voice_clone(
            text=req.input,
            language=req.language or DEFAULT_LANGUAGE,
            voice_clone_prompt=prompts[req.voice],
        )
    body, media = _encode(wavs[0], sr, req.response_format)
    return Response(content=body, media_type=media)
