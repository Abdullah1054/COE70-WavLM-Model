import io
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModel
from pydub import AudioSegment

# =========================
# CONFIG — update if needed
# =========================
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path(
    os.getenv(
        "CHECKPOINT_PATH",
        BASE_DIR / "checkpoints" / "best.pt"
    )
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Model definition (matches your training)
# =========================
class WavLMFineTuner(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int = 8, dropout: float = 0.1):
        super().__init__()
        # Force safetensors to avoid the torch.load security restriction
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden = self.encoder.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_classes),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, 1, T) -> (B, T)
        x = wav.squeeze(1)
        out = self.encoder(input_values=x)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.head(pooled)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


# =========================
# Helpers
# =========================
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def load_checkpoint() -> Dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found at: {CHECKPOINT_PATH.resolve()}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = ckpt.get("config", {})
    return {"ckpt": ckpt, "cfg": cfg}

def decode_audio(file_bytes: bytes):
    # 1) Try soundfile first (works for WAV/FLAC/etc.)
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
        return mono, sr
    except Exception:
        pass

    # 2) Fallback: use ffmpeg via pydub (works for webm/opus)
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_channels(1)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # normalize int PCM to float
    samples /= (np.iinfo(np.int16).max + 1e-6) if audio.sample_width == 2 else (np.max(np.abs(samples)) + 1e-6)
    return samples, sr

def resample_to_16k(wav: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr == target_sr:
        return wav
    wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, T)
    wav_rs = torchaudio.transforms.Resample(sr, target_sr)(wav_t).squeeze(0)
    return wav_rs.numpy()

def pad_or_trim(wav: np.ndarray, target_len: int) -> np.ndarray:
    if len(wav) > target_len:
        return wav[:target_len]
    if len(wav) < target_len:
        pad = np.zeros(target_len - len(wav), dtype=np.float32)
        return np.concatenate([wav, pad], axis=0)
    return wav

def normalize(wav: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(wav)) + 1e-6
    return (wav / m).astype(np.float32)


# =========================
# App
# =========================
app = FastAPI(title="SER API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev. lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"model": None, "cfg": None}

@app.on_event("startup")
def startup():
    info = load_checkpoint()
    ckpt = info["ckpt"]
    cfg = info["cfg"]

    model_name = cfg.get("wavlm_model_name", "microsoft/wavlm-base")
    dropout = float(cfg.get("wavlm_dropout", 0.1))
    max_seconds = float(cfg.get("max_seconds", 5.0))
    sample_rate = int(cfg.get("sample_rate", 16000))

    model = WavLMFineTuner(model_name=model_name, num_classes=8, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    STATE["model"] = model
    STATE["cfg"] = {"max_seconds": max_seconds, "sample_rate": sample_rate}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": STATE["model"] is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()

    wav, sr = decode_audio(file_bytes)
    wav = resample_to_16k(wav, sr, target_sr=STATE["cfg"]["sample_rate"])
    wav = normalize(wav)

    target_len = int(STATE["cfg"]["sample_rate"] * STATE["cfg"]["max_seconds"])
    wav = pad_or_trim(wav, target_len)

    x = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,T)

    with torch.no_grad():
        logits = STATE["model"](x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()

    # Build response
    pairs = list(zip(LABELS, probs))
    pairs.sort(key=lambda p: p[1], reverse=True)

    top_label, top_prob = pairs[0]
    return {
        "topEmotion": top_label,
        "confidence": float(top_prob),
        "probabilities": [{"label": k, "prob": float(v)} for k, v in pairs]
    }