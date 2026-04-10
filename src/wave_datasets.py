from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

# Same mapping as your other dataset file
EMOTION_ID_TO_NAME = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}
EMOTION_ID_TO_INDEX = {k: i for i, k in enumerate(sorted(EMOTION_ID_TO_NAME.keys()))}
INDEX_TO_NAME = {v: EMOTION_ID_TO_NAME[k] for k, v in EMOTION_ID_TO_INDEX.items()}


def filename_to_emotion_id(filename: str) -> Optional[int]:
    # RAVDESS: 03-01-05-02-... => third field is emotion id
    name = Path(filename).name
    parts = name.split("-")
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except Exception:
        return None


def list_wavs(data_root: str) -> List[str]:
    root = Path(data_root)
    return sorted([str(p) for p in root.rglob("*.wav")])


def pad_or_trim_1d(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    # wav shape: (1, T)
    T = wav.shape[1]
    if T == target_len:
        return wav
    if T > target_len:
        return wav[:, :target_len]
    pad_amt = target_len - T
    return torch.nn.functional.pad(wav, (0, pad_amt), mode="constant", value=0.0)


class RAVDESSWaveDataset(Dataset):
    """
    Returns (waveform, label)
    waveform: (1, target_len) float32
    """
    def __init__(self, files: List[str], sample_rate: int = 16000, max_seconds: float = 5.0):
        self.files = files
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * max_seconds)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.files[idx]

        eid = filename_to_emotion_id(path)
        if eid is None or eid not in EMOTION_ID_TO_INDEX:
            raise ValueError(f"Could not parse emotion id from: {path}")
        y = EMOTION_ID_TO_INDEX[eid]

        # IMPORTANT: backend="soundfile" fixes Windows backend issues
        wav, sr = torchaudio.load(path, backend="soundfile")  # (C, T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono

        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)

        # normalize waveform a bit (simple)
        wav = wav / (wav.abs().max() + 1e-6)

        # fixed-length for batching
        wav = pad_or_trim_1d(wav, self.target_len)

        return wav, y