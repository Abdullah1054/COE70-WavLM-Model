from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

# RAVDESS emotion id mapping (3rd field in filename)
# 1 neutral, 2 calm, 3 happy, 4 sad, 5 angry, 6 fearful, 7 disgust, 8 surprised
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
EMOTION_ID_TO_INDEX = {k: i for i, k in enumerate(sorted(EMOTION_ID_TO_NAME.keys()))}  # 1->0 ... 8->7
INDEX_TO_NAME = {v: EMOTION_ID_TO_NAME[k] for k, v in EMOTION_ID_TO_INDEX.items()}


def filename_to_emotion_id(filename: str) -> Optional[int]:
    """RAVDESS: 03-01-05-02-01-01-12.wav => third field is emotion ID (05)."""
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


def pad_or_trim(spec: torch.Tensor, max_frames: int) -> torch.Tensor:
    """spec: (n_mels, time) -> (n_mels, max_frames)"""
    n_mels, t = spec.shape
    if t == max_frames:
        return spec
    if t > max_frames:
        return spec[:, :max_frames]
    pad_amt = max_frames - t
    return torch.nn.functional.pad(spec, (0, pad_amt), mode="constant", value=0.0)


class RAVDESSMelDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        f_min: int = 0,
        f_max: Optional[int] = None,
        max_frames: int = 256,
        use_specaugment: bool = False,
        specaugment_time_mask: int = 20,
        specaugment_freq_mask: int = 8,
    ):
        self.files = files
        self.sample_rate = sample_rate
        self.max_frames = max_frames

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        self.use_specaugment = use_specaugment
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=specaugment_time_mask)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=specaugment_freq_mask)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.files[idx]

        emotion_id = filename_to_emotion_id(path)
        if emotion_id is None or emotion_id not in EMOTION_ID_TO_INDEX:
            raise ValueError(f"Could not parse emotion id from: {path}")
        y = EMOTION_ID_TO_INDEX[emotion_id]  # 0..7

        wav, sr = torchaudio.load(path, backend="soundfile")

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # mono
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)

        mel = self.melspec(wav)          # (1, n_mels, time)
        mel = self.amp_to_db(mel)        # log scale
        mel = mel.squeeze(0)             # (n_mels, time)

        # normalize per-sample (z-score)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # pad/trim to fixed time frames (matches your MCR narrative)
        mel = pad_or_trim(mel, self.max_frames)

        # optional SpecAugment
        if self.use_specaugment:
            mel_aug = mel.unsqueeze(0)
            mel_aug = self.time_mask(mel_aug)
            mel_aug = self.freq_mask(mel_aug)
            mel = mel_aug.squeeze(0)

        x = mel.unsqueeze(0)  # (1, n_mels, max_frames)
        return x, y
