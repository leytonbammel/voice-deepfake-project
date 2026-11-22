"""Audio preprocessing shared by verification and detection.

load_audio():  any file -> 16 kHz mono float32 tensor
trim_silence(): energy-based voice activity trimming of leading/trailing silence
normalize_loudness(): RMS normalisation to a target dBFS (peak-limited)
preprocess(): all three in order
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

TARGET_SR = 16_000


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)  # mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.float()


def trim_silence(wav: torch.Tensor, sr: int = TARGET_SR, frame_ms: float = 25.0,
                 threshold_db: float = -40.0, pad_ms: float = 100.0) -> torch.Tensor:
    """Drop leading/trailing frames whose energy is threshold_db below the loudest frame."""
    frame = int(sr * frame_ms / 1000)
    if wav.numel() < frame:
        return wav
    frames = wav[: wav.numel() // frame * frame].reshape(-1, frame)
    energy_db = 10 * torch.log10(frames.pow(2).mean(dim=1) + 1e-10)
    voiced = torch.nonzero(energy_db > energy_db.max() + threshold_db).flatten()
    if voiced.numel() == 0:
        return wav
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(voiced[0]) * frame - pad)
    end = min(wav.numel(), (int(voiced[-1]) + 1) * frame + pad)
    return wav[start:end]


def normalize_loudness(wav: torch.Tensor, target_dbfs: float = -20.0) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt()
    if rms < 1e-8:
        return wav
    gain = 10 ** (target_dbfs / 20) / rms
    out = wav * gain
    peak = out.abs().max()
    return out / peak * 0.99 if peak > 0.99 else out


def preprocess(path: str | Path) -> torch.Tensor:
    return normalize_loudness(trim_silence(load_audio(path)))
