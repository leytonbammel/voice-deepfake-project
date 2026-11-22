"""Inference test for the deepfake-speech detector (NII AntiDeepfake, wav2vec-small).

Prints the bona fide / spoof probabilities for each clip.

Usage:
  python scripts/test_detection.py real.wav fake.wav ...
With no arguments it uses the public In-the-Wild clips from scripts/fetch_samples.py.
"""
import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.detector import MODEL_ID, load_detector, score_waveform  # noqa: E402
from src.preprocess import load_audio  # noqa: E402


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    args = ap.parse_args()
    files = args.files or sorted((ROOT / "data" / "samples" / "itw").glob("*.wav"))

    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"Loading deepfake detector: {MODEL_ID}")
    model = load_detector("cpu")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M   device: cpu\n")

    print(f"{'clip':<40}{'dur(s)':>7}{'P(real)':>10}{'P(fake)':>10}  prediction   time(s)")
    print("-" * 92)
    for f in files:
        wav = load_audio(f)
        t0 = time.time()
        p = score_waveform(model, wav)
        dt = time.time() - t0
        label = "BONA FIDE" if p["real"] >= p["fake"] else "SPOOF"
        print(f"{rel(f):<40}{wav.numel() / 16000:>7.1f}{p['real']:>10.4f}{p['fake']:>10.4f}  {label:<11}{dt:>8.2f}")


if __name__ == "__main__":
    main()
