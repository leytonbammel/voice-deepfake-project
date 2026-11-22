"""Inference test for the speaker-verification model (SpeechBrain ECAPA-TDNN).

Scores one same-speaker pair and one different-speaker pair with cosine similarity.

Usage:
  python scripts/test_verification.py --same A1.wav A2.wav --diff A1.wav B1.wav
With no arguments it uses the public clips from scripts/fetch_samples.py.
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speechbrain.inference.speaker import SpeakerRecognition  # noqa: E402

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
THRESHOLD = 0.25  # starting point; we will tune it on our own same/different-speaker pairs
S = ROOT / "data" / "samples"


def rel(p: str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--same", nargs=2, default=[S / "librispeech/spk1272_1.wav", S / "librispeech/spk1272_2.wav"])
    ap.add_argument("--diff", nargs=2, default=[S / "librispeech/spk1272_1.wav", S / "itw/real_1.wav"])
    args = ap.parse_args()

    print(f"Loading speaker-verification model: {MODEL_ID}")
    v = SpeakerRecognition.from_hparams(source=MODEL_ID, savedir=str(ROOT / "pretrained_models" / "ecapa"))
    print(f"Decision threshold (cosine, SpeechBrain default): {THRESHOLD}\n")

    print(f"{'pair':<14}{'clip 1':<40}{'clip 2':<40}{'cosine':>8}  same speaker?")
    print("-" * 116)
    for name, (a, b) in (("SAME speaker", args.same), ("DIFF speakers", args.diff)):
        score, _ = v.verify_files(str(a), str(b))  # cosine similarity of ECAPA embeddings
        score = score.item()
        print(f"{name:<14}{rel(str(a)):<40}{rel(str(b)):<40}{score:>8.4f}  {score > THRESHOLD}")


if __name__ == "__main__":
    main()
