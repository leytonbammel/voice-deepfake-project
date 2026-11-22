"""Download a few public benchmark clips for the inference tests.

Writes into data/samples/ (gitignored):
  librispeech/<speaker>_<n>.wav   real read speech, several speakers (LibriSpeech dev-clean subset)
  itw/real_<n>.wav, itw/fake_<n>.wav   bona fide and spoofed clips from In-the-Wild

In-the-Wild ships as one large zip on Hugging Face. We open it with HTTP range
reads (HfFileSystem) and pull out only the members we need, so the full
archive is never downloaded.

Usage:  python scripts/fetch_samples.py [--n 2]
"""
import argparse
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from huggingface_hub import HfFileSystem, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples"


def fetch_librispeech(per_speaker: int = 2, speakers: int = 3) -> None:
    import pyarrow.parquet as pq

    path = hf_hub_download(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean/validation-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(path).to_pylist()
    by_spk = defaultdict(list)
    for row in table:
        by_spk[row["speaker_id"]].append(row)
    out = OUT / "librispeech"
    out.mkdir(parents=True, exist_ok=True)
    # most-represented speaker first so we get a same-speaker pair, then others for different-speaker pairs
    chosen = sorted(by_spk, key=lambda s: -len(by_spk[s]))[:speakers]
    for spk in chosen:
        for i, row in enumerate(by_spk[spk][:per_speaker], 1):
            wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            dst = out / f"spk{spk}_{i}.wav"
            sf.write(dst, wav, sr)
            print(f"wrote {dst.relative_to(ROOT)}  ({len(wav) / sr:.1f}s, {sr} Hz)")


def fetch_in_the_wild(n: int = 2, min_sec: float = 3.0) -> None:
    fs = HfFileSystem()
    url = "datasets/mueller91/In-The-Wild/release_in_the_wild.zip"
    out = OUT / "itw"
    out.mkdir(parents=True, exist_ok=True)
    with fs.open(url, "rb", block_size=1 << 20) as fh, zipfile.ZipFile(fh) as zf:
        names = zf.namelist()
        meta_name = next(m for m in names if m.endswith("meta.csv"))
        prefix = meta_name[: -len("meta.csv")]
        rows = list(csv.DictReader(io.TextIOWrapper(zf.open(meta_name), encoding="utf-8")))
        picked = {"bona-fide": [], "spoof": []}
        for r in rows:
            label = r["label"]
            if len(picked[label]) >= n:
                if all(len(v) >= n for v in picked.values()):
                    break
                continue
            data = zf.read(prefix + r["file"])
            info = sf.info(io.BytesIO(data))
            if info.duration < min_sec:  # skip very short clips
                continue
            picked[label].append(r)
            tag = "real" if label == "bona-fide" else "fake"
            dst = out / f"{tag}_{len(picked[label])}.wav"
            dst.write_bytes(data)
            print(f"wrote {dst.relative_to(ROOT)}  label={label}  speaker={r['speaker']}  "
                  f"src={r['file']}  ({info.duration:.1f}s, {info.samplerate} Hz)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="clips per class from In-the-Wild")
    args = ap.parse_args()
    fetch_librispeech()
    fetch_in_the_wild(args.n)
