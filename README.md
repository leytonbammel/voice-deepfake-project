# Deepfake-Resistant Voice Authentication

**DSC 4354 - Project 3 - Modality: Voice authentication**
Team: Leyton Bammel, Connor Hurst

## Setup

Requires **Python 3.9**, which fairseq 0.12.2 needs. Runs on CPU.

```bash
# 1. Create the environment
uv venv --python 3.9 --seed .venv          # or: conda create -n voiceauth python=3.9
source .venv/bin/activate

# 2. Install (pip 24.0 is needed: newer pip rejects fairseq's omegaconf 2.0.6 dependency)
python -m pip install "pip==24.0"
pip install -r requirements.txt

# 3. Run the inference tests
python scripts/fetch_samples.py            # downloads public test clips into data/ (gitignored)
python scripts/test_verification.py        # ECAPA-TDNN: same- vs different-speaker cosine scores
python scripts/test_detection.py           # AntiDeepfake: P(real) / P(fake) per clip
```

Both test scripts also accept your own files:
`test_verification.py --same a1.wav a2.wav --diff a1.wav b1.wav` and `test_detection.py real.wav fake.wav`.

## Models

| Role | Model | Checkpoint |
|---|---|---|
| Speaker verification | ECAPA-TDNN (SpeechBrain) | [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| Deepfake detection | NII AntiDeepfake, wav2vec 2.0 Small | [`nii-yamagishilab/wav2vec-small-anti-deepfake`](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake) |

**Citations**

- B. Desplanques, J. Thienpondt, K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," *Proc. Interspeech 2020*, pp. 3830–3834. arXiv:2005.07143.
- M. Ravanelli et al., "SpeechBrain: A General-Purpose Speech Toolkit," arXiv:2106.04624, 2021.
- W. Ge, X. Wang, X. Liu, J. Yamagishi, "Post-training for Deepfake Speech Detection," *2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)*. arXiv:2506.21090. Code: https://github.com/nii-yamagishilab/AntiDeepfake. Weights © NII Yamagishi Lab, CC BY-NC-SA 4.0. `src/detector.py` follows the model card's inference code.
- Test clips: In-the-Wild (N. Müller et al., "Does Audio Deepfake Detection Generalize?," *Proc. Interspeech 2022*, arXiv:2203.16263) and LibriSpeech (V. Panayotov et al., *Proc. ICASSP 2015*). Neither is redistributed in this repo.

## Labor division

| Member | Responsibilities |
|---|---|
| **Leyton Bammel** | Environment and repository setup. Audio preprocessing. Speaker verification with ECAPA-TDNN: enrollment flow, template averaging, threshold calibration. Web application. Final accept/reject decision logic. |
| **Connor Hurst** | Deepfake detection with AntiDeepfake: integration, threshold tuning. Liveness challenge (random phrase + Whisper transcription). Attack evaluation: replays, TTS / voice-clone attacks, benchmark runs, failure-case analysis. |
| **Shared** | Consent forms and data collection, README and AI usage log, prototype submission, presentation and live demo. |

## AI tool usage log

| Date | Tool | What it was used for | Human review |
|---|---|---|---|
| 2026-11-22 | Claude Code (Claude Opus 5.5) | Model research and comparison. Python 3.9 / fairseq environment setup. Wrote `src/preprocess.py`, `src/detector.py` (adapted from the AntiDeepfake model card), `scripts/fetch_samples.py` and the two inference test scripts. Checked citation metadata. Drafted this README and the Nov 22 report. | Ran the scripts and checked the outputs. Reviewed the model choices, citations and labor division. |
