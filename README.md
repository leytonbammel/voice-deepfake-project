# Deepfake-Resistant Voice Authentication

**DSC 4354 – Project 3 · Modality: Voice authentication (Option B)**
Team: Leyton Bammel, Connor Hurst

A web application that verifies a user by voice and rejects replayed recordings and AI-generated (cloned / text-to-speech) voices. Each login attempt returns one of **Accept**, **Reject (wrong person)** or **Reject (suspected fake or spoof)**. It combines four checks on a single microphone recording:

1. **Enrollment:** the user records several short voice samples. Their ECAPA-TDNN embeddings are averaged into a speaker template.
2. **Identity verification:** at login, the cosine similarity between the new sample and the template is compared against a calibrated threshold.
3. **Deepfake / spoof detection:** NII AntiDeepfake (wav2vec 2.0) estimates whether the audio is bona fide or synthetic.
4. **Liveness challenge:** the user reads a randomly generated phrase. Whisper ASR confirms the phrase was spoken, so a pre-recorded clip can't pass.

> Status (Nov 22): environment set up, both models running inference on CPU, preprocessing implemented. Enrollment flow, web UI and liveness come next (see the [timeline](#timeline)).

---

## Setup

Requires **Python 3.9** (fairseq 0.12.2 doesn't build on newer Pythons) and about 1 GB of disk for model weights. No GPU is needed.

```bash
git clone <this-repo-url> voice-auth && cd voice-auth

# Option 1: uv (installs Python 3.9 for you)
uv venv --python 3.9 --seed .venv
# Option 2: conda
#   conda create -n voiceauth python=3.9 && conda activate voiceauth

source .venv/bin/activate
python -m pip install "pip==24.0"     # pip >= 24.1 rejects omegaconf 2.0.6's metadata (fairseq dependency)
pip install -r requirements.txt       # CPU builds of torch 2.6.0 / torchaudio 2.6.0, fairseq 0.12.2, speechbrain 1.0.2
```

### Run the inference tests

```bash
python scripts/fetch_samples.py        # downloads a few public clips into data/samples/ (gitignored)
python scripts/test_verification.py    # ECAPA-TDNN: same-speaker vs different-speaker cosine scores
python scripts/test_detection.py       # AntiDeepfake: P(real) / P(fake) for bona fide and spoofed clips
```

Both scripts accept your own files:

```bash
python scripts/test_verification.py --same me_1.wav me_2.wav --diff me_1.wav other.wav
python scripts/test_detection.py me_1.wav some_tts_clip.wav
```

Model weights download on first run, from Hugging Face into `pretrained_models/` and `~/.cache/huggingface/`. Both locations are gitignored.

## Repository layout

```
src/
  preprocess.py     load -> 16 kHz mono, energy-based silence trimming, loudness normalisation
  detector.py       AntiDeepfake model definition + scoring (from the official model card)
scripts/
  fetch_samples.py      pulls LibriSpeech + In-the-Wild sample clips (range reads, no full download)
  test_verification.py  speaker-verification inference test
  test_detection.py     deepfake-detection inference test
requirements.txt    pinned environment (Python 3.9, CPU)
```

**No biometric data is stored in this repository.** `.gitignore` excludes all audio (`*.wav`, `*.flac`, …), `data/`, enrollment templates, consent forms, model weights and `.env`.

---

## Models

| Role | Model | Checkpoint | Size | License |
|---|---|---|---|---|
| Speaker verification | ECAPA-TDNN (SpeechBrain) | [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ~20M | Apache 2.0 |
| Deepfake detection | AntiDeepfake, wav2vec 2.0 Small | [`nii-yamagishilab/wav2vec-small-anti-deepfake`](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake) | 95M | Code BSD-3, weights CC BY-NC-SA 4.0 |
| Liveness ASR (planned) | Whisper tiny/base | [`openai/whisper`](https://github.com/openai/whisper) | 39–74M | MIT |
| Backup detector (optional) | AASIST-L | [`clovaai/aasist`](https://github.com/clovaai/aasist) | 85K | MIT |

**Speaker verification.** ECAPA-TDNN turns a 16 kHz utterance into a 192-dimensional speaker embedding. It was trained on VoxCeleb1+2 and reports 0.80% EER on VoxCeleb1-test (cleaned). Two utterances are compared by cosine similarity. We start from SpeechBrain's default threshold of 0.25 and will recalibrate it on same- and different-speaker pairs recorded on our own microphones.

**Deepfake detection.** AntiDeepfake feeds a raw 16 kHz waveform into a wav2vec 2.0 Base encoder: a convolutional feature encoder followed by 12 Transformer layers. It average-pools the frame features over time and passes them to a linear layer that outputs bona fide vs. spoof probabilities. The self-supervised encoder was *post-trained* on about 74k hours of speech from 29 corpora: 56k hours real and 18k hours fake. These include ASVspoof 2019 LA, ASVspoof 2021 LA/DF, ASVspoof 5, MLAAD, CodecFake, CVoiceFake, WaveFake and vocoded LibriTTS/VoxCeleb2. As a result, it learns general traces of TTS, voice-conversion, vocoder and neural-codec synthesis. It reports 4.24% EER on the out-of-domain In-the-Wild benchmark. It was not trained on replayed (loudspeaker re-recorded) audio, so replay attacks are handled by the liveness challenge.

### Citations

- B. Desplanques, J. Thienpondt, K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," *Proc. Interspeech 2020*, pp. 3830–3834. arXiv:2005.07143.
- M. Ravanelli, T. Parcollet, P. Plantinga, A. Rouhe, S. Cornell, L. Lugosch, C. Subakan, N. Dawalatabad, A. Heba, J. Zhong, et al., "SpeechBrain: A General-Purpose Speech Toolkit," arXiv:2106.04624, 2021.
- W. Ge, X. Wang, X. Liu, J. Yamagishi, "Post-training for Deepfake Speech Detection," *2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)*. arXiv:2506.21090. Code: https://github.com/nii-yamagishilab/AntiDeepfake
- J.-w. Jung, H.-S. Heo, H. Tak, H.-j. Shim, J. S. Chung, B.-J. Lee, H.-J. Yu, N. Evans, "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," *Proc. ICASSP 2022*. arXiv:2110.01200.
- A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," *Proc. ICML 2023*. arXiv:2212.04356.

### Datasets used for testing

- In-the-Wild: N. Müller, P. Czempin, F. Dieckmann, A. Froghyar, K. Böttinger, "Does Audio Deepfake Detection Generalize?," *Proc. Interspeech 2022*. arXiv:2203.16263. HF: `mueller91/In-The-Wild`.
- LibriSpeech: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books," *Proc. ICASSP 2015*. The sample clips come from the `hf-internal-testing/librispeech_asr_dummy` subset.

---

## Labor division

| Member | Responsibilities |
|---|---|
| **Leyton Bammel** | Environment and repository setup. Audio preprocessing (resampling, silence trimming, loudness normalisation). Speaker verification with ECAPA-TDNN: enrollment flow, template averaging, threshold calibration on our own same/different-speaker pairs. Web application (backend API, microphone recording UI, score display). Final accept/reject decision logic. |
| **Connor Hurst** | Deepfake detection with AntiDeepfake (and AASIST-L as an optional second vote): integration, threshold tuning. Liveness challenge (random-phrase generator, Whisper transcription, word-error-rate check). Attack evaluation: Level 1–2 phone replays, TTS / voice-clone attacks, In-the-Wild and ReplayDF benchmark runs, failure-case analysis. |
| **Shared** | Consent forms and enrollment-data collection, README and AI-usage log, Dec 6 prototype submission, Dec 12 presentation and live demo. |

## Timeline

| Date | Milestone |
|---|---|
| Nov 22 | Models selected and cited, inference tests passing, environment + repo, preprocessing ✅ |
| Dec 6 | Prototype: enrollment + login with real verification scores, detector integrated, liveness in progress, first replay-attack results |
| Dec 12 | Final app, attack evaluation, presentation + live demo |

---

## AI tool usage log

| Date | Tool | What it was used for | Human review |
|---|---|---|---|
| 2026-11-22 | Claude Code (Claude Opus 5.5) | Model research and comparison. Python 3.9 / fairseq environment setup. Wrote `src/preprocess.py`, `src/detector.py` (adapted from the AntiDeepfake model card), `scripts/fetch_samples.py` and the two inference test scripts. Checked citation metadata against arXiv and the authors' BibTeX. Drafted this README and the Nov 22 report. | Ran the scripts and checked the outputs. Reviewed the model choices, citations and labor division. |

## Acknowledgements

AntiDeepfake models © National Institute of Informatics (Yamagishi Lab), released under CC BY-NC-SA 4.0 for research and education. The detector class in `src/detector.py` follows the inference code on the official model card. ECAPA-TDNN weights are from SpeechBrain (Apache 2.0). Sample clips come from In-the-Wild and LibriSpeech and are not redistributed in this repo.
