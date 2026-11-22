"""NII AntiDeepfake (wav2vec-small) deepfake-speech detector.

Model definition follows the official Hugging Face model card:
https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake
Ge, Wang, Liu & Yamagishi, "Post-training for Deepfake Speech Detection", IEEE ASRU 2025.
"""
import torch
from fairseq.models.wav2vec import Wav2Vec2Config, Wav2Vec2Model
from huggingface_hub import PyTorchModelHubMixin

MODEL_ID = "nii-yamagishilab/wav2vec-small-anti-deepfake"


class SSLModel(torch.nn.Module):
    """wav2vec 2.0 Base front end (CNN feature encoder + 12 Transformer layers)."""

    def __init__(self):
        super().__init__()
        cfg = Wav2Vec2Config(
            encoder_layers=12,
            encoder_embed_dim=768,
            quantize_targets=True,
            latent_dim=256,
            final_dim=256,
        )
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 3:
            wav = wav[:, :, 0]
        return self.model(wav, mask=False, features_only=True)["x"]  # [B, T, 768]


class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    """SSL front end -> average pooling over time -> linear layer -> [fake, real] logits."""

    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 768
        self.num_classes = 2
        self.m_ssl = SSLModel()
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(self.ssl_orig_output_dim, self.num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        emb = self.m_ssl.extract_feat(wav).transpose(1, 2)  # [B, D, T]
        pooled = self.adap_pool1d(emb).squeeze(-1)          # [B, D]
        return self.proj_fc(pooled)                         # [B, 2]  (index 0 = fake, 1 = real)


def load_detector(device: str = "cpu") -> DeepfakeDetector:
    model = DeepfakeDetector.from_pretrained(MODEL_ID)
    return model.to(device).eval()


@torch.no_grad()
def score_waveform(model: DeepfakeDetector, wav: torch.Tensor) -> dict:
    """wav: 1-D float tensor, 16 kHz mono. Returns softmax probabilities."""
    wav = torch.nn.functional.layer_norm(wav, wav.shape)  # per-utterance normalisation, as in the model card
    probs = torch.softmax(model(wav.unsqueeze(0)), dim=1)[0]
    return {"fake": float(probs[0]), "real": float(probs[1])}
