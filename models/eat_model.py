"""
EAT model implementation.

This module wraps the Hugging Face EAT checkpoints and adapts them to the
framework's waveform-in/classification-out interface.
"""

import torch
import torch.nn as nn
import torchaudio
import os

from .base_model import BaseAudioModel
from .beats_model import BatchNorm1d, ArcMarginProduct
from .eat_local import load_local_eat


class EATAudioModel(BaseAudioModel):
    """
    Efficient Audio Transformer model implementation.

    EAT expects normalized Kaldi fbank features with shape [B, 1, T, 128].
    This wrapper accepts waveform tensors [B, samples] to stay compatible with
    the existing training and validation pipelines.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        feature_dim: int = 768,
        dropout: float = 0.1,
        sample_rate: int = 16000,
        target_length: int = 1024,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        pooling: str = "cls",
        trust_remote_code: bool = True,
    ):
        super().__init__()

        self.eat = self._load_eat_model(model_name, trust_remote_code)

        self.sample_rate = sample_rate
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.pooling = pooling.lower()
        if self.pooling != "cls":
            raise ValueError(f"Unsupported EAT pooling: {pooling}. Expected 'cls'.")

        self.bn = BatchNorm1d(input_size=feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.arc_margin = ArcMarginProduct(
            in_features=feature_dim,
            out_features=num_classes,
            s=30.0,
            m=0.5,
        )

        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def _load_eat_model(self, model_name: str, trust_remote_code: bool) -> nn.Module:
        if os.path.isfile(model_name) and model_name.endswith(".safetensors"):
            return load_local_eat(model_name)

        if (
            os.path.isfile(os.path.join(model_name, "config.json"))
            and os.path.isfile(os.path.join(model_name, "model.safetensors"))
        ):
            return load_local_eat(model_name)

        try:
            from transformers import AutoModel

            return AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        except Exception as auto_model_error:
            try:
                return load_local_eat(model_name)
            except Exception as local_error:
                raise RuntimeError(
                    f"Failed to load EAT model: {model_name}\n"
                    f"AutoModel error: {str(auto_model_error)}\n"
                    f"Local loader error: {str(local_error)}\n"
                    f"Please verify the model name, cache, and network access."
                )

    def _waveform_to_fbank(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform batch [B, T] to EAT fbank input [B, 1, target_length, 128].
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        features = []
        for waveform in audio:
            waveform = waveform.float()
            waveform = waveform - waveform.mean()

            mel = torchaudio.compliance.kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.sample_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )

            n_frames = mel.shape[0]
            if n_frames < self.target_length:
                pad = self.target_length - n_frames
                mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
            else:
                mel = mel[: self.target_length, :]

            mel = (mel - self.norm_mean) / (self.norm_std * 2)
            features.append(mel)

        return torch.stack(features, dim=0).unsqueeze(1)

    def _extract_sequence_features(self, audio: torch.Tensor, layer: int = None) -> torch.Tensor:
        fbank = self._waveform_to_fbank(audio)
        if layer is None:
            return self.eat.extract_features(fbank)
        return self.eat.extract_features(fbank, layer=layer)

    def _pool_sequence_features(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence[:, 0]

    def forward(
        self,
        audio: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        """
        sequence = self._extract_sequence_features(audio)
        feature = self._pool_sequence_features(sequence)

        feature = self.bn(feature)
        feature = self.dropout(feature)
        emb = self.fc(feature)
        logits = self.arc_margin(emb, labels)
        return logits

    def extract_embedding(
        self,
        audio: torch.Tensor,
        padding_mask: torch.Tensor = None,
        layer: int = None,
    ) -> torch.Tensor:
        """
        Extract audio embeddings for inference.
        """
        self.eval()
        with torch.no_grad():
            sequence = self._extract_sequence_features(audio, layer=layer)
            feature = self._pool_sequence_features(sequence)
            feature = self.bn(feature)
            emb = self.fc(feature)
        return emb

    def get_feature_dim(self) -> int:
        return self.feature_dim
