"""
Dasheng model implementation.

This module wraps the local Dasheng checkpoint and adapts it to the framework's
waveform-in/classification-out interface.
"""

from pathlib import Path
import sys
from importlib import import_module, invalidate_caches

import torch
import torch.nn as nn

from .base_model import BaseAudioModel
from .beats_model import ArcMarginProduct, BatchNorm1d


def _load_dasheng_constructor(model_size: str):
    repo_root = Path(__file__).resolve().parents[1]
    local_dasheng = repo_root / "dasheng"
    if local_dasheng.exists() and str(local_dasheng) not in sys.path:
        sys.path.insert(0, str(local_dasheng))
    invalidate_caches()

    try:
        pretrained = import_module("dasheng.pretrained.pretrained")
        dasheng_base = pretrained.dasheng_base
        dasheng_06B = pretrained.dasheng_06B
        dasheng_12B = pretrained.dasheng_12B
    except ImportError:
        from dasheng import dasheng_06B, dasheng_12B, dasheng_base

    constructors = {
        "base": dasheng_base,
        "06b": dasheng_06B,
        "0.6b": dasheng_06B,
        "12b": dasheng_12B,
        "1.2b": dasheng_12B,
    }
    key = model_size.lower()
    if key not in constructors:
        raise ValueError(
            f"Unsupported Dasheng model_size: {model_size}. "
            "Expected one of: base, 0.6b, 06b, 1.2b, 12b"
        )
    return constructors[key]


class DashengAudioModel(BaseAudioModel):
    """
    Dasheng encoder with the same ArcMargin classification head as BEATs/EAT.

    Dasheng returns frame/token embeddings shaped [B, T, D]. The official
    downstream example mean-pools this sequence before classification, so this
    wrapper follows the same pooling.
    """

    def __init__(
        self,
        model_path: str,
        model_size: str,
        num_classes: int,
        feature_dim: int = 1536,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        constructor = _load_dasheng_constructor(model_size)
        self.dasheng = constructor(path=model_path)

        if freeze_encoder:
            for param in self.dasheng.parameters():
                param.requires_grad = False

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
        self.freeze_encoder = freeze_encoder

    def _extract_sequence_features(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio.float()

        if self.freeze_encoder:
            with torch.no_grad():
                return self.dasheng(audio)
        return self.dasheng(audio)

    def _extract_feature(self, audio: torch.Tensor) -> torch.Tensor:
        sequence = self._extract_sequence_features(audio)
        return sequence.mean(dim=1)

    def forward(
        self,
        audio: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        feature = self._extract_feature(audio)
        feature = self.bn(feature)
        feature = self.dropout(feature)
        emb = self.fc(feature)
        return self.arc_margin(emb, labels)

    def extract_embedding(
        self,
        audio: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            feature = self._extract_feature(audio)
            feature = self.bn(feature)
            emb = self.fc(feature)
        return emb

    def get_feature_dim(self) -> int:
        return self.feature_dim
