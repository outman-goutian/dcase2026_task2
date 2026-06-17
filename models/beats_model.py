"""
BEATs model implementation.

This module provides a wrapper for the BEATs (Bidirectional Encoder representation
from Audio Transformers) pretrained model, implementing the BaseAudioModel interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseAudioModel
from beats.BEATs import BEATs, BEATsConfig


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=True,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)
        if x.size(0) == 1 and self.training:
            x_n = x
        else:
            x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    
    This module implements the ArcMarginProduct layer which adds an angular margin
    to the classification loss, improving the discriminative power of the embeddings.
    """
    
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.s = s  # scale
        self.m = m  # margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        """
        Forward pass with angular margin.
        
        Args:
            input: Feature embeddings (B, D)
            label: One-hot encoded labels (B, C)
            
        Returns:
            output: Logits with angular margin applied (B, C)
        """
        # input shape: (B, D)
        # weight shape: (C, D)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # (B, C)
        sine = torch.sqrt(1.0 - cosine ** 2).clamp(min=1e-9)
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (label * phi) + ((1.0 - label) * cosine)  # only add margin to correct class
        output *= self.s
        return output


class BEATsModel(BaseAudioModel):
    """
    BEATs model implementation.
    
    This class wraps the BEATs pretrained model and implements the BaseAudioModel
    interface for use in the audio finetuning framework.
    """
    
    def __init__(self, checkpoint_path: str, num_classes: int, 
                 feature_dim: int = 768, dropout: float = 0.1):
        """
        Initialize BEATs model.
        
        Args:
            checkpoint_path: Path to BEATs pretrained checkpoint
            num_classes: Number of classification classes
            feature_dim: Feature dimension (default: 768)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        # Load BEATs pretrained model
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.beats = BEATs(cfg)
            self.beats.load_state_dict(checkpoint['model'])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model checkpoint not found at: {checkpoint_path}\n"
                f"Please verify the path exists and is accessible."
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model checkpoint from: {checkpoint_path}\n"
                f"Error: {str(e)}"
            )
        
        # Classification head
        self.bn = BatchNorm1d(input_size=feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.arc_margin = ArcMarginProduct(
            in_features=feature_dim,
            out_features=num_classes,
            s=30.0,
            m=0.5
        )
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
    
    def forward(self, audio: torch.Tensor, labels: torch.Tensor, 
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            audio: Input audio tensor (B, T)
            labels: One-hot encoded labels (B, C)
            padding_mask: Boolean mask for padding (B, T)
            
        Returns:
            logits: Classification logits (B, C)
        """
        # Extract features from BEATs
        feature, _, _ = self.beats.extract_features(audio, padding_mask=padding_mask)
        feature = feature.mean(dim=1)  # (B, T, D) -> (B, D)
        
        # Apply classification head
        feature = self.bn(feature)
        emb = self.fc(feature)
        logits = self.arc_margin(emb, labels)
        return logits
    
    def extract_embedding(self, audio: torch.Tensor, 
                         padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Extract audio embeddings for inference.
        
        Args:
            audio: Input audio tensor (B, T)
            padding_mask: Boolean mask for padding (B, T)
            
        Returns:
            embeddings: Audio embeddings (B, D)
        """
        self.eval()
        with torch.no_grad():
            feature, _, _ = self.beats.extract_features(audio, padding_mask=padding_mask)
            feature = feature.mean(dim=1)  # (B, T, D) -> (B, D)
            feature = self.bn(feature)
            emb = self.fc(feature)
        return emb
    
    def get_feature_dim(self) -> int:
        """
        Return the feature dimension of the model.
        
        Returns:
            feature_dim: The dimension of the embedding vector
        """
        return self.feature_dim
