"""
Base model interface for audio models.

This module defines the abstract base class that all audio models must implement
to ensure a consistent API across different model implementations.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseAudioModel(ABC, nn.Module):
    """
    Abstract base class for audio models.
    
    All audio model implementations (BEATs, AST, etc.) must inherit from this class
    and implement the required abstract methods to ensure compatibility with the
    training and inference pipeline.
    """
    
    @abstractmethod
    def forward(self, audio: torch.Tensor, labels: torch.Tensor, 
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            audio: Input audio tensor (B, T) where B=batch size, T=time steps
            labels: One-hot encoded labels (B, C) where C=num_classes
            padding_mask: Boolean mask for padding (B, T), optional
            
        Returns:
            logits: Classification logits (B, C)
        """
        pass
    
    @abstractmethod
    def extract_embedding(self, audio: torch.Tensor, 
                         padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Extract audio embeddings for inference.
        
        Args:
            audio: Input audio tensor (B, T) where B=batch size, T=time steps
            padding_mask: Boolean mask for padding (B, T), optional
            
        Returns:
            embeddings: Audio embeddings (B, D) where D=embedding_dim
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Return the feature dimension of the model.
        
        Returns:
            feature_dim: The dimension of the embedding/feature vector
        """
        pass
