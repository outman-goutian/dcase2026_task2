"""
Full parameter finetuning strategy.

This module implements the full parameter finetuning strategy where all model
parameters are enabled for gradient updates during training.
"""

from typing import Iterator
from types import SimpleNamespace
import torch.nn as nn
from .base_strategy import FinetuningStrategy


class FullFinetuningStrategy(FinetuningStrategy):
    """
    Full parameter finetuning strategy.
    
    This strategy enables all model parameters for training, allowing gradient
    updates for the entire model during backpropagation.
    """
    
    def apply(self, model: nn.Module, config: SimpleNamespace) -> nn.Module:
        """
        Enable all parameters for training.
        
        Args:
            model: The audio model to apply strategy to
            config: Configuration namespace (not used for full finetuning)
            
        Returns:
            model: Model with all parameters enabled for training
        """
        # Enable gradient updates for all parameters
        for param in model.parameters():
            param.requires_grad = True
        
        print(f"Full parameter finetuning enabled: all {sum(p.numel() for p in model.parameters())} parameters are trainable")
        
        return model
    
    def get_trainable_parameters(self, model: nn.Module) -> Iterator:
        """
        Return all model parameters.
        
        Args:
            model: The model to get trainable parameters from
            
        Returns:
            Iterator of all model parameters
        """
        return model.parameters()
