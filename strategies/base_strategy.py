"""
Base finetuning strategy interface.

This module defines the abstract base class for all finetuning strategies.
"""

from abc import ABC, abstractmethod
from typing import Iterator
from types import SimpleNamespace
import torch.nn as nn


class FinetuningStrategy(ABC):
    """
    Abstract base class for finetuning strategies.
    
    All finetuning strategy implementations (full parameter, LoRA, etc.) must
    inherit from this class and implement the required abstract methods.
    """
    
    @abstractmethod
    def apply(self, model: nn.Module, config: SimpleNamespace) -> nn.Module:
        """
        Apply finetuning strategy to model.
        
        Args:
            model: The audio model to apply strategy to
            config: Configuration namespace containing strategy parameters
            
        Returns:
            model: Model with strategy applied
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self, model: nn.Module) -> Iterator:
        """
        Return iterator of trainable parameters.
        
        Args:
            model: The model to get trainable parameters from
            
        Returns:
            Iterator of trainable parameters
        """
        pass
