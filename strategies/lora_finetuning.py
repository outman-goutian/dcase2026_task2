"""
LoRA (Low-Rank Adaptation) finetuning strategy.

This module implements the LoRA finetuning strategy using the Hugging Face PEFT library.
LoRA freezes the base model parameters and injects trainable low-rank adaptation matrices.
"""

from typing import Iterator
from types import SimpleNamespace
import torch.nn as nn
from .base_strategy import FinetuningStrategy

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    raise ImportError(
        "peft library is required for LoRA finetuning. "
        "Install it with: pip install peft"
    )


class LoRAFinetuningStrategy(FinetuningStrategy):
    """
    LoRA (Low-Rank Adaptation) finetuning strategy.
    
    This strategy freezes the base model parameters and injects low-rank adaptation
    matrices that are trained instead. This significantly reduces the number of
    trainable parameters and memory usage.
    """
    
    def apply(self, model: nn.Module, config: SimpleNamespace) -> nn.Module:
        """
        Apply LoRA adapters to model.
        
        Args:
            model: The audio model to apply strategy to
            config: Configuration namespace containing LoRA parameters
            
        Returns:
            model: Model with LoRA adapters applied
        """
        # Validate LoRA configuration
        if not hasattr(config.finetuning, 'lora'):
            raise ValueError(
                "Missing LoRA configuration when strategy='lora'. "
                "Please add 'finetuning.lora' section to config with rank, alpha, etc."
            )
        
        lora_config_params = config.finetuning.lora
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=lora_config_params.rank,
            lora_alpha=lora_config_params.alpha,
            target_modules=lora_config_params.target_modules,
            lora_dropout=lora_config_params.dropout,
            bias="none",
            modules_to_save=getattr(
                lora_config_params,
                'modules_to_save',
                ['bn', 'fc', 'arc_margin']
            ),
            task_type=None
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters info
        model.print_trainable_parameters()
        
        return model
    
    def get_trainable_parameters(self, model: nn.Module) -> Iterator:
        """
        Return only LoRA adapter parameters.
        
        Args:
            model: The model to get trainable parameters from
            
        Returns:
            Iterator of only LoRA adapter parameters (base model frozen)
        """
        return filter(lambda p: p.requires_grad, model.parameters())
