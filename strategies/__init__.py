"""
Finetuning strategies module.

This module provides different finetuning strategies (full parameter, LoRA)
for audio models through the FinetuningStrategy abstract class.
"""

__all__ = ['FinetuningStrategy', 'FullFinetuningStrategy', 'LoRAFinetuningStrategy']


def __getattr__(name):
    if name == 'FinetuningStrategy':
        from .base_strategy import FinetuningStrategy
        return FinetuningStrategy
    if name == 'FullFinetuningStrategy':
        from .full_finetuning import FullFinetuningStrategy
        return FullFinetuningStrategy
    if name == 'LoRAFinetuningStrategy':
        from .lora_finetuning import LoRAFinetuningStrategy
        return LoRAFinetuningStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
