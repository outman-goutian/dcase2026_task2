"""
Model factory for creating audio models with finetuning strategies.

This module provides a factory function to create audio models based on
configuration parameters, with the appropriate finetuning strategy applied.
"""

from types import SimpleNamespace
from typing import Tuple
import torch.nn as nn

from strategies import FinetuningStrategy


def create_model(config: SimpleNamespace, num_classes: int) -> Tuple[nn.Module, FinetuningStrategy]:
    """
    Factory function to create audio model based on configuration.
    
    This function creates the appropriate audio model (BEATs, AST, etc.) based on
    the configuration and applies the specified finetuning strategy (full, LoRA).
    
    Args:
        config: Configuration namespace containing model and finetuning parameters
        num_classes: Number of classification classes
        
    Returns:
        model: Configured audio model with finetuning strategy applied
        strategy: The finetuning strategy instance that was applied
        
    Raises:
        ValueError: If model type or finetuning strategy is unsupported
        
    Example:
        >>> config = load_config('config.yaml')
        >>> model, strategy = create_model(config, num_classes=100)
        >>> optimizer = optim.AdamW(strategy.get_trainable_parameters(model), lr=1e-4)
    """
    
    # Validate model type exists in config
    if not hasattr(config.model, 'type'):
        raise ValueError(
            "Missing required parameter: model.type\n"
            "Please specify model type in config (e.g., 'beats', 'ast', 'eat', or 'dasheng')"
        )
    
    # Create base model based on model type
    model_type = config.model.type.lower()
    
    if model_type == "beats":
        from models import BEATsModel

        # Validate BEATs-specific parameters
        if not hasattr(config.model, 'checkpoint_path'):
            raise ValueError(
                "Missing required parameter: model.checkpoint_path\n"
                "Please specify BEATs checkpoint path in config"
            )
        
        model = BEATsModel(
            checkpoint_path=config.model.checkpoint_path,
            num_classes=num_classes,
            feature_dim=config.model.feature_dim,
            dropout=config.model.dropout
        )
        print(f"Created BEATs model with {num_classes} classes")
        
    elif model_type == "ast":
        from models import ASTAudioModel

        # Validate AST-specific parameters
        if not hasattr(config.model, 'checkpoint_path') and not hasattr(config.model, 'model_name'):
            raise ValueError(
                "Missing required parameter: model.checkpoint_path\n"
                "Please specify local official AST checkpoint path in config"
            )
        
        model = ASTAudioModel(
            checkpoint_path=getattr(config.model, 'checkpoint_path', None),
            model_name=getattr(config.model, 'model_name', None),
            num_classes=num_classes,
            feature_dim=config.model.feature_dim,
            dropout=config.model.dropout,
            sample_rate=config.data.sample_rate,
            target_length=getattr(config.model, 'target_length', 1024),
            norm_mean=getattr(config.model, 'norm_mean', -4.2677393),
            norm_std=getattr(config.model, 'norm_std', 4.5689974),
            fstride=getattr(config.model, 'fstride', 10),
            tstride=getattr(config.model, 'tstride', 10),
            input_fdim=getattr(config.model, 'input_fdim', 128),
            model_size=getattr(config.model, 'model_size', 'base384'),
            imagenet_pretrain=getattr(config.model, 'imagenet_pretrain', False),
            pooling=getattr(config.model, 'pooling', 'cls_distill')
        )
        print(f"Created AST model with {num_classes} classes")

    elif model_type == "eat":
        from models import EATAudioModel

        # Validate EAT-specific parameters
        if not hasattr(config.model, 'model_name'):
            raise ValueError(
                "Missing required parameter: model.model_name\n"
                "Please specify Hugging Face model name for EAT in config"
            )

        model = EATAudioModel(
            model_name=config.model.model_name,
            num_classes=num_classes,
            feature_dim=config.model.feature_dim,
            dropout=config.model.dropout,
            sample_rate=config.data.sample_rate,
            target_length=getattr(config.model, 'target_length', 1024),
            norm_mean=getattr(config.model, 'norm_mean', -4.268),
            norm_std=getattr(config.model, 'norm_std', 4.569),
            pooling=getattr(config.model, 'pooling', 'cls'),
            trust_remote_code=getattr(config.model, 'trust_remote_code', True)
        )
        print(f"Created EAT model with {num_classes} classes")

    elif model_type == "dasheng":
        from models import DashengAudioModel

        if not hasattr(config.model, 'model_path'):
            raise ValueError(
                "Missing required parameter: model.model_path\n"
                "Please specify local Dasheng checkpoint path in config"
            )

        model = DashengAudioModel(
            model_path=config.model.model_path,
            model_size=getattr(config.model, 'model_size', '1.2b'),
            num_classes=num_classes,
            feature_dim=config.model.feature_dim,
            dropout=config.model.dropout,
            freeze_encoder=getattr(config.model, 'freeze_encoder', False)
        )
        print(f"Created Dasheng model with {num_classes} classes")
        
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}\n"
            f"Expected one of: ['beats', 'ast', 'eat', 'dasheng']"
        )
    
    # Validate finetuning strategy exists in config
    if not hasattr(config.finetuning, 'strategy'):
        raise ValueError(
            "Missing required parameter: finetuning.strategy\n"
            "Please specify finetuning strategy in config (e.g., 'full' or 'lora')"
        )
    
    # Apply finetuning strategy
    strategy_type = config.finetuning.strategy.lower()
    
    if strategy_type == "full":
        from strategies import FullFinetuningStrategy

        strategy = FullFinetuningStrategy()
        model = strategy.apply(model, config)
        print("Applied full parameter finetuning strategy")
        
    elif strategy_type == "lora":
        from strategies import LoRAFinetuningStrategy

        strategy = LoRAFinetuningStrategy()
        model = strategy.apply(model, config)
        print("Applied LoRA finetuning strategy")
        
    else:
        raise ValueError(
            f"Unsupported finetuning strategy: {strategy_type}\n"
            f"Expected one of: ['full', 'lora']"
        )
    
    return model, strategy
