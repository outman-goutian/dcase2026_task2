"""
Configuration validation module.

This module provides functions to validate configuration parameters before
training to ensure all required parameters are present and valid.
"""

import os
from types import SimpleNamespace
from typing import List


def validate_config(config: SimpleNamespace) -> None:
    """
    Validate configuration parameters.
    
    This function checks that all required parameters are present and valid,
    raising descriptive errors for any validation failures.
    
    Args:
        config: Configuration namespace to validate
        
    Raises:
        ValueError: If any configuration parameters are invalid or missing
        
    Example:
        >>> config = load_config('config.yaml')
        >>> validate_config(config)  # Raises ValueError if invalid
    """
    errors = []
    
    # Validate data configuration
    if not hasattr(config, 'data'):
        errors.append("Missing required section: data")
    else:
        if not hasattr(config.data, 'train_txt'):
            errors.append("Missing required parameter: data.train_txt")
        if not hasattr(config.data, 'sample_rate'):
            errors.append("Missing required parameter: data.sample_rate")
        elif config.data.sample_rate <= 0:
            errors.append(f"data.sample_rate must be positive, got: {config.data.sample_rate}")
        if not hasattr(config.data, 'batch_size'):
            errors.append("Missing required parameter: data.batch_size")
        elif config.data.batch_size <= 0:
            errors.append(f"data.batch_size must be positive, got: {config.data.batch_size}")
    
    # Validate model configuration
    if not hasattr(config, 'model'):
        errors.append("Missing required section: model")
    else:
        # Validate model type
        if not hasattr(config.model, 'type'):
            errors.append("Missing required parameter: model.type")
        elif config.model.type not in ['beats', 'ast', 'eat', 'dasheng']:
            errors.append(
                f"Invalid model type: {config.model.type}. "
                f"Expected one of: ['beats', 'ast', 'eat', 'dasheng']"
            )
        else:
            # Validate model-specific parameters
            if config.model.type == 'beats':
                if not hasattr(config.model, 'checkpoint_path'):
                    errors.append("Missing required parameter: model.checkpoint_path (required for BEATs model)")
            elif config.model.type == 'ast':
                if not hasattr(config.model, 'checkpoint_path') and not hasattr(config.model, 'model_name'):
                    errors.append("Missing required parameter: model.checkpoint_path (required for AST model)")
                elif hasattr(config.model, 'checkpoint_path') and not os.path.exists(config.model.checkpoint_path):
                    errors.append(f"AST checkpoint file not found: {config.model.checkpoint_path}")
            elif config.model.type == 'eat':
                if not hasattr(config.model, 'model_name'):
                    errors.append("Missing required parameter: model.model_name (required for EAT model)")
                if hasattr(config.model, 'pooling') and config.model.pooling != 'cls':
                    errors.append(
                        f"Invalid EAT pooling: {config.model.pooling}. "
                        "Only 'cls' is supported for EAT."
                    )
            elif config.model.type == 'dasheng':
                if not hasattr(config.model, 'model_path'):
                    errors.append("Missing required parameter: model.model_path (required for Dasheng model)")
        
        # Validate common model parameters
        if not hasattr(config.model, 'num_classes'):
            errors.append("Missing required parameter: model.num_classes")
        elif config.model.num_classes <= 0:
            errors.append(f"model.num_classes must be positive, got: {config.model.num_classes}")
        
        if not hasattr(config.model, 'feature_dim'):
            errors.append("Missing required parameter: model.feature_dim")
        elif config.model.feature_dim <= 0:
            errors.append(f"model.feature_dim must be positive, got: {config.model.feature_dim}")
        
        if not hasattr(config.model, 'dropout'):
            errors.append("Missing required parameter: model.dropout")
        elif not (0 <= config.model.dropout < 1):
            errors.append(f"model.dropout must be in [0, 1), got: {config.model.dropout}")
    
    # Validate finetuning configuration
    if not hasattr(config, 'finetuning'):
        errors.append("Missing required section: finetuning")
    else:
        # Validate finetuning strategy
        if not hasattr(config.finetuning, 'strategy'):
            errors.append("Missing required parameter: finetuning.strategy")
        elif config.finetuning.strategy not in ['full', 'lora']:
            errors.append(
                f"Invalid finetuning strategy: {config.finetuning.strategy}. "
                f"Expected one of: ['full', 'lora']"
            )
        
        # Validate LoRA parameters if LoRA is selected
        if hasattr(config.finetuning, 'strategy') and config.finetuning.strategy == 'lora':
            if not hasattr(config.finetuning, 'lora'):
                errors.append("Missing LoRA configuration when strategy='lora'")
            else:
                if not hasattr(config.finetuning.lora, 'rank'):
                    errors.append("Missing required parameter: finetuning.lora.rank")
                elif config.finetuning.lora.rank <= 0:
                    errors.append(f"LoRA rank must be positive, got: {config.finetuning.lora.rank}")
                
                if not hasattr(config.finetuning.lora, 'alpha'):
                    errors.append("Missing required parameter: finetuning.lora.alpha")
                elif config.finetuning.lora.alpha <= 0:
                    errors.append(f"LoRA alpha must be positive, got: {config.finetuning.lora.alpha}")
                
                if not hasattr(config.finetuning.lora, 'dropout'):
                    errors.append("Missing required parameter: finetuning.lora.dropout")
                elif not (0 <= config.finetuning.lora.dropout < 1):
                    errors.append(f"LoRA dropout must be in [0, 1), got: {config.finetuning.lora.dropout}")
                
                if not hasattr(config.finetuning.lora, 'target_modules'):
                    errors.append("Missing required parameter: finetuning.lora.target_modules")
                elif not isinstance(config.finetuning.lora.target_modules, list):
                    errors.append("finetuning.lora.target_modules must be a list")
                elif len(config.finetuning.lora.target_modules) == 0:
                    errors.append("finetuning.lora.target_modules must not be empty")
    
    # Validate training configuration
    if not hasattr(config, 'training'):
        errors.append("Missing required section: training")
    else:
        if not hasattr(config.training, 'num_epochs'):
            errors.append("Missing required parameter: training.num_epochs")
        elif config.training.num_epochs <= 0:
            errors.append(f"training.num_epochs must be positive, got: {config.training.num_epochs}")
        
        if not hasattr(config.training, 'learning_rate'):
            errors.append("Missing required parameter: training.learning_rate")
        elif config.training.learning_rate <= 0:
            errors.append(f"training.learning_rate must be positive, got: {config.training.learning_rate}")
        
        if not hasattr(config.training, 'weight_decay'):
            errors.append("Missing required parameter: training.weight_decay")
        elif config.training.weight_decay < 0:
            errors.append(f"training.weight_decay must be non-negative, got: {config.training.weight_decay}")
        
        if not hasattr(config.training, 'max_grad_norm'):
            errors.append("Missing required parameter: training.max_grad_norm")
        elif config.training.max_grad_norm <= 0:
            errors.append(f"training.max_grad_norm must be positive, got: {config.training.max_grad_norm}")
    
    # Validate logging configuration
    if not hasattr(config, 'logging'):
        errors.append("Missing required section: logging")
    else:
        if not hasattr(config.logging, 'save_dir'):
            errors.append("Missing required parameter: logging.save_dir")
        if not hasattr(config.logging, 'log_dir'):
            errors.append("Missing required parameter: logging.log_dir")
    
    # Validate checkpoint configuration
    if hasattr(config, 'checkpoint'):
        if hasattr(config.checkpoint, 'resume_epoch'):
            resume_epoch = config.checkpoint.resume_epoch
            # Allow null, empty string, or positive integer
            if resume_epoch is not None and resume_epoch != '':
                if not isinstance(resume_epoch, int):
                    errors.append(
                        f"Invalid checkpoint.resume_epoch value: {resume_epoch}. "
                        f"Expected null, empty, or positive integer."
                    )
                elif resume_epoch <= 0:
                    errors.append(
                        f"Invalid checkpoint.resume_epoch value: {resume_epoch}. "
                        f"Expected positive integer."
                    )
    
    # Validate validation configuration
    if hasattr(config, 'validation'):
        # Validate validation.scp_path existence
        if hasattr(config.validation, 'scp_path'):
            scp_path = config.validation.scp_path
            if not os.path.exists(scp_path):
                errors.append(
                    f"Validation SCP file not found: {scp_path}. "
                    f"Please verify the path in validation.scp_path."
                )
        
        # Validate validation.score_method
        if hasattr(config.validation, 'score_method'):
            score_method = config.validation.score_method
            valid_score_methods = [
                'knn',
                'consin_distance',
                'knn_domain_zscore',
                'knn_local_density',
                'knn_domain_local_density',
            ]
            if score_method not in valid_score_methods:
                errors.append(
                    f"Invalid validation.score_method: {score_method}. "
                    f"Expected one of: {valid_score_methods}"
                )
    
    # Raise error if any validation failures
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + 
            "\n".join(f"  - {error}" for error in errors)
        )
    
    print("Configuration validation passed ✓")
