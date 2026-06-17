"""
Checkpoint utilities for managing model checkpoints with full state restoration.

This module provides the CheckpointManager class for saving and loading training
checkpoints with support for model weights, optimizer state, scheduler state, and
training metadata. It enables seamless training resumption from any saved epoch.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving and loading with full state restoration."""
    
    def __init__(self, save_dir: str, config: SimpleNamespace):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save/load checkpoints
            config: Configuration namespace
        """
        self.save_dir = save_dir
        self.config = config
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"CheckpointManager initialized with save_dir: {save_dir}")
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        loss: float,
        accuracy: float,
        num_classes: int,
        label_encoder_classes: list,
        config: SimpleNamespace
    ) -> str:
        """
        Save checkpoint with full training state.
        
        Args:
            epoch: Current epoch number
            model: Model instance (handles DataParallel)
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler instance
            loss: Current loss value
            accuracy: Current accuracy value
            num_classes: Number of classification classes
            label_encoder_classes: List of label encoder classes
            config: Configuration namespace
            
        Returns:
            Path to saved checkpoint file
        """
        # Handle DataParallel model state extraction
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        # Extract critical training configuration for inference consistency
        training_config = self._extract_training_config(config)
        
        # Build checkpoint dictionary with all state information
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss': loss,
            'accuracy': accuracy,
            'num_classes': num_classes,
            'label_encoder_classes': label_encoder_classes,
            # Don't save config to avoid pickle issues - it's not needed for resumption
            'config': None,
            # Save critical training configuration for inference consistency
            'training_config': training_config
        }
        
        # Add model-specific metadata for LoRA
        if hasattr(config, 'finetuning') and hasattr(config.finetuning, 'strategy') and config.finetuning.strategy == 'lora':
            if hasattr(config, 'model') and hasattr(config.model, 'type'):
                if config.model.type == 'beats' and hasattr(config.model, 'checkpoint_path'):
                    checkpoint['base_model_path'] = config.model.checkpoint_path
                elif config.model.type == 'ast' and hasattr(config.model, 'checkpoint_path'):
                    checkpoint['base_model_path'] = config.model.checkpoint_path
                elif config.model.type in ['ast', 'eat'] and hasattr(config.model, 'model_name'):
                    checkpoint['base_model_name'] = config.model.model_name
                elif config.model.type == 'dasheng' and hasattr(config.model, 'model_path'):
                    checkpoint['base_model_path'] = config.model.model_path
        
        # Save checkpoint to disk
        checkpoint_path = self.get_checkpoint_path(epoch)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path} (epoch={epoch}, loss={loss:.4f}, accuracy={accuracy:.4f})")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            epoch: Epoch number to load
            model: Model instance to load weights into
            optimizer: Optimizer instance to load state into
            scheduler: Scheduler instance to load state into
            
        Returns:
            Dictionary containing checkpoint metadata
            
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            RuntimeError: If checkpoint file is corrupted
        """
        checkpoint_path = self.get_checkpoint_path(epoch)
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file for epoch {epoch} not found at {checkpoint_path}"
            )
        
        try:
            # Load checkpoint file
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Restore model state (handle DataParallel)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler state (backward compatibility)
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                logger.warning(
                    f"Checkpoint from epoch {epoch} does not contain scheduler state. "
                    f"Initializing scheduler from current epoch."
                )
            
            logger.info(
                f"Checkpoint loaded: {checkpoint_path} "
                f"(epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss', 0):.4f}, "
                f"accuracy={checkpoint.get('accuracy', 0):.4f})"
            )
            
            return checkpoint
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}: {str(e)}"
            )
    
    def checkpoint_exists(self, epoch: int) -> bool:
        """
        Check if checkpoint file exists for given epoch.
        
        Args:
            epoch: Epoch number to check
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        checkpoint_path = self.get_checkpoint_path(epoch)
        return os.path.exists(checkpoint_path)
    
    def get_checkpoint_path(self, epoch: int) -> str:
        """
        Get path to checkpoint file for given epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Path to checkpoint file
        """
        return os.path.join(self.save_dir, f"model_epoch{epoch}.pth")
    
    def _config_to_dict(self, config: SimpleNamespace) -> Optional[Dict[str, Any]]:
        """
        Convert SimpleNamespace config to dictionary.
        
        Args:
            config: Configuration namespace
            
        Returns:
            Dictionary representation of config, or None if conversion fails
        """
        if hasattr(config, 'to_dict'):
            return config.to_dict()
        
        try:
            return self._namespace_to_dict(config)
        except Exception as e:
            logger.warning(f"Failed to convert config to dict: {e}")
            return None
    
    def _namespace_to_dict(self, ns: SimpleNamespace) -> Dict[str, Any]:
        """
        Recursively convert SimpleNamespace to dictionary.
        
        Args:
            ns: SimpleNamespace object
            
        Returns:
            Dictionary representation
        """
        if isinstance(ns, SimpleNamespace):
            result = {}
            for k, v in vars(ns).items():
                # Skip unpicklable objects like functions
                if callable(v) and not isinstance(v, type):
                    continue
                try:
                    result[k] = self._namespace_to_dict(v)
                except Exception:
                    # Skip items that can't be converted
                    continue
            return result
        elif isinstance(ns, list):
            return [self._namespace_to_dict(item) for item in ns]
        elif isinstance(ns, (str, int, float, bool, type(None))):
            return ns
        else:
            # For other types, try to convert to string
            try:
                return str(ns)
            except Exception:
                return None
    
    def _extract_training_config(self, config: SimpleNamespace) -> Dict[str, Any]:
        """
        Extract critical training configuration parameters for inference consistency.
        
        This ensures that inference uses the same preprocessing and model configuration
        as training, preventing subtle bugs from configuration mismatches.
        
        Args:
            config: Configuration namespace
            
        Returns:
            Dictionary containing critical training configuration
        """
        training_config = {}
        
        # Data preprocessing configuration
        if hasattr(config, 'data'):
            training_config['sample_rate'] = getattr(config.data, 'sample_rate', 16000)
            # Calculate audio_length from sample_rate (10 seconds default)
            training_config['audio_length'] = training_config['sample_rate'] * 10
        
        # Model configuration
        if hasattr(config, 'model'):
            training_config['model_type'] = getattr(config.model, 'type', None)
            training_config['feature_dim'] = getattr(config.model, 'feature_dim', 768)
            training_config['dropout'] = getattr(config.model, 'dropout', 0.1)
            
            # Model-specific paths/names
            if hasattr(config.model, 'checkpoint_path'):
                training_config['model_checkpoint_path'] = config.model.checkpoint_path
            if hasattr(config.model, 'model_path'):
                training_config['model_path'] = config.model.model_path
            if hasattr(config.model, 'model_size'):
                training_config['model_size'] = config.model.model_size
            if hasattr(config.model, 'model_name'):
                training_config['model_name'] = config.model.model_name
        
        # Finetuning strategy configuration
        if hasattr(config, 'finetuning'):
            training_config['finetuning_strategy'] = getattr(config.finetuning, 'strategy', None)
            
            # LoRA-specific configuration
            if training_config['finetuning_strategy'] == 'lora' and hasattr(config.finetuning, 'lora'):
                training_config['lora'] = {
                    'rank': getattr(config.finetuning.lora, 'rank', 8),
                    'alpha': getattr(config.finetuning.lora, 'alpha', 16),
                    'dropout': getattr(config.finetuning.lora, 'dropout', 0.1),
                    'target_modules': getattr(config.finetuning.lora, 'target_modules', [])
                }
        
        logger.debug(f"Extracted training config: {training_config}")
        
        return training_config
