"""
独立推理脚本 - 使用训练好的模型进行推理

使用方法:
    python inference.py --config inference_config.yaml
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config_loader import load_config
from utils.model_factory import create_model
from utils.checkpoint_utils import CheckpointManager
from scripts.validation_inference import ValidationInference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_config_consistency(config, checkpoint):
    """
    验证推理配置与训练配置的一致性
    
    Args:
        config: 推理配置对象
        checkpoint: checkpoint字典
        
    Raises:
        ValueError: 如果配置不一致
    """
    if 'training_config' not in checkpoint:
        logger.warning(
            "⚠️  Checkpoint does not contain training_config. "
            "Cannot verify consistency with training configuration. "
            "This checkpoint was created with an older version."
        )
        return
    
    train_cfg = checkpoint['training_config']
    errors = []
    warnings = []
    
    # 验证sample_rate
    if hasattr(config, 'data') and 'sample_rate' in train_cfg:
        if config.data.sample_rate != train_cfg['sample_rate']:
            errors.append(
                f"Sample rate mismatch! "
                f"Training: {train_cfg['sample_rate']}, "
                f"Inference: {config.data.sample_rate}"
            )
    
    # 验证model_type
    if hasattr(config, 'model') and 'model_type' in train_cfg:
        if config.model.type != train_cfg['model_type']:
            errors.append(
                f"Model type mismatch! "
                f"Training: {train_cfg['model_type']}, "
                f"Inference: {config.model.type}"
            )
    
    # 验证feature_dim
    if hasattr(config, 'model') and 'feature_dim' in train_cfg:
        if config.model.feature_dim != train_cfg['feature_dim']:
            errors.append(
                f"Feature dimension mismatch! "
                f"Training: {train_cfg['feature_dim']}, "
                f"Inference: {config.model.feature_dim}"
            )
    
    # 验证dropout
    if hasattr(config, 'model') and 'dropout' in train_cfg:
        if config.model.dropout != train_cfg['dropout']:
            warnings.append(
                f"Dropout mismatch (usually OK for inference). "
                f"Training: {train_cfg['dropout']}, "
                f"Inference: {config.model.dropout}"
            )
    
    # 验证finetuning_strategy
    if hasattr(config, 'finetuning') and 'finetuning_strategy' in train_cfg:
        if config.finetuning.strategy != train_cfg['finetuning_strategy']:
            errors.append(
                f"Finetuning strategy mismatch! "
                f"Training: {train_cfg['finetuning_strategy']}, "
                f"Inference: {config.finetuning.strategy}"
            )
    
    # 如果是LoRA，验证LoRA配置
    if (hasattr(config, 'finetuning') and 
        config.finetuning.strategy == 'lora' and 
        'lora' in train_cfg):
        
        train_lora = train_cfg['lora']
        infer_lora = config.finetuning.lora
        
        if infer_lora.rank != train_lora['rank']:
            errors.append(
                f"LoRA rank mismatch! "
                f"Training: {train_lora['rank']}, "
                f"Inference: {infer_lora.rank}"
            )
        
        if infer_lora.alpha != train_lora['alpha']:
            errors.append(
                f"LoRA alpha mismatch! "
                f"Training: {train_lora['alpha']}, "
                f"Inference: {infer_lora.alpha}"
            )
        
        if infer_lora.target_modules != train_lora['target_modules']:
            errors.append(
                f"LoRA target_modules mismatch! "
                f"Training: {train_lora['target_modules']}, "
                f"Inference: {infer_lora.target_modules}"
            )
    
    # 打印警告
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  ⚠️  {warning}")
    
    # 如果有错误，抛出异常
    if errors:
        error_msg = "Configuration consistency check failed:\n"
        for error in errors:
            error_msg += f"  ❌ {error}\n"
        error_msg += "\nPlease ensure your inference config matches the training config."
        raise ValueError(error_msg)
    
    logger.info("✅ Configuration consistency check passed")


def load_trained_model(config, checkpoint_path: str, device: torch.device):
    """
    加载训练好的模型
    
    Args:
        config: 配置对象
        checkpoint_path: checkpoint文件路径
        device: 设备
        
    Returns:
        model: 加载好的模型
        checkpoint: checkpoint字典（包含元数据）
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 验证配置一致性
    try:
        validate_config_consistency(config, checkpoint)
    except ValueError as e:
        logger.error(str(e))
        raise
    
    # 获取模型配置
    num_classes = checkpoint.get('num_classes', config.model.num_classes)
    logger.info(f"Model has {num_classes} classes")
    
    # 创建模型
    model, strategy = create_model(config, num_classes)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 兼容旧格式
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # 打印checkpoint信息
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        logger.info(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    if 'accuracy' in checkpoint:
        logger.info(f"Checkpoint accuracy: {checkpoint['accuracy']:.4f}")
    
    return model, checkpoint


def run_inference(config, checkpoint_path: str, output_dir: str = None):
    """
    运行推理
    
    Args:
        config: 配置对象
        checkpoint_path: checkpoint文件路径
        output_dir: 输出目录（可选）
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载模型
    model, checkpoint = load_trained_model(config, checkpoint_path, device)
    
    # 创建ValidationInference实例
    logger.info("Initializing validation inference...")
    validation_inference = ValidationInference(
        model=model,
        config=config,
        device=device
    )
    
    # 运行验证推理
    logger.info("Running inference...")
    metrics = validation_inference.run_validation(epoch=0)
    
    # 打印结果
    logger.info("\n" + "="*60)
    logger.info("Inference Results:")
    logger.info("="*60)
    logger.info(f"Overall Metrics:")
    logger.info(f"  AUC:  {metrics['auc']:.4f}")
    logger.info(f"  pAUC: {metrics['pauc']:.4f}")
    logger.info(f"  F1:   {metrics['f1']:.4f}")
    logger.info(f"\nSource Domain:")
    logger.info(f"  AUC:  {metrics['source_auc']:.4f}")
    logger.info(f"  pAUC: {metrics['source_pauc']:.4f}")
    logger.info(f"\nTarget Domain:")
    logger.info(f"  AUC:  {metrics['target_auc']:.4f}")
    logger.info(f"  pAUC: {metrics['target_pauc']:.4f}")
    logger.info("="*60)
    
    # 保存结果
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成结果文件名
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        result_file = os.path.join(output_dir, f"inference_results_{checkpoint_name}_{timestamp}.txt")
        
        # 保存到文件
        with open(result_file, 'w') as f:
            f.write(f"Inference Results\n")
            f.write(f"="*60 + "\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Timestamp: {metrics['timestamp']}\n")
            f.write(f"\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  AUC:  {metrics['auc']:.4f}\n")
            f.write(f"  pAUC: {metrics['pauc']:.4f}\n")
            f.write(f"  F1:   {metrics['f1']:.4f}\n")
            f.write(f"\n")
            f.write(f"Source Domain:\n")
            f.write(f"  AUC:  {metrics['source_auc']:.4f}\n")
            f.write(f"  pAUC: {metrics['source_pauc']:.4f}\n")
            f.write(f"\n")
            f.write(f"Target Domain:\n")
            f.write(f"  AUC:  {metrics['target_auc']:.4f}\n")
            f.write(f"  pAUC: {metrics['target_pauc']:.4f}\n")
        
        logger.info(f"\nResults saved to: {result_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help="Path to inference configuration file"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help="Path to checkpoint file (overrides config)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help="Output directory for results (overrides config)"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # 获取checkpoint路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = config.inference.checkpoint_path
    
    # 获取输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = getattr(config.inference, 'output_dir', 'inference_results')
    
    # 运行推理
    try:
        metrics = run_inference(config, checkpoint_path, output_dir)
        logger.info("\nInference completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"\nInference failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
