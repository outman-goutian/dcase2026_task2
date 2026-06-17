import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
import torchaudio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config_loader import load_config
from utils.config_validator import validate_config
from utils.model_factory import create_model
from utils.checkpoint_utils import CheckpointManager
from scripts.validation_inference import ValidationInference
from utils.audio_dataset import create_train_dataset
import os
import random
import logging
import traceback
import torch.backends.cudnn as cudnn

# ===================== Mixup 函数 =====================

def mixup_data(x, y, alpha=0.4):
    """
    实现 mixup 数据增强
    x: 输入数据
    y: 标签
    alpha: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b
    return mixed_x, mixed_y

# ===================== 数据读取 =====================

def load_txt(file_path):
    """
    从 txt 文件中加载路径和标签
    每行：label \t path
    """
    data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'path'])
    paths = data['path'].tolist()
    labels = data['label'].tolist()
    return paths, labels


def prepare_training_samples(paths, labels, config):
    """
    Prepare training sample list with optional per-channel label expansion.

    Returns:
        Tuple of (paths, labels, channel_indices)
    """
    duplicate_channel2 = getattr(config.data, 'duplicate_channel2_as_new_class', False)
    if not duplicate_channel2:
        return paths, labels, None

    suffix = getattr(config.data, 'channel_label_suffix', '__ch2')
    ch2_labels = [f"{label}{suffix}" for label in labels]

    expanded_paths = list(paths) + list(paths)
    expanded_labels = list(labels) + ch2_labels
    channel_indices = [0] * len(paths) + [1] * len(paths)

    return expanded_paths, expanded_labels, channel_indices


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

# ===================== 主流程 =====================

def train_epoch(model, train_loader, optimizer, criterion, device, config, strategy, log_file=None, epoch=None):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        config: 配置
        strategy: 微调策略
        log_file: 日志文件
        epoch: 当前epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (audio, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        audio, labels = audio.to(device), labels.to(device)
        padding_mask = (audio == 0)
        
        # 应用 Mixup
        audio, labels = mixup_data(audio, labels, alpha=config.training.mixup_alpha)
         
        optimizer.zero_grad()
        outputs = model(audio, labels, padding_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            strategy.get_trainable_parameters(model), 
            max_norm=config.training.max_grad_norm
        )
        
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
        total += labels.size(0)
        
        # 记录每个batch的损失
        if log_file is not None and batch_idx % config.logging.log_interval == 0:
            log_file.write(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}\n")

    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    
    # 记录每个epoch的损失和准确率
    if log_file is not None:
        log_file.write(f"Epoch {epoch+1}, Loss: {avg_loss}, Accuracy: {acc}\n")

    return avg_loss, acc

def main(config_path: str = None):
    """
    主训练函数
    
    Args:
        config_path: 配置文件路径
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # 加载配置
    print("Loading configuration...")
    config = load_config(config_path)

    seed = getattr(config.training, 'seed', None)
    if seed is not None:
        print(f"Setting random seed: {seed}")
        set_random_seed(seed)
    
    # 验证配置
    print("Validating configuration...")
    validate_config(config)
    
    # Validate resume_epoch parameter
    resume_epoch = getattr(getattr(config, 'checkpoint', None), 'resume_epoch', None)
    if resume_epoch is not None and not isinstance(resume_epoch, int):
        raise ValueError(
            f"Invalid resume_epoch value: {resume_epoch}. "
            f"Expected null, empty, or positive integer."
        )
    if resume_epoch is not None and resume_epoch <= 0:
        raise ValueError(
            f"Invalid resume_epoch value: {resume_epoch}. "
            f"Expected positive integer."
        )

    # 加载训练数据
    print(f"Loading training data from {config.data.train_txt}...")
    train_paths, train_labels = load_txt(config.data.train_txt)
    train_paths, train_labels, channel_indices = prepare_training_samples(
        train_paths, train_labels, config
    )

    # 编码标签
    label_encoder = LabelEncoder()
    train_encoded = label_encoder.fit_transform(train_labels)

    # one-hot 编码
    onehot_encoder = OneHotEncoder()
    train_onehot = onehot_encoder.fit_transform(train_encoded.reshape(-1, 1))

    # 构造数据集和 DataLoader
    train_dataset = create_train_dataset(
        paths=train_paths, 
        labels=train_onehot, 
        sample_rate=config.data.sample_rate,
        channel_indices=channel_indices,
        train_channel_mode=getattr(config.data, 'train_channel_mode', 'mix'),
        channel_mix_alpha_min=getattr(config.data, 'channel_mix_alpha_min', 0.0),
        channel_mix_alpha_max=getattr(config.data, 'channel_mix_alpha_max', 0.5)
    )
    
    # 调整batch size for multi-GPU
    if isinstance(config.training.gpu_id, list) and len(config.training.gpu_id) > 1:
        original_batch_size = config.data.batch_size
        config.data.batch_size = config.data.batch_size * len(config.training.gpu_id)
        print(f"Multi-GPU training: adjusted batch size from {original_batch_size} to {config.data.batch_size}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers, 
        pin_memory=config.data.pin_memory
    )

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    num_classes = len(label_encoder.classes_)
    print(f"\nCreating model with {num_classes} classes...")
    model, strategy = create_model(config, num_classes)
    
    # 多GPU支持
    if isinstance(config.training.gpu_id, list) and len(config.training.gpu_id) > 1:
        print(f"Using DataParallel with GPUs: {config.training.gpu_id}")
        model = nn.DataParallel(model, device_ids=config.training.gpu_id)
    
    model = model.to(device)

    # 损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 优化器 - 使用strategy获取可训练参数
    if isinstance(model, nn.DataParallel):
        trainable_params = strategy.get_trainable_parameters(model.module)
    else:
        trainable_params = strategy.get_trainable_parameters(model)
    
    optimizer = optim.AdamW(
        trainable_params, 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.training.scheduler.step_size, 
        gamma=config.training.scheduler.gamma
    )
    
    # 获取实验标签并创建带tag的目录
    experiment_tag = getattr(config.logging, 'tag', 'default')
    save_dir_with_tag = os.path.join(config.logging.save_dir, experiment_tag)
    log_dir_with_tag = os.path.join(config.logging.log_dir, experiment_tag)
    
    # 创建目录
    os.makedirs(save_dir_with_tag, exist_ok=True)
    os.makedirs(log_dir_with_tag, exist_ok=True)
    
    # Initialize checkpoint manager with tagged directory
    checkpoint_manager = CheckpointManager(
        save_dir=save_dir_with_tag,
        config=config
    )
    
    # Determine starting epoch
    start_epoch = 0
    
    # Load checkpoint if resuming
    if resume_epoch is not None:
        logger.info(f"Resuming training from epoch {resume_epoch}")
        
        # Check if checkpoint exists
        if not checkpoint_manager.checkpoint_exists(resume_epoch):
            raise FileNotFoundError(
                f"Checkpoint file for epoch {resume_epoch} not found at "
                f"{checkpoint_manager.get_checkpoint_path(resume_epoch)}"
            )
        
        try:
            # Load checkpoint and restore state
            checkpoint = checkpoint_manager.load_checkpoint(
                epoch=resume_epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler
            )
            
            # Set starting epoch to resume from next epoch
            start_epoch = resume_epoch
            
            logger.info(
                f"Successfully loaded checkpoint from epoch {resume_epoch}. "
                f"Resuming training from epoch {start_epoch + 1}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
    
    # Initialize validation inference if enabled
    validation_inference = None
    validation_enabled = getattr(getattr(config, 'validation', None), 'enabled', False)
    
    if validation_enabled:
        try:
            validation_inference = ValidationInference(
                model=model,
                config=config,
                device=device
            )
            logger.info("Validation inference enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize validation inference: {str(e)}")
            logger.warning("Training will continue without validation")
            validation_inference = None

    # 创建更具描述性的日志文件名
    # 格式: training_YYYYMMDD-HHMM_model_strategy.log
    # 例如: training_20260412-1152_beats_lora.log
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    model_type = config.model.type
    strategy_name = config.finetuning.strategy
    log_file_path = f"{log_dir_with_tag}/training_{current_time}_{model_type}_{strategy_name}.log"
    log_file = open(log_file_path, 'w')
    
    # 创建CSV文件保存训练和验证指标
    metrics_csv_path = f"{log_dir_with_tag}/metrics_{current_time}_{model_type}_{strategy_name}.csv"
    metrics_csv = open(metrics_csv_path, 'w')
    metrics_csv.write("epoch,train_loss,train_acc,val_auc,val_pauc,val_f1,val_source_auc,val_source_pauc,val_target_auc,val_target_pauc\n")
    
    logger.info(f"Experiment tag: {experiment_tag}")
    logger.info(f"Log directory: {log_dir_with_tag}")
    logger.info(f"Save directory: {save_dir_with_tag}")
    logger.info(f"Training log: {log_file_path}")
    logger.info(f"Metrics CSV: {metrics_csv_path}")
    
    log_file.write(f"Experiment tag: {experiment_tag}\n")
    log_file.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Model type: {config.model.type}\n")
    log_file.write(f"Finetuning strategy: {config.finetuning.strategy}\n")
    log_file.write(f"Number of classes: {num_classes}\n")
    log_file.write(f"Number of epochs: {config.training.num_epochs}\n\n")
    if seed is not None:
        log_file.write(f"Random seed: {seed}\n\n")

    # 训练循环 (单阶段)
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch + 1}")
    print("=" * 60)
    
    for epoch in range(start_epoch, config.training.num_epochs):
        loss, acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config, 
            strategy, log_file=log_file, epoch=epoch
        )
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config.training.num_epochs} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        # 按间隔保存模型
        if (epoch + 1) % config.training.save_interval == 0:
            try:
                # Use checkpoint manager to save checkpoint
                save_path = checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=loss,
                    accuracy=acc,
                    num_classes=num_classes,
                    label_encoder_classes=label_encoder.classes_.tolist(),
                    config=config
                )
                print(f"Model checkpoint saved at {save_path}")
                log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved: {save_path}\n")
            except Exception as e:
                logger.error(f"Failed to save checkpoint for epoch {epoch+1}: {str(e)}")
                log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint save failed: {str(e)}\n")
        
        # Run validation inference after each epoch
        if validation_inference is not None:
            try:
                logger.info(f"Running validation inference for epoch {epoch+1}")
                metrics = validation_inference.run_validation(epoch=epoch+1)
                
                # Log validation metrics
                validation_log = (
                    f"[{metrics['timestamp']}] Epoch {epoch+1} Validation - "
                    f"AUC: {metrics['auc']:.4f}, pAUC: {metrics['pauc']:.4f}, F1: {metrics['f1']:.4f}, "
                    f"source_AUC: {metrics['source_auc']:.4f}, source_pAUC: {metrics['source_pauc']:.4f}, "
                    f"target_AUC: {metrics['target_auc']:.4f}, target_pAUC: {metrics['target_pauc']:.4f}"
                )
                print(validation_log)
                log_file.write(validation_log + "\n")
                log_file.flush()
                
                # Write metrics to CSV
                metrics_csv.write(
                    f"{epoch+1},{loss:.6f},{acc:.6f},"
                    f"{metrics['auc']:.6f},{metrics['pauc']:.6f},{metrics['f1']:.6f},"
                    f"{metrics['source_auc']:.6f},{metrics['source_pauc']:.6f},"
                    f"{metrics['target_auc']:.6f},{metrics['target_pauc']:.6f}\n"
                )
                metrics_csv.flush()
                
            except Exception as e:
                error_msg = (
                    f"Validation inference failed for epoch {epoch+1}: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                logger.error(error_msg)
                log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
                log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Continuing training despite validation failure\n")
                log_file.flush()
        else:
            # If validation is disabled, still write training metrics to CSV
            metrics_csv.write(f"{epoch+1},{loss:.6f},{acc:.6f},,,,,,,\n")
            metrics_csv.flush()

    # 保存最终模型
    try:
        final_save_path = checkpoint_manager.save_checkpoint(
            epoch=config.training.num_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            accuracy=acc,
            num_classes=num_classes,
            label_encoder_classes=label_encoder.classes_.tolist(),
            config=config
        )
        print(f"\nFinal model saved at {final_save_path}")
        log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final model saved: {final_save_path}\n")
    except Exception as e:
        logger.error(f"Failed to save final model: {str(e)}")
        log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final model save failed: {str(e)}\n")
    
    log_file.write(f"\nTraining completed at {datetime.now().strftime('%Y%m%d-%H%M%S')}\n")
    log_file.close()
    metrics_csv.close()
    print("=" * 60)
    print("Training completed successfully!")
    print(f"Training log saved to: {log_file_path}")
    print(f"Metrics CSV saved to: {metrics_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train audio classification model")
    parser.add_argument('--config_file', type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    main(args.config_file)
