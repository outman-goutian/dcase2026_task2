"""
统一的音频数据集类，用于训练和验证

这个模块提供了一个统一的AudioDataset类，确保训练和验证使用相同的数据预处理逻辑。
"""

import torch
import torchaudio
import random
from typing import Union, List
import numpy as np


class AudioDataset(torch.utils.data.Dataset):
    """
    统一的音频数据集类
    
    用于训练和验证，支持两种模式：
    - 训练模式：随机裁剪、随机padding
    - 验证模式：中心裁剪、固定padding
    
    Args:
        paths: 音频文件路径列表
        labels: 标签列表（可以是one-hot编码的稀疏矩阵或普通列表）
        sample_rate: 目标采样率
        max_len: 音频最大长度（采样点数）
        mode: 'train' 或 'val'，控制数据增强行为
    """
    
    def __init__(
        self, 
        paths: List[str], 
        labels: Union[List, np.ndarray], 
        sample_rate: int = 16000, 
        max_len: int = 160000,
        mode: str = 'train',
        channel_indices: List[int] = None,
        train_channel_mode: str = 'mix',
        val_channel_mode: str = 'ch1',
        channel_mix_alpha_min: float = 0.0,
        channel_mix_alpha_max: float = 0.5
    ):
        self.paths = paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.mode = mode
        self.channel_indices = channel_indices
        self.train_channel_mode = train_channel_mode
        self.val_channel_mode = val_channel_mode
        self.channel_mix_alpha_min = channel_mix_alpha_min
        self.channel_mix_alpha_max = channel_mix_alpha_max
        
        # 检测labels的类型
        self.is_sparse = hasattr(labels, 'toarray')  # 稀疏矩阵
        
    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 重采样
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # ===== 保证双通道 =====
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        ch1, ch2 = waveform[0], waveform[1]

        # ===== train / val 不同策略 =====
        sample_channel = None if self.channel_indices is None else self.channel_indices[idx]
        if sample_channel is not None:
            waveform = ch1 if sample_channel == 0 else ch2
        else:
            channel_mode = self.train_channel_mode if self.mode == 'train' else self.val_channel_mode

            if channel_mode == 'mix':
                alpha = random.uniform(self.channel_mix_alpha_min, self.channel_mix_alpha_max)
                waveform = ch1 + alpha * ch2
            elif channel_mode == 'ch1':
                waveform = ch1
            elif channel_mode == 'ch2':
                waveform = ch2
            elif channel_mode == 'random_single':
                waveform = ch1 if random.random() < 0.5 else ch2
            else:
                raise ValueError(f"Unsupported channel mode: {channel_mode}")

        # ✅ 一定放在这里（所有分支之后）
        length = waveform.shape[0]

        # ===== 裁剪 / padding =====
        if length > self.max_len:
            if self.mode == 'train':
                start = random.randint(0, length - self.max_len)
            else:
                start = (length - self.max_len) // 2
            waveform = waveform[start:start + self.max_len]
        else:
            pad_len = self.max_len - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # ===== normalize（最后做）=====
        waveform = waveform / (waveform.abs().max() + 1e-6)
        waveform = waveform[:self.max_len]
        
        # 处理标签
        if self.is_sparse:
            # 稀疏矩阵（训练时的one-hot编码）
            label = torch.tensor(self.labels[idx].toarray(), dtype=torch.float)
            label = label.squeeze(0)
        else:
            # 普通列表（验证时的整数标签）
            label = torch.tensor([self.labels[idx]], dtype=torch.long)
        
        # 返回格式统一
        if self.mode == 'train':
            # 训练模式：返回元组（兼容现有train.py）
            return waveform, label
        else:
            # 验证模式：返回字典（兼容validation_inference.py）
            return {
                "input": waveform,
                "label": label
            }
    
    def __len__(self):
        return len(self.paths)


def create_train_dataset(
    paths,
    labels,
    sample_rate=16000,
    max_len=160000,
    channel_indices=None,
    train_channel_mode='mix',
    channel_mix_alpha_min=0.0,
    channel_mix_alpha_max=0.5
):
    """
    创建训练数据集的便捷函数
    
    Args:
        paths: 音频文件路径列表
        labels: one-hot编码的标签（稀疏矩阵）
        sample_rate: 采样率
        max_len: 最大长度
        
    Returns:
        AudioDataset实例（训练模式）
    """
    return AudioDataset(
        paths=paths,
        labels=labels,
        sample_rate=sample_rate,
        max_len=max_len,
        mode='train',
        channel_indices=channel_indices,
        train_channel_mode=train_channel_mode,
        channel_mix_alpha_min=channel_mix_alpha_min,
        channel_mix_alpha_max=channel_mix_alpha_max
    )


def create_val_dataset(paths, labels, sample_rate=16000, max_len=160000):
    """
    创建验证数据集的便捷函数
    
    Args:
        paths: 音频文件路径列表
        labels: 整数标签列表
        sample_rate: 采样率
        max_len: 最大长度
        
    Returns:
        AudioDataset实例（验证模式）
    """
    return AudioDataset(
        paths=paths,
        labels=labels,
        sample_rate=sample_rate,
        max_len=max_len,
        mode='val'
    )
