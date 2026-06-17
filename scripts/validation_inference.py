"""
Validation inference module for automatic model evaluation during training.

This module provides the ValidationInference class for executing validation inference
and computing metrics (AUC, pAUC, F1) after each training epoch. It uses the same
feature extraction and scoring logic as infer_2025.py for consistency.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN
from datetime import datetime
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from utils.audio_dataset import create_val_dataset

logger = logging.getLogger(__name__)


class ValidationInference:
    """Executes validation inference and computes metrics."""
    
    def __init__(
        self,
        model: nn.Module,
        config: SimpleNamespace,
        device: torch.device
    ):
        """
        Initialize validation inference module.
        
        Args:
            model: Trained model instance
            config: Configuration namespace
            device: Device to run inference on
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Extract validation configuration
        self.scp_path = getattr(config.validation, 'scp_path', None)
        self.train_scp_path = getattr(config.validation, 'train_scp_path', None)
        self.score_method = getattr(config.validation, 'score_method', 'consin_distance')
        self.top_k = getattr(config.validation, 'top_k', 1)
        self.n_clusters = getattr(config.validation, 'n_clusters', 10)
        self.local_density_k = getattr(config.validation, 'local_density_k', 16)
        self.local_density_source_k = getattr(config.validation, 'local_density_source_k', 16)
        self.local_density_target_k = getattr(config.validation, 'local_density_target_k', 9)
        self.local_density_scale_mode = getattr(config.validation, 'local_density_scale_mode', 'sum')
        self.local_density_distance = getattr(config.validation, 'local_density_distance', 'l2')
        self.local_density_normalize_embedding = getattr(
            config.validation,
            'local_density_normalize_embedding',
            False
        )
        self.embedding_layer = getattr(config.validation, 'embedding_layer', None)
        self.embedding_layers = getattr(config.validation, 'embedding_layers', None)
        self.layer_ensemble_mode = getattr(config.validation, 'layer_ensemble_mode', 'score_mean')
        self.score_eps = getattr(config.validation, 'score_eps', 1e-8)
        self.batch_size = getattr(config.validation, 'batch_size', 16)
        self.num_workers = getattr(config.validation, 'num_workers', 4)  # DataLoader workers
        self.parallel_machines = getattr(config.validation, 'parallel_machines', False)  # Parallel machine processing
        self.audio_length = getattr(config.data, 'sample_rate', 16000) * 10  # 10 seconds default
        
        # Extract augmentation configuration
        augmentation_config = getattr(config.validation, 'augmentation', None)
        if augmentation_config is not None:
            self.use_adasyn = getattr(augmentation_config, 'use_adasyn', True)
            self.use_mixup = getattr(augmentation_config, 'use_mixup', True)
            self.mixup_alpha = getattr(augmentation_config, 'mixup_alpha', 0.25)
            self.skip_adasyn_machines = getattr(augmentation_config, 'skip_adasyn_machines', ['fan'])
        else:
            # Default values if augmentation config is not present
            self.use_adasyn = True
            self.use_mixup = True
            self.mixup_alpha = 0.25
            self.skip_adasyn_machines = ['fan']
        
        logger.info(
            f"ValidationInference initialized: "
            f"scp_path={self.scp_path}, train_scp_path={self.train_scp_path}, "
            f"score_method={self.score_method}, "
            f"top_k={self.top_k}, n_clusters={self.n_clusters}, "
            f"local_density_k={self.local_density_k}, "
            f"local_density_source_k={self.local_density_source_k}, "
            f"local_density_target_k={self.local_density_target_k}, "
            f"local_density_scale_mode={self.local_density_scale_mode}, "
            f"local_density_distance={self.local_density_distance}, "
            f"local_density_normalize_embedding={self.local_density_normalize_embedding}, "
            f"embedding_layer={self.embedding_layer}, "
            f"embedding_layers={self.embedding_layers}, "
            f"layer_ensemble_mode={self.layer_ensemble_mode}, "
            f"num_workers={self.num_workers}, parallel_machines={self.parallel_machines}"
        )
        logger.info(
            f"Augmentation config: "
            f"use_adasyn={self.use_adasyn}, use_mixup={self.use_mixup}, "
            f"mixup_alpha={self.mixup_alpha}, skip_adasyn_machines={self.skip_adasyn_machines}"
        )
    
    def run_validation(self, epoch: int, max_samples: int = None) -> Dict[str, float]:
        """
        Execute validation inference and compute metrics per machine type.
        
        Args:
            epoch: Current epoch number
            max_samples: Maximum number of samples to process (for testing)
            
        Returns:
            Dictionary containing validation metrics:
            {
                'auc': float,
                'pauc': float,
                'f1': float,
                'source_auc': float,
                'source_pauc': float,
                'target_auc': float,
                'target_pauc': float,
                'timestamp': str
            }
            
        Raises:
            FileNotFoundError: If validation or training SCP file does not exist
            RuntimeError: If validation inference fails
        """
        try:
            logger.info(f"Starting validation inference for epoch {epoch}")
            
            # Load training and validation data from SCP files
            train_file_list, train_label_list, train_machine_list, train_source_list = self._load_training_data()
            val_file_list, val_label_list, val_machine_list, val_source_list = self._load_validation_data()
            
            # Limit samples for testing if specified
            if max_samples is not None and max_samples > 0:
                val_file_list = val_file_list[:max_samples]
                val_label_list = val_label_list[:max_samples]
                val_machine_list = val_machine_list[:max_samples]
                val_source_list = val_source_list[:max_samples]
                logger.info(f"Limited validation to {max_samples} samples for testing")
            
            # Get unique machine types
            machine_names = np.unique(val_machine_list)
            logger.info(f"Processing {len(machine_names)} machine types: {machine_names}")
            
            # Process each machine type independently
            # Use parallel processing if enabled and multiple machines exist
            if self.parallel_machines and len(machine_names) > 1:
                logger.info("Using parallel processing for multiple machines")
                per_machine_metrics = self._process_machines_parallel(
                    machine_names,
                    train_file_list, train_label_list, train_machine_list, train_source_list,
                    val_file_list, val_label_list, val_machine_list, val_source_list
                )
            else:
                logger.info("Using sequential processing for machines")
                per_machine_metrics = self._process_machines_sequential(
                    machine_names,
                    train_file_list, train_label_list, train_machine_list, train_source_list,
                    val_file_list, val_label_list, val_machine_list, val_source_list
                )
            
            # Aggregate metrics across machines using harmonic mean
            logger.info("\nAggregating metrics across all machines...")
            aggregated_metrics = self._aggregate_metrics(per_machine_metrics)
            
            # Create result dictionary
            result = {
                'auc': aggregated_metrics['auc'],
                'pauc': aggregated_metrics['pauc'],
                'f1': aggregated_metrics['f1'],
                'source_auc': aggregated_metrics['source_auc'],
                'source_pauc': aggregated_metrics['source_pauc'],
                'target_auc': aggregated_metrics['target_auc'],
                'target_pauc': aggregated_metrics['target_pauc'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logger.info(
                f"\nValidation complete for epoch {epoch}:\n"
                f"  AUC={result['auc']:.4f}, pAUC={result['pauc']:.4f}, F1={result['f1']:.4f}\n"
                f"  source_AUC={result['source_auc']:.4f}, source_pAUC={result['source_pauc']:.4f}\n"
                f"  target_AUC={result['target_auc']:.4f}, target_pAUC={result['target_pauc']:.4f}"
            )
            
            return result
            
        except FileNotFoundError as e:
            logger.error(f"SCP file not found: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Validation inference failed for epoch {epoch}: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(f"Validation inference failed: {str(e)}")
    
    def _process_machines_sequential(
        self,
        machine_names: np.ndarray,
        train_file_list: list,
        train_label_list: list,
        train_machine_list: list,
        train_source_list: list,
        val_file_list: list,
        val_label_list: list,
        val_machine_list: list,
        val_source_list: list
    ) -> List[Tuple]:
        """
        Process machines sequentially (original implementation).
        
        Returns:
            List of metric tuples for each machine
        """
        per_machine_metrics = []
        
        for machine_name in machine_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing machine: {machine_name}")
            logger.info(f"{'='*60}")
            
            metrics = self._process_single_machine(
                machine_name,
                train_file_list, train_label_list, train_machine_list, train_source_list,
                val_file_list, val_label_list, val_machine_list, val_source_list
            )
            
            per_machine_metrics.append(metrics)
            
            logger.info(
                f"Machine {machine_name} metrics: "
                f"AUC={metrics[0]:.4f}, pAUC={metrics[1]:.4f}, F1={metrics[2]:.4f}, "
                f"source_AUC={metrics[3]:.4f}, source_pAUC={metrics[4]:.4f}, "
                f"target_AUC={metrics[5]:.4f}, target_pAUC={metrics[6]:.4f}"
            )
        
        return per_machine_metrics
    
    def _process_machines_parallel(
        self,
        machine_names: np.ndarray,
        train_file_list: list,
        train_label_list: list,
        train_machine_list: list,
        train_source_list: list,
        val_file_list: list,
        val_label_list: list,
        val_machine_list: list,
        val_source_list: list
    ) -> List[Tuple]:
        """
        Process machines in parallel using multiprocessing.
        
        Note: Due to CUDA limitations, we cannot share GPU models across processes.
        This implementation uses CPU for parallel processing or requires careful GPU management.
        
        Returns:
            List of metric tuples for each machine
        """
        logger.warning(
            "Parallel machine processing is experimental. "
            "GPU models cannot be easily shared across processes. "
            "Consider using sequential processing with increased num_workers instead."
        )
        
        # For now, fall back to sequential processing
        # TODO: Implement proper parallel processing with model serialization
        logger.info("Falling back to sequential processing (parallel not fully implemented)")
        return self._process_machines_sequential(
            machine_names,
            train_file_list, train_label_list, train_machine_list, train_source_list,
            val_file_list, val_label_list, val_machine_list, val_source_list
        )
    
    def _process_single_machine(
        self,
        machine_name: str,
        train_file_list: list,
        train_label_list: list,
        train_machine_list: list,
        train_source_list: list,
        val_file_list: list,
        val_label_list: list,
        val_machine_list: list,
        val_source_list: list
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Process a single machine type.
        
        Returns:
            Tuple of (auc, pauc, f1, auc_source, pauc_source, auc_target, pauc_target)
        """
        # Filter training samples by machine type
        train_indices = [i for i, m in enumerate(train_machine_list) if m == machine_name]
        train_files_machine = [train_file_list[i] for i in train_indices]
        train_labels_machine = [train_label_list[i] for i in train_indices]
        train_sources_machine = [train_source_list[i] for i in train_indices]
        
        # Filter validation samples by machine type
        val_indices = [i for i, m in enumerate(val_machine_list) if m == machine_name]
        val_files_machine = [val_file_list[i] for i in val_indices]
        val_labels_machine = [val_label_list[i] for i in val_indices]
        val_sources_machine = [val_source_list[i] for i in val_indices]
        
        logger.info(
            f"Machine {machine_name}: "
            f"{len(train_files_machine)} training samples, "
            f"{len(val_files_machine)} validation samples"
        )
        
        # Create training dataset and dataloader
        train_dataset = create_val_dataset(
            paths=train_files_machine,
            labels=train_labels_machine,
            sample_rate=self.config.data.sample_rate,
            max_len=self.audio_length
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # Use configured num_workers
            pin_memory=True if self.device.type == 'cuda' else False,
            timeout=0
        )
        
        # Create validation dataset and dataloader
        val_dataset = create_val_dataset(
            paths=val_files_machine,
            labels=val_labels_machine,
            sample_rate=self.config.data.sample_rate,
            max_len=self.audio_length
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # Use configured num_workers
            pin_memory=True if self.device.type == 'cuda' else False,
            timeout=0
        )
        
        if self.embedding_layers:
            scores, val_gt_labels = self._compute_layer_ensemble_scores(
                machine_name,
                train_dataloader,
                val_dataloader,
                train_sources_machine
            )
        else:
            # Extract training features
            logger.info(f"Extracting training features for {machine_name}...")
            train_features, _ = self._extract_features(train_dataloader)
            
            # Extract validation features
            logger.info(f"Extracting validation features for {machine_name}...")
            val_features, val_gt_labels = self._extract_features(val_dataloader)
            
            # Apply ADASYN+Mixup augmentation to training embeddings
            logger.info(f"Augmenting training embeddings for {machine_name}...")
            augmented_train_features = self._augment_training_embeddings(
                train_features,
                train_sources_machine,
                machine_name
            )
            
            # Compute anomaly scores using augmented training embeddings
            logger.info(f"Computing anomaly scores for {machine_name}...")
            scores = self._compute_anomaly_scores(
                val_features,
                augmented_train_features,
                train_sources_machine
            )
        
        # Evaluate per-machine metrics (6 metrics)
        logger.info(f"Evaluating metrics for {machine_name}...")
        metrics = self._evaluate_metrics(
            val_gt_labels,
            scores,
            val_sources_machine
        )
        
        return metrics

    def _compute_layer_ensemble_scores(
        self,
        machine_name: str,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        train_sources_machine: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute anomaly scores from multiple EAT layers."""
        layers = list(self.embedding_layers)
        original_layer = self.embedding_layer
        logger.info(
            f"Using layer ensemble for {machine_name}: "
            f"layers={layers}, mode={self.layer_ensemble_mode}"
        )

        try:
            if self.layer_ensemble_mode == 'score_mean':
                layer_scores = []
                val_gt_labels = None
                for layer in layers:
                    self.embedding_layer = layer
                    logger.info(f"Extracting layer {layer} training features for {machine_name}...")
                    train_features, _ = self._extract_features(train_dataloader)
                    logger.info(f"Extracting layer {layer} validation features for {machine_name}...")
                    val_features, labels = self._extract_features(val_dataloader)
                    if val_gt_labels is None:
                        val_gt_labels = labels

                    logger.info(f"Augmenting layer {layer} training embeddings for {machine_name}...")
                    augmented_train_features = self._augment_training_embeddings(
                        train_features,
                        train_sources_machine,
                        machine_name
                    )
                    logger.info(f"Computing layer {layer} anomaly scores for {machine_name}...")
                    layer_scores.append(
                        self._compute_anomaly_scores(
                            val_features,
                            augmented_train_features,
                            train_sources_machine
                        )
                    )

                return np.mean(np.stack(layer_scores, axis=0), axis=0), val_gt_labels

            if self.layer_ensemble_mode == 'embedding_mean':
                train_feature_layers = []
                val_feature_layers = []
                val_gt_labels = None
                for layer in layers:
                    self.embedding_layer = layer
                    logger.info(f"Extracting layer {layer} training features for {machine_name}...")
                    train_features, _ = self._extract_features(train_dataloader)
                    logger.info(f"Extracting layer {layer} validation features for {machine_name}...")
                    val_features, labels = self._extract_features(val_dataloader)
                    if val_gt_labels is None:
                        val_gt_labels = labels
                    train_feature_layers.append(train_features)
                    val_feature_layers.append(val_features)

                train_features = np.mean(np.stack(train_feature_layers, axis=0), axis=0)
                val_features = np.mean(np.stack(val_feature_layers, axis=0), axis=0)

                logger.info(f"Augmenting averaged training embeddings for {machine_name}...")
                augmented_train_features = self._augment_training_embeddings(
                    train_features,
                    train_sources_machine,
                    machine_name
                )
                logger.info(f"Computing averaged-embedding anomaly scores for {machine_name}...")
                scores = self._compute_anomaly_scores(
                    val_features,
                    augmented_train_features,
                    train_sources_machine
                )
                return scores, val_gt_labels

            raise ValueError(f"Unknown layer_ensemble_mode: {self.layer_ensemble_mode}")
        finally:
            self.embedding_layer = original_layer
    
    def _load_validation_data(self) -> Tuple[list, list, list, list]:
        """
        Load validation data from SCP file.
        
        Returns:
            Tuple of (file_list, label_list, machine_list, source_list)
            
        Raises:
            FileNotFoundError: If SCP file does not exist
        """
        if self.scp_path is None:
            raise ValueError("validation.scp_path not configured")
        
        if not os.path.exists(self.scp_path):
            raise FileNotFoundError(
                f"Validation SCP file not found: {self.scp_path}"
            )
        
        logger.info(f"Loading validation data from: {self.scp_path}")
        
        file_list = []
        label_list = []
        machine_list = []
        source_list = []
        
        # Read SCP file (format: label file_path)
        with open(self.scp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Skipping invalid line: {line}")
                    continue
                
                label_str = parts[0]
                file_path = parts[1]
                
                # Parse label to extract machine type, domain, and status
                # Expected format: machine_section_XX_domain_split_status_XXXX
                label_parts = label_str.split('_')
                
                if len(label_parts) < 6:
                    logger.warning(f"Skipping invalid label format: {label_str}")
                    continue
                
                # Extract components
                machine_type = label_parts[0]  # e.g., "ToyCar"
                
                # Find domain (source/target)
                domain = None
                for part in label_parts:
                    if part in ['source', 'target']:
                        domain = part
                        break
                
                # Find status (normal/anomaly)
                status = None
                for part in label_parts:
                    if part in ['normal', 'anomaly']:
                        status = part
                        break
                
                # Convert status to binary label (0=normal, 1=anomaly)
                if status == 'normal':
                    label = 0
                elif status == 'anomaly':
                    label = 1
                else:
                    logger.warning(f"Unknown status in label: {label_str}, defaulting to normal")
                    label = 0
                
                # Default domain if not found
                if domain is None:
                    domain = 'source'
                
                file_list.append(file_path)
                label_list.append(label)
                machine_list.append(machine_type)
                source_list.append(domain)
        
        logger.info(
            f"Loaded {len(file_list)} samples from validation SCP file "
            f"(machines: {len(set(machine_list))}, "
            f"normal: {label_list.count(0)}, anomaly: {label_list.count(1)})"
        )
        
        return file_list, label_list, machine_list, source_list
    
    def _load_training_data(self) -> Tuple[list, list, list, list]:
        """
        Load training data from SCP file.
        
        Returns:
            Tuple of (file_list, label_list, machine_list, source_list)
            
        Raises:
            FileNotFoundError: If training SCP file does not exist
        """
        if self.train_scp_path is None:
            raise ValueError("validation.train_scp_path not configured")
        
        if not os.path.exists(self.train_scp_path):
            raise FileNotFoundError(
                f"Training SCP file not found: {self.train_scp_path}"
            )
        
        logger.info(f"Loading training data from: {self.train_scp_path}")
        
        file_list = []
        label_list = []
        machine_list = []
        source_list = []
        
        # Read SCP file (format: label file_path)
        # Use same parsing logic as _load_validation_data()
        with open(self.train_scp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Skipping invalid line: {line}")
                    continue
                
                label_str = parts[0]
                file_path = parts[1]
                
                # Parse label to extract machine type, domain, and status
                # Expected format: machine_section_XX_domain_split_status_XXXX
                label_parts = label_str.split('_')
                
                if len(label_parts) < 6:
                    logger.warning(f"Skipping invalid label format: {label_str}")
                    continue
                
                # Extract components
                machine_type = label_parts[0]  # e.g., "ToyCar"
                
                # Find domain (source/target)
                domain = None
                for part in label_parts:
                    if part in ['source', 'target']:
                        domain = part
                        break
                
                # Find status (normal/anomaly)
                status = None
                for part in label_parts:
                    if part in ['normal', 'anomaly']:
                        status = part
                        break
                
                # Convert status to binary label (0=normal, 1=anomaly)
                if status == 'normal':
                    label = 0
                elif status == 'anomaly':
                    label = 1
                else:
                    logger.warning(f"Unknown status in label: {label_str}, defaulting to normal")
                    label = 0
                
                # Default domain if not found
                if domain is None:
                    domain = 'source'
                
                file_list.append(file_path)
                label_list.append(label)
                machine_list.append(machine_type)
                source_list.append(domain)
        
        logger.info(
            f"Loaded {len(file_list)} samples from training SCP file "
            f"(machines: {len(set(machine_list))}, "
            f"normal: {label_list.count(0)}, anomaly: {label_list.count(1)})"
        )
        
        return file_list, label_list, machine_list, source_list
    
    def _extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features using model.
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        self.model.eval()
        
        feature_list = []
        label_list = []
        
        logger.info(f"Starting feature extraction with {len(dataloader)} batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Extracting features')):
                try:
                    if batch_idx % 10 == 0:
                        logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
                    
                    x = batch['input']
                    x = x.to(self.device)
                    y = batch['label']
                    
                    logger.debug(f"Batch {batch_idx}: input shape={x.shape}, label shape={y.shape}")
                    
                    # Create padding mask
                    padding_mask = torch.zeros(x.shape).bool().to(self.device)
                    
                    # Extract embeddings
                    # Handle DataParallel model
                    if isinstance(self.model, nn.DataParallel):
                        if self.embedding_layer is None:
                            emb = self.model.module.extract_embedding(x, padding_mask=padding_mask)
                        else:
                            emb = self.model.module.extract_embedding(
                                x,
                                padding_mask=padding_mask,
                                layer=self.embedding_layer
                            )
                    else:
                        if self.embedding_layer is None:
                            emb = self.model.extract_embedding(x, padding_mask=padding_mask)
                        else:
                            emb = self.model.extract_embedding(
                                x,
                                padding_mask=padding_mask,
                                layer=self.embedding_layer
                            )
                    
                    logger.debug(f"Batch {batch_idx}: embedding shape={emb.shape}")
                    
                    feature_list.append(emb.cpu())
                    label_list.extend(y.cpu().numpy())
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
        
        if len(feature_list) == 0:
            raise RuntimeError("No features extracted - all batches failed")
        
        # Concatenate all features
        features = torch.cat(feature_list, dim=0).numpy()
        labels = np.array(label_list).reshape(-1)
        
        logger.info(f"Extracted features shape: {features.shape}, labels shape: {labels.shape}")
        
        return features, labels
    
    def _augment_training_embeddings(
        self,
        train_embeddings: np.ndarray,
        domain_list: list,
        machine_name: str
    ) -> np.ndarray:
        """
        Apply ADASYN+Mixup augmentation to training embeddings.
        
        Args:
            train_embeddings: Training embeddings array
            domain_list: List of domain labels ('source' or 'target')
            machine_name: Machine type name
            
        Returns:
            Augmented training embeddings
        """
        logger.info(f"Augmenting training embeddings for {machine_name}...")
        logger.info(f"Augmentation enabled: ADASYN={self.use_adasyn}, Mixup={self.use_mixup}")
        
        # Separate embeddings by domain
        domain_array = np.array(domain_list)
        source_indices = np.where(domain_array == 'source')[0]
        target_indices = np.where(domain_array == 'target')[0]
        
        x_train_source = train_embeddings[source_indices]
        x_train_target = train_embeddings[target_indices]
        
        logger.info(
            f"Source samples: {len(x_train_source)}, "
            f"Target samples: {len(x_train_target)}"
        )
        
        # Handle edge case: no source or target samples
        if len(x_train_source) == 0 or len(x_train_target) == 0:
            logger.warning(f"Missing source or target samples for {machine_name}, returning original embeddings")
            return train_embeddings
        
        # === ADASYN Augmentation (可配置) ===
        x_train_target_adasyn = x_train_target
        
        if self.use_adasyn:
            # Check if this machine should skip ADASYN
            skip_adasyn = machine_name in self.skip_adasyn_machines
            
            # Apply ADASYN if source and target counts differ and not in skip list
            if not skip_adasyn and len(x_train_source) != len(x_train_target):
                try:
                    logger.info(f"Applying ADASYN oversampling for {machine_name}...")
                    
                    # Create labels for ADASYN (0=source, 1=target)
                    y_source = np.zeros(x_train_source.shape[0])
                    y_target = np.ones(x_train_target.shape[0])
                    x = np.concatenate([x_train_source, x_train_target], axis=0)
                    y = np.concatenate([y_source, y_target])
                    
                    oversampler = ADASYN(sampling_strategy='auto', random_state=42)
                    x_samp, y_samp = oversampler.fit_resample(x, y)
                    
                    # Extract oversampled target samples
                    x_train_target_adasyn = x_samp[x_train_source.shape[0]:, :]
                    logger.info(f"After ADASYN: {len(x_train_target_adasyn)} target samples")
                except Exception as e:
                    logger.warning(f"ADASYN failed for {machine_name}: {e}, using original target samples")
                    x_train_target_adasyn = x_train_target
            else:
                if skip_adasyn:
                    logger.info(f"Skipping ADASYN for {machine_name} (in skip list)")
                else:
                    logger.info(f"Skipping ADASYN for {machine_name} (already balanced)")
        else:
            logger.info(f"ADASYN disabled by configuration")
        
        # === Mixup Augmentation (可配置) ===
        x_mix = np.array([])
        
        if self.use_mixup:
            logger.info(f"Applying Mixup augmentation with alpha={self.mixup_alpha}")
            
            # Strategy 1: mixup_alpha * source + (1-mixup_alpha) * target (if ADASYN balanced)
            if x_train_target_adasyn.shape[0] == x_train_source.shape[0]:
                logger.info("Using ADASYN-balanced mixup strategy")
                x_mix1 = x_train_source * self.mixup_alpha + (1 - self.mixup_alpha) * x_train_target_adasyn
                
                # Strategy 2: 0.5 source + 0.5 random target
                rand_target_idcs = np.random.randint(0, len(x_train_target), len(x_train_source))
                x_train_target2 = x_train_target[rand_target_idcs]
                x_mix2 = x_train_source * 0.5 + 0.5 * x_train_target2
                
                x_mix = np.concatenate([x_mix1, x_mix2], axis=0)
            else:
                logger.info("Using random target mixup strategy")
                # Only use random target mixup if ADASYN didn't balance
                if len(x_train_target) > 0:
                    rand_target_idcs = np.random.randint(0, len(x_train_target), len(x_train_source))
                    x_train_target2 = x_train_target[rand_target_idcs]
                    x_mix = x_train_source * self.mixup_alpha + (1 - self.mixup_alpha) * x_train_target2
                else:
                    logger.warning(f"No target samples for {machine_name}, skipping mixup")
        else:
            logger.info(f"Mixup disabled by configuration")
        
        # === Concatenate all embeddings ===
        if len(x_mix) > 0:
            emb_train = np.concatenate([x_train_source, x_train_target_adasyn, x_mix], axis=0)
        else:
            emb_train = np.concatenate([x_train_source, x_train_target_adasyn], axis=0)
        
        logger.info(
            f"Final augmented embeddings: {emb_train.shape[0]} total "
            f"(source: {len(x_train_source)}, target: {len(x_train_target_adasyn)}, "
            f"mixup: {len(x_mix)})"
        )
        
        return emb_train
    
    def _compute_anomaly_scores(
        self,
        test_features: np.ndarray,
        train_features: np.ndarray,
        train_source_list: list = None
    ) -> np.ndarray:
        """
        Compute anomaly scores using configured method.
        
        Args:
            test_features: Test feature array
            train_features: Training feature array
            
        Returns:
            Anomaly scores as numpy array
        """
        if self.score_method == 'knn':
            scores = self._knn_anomaly_score(test_features, train_features, self.top_k)
        elif self.score_method == 'knn_domain_zscore':
            scores = self._domain_zscore_anomaly_score(
                test_features,
                train_features,
                train_source_list,
                self.top_k
            )
        elif self.score_method == 'knn_local_density':
            scores = self._local_density_anomaly_score(
                test_features,
                train_features,
                self.local_density_k
            )
        elif self.score_method == 'knn_domain_local_density':
            scores = self._domain_local_density_anomaly_score(
                test_features,
                train_features,
                train_source_list,
                self.local_density_source_k,
                self.local_density_target_k
            )
        elif self.score_method == 'consin_distance':
            scores = self._cluster_anomaly_score(test_features, train_features, self.n_clusters)
        else:
            raise ValueError(f"Unknown score method: {self.score_method}")
        
        return scores
    
    def _knn_anomaly_score(
        self,
        test_features: np.ndarray,
        train_features: np.ndarray,
        top_k: int
    ) -> np.ndarray:
        """
        Compute KNN-based anomaly scores.
        
        Args:
            test_features: Test feature array
            train_features: Training feature array
            top_k: Number of nearest neighbors
            
        Returns:
            Anomaly scores
        """
        # Flatten features
        test_features_flat = test_features.reshape(test_features.shape[0], -1)
        train_features_flat = train_features.reshape(train_features.shape[0], -1)
        
        # Compute distance matrix
        dist_matrix = self._calc_dist_matrix(test_features_flat, train_features_flat, bs=64)
        
        # Get top-k nearest neighbors
        topk_indices = np.argpartition(dist_matrix, top_k, axis=1)[:, :top_k]
        topk_values = np.take_along_axis(dist_matrix, topk_indices, axis=1)
        
        # Return mean distance to top-k neighbors
        return np.mean(topk_values, axis=1)

    def _split_source_target_banks(
        self,
        train_features: np.ndarray,
        train_source_list: list = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split a machine memory bank into source and target domains."""
        train_features_flat = train_features.reshape(train_features.shape[0], -1)
        if train_source_list is None or len(train_source_list) != train_features_flat.shape[0]:
            logger.warning(
                "Missing or mismatched train_source_list for domain-wise scoring; "
                "using the full bank for both source and target."
            )
            return train_features_flat, train_features_flat

        domain_array = np.asarray(train_source_list)
        source_bank = train_features_flat[domain_array == 'source']
        target_bank = train_features_flat[domain_array == 'target']

        if len(source_bank) == 0:
            logger.warning("No source samples found; falling back to full bank for source.")
            source_bank = train_features_flat
        if len(target_bank) == 0:
            logger.warning("No target samples found; falling back to full bank for target.")
            target_bank = train_features_flat

        return source_bank, target_bank

    def _prepare_density_features(self, features: np.ndarray) -> np.ndarray:
        """Flatten embeddings and optionally L2-normalize them for density scoring."""
        features_flat = features.reshape(features.shape[0], -1).astype(np.float64, copy=False)
        if self.local_density_normalize_embedding:
            norms = np.linalg.norm(features_flat, axis=1, keepdims=True)
            features_flat = features_flat / np.maximum(norms, self.score_eps)
        return features_flat

    def _density_distances(self, queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
        """Pairwise distances for normalization-based scoring."""
        if self.local_density_distance == 'l2':
            return euclidean_distances(queries, bank)
        if self.local_density_distance == 'cosine':
            return cosine_distances(queries, bank)
        raise ValueError(f"Unknown local_density_distance: {self.local_density_distance}")

    def _topk_mean_distances(
        self,
        queries: np.ndarray,
        bank: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Mean configured distance from each query to its k nearest samples in bank."""
        k_eff = max(1, min(k, bank.shape[0]))
        dist_matrix = self._density_distances(queries, bank)
        topk_indices = np.argpartition(dist_matrix, k_eff - 1, axis=1)[:, :k_eff]
        topk_values = np.take_along_axis(dist_matrix, topk_indices, axis=1)
        return np.mean(topk_values, axis=1)

    def _intra_domain_knn_scores(
        self,
        bank: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Intra-bank kNN distances for normal samples, excluding each sample itself."""
        if bank.shape[0] <= 1:
            return np.zeros(bank.shape[0], dtype=np.float64)

        k_eff = max(1, min(k + 1, bank.shape[0]))
        dist_matrix = self._density_distances(bank, bank)
        nearest = np.argpartition(dist_matrix, k_eff - 1, axis=1)[:, :k_eff]
        scores = []
        for i, indices in enumerate(nearest):
            indices = indices[indices != i][:k]
            if len(indices) == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.mean(dist_matrix[i, indices])))
        return np.asarray(scores, dtype=np.float64)

    def _local_scales(
        self,
        bank: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Local scale from k nearest in-bank neighbors, excluding self."""
        if bank.shape[0] <= 1:
            return np.full(bank.shape[0], self.score_eps, dtype=np.float64)

        k_eff = max(1, min(k + 1, bank.shape[0]))
        dist_matrix = self._density_distances(bank, bank)
        nearest = np.argpartition(dist_matrix, k_eff - 1, axis=1)[:, :k_eff]
        scales = []
        for i, indices in enumerate(nearest):
            indices = indices[indices != i][:k]
            if len(indices) == 0:
                scale = self.score_eps
            elif self.local_density_scale_mode == 'sum':
                scale = float(np.sum(dist_matrix[i, indices]))
            elif self.local_density_scale_mode == 'mean':
                scale = float(np.mean(dist_matrix[i, indices]))
            else:
                raise ValueError(f"Unknown local_density_scale_mode: {self.local_density_scale_mode}")
            scales.append(max(scale, self.score_eps))
        return np.asarray(scales, dtype=np.float64)

    def _domain_zscore_anomaly_score(
        self,
        test_features: np.ndarray,
        train_features: np.ndarray,
        train_source_list: list,
        top_k: int
    ) -> np.ndarray:
        """Domain-wise z-score normalized kNN anomaly scores."""
        test_features_flat = self._prepare_density_features(test_features)
        source_bank, target_bank = self._split_source_target_banks(train_features, train_source_list)
        source_bank = self._prepare_density_features(source_bank)
        target_bank = self._prepare_density_features(target_bank)

        source_train_scores = self._intra_domain_knn_scores(source_bank, top_k)
        target_train_scores = self._intra_domain_knn_scores(target_bank, top_k)
        mu_s = float(np.mean(source_train_scores))
        mu_t = float(np.mean(target_train_scores))
        sigma_t = max(float(np.std(target_train_scores)), self.score_eps)

        ds = self._topk_mean_distances(test_features_flat, source_bank, top_k)
        dt = self._topk_mean_distances(test_features_flat, target_bank, top_k)
        zs = (ds - mu_s) / sigma_t
        zt = (dt - mu_t) / sigma_t
        return np.minimum(zs, zt)

    def _local_density_anomaly_score(
        self,
        test_features: np.ndarray,
        train_features: np.ndarray,
        local_k: int
    ) -> np.ndarray:
        """Local density normalized kNN anomaly scores using a combined bank."""
        test_features_flat = self._prepare_density_features(test_features)
        train_features_flat = self._prepare_density_features(train_features)
        local_scales = self._local_scales(train_features_flat, local_k)
        dist_matrix = self._density_distances(test_features_flat, train_features_flat)
        normalized = dist_matrix / (local_scales.reshape(1, -1) + self.score_eps)
        return np.min(normalized, axis=1)

    def _domain_local_density_anomaly_score(
        self,
        test_features: np.ndarray,
        train_features: np.ndarray,
        train_source_list: list,
        source_k: int,
        target_k: int
    ) -> np.ndarray:
        """Domain-wise local density normalized kNN anomaly scores."""
        test_features_flat = self._prepare_density_features(test_features)
        source_bank, target_bank = self._split_source_target_banks(train_features, train_source_list)
        source_bank = self._prepare_density_features(source_bank)
        target_bank = self._prepare_density_features(target_bank)

        source_scales = self._local_scales(source_bank, source_k)
        target_scales = self._local_scales(target_bank, target_k)

        source_dists = self._density_distances(test_features_flat, source_bank)
        target_dists = self._density_distances(test_features_flat, target_bank)
        source_scores = np.min(source_dists / (source_scales.reshape(1, -1) + self.score_eps), axis=1)
        target_scores = np.min(target_dists / (target_scales.reshape(1, -1) + self.score_eps), axis=1)
        return np.minimum(source_scores, target_scores)
    
    def _cluster_anomaly_score(
        self,
        test_features: np.ndarray,
        train_features: np.ndarray,
        n_clusters: int = 10
    ) -> np.ndarray:
        """
        Compute cluster-based anomaly scores using cosine distance.
        
        Args:
            test_features: Test feature array
            train_features: Training feature array
            n_clusters: Number of clusters
            
        Returns:
            Anomaly scores
        """
        # Flatten features
        train_features_flat = train_features.reshape(train_features.shape[0], -1)
        test_features_flat = test_features.reshape(test_features.shape[0], -1)
        
        # Fit KMeans on training features
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(train_features_flat)
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        
        # Compute cosine distances to all centers
        cos_distances = cosine_distances(test_features_flat, centers)
        
        # Return minimum distance to any center
        return np.min(cos_distances, axis=1)
    
    def _calc_dist_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bs: int = 16
    ) -> np.ndarray:
        """
        Calculate Euclidean distance matrix in batches.
        
        Args:
            x: First feature array
            y: Second feature array
            bs: Batch size
            
        Returns:
            Distance matrix
        """
        if bs > x.shape[0]:
            bs = x.shape[0]
        
        dist_matrices = np.zeros((x.shape[0], y.shape[0]))
        
        for i in range(y.shape[0]):
            dist_x_list = np.zeros(x.shape[0])
            x_batch = x.shape[0] // bs + (1 if x.shape[0] % bs != 0 else 0)
            
            for j in range(x_batch):
                x_i = x[j*bs:j*bs+bs]
                # Compute Euclidean distance
                dist_matrix = euclidean_distances(x_i, y[i].reshape(1, -1))
                dist_x_list[j*bs:j*bs+bs] = dist_matrix.reshape(-1)
            
            dist_matrices[:, i] = dist_x_list
        
        return dist_matrices
    
    def _evaluate_metrics(
        self,
        gt_list: np.ndarray,
        scores: np.ndarray,
        domain_list: list = None
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Evaluate AUC, pAUC, F1, and per-domain metrics.
        
        Args:
            gt_list: Ground truth labels (0=normal, 1=anomaly)
            scores: Anomaly scores
            domain_list: List of domain labels ('source' or 'target'), optional
            
        Returns:
            Tuple of (auc, pauc, f1, auc_source, pauc_source, auc_target, pauc_target)
        """
        gt_list = np.asarray(gt_list)
        
        # Compute overall ROC curve and AUC
        fpr, tpr, _ = roc_curve(gt_list, scores)
        auc = roc_auc_score(gt_list, scores)
        
        # Compute overall partial AUC (max_fpr=0.1)
        pauc = roc_auc_score(gt_list, scores, max_fpr=0.1)
        
        # Compute overall precision-recall curve and F1
        precision, recall, thresholds = precision_recall_curve(gt_list, scores)
        f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
        f1 = np.max(f1_scores)
        
        # Compute per-domain metrics if domain_list is provided
        if domain_list is not None:
            domain_array = np.array(domain_list)
            
            # Filter by source domain
            source_indices = np.where(domain_array == 'source')[0]
            if len(source_indices) > 0 and len(np.unique(gt_list[source_indices])) > 1:
                auc_source = roc_auc_score(gt_list[source_indices], scores[source_indices])
                pauc_source = roc_auc_score(gt_list[source_indices], scores[source_indices], max_fpr=0.1)
            else:
                logger.warning("Not enough source samples or only one class, setting source metrics to 0.5")
                auc_source = 0.5
                pauc_source = 0.5
            
            # Filter by target domain
            target_indices = np.where(domain_array == 'target')[0]
            if len(target_indices) > 0 and len(np.unique(gt_list[target_indices])) > 1:
                auc_target = roc_auc_score(gt_list[target_indices], scores[target_indices])
                pauc_target = roc_auc_score(gt_list[target_indices], scores[target_indices], max_fpr=0.1)
            else:
                logger.warning("Not enough target samples or only one class, setting target metrics to 0.5")
                auc_target = 0.5
                pauc_target = 0.5
        else:
            # If no domain list provided, set per-domain metrics to overall metrics
            auc_source = auc
            pauc_source = pauc
            auc_target = auc
            pauc_target = pauc
        
        return auc, pauc, f1, auc_source, pauc_source, auc_target, pauc_target
    
    def _aggregate_metrics(
        self,
        per_machine_metrics: list
    ) -> Dict[str, float]:
        """
        Aggregate per-machine metrics using harmonic mean.
        
        Args:
            per_machine_metrics: List of metric tuples from each machine
                Each tuple: (auc, pauc, f1, auc_source, pauc_source, auc_target, pauc_target)
            
        Returns:
            Dictionary with aggregated metrics
        """
        from scipy.stats import hmean
        
        # Collect metrics in lists
        aucs = []
        paucs = []
        f1s = []
        aucs_source = []
        paucs_source = []
        aucs_target = []
        paucs_target = []
        
        for metrics in per_machine_metrics:
            auc, pauc, f1, auc_source, pauc_source, auc_target, pauc_target = metrics
            aucs.append(auc)
            paucs.append(pauc)
            f1s.append(f1)
            aucs_source.append(auc_source)
            paucs_source.append(pauc_source)
            aucs_target.append(auc_target)
            paucs_target.append(pauc_target)
        
        # Compute harmonic mean for each metric
        mean_auc = float(hmean(aucs))
        mean_pauc = float(hmean(paucs))
        mean_f1 = float(hmean(f1s))
        mean_auc_source = float(hmean(aucs_source))
        mean_pauc_source = float(hmean(paucs_source))
        mean_auc_target = float(hmean(aucs_target))
        mean_pauc_target = float(hmean(paucs_target))
        
        return {
            'auc': mean_auc,
            'pauc': mean_pauc,
            'f1': mean_f1,
            'source_auc': mean_auc_source,
            'source_pauc': mean_pauc_source,
            'target_auc': mean_auc_target,
            'target_pauc': mean_pauc_target
        }
