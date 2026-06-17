"""
Compare ch1/ch2 anomaly-score fusion for a fixed BEATs layer.

For each channel, the script extracts embeddings from the selected BEATs layer,
computes per-machine anomaly scores with the existing validation scorer, then
sweeps score fusion weights:

    fused_score = alpha * ch1_score + (1 - alpha) * ch2_score
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.compare_beats_layers import parse_scp
from scripts.inference import load_trained_model
from scripts.validation_inference import ValidationInference
from utils.config_loader import load_config


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ChannelValDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        sample_rate: int,
        max_len: int,
        channel: int,
    ):
        if channel not in (0, 1):
            raise ValueError("channel must be 0 or 1")
        self.paths = paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.channel = channel

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.paths[idx])
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        waveform = waveform[self.channel]
        length = waveform.shape[0]
        if length > self.max_len:
            start = (length - self.max_len) // 2
            waveform = waveform[start:start + self.max_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_len - length))

        waveform = waveform / (waveform.abs().max() + 1e-6)
        return {
            "input": waveform[:self.max_len],
            "label": torch.tensor([self.labels[idx]], dtype=torch.long),
        }


def parse_alphas(spec: str) -> List[float]:
    if ":" in spec:
        start, stop, step = [float(x) for x in spec.split(":")]
        count = int(round((stop - start) / step)) + 1
        values = [start + i * step for i in range(count)]
    else:
        values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    values = sorted(set(round(v, 10) for v in values))
    invalid = [v for v in values if v < 0.0 or v > 1.0]
    if invalid:
        raise ValueError(f"alpha values must be in [0, 1], got {invalid}")
    return values


def extract_channel_embeddings(
    model: torch.nn.Module,
    paths: List[str],
    labels: List[int],
    config: SimpleNamespace,
    device: torch.device,
    layer: int,
    num_encoder_layers: int,
    channel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = ChannelValDataset(
        paths=paths,
        labels=labels,
        sample_rate=config.data.sample_rate,
        max_len=getattr(config.data, "sample_rate", 16000) * 10,
        channel=channel,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=getattr(config.validation, "batch_size", 32),
        shuffle=False,
        num_workers=getattr(config.validation, "num_workers", 4),
        pin_memory=device.type == "cuda",
        timeout=0,
    )

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    target_layer = num_encoder_layers - 1
    features = []
    label_chunks = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting ch{channel + 1} layer {layer}"):
            x = batch["input"].to(device)
            padding_mask = torch.zeros(x.shape, dtype=torch.bool, device=device)
            _, _, layer_results = base_model.beats.extract_features(
                x,
                padding_mask=padding_mask,
                layer=target_layer,
            )
            feature = layer_results[layer][0].transpose(0, 1).mean(dim=1)
            emb = base_model.fc(base_model.bn(feature))
            features.append(emb.cpu())
            label_chunks.append(batch["label"].cpu().numpy())

    return torch.cat(features, dim=0).numpy(), np.concatenate(label_chunks).reshape(-1)


def compute_machine_scores(
    validation: ValidationInference,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_machine_list: List[str],
    train_source_list: List[str],
    val_machine_list: List[str],
) -> Dict[str, Tuple[List[int], np.ndarray]]:
    scores_by_machine = {}
    for machine_name in sorted(set(val_machine_list)):
        train_indices = [i for i, m in enumerate(train_machine_list) if m == machine_name]
        val_indices = [i for i, m in enumerate(val_machine_list) if m == machine_name]
        train_sources = [train_source_list[i] for i in train_indices]
        augmented_train = validation._augment_training_embeddings(
            train_features[train_indices],
            train_sources,
            machine_name,
        )
        scores = validation._compute_anomaly_scores(val_features[val_indices], augmented_train)
        scores_by_machine[machine_name] = (val_indices, scores)
    return scores_by_machine


def evaluate_fusion(
    validation: ValidationInference,
    ch1_scores: Dict[str, Tuple[List[int], np.ndarray]],
    ch2_scores: Dict[str, Tuple[List[int], np.ndarray]],
    val_labels: np.ndarray,
    val_source_list: List[str],
    alphas: List[float],
) -> List[Dict[str, float]]:
    rows = []
    for alpha in alphas:
        per_machine_metrics = []
        for machine_name in sorted(ch1_scores):
            val_indices, scores1 = ch1_scores[machine_name]
            val_indices2, scores2 = ch2_scores[machine_name]
            if val_indices != val_indices2:
                raise RuntimeError(f"Validation indices mismatch for {machine_name}")

            scores = alpha * scores1 + (1.0 - alpha) * scores2
            labels = val_labels[val_indices]
            domains = [val_source_list[i] for i in val_indices]
            per_machine_metrics.append(validation._evaluate_metrics(labels, scores, domains))

        metrics = validation._aggregate_metrics(per_machine_metrics)
        row = {
            "alpha_ch1": alpha,
            "alpha_ch2": 1.0 - alpha,
            "auc": metrics["auc"],
            "pauc": metrics["pauc"],
            "f1": metrics["f1"],
            "source_auc": metrics["source_auc"],
            "source_pauc": metrics["source_pauc"],
            "target_auc": metrics["target_auc"],
            "target_pauc": metrics["target_pauc"],
        }
        rows.append(row)
        logger.info(
            "alpha_ch1=%.3f alpha_ch2=%.3f: AUC=%.6f pAUC=%.6f F1=%.6f",
            row["alpha_ch1"],
            row["alpha_ch2"],
            row["auc"],
            row["pauc"],
            row["f1"],
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ch1/ch2 score fusion")
    parser.add_argument("--config", default="configs/beats_full.yaml")
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints_sweep_mixup_lr_train_scp_ep10/beats_full/"
            "beats_full_trainscp_lr1e-4_wd5e-3_mix035_ep10/model_epoch10.pth"
        ),
    )
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument(
        "--alphas",
        default="0:1:0.05",
        help="Comma list like 0,0.25,0.5,0.75,1 or range start:stop:step.",
    )
    parser.add_argument("--output-dir", default="channel_fusion_results")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    if args.batch_size is not None:
        config.validation.batch_size = args.batch_size
    if args.num_workers is not None:
        config.validation.num_workers = args.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_trained_model(config, args.checkpoint, device)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    num_encoder_layers = len(base_model.beats.encoder.layers)
    if args.layer < 0 or args.layer > num_encoder_layers:
        raise ValueError(f"layer must be in 0..{num_encoder_layers}")

    validation = ValidationInference(model=model, config=config, device=device)
    train_files, train_labels, train_machines, train_sources = parse_scp(config.validation.train_scp_path)
    val_files, val_labels, val_machines, val_sources = parse_scp(config.validation.scp_path)

    channel_features = {}
    channel_val_features = {}
    val_labels_np = None
    for channel in (0, 1):
        logger.info("Extracting channel %d training embeddings", channel + 1)
        channel_features[channel], _ = extract_channel_embeddings(
            model,
            train_files,
            train_labels,
            config,
            device,
            args.layer,
            num_encoder_layers,
            channel,
        )
        logger.info("Extracting channel %d validation embeddings", channel + 1)
        channel_val_features[channel], labels_np = extract_channel_embeddings(
            model,
            val_files,
            val_labels,
            config,
            device,
            args.layer,
            num_encoder_layers,
            channel,
        )
        if val_labels_np is None:
            val_labels_np = labels_np
        elif not np.array_equal(val_labels_np, labels_np):
            raise RuntimeError("Validation labels mismatch between channels")

    logger.info("Computing ch1 anomaly scores")
    ch1_scores = compute_machine_scores(
        validation,
        channel_features[0],
        channel_val_features[0],
        train_machines,
        train_sources,
        val_machines,
    )
    logger.info("Computing ch2 anomaly scores")
    ch2_scores = compute_machine_scores(
        validation,
        channel_features[1],
        channel_val_features[1],
        train_machines,
        train_sources,
        val_machines,
    )

    alphas = parse_alphas(args.alphas)
    rows = evaluate_fusion(
        validation,
        ch1_scores,
        ch2_scores,
        val_labels_np,
        val_sources,
        alphas,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.output_dir, f"ch_fusion_layer{args.layer}_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"ch_fusion_layer{args.layer}_{timestamp}.json")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "config": args.config,
                "layer": args.layer,
                "score_method": config.validation.score_method,
                "alphas": alphas,
                "results": rows,
            },
            f,
            indent=2,
        )

    best_auc = max(rows, key=lambda row: row["auc"])
    best_pauc = max(rows, key=lambda row: row["pauc"])
    logger.info(
        "Best AUC: alpha_ch1=%.3f alpha_ch2=%.3f AUC=%.6f",
        best_auc["alpha_ch1"],
        best_auc["alpha_ch2"],
        best_auc["auc"],
    )
    logger.info(
        "Best pAUC: alpha_ch1=%.3f alpha_ch2=%.3f pAUC=%.6f",
        best_pauc["alpha_ch1"],
        best_pauc["alpha_ch2"],
        best_pauc["pauc"],
    )
    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved JSON: %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
