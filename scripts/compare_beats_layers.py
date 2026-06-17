"""
Compare BEATs intermediate layer embeddings on validation inference.

The script selects the best checkpoint from metrics CSV files by default, extracts
pre-encoder plus all transformer-layer embeddings in one BEATs forward pass, and
evaluates each layer with the existing anomaly scoring/metric code.
"""

import argparse
import csv
import glob
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.inference import load_trained_model
from scripts.validation_inference import ValidationInference
from utils.audio_dataset import create_val_dataset
from utils.config_loader import load_config


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(
    metrics_glob: str,
    metric: str,
    checkpoint_roots: List[str],
) -> Tuple[str, Dict[str, str]]:
    candidates = []
    for metrics_path in glob.glob(metrics_glob, recursive=True):
        with open(metrics_path, newline="") as f:
            for row in csv.DictReader(f):
                if not row or metric not in row:
                    continue
                try:
                    value = float(row[metric])
                    epoch = int(row["epoch"])
                except (KeyError, TypeError, ValueError):
                    continue
                candidates.append((value, epoch, metrics_path, row))

    if not candidates:
        raise RuntimeError(f"No metric rows with '{metric}' found under {metrics_glob}")

    reverse = "loss" not in metric.lower()
    value, epoch, metrics_path, row = sorted(candidates, key=lambda x: x[0], reverse=reverse)[0]
    exp_tag = os.path.basename(os.path.dirname(metrics_path))
    checkpoint_path = None
    checked_paths = []
    for checkpoint_root in checkpoint_roots:
        path = os.path.join(checkpoint_root, exp_tag, f"model_epoch{epoch}.pth")
        checked_paths.append(path)
        if os.path.exists(path):
            checkpoint_path = path
            break
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Best metrics point to a checkpoint that was not found. Checked:\n"
            + "\n".join(f"  {path}" for path in checked_paths)
        )

    info = dict(row)
    info.update(
        {
            "selected_metric": metric,
            "selected_metric_value": value,
            "metrics_path": metrics_path,
            "experiment_tag": exp_tag,
            "epoch": epoch,
        }
    )
    return checkpoint_path, info


def parse_scp(path: str) -> Tuple[List[str], List[int], List[str], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    file_list, label_list, machine_list, source_list = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                logger.warning("Skipping invalid SCP line: %s", line)
                continue

            label_str, file_path = parts[0], parts[1]
            label_parts = label_str.split("_")
            if len(label_parts) < 6:
                logger.warning("Skipping invalid label format: %s", label_str)
                continue

            machine = label_parts[0]
            domain = next((p for p in label_parts if p in ("source", "target")), "source")
            status = next((p for p in label_parts if p in ("normal", "anomaly")), "normal")
            label = 1 if status == "anomaly" else 0

            file_list.append(file_path)
            label_list.append(label)
            machine_list.append(machine)
            source_list.append(domain)

    return file_list, label_list, machine_list, source_list


def parse_layers(spec: str, max_layer: int) -> List[int]:
    if spec == "all":
        return list(range(max_layer + 1))

    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))

    layers = sorted(set(layers))
    invalid = [layer for layer in layers if layer < 0 or layer > max_layer]
    if invalid:
        raise ValueError(f"Invalid layers {invalid}; valid range is 0..{max_layer}")
    return layers


def extract_layer_embeddings(
    model: torch.nn.Module,
    paths: List[str],
    labels: List[int],
    config: SimpleNamespace,
    device: torch.device,
    layers: List[int],
    num_encoder_layers: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    dataset = create_val_dataset(
        paths=paths,
        labels=labels,
        sample_rate=config.data.sample_rate,
        max_len=getattr(config.data, "sample_rate", 16000) * 10,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=getattr(config.validation, "batch_size", 32),
        shuffle=False,
        num_workers=getattr(config.validation, "num_workers", 4),
        pin_memory=device.type == "cuda",
        timeout=0,
    )

    feature_chunks = {layer: [] for layer in layers}
    label_chunks = []
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    target_layer = num_encoder_layers - 1

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting layer embeddings"):
            x = batch["input"].to(device)
            y = batch["label"].cpu().numpy()
            padding_mask = torch.zeros(x.shape, dtype=torch.bool, device=device)

            _, _, layer_results = base_model.beats.extract_features(
                x,
                padding_mask=padding_mask,
                layer=target_layer,
            )
            if len(layer_results) != num_encoder_layers + 1:
                raise RuntimeError(
                    f"Expected {num_encoder_layers + 1} layer outputs, got {len(layer_results)}"
                )

            for layer in layers:
                feature = layer_results[layer][0].transpose(0, 1).mean(dim=1)
                emb = base_model.fc(base_model.bn(feature))
                feature_chunks[layer].append(emb.cpu())

            label_chunks.append(y)

    features = {
        layer: torch.cat(chunks, dim=0).numpy()
        for layer, chunks in feature_chunks.items()
    }
    labels_np = np.concatenate(label_chunks).reshape(-1)
    return features, labels_np


def evaluate_layers(
    validation: ValidationInference,
    train_features_by_layer: Dict[int, np.ndarray],
    val_features_by_layer: Dict[int, np.ndarray],
    val_labels: np.ndarray,
    train_machine_list: List[str],
    train_source_list: List[str],
    val_machine_list: List[str],
    val_source_list: List[str],
) -> List[Dict[str, float]]:
    rows = []
    machine_names = sorted(set(val_machine_list))

    for layer in sorted(val_features_by_layer):
        logger.info("Evaluating layer %s", layer)
        per_machine_metrics = []
        for machine_name in machine_names:
            train_indices = [i for i, m in enumerate(train_machine_list) if m == machine_name]
            val_indices = [i for i, m in enumerate(val_machine_list) if m == machine_name]

            train_features = train_features_by_layer[layer][train_indices]
            val_features = val_features_by_layer[layer][val_indices]
            labels = val_labels[val_indices]
            val_sources = [val_source_list[i] for i in val_indices]
            train_sources = [train_source_list[i] for i in train_indices]

            augmented_train = validation._augment_training_embeddings(
                train_features,
                train_sources,
                machine_name,
            )
            scores = validation._compute_anomaly_scores(val_features, augmented_train)
            per_machine_metrics.append(validation._evaluate_metrics(labels, scores, val_sources))

        metrics = validation._aggregate_metrics(per_machine_metrics)
        row = {
            "layer": layer,
            "layer_name": "pre_encoder" if layer == 0 else f"encoder_{layer}",
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
            "Layer %s: AUC=%.6f pAUC=%.6f F1=%.6f",
            row["layer_name"],
            row["auc"],
            row["pauc"],
            row["f1"],
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare BEATs layer embeddings")
    parser.add_argument("--config", default="configs/beats_full.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--metrics-glob", default="runs/**/*.csv")
    parser.add_argument("--select-metric", default="val_auc")
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=[
            "checkpoints_beats_full",
            "checkpoints_lr_sweep_train_scp/beats_full",
            "checkpoints_sweep_train_scp_ep10/beats_full",
            "checkpoints_sweep_mixup_lr_train_scp_ep10/beats_full",
        ],
        help="Checkpoint root to search. Can be passed multiple times.",
    )
    parser.add_argument("--layers", default="all", help="all, comma list, or range like 0,6,12")
    parser.add_argument("--output-dir", default="layer_inference_results")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    if args.batch_size is not None:
        config.validation.batch_size = args.batch_size
    if args.num_workers is not None:
        config.validation.num_workers = args.num_workers

    selected = {}
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path, selected = find_best_checkpoint(
            args.metrics_glob,
            args.select_metric,
            args.checkpoint_root,
        )
        logger.info(
            "Selected checkpoint: %s (%s=%s, metrics=%s)",
            checkpoint_path,
            selected["selected_metric"],
            selected["selected_metric_value"],
            selected["metrics_path"],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(config, checkpoint_path, device)
    num_encoder_layers = len((model.module if isinstance(model, torch.nn.DataParallel) else model).beats.encoder.layers)
    layers = parse_layers(args.layers, num_encoder_layers)

    validation = ValidationInference(model=model, config=config, device=device)
    train_files, train_labels, train_machines, train_sources = parse_scp(config.validation.train_scp_path)
    val_files, val_labels, val_machines, val_sources = parse_scp(config.validation.scp_path)

    logger.info("Extracting train embeddings for %d layers", len(layers))
    train_features, _ = extract_layer_embeddings(
        model, train_files, train_labels, config, device, layers, num_encoder_layers
    )
    logger.info("Extracting validation embeddings for %d layers", len(layers))
    val_features, val_labels_np = extract_layer_embeddings(
        model, val_files, val_labels, config, device, layers, num_encoder_layers
    )

    rows = evaluate_layers(
        validation,
        train_features,
        val_features,
        val_labels_np,
        train_machines,
        train_sources,
        val_machines,
        val_sources,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    csv_path = os.path.join(args.output_dir, f"beats_layer_compare_{stem}_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"beats_layer_compare_{stem}_{timestamp}.json")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = {
        "checkpoint": checkpoint_path,
        "selected": selected,
        "config": args.config,
        "score_method": config.validation.score_method,
        "train_scp_path": config.validation.train_scp_path,
        "val_scp_path": config.validation.scp_path,
        "layers": layers,
        "results": rows,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    best = max(rows, key=lambda row: row["auc"])
    logger.info("Best layer by AUC: %s AUC=%.6f", best["layer_name"], best["auc"])
    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved JSON: %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
