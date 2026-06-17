"""
Create DCASE challenge submission files for EAT final inference.

Default scoring mirrors the best EAT validation setting in this repo:
  top1 checkpoint, EAT layers 8/10/12, score_mean ensemble,
  domain-wise local density, sum scale, L2 distance, Ks=16, Kt=9.
"""

import argparse
import csv
import json
import logging
import os
import sys
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import hmean
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.inference import load_trained_model
from utils.audio_dataset import create_val_dataset
from utils.config_loader import load_config


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_scp_with_section(path: str) -> Tuple[List[str], List[int], List[str], List[str], List[str]]:
    files = []
    labels = []
    machines = []
    domains = []
    sections = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid SCP line: {line}")

            label_str, file_path = parts[0], parts[1]
            label_parts = label_str.split("_")
            if len(label_parts) < 3:
                raise ValueError(f"Invalid label in SCP: {label_str}")

            machine = label_parts[0]
            domain = next((part for part in label_parts if part in ("source", "target")), "source")
            status = next((part for part in label_parts if part in ("normal", "anomaly")), "normal")
            label = 1 if status == "anomaly" else 0

            section = "section_00"
            for idx in range(len(label_parts) - 1):
                if label_parts[idx] == "section":
                    section = f"section_{label_parts[idx + 1]}"
                    break

            files.append(file_path)
            labels.append(label)
            machines.append(machine)
            domains.append(domain)
            sections.append(section)

    return files, labels, machines, domains, sections


def extract_embeddings(
    model: nn.Module,
    files: List[str],
    labels: List[int],
    config,
    device: torch.device,
    layer: int,
) -> np.ndarray:
    audio_length = getattr(config.data, "sample_rate", 16000) * 10
    dataset = create_val_dataset(
        paths=files,
        labels=labels,
        sample_rate=config.data.sample_rate,
        max_len=audio_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=getattr(config.validation, "batch_size", 32),
        shuffle=False,
        num_workers=getattr(config.validation, "num_workers", 2),
        pin_memory=device.type == "cuda",
        timeout=0,
    )

    model.eval()
    feature_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting layer {layer}"):
            x = batch["input"].to(device)
            padding_mask = torch.zeros(x.shape).bool().to(device)
            if isinstance(model, nn.DataParallel):
                emb = model.module.extract_embedding(x, padding_mask=padding_mask, layer=layer)
            else:
                emb = model.extract_embedding(x, padding_mask=padding_mask, layer=layer)
            feature_list.append(emb.cpu())

    if not feature_list:
        raise RuntimeError("No embeddings extracted")
    return torch.cat(feature_list, dim=0).numpy()


def prepare_density_features(
    features: np.ndarray,
    normalize_embedding: bool,
    eps: float,
) -> np.ndarray:
    features_flat = features.reshape(features.shape[0], -1).astype(np.float64, copy=False)
    if normalize_embedding:
        norms = np.linalg.norm(features_flat, axis=1, keepdims=True)
        features_flat = features_flat / np.maximum(norms, eps)
    return features_flat


def density_distances(queries: np.ndarray, bank: np.ndarray, distance: str) -> np.ndarray:
    if distance == "l2":
        return euclidean_distances(queries, bank)
    if distance == "cosine":
        return cosine_distances(queries, bank)
    raise ValueError(f"Unknown local density distance: {distance}")


def local_scales(
    bank: np.ndarray,
    k: int,
    scale_mode: str,
    distance: str,
    eps: float,
) -> np.ndarray:
    if bank.shape[0] <= 1:
        return np.full(bank.shape[0], eps, dtype=np.float64)

    k_eff = max(1, min(k + 1, bank.shape[0]))
    dist_matrix = density_distances(bank, bank, distance)
    nearest = np.argpartition(dist_matrix, k_eff - 1, axis=1)[:, :k_eff]
    scales = []
    for i, indices in enumerate(nearest):
        indices = indices[indices != i][:k]
        if len(indices) == 0:
            scale = eps
        elif scale_mode == "sum":
            scale = float(np.sum(dist_matrix[i, indices]))
        elif scale_mode == "mean":
            scale = float(np.mean(dist_matrix[i, indices]))
        else:
            raise ValueError(f"Unknown local density scale mode: {scale_mode}")
        scales.append(max(scale, eps))
    return np.asarray(scales, dtype=np.float64)


def domain_local_density_scores(
    query_features: np.ndarray,
    train_features: np.ndarray,
    train_sources: List[str],
    source_k: int,
    target_k: int,
    scale_mode: str,
    distance: str,
    normalize_embedding: bool,
    eps: float,
    exclude_self_prefix: bool = False,
) -> np.ndarray:
    query_flat = prepare_density_features(query_features, normalize_embedding, eps)
    train_flat = prepare_density_features(train_features, normalize_embedding, eps)
    domain_array = np.asarray(train_sources)
    source_indices = np.where(domain_array == "source")[0]
    target_indices = np.where(domain_array == "target")[0]
    if len(source_indices) == 0:
        source_indices = np.arange(len(train_flat))
    if len(target_indices) == 0:
        target_indices = np.arange(len(train_flat))

    source_bank = train_flat[source_indices]
    target_bank = train_flat[target_indices]
    source_scales = local_scales(source_bank, source_k, scale_mode, distance, eps)
    target_scales = local_scales(target_bank, target_k, scale_mode, distance, eps)
    source_dists = density_distances(query_flat, source_bank, distance)
    target_dists = density_distances(query_flat, target_bank, distance)

    if exclude_self_prefix:
        source_pos = {idx: pos for pos, idx in enumerate(source_indices)}
        target_pos = {idx: pos for pos, idx in enumerate(target_indices)}
        for query_idx in range(min(len(query_flat), len(train_flat))):
            if query_idx in source_pos:
                source_dists[query_idx, source_pos[query_idx]] = np.inf
            if query_idx in target_pos:
                target_dists[query_idx, target_pos[query_idx]] = np.inf

    source_scores = np.min(source_dists / (source_scales.reshape(1, -1) + eps), axis=1)
    target_scores = np.min(target_dists / (target_scales.reshape(1, -1) + eps), axis=1)
    return np.minimum(source_scores, target_scores)


def knn_scores(
    query_features: np.ndarray,
    train_features: np.ndarray,
    top_k: int,
    exclude_self_prefix: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    scores = []
    query_flat = query_features.reshape(query_features.shape[0], -1)
    train_flat = train_features.reshape(train_features.shape[0], -1)
    for start in range(0, len(query_flat), batch_size):
        end = min(start + batch_size, len(query_flat))
        distances = pairwise_distances(query_flat[start:end], train_flat, metric="euclidean")
        if exclude_self_prefix:
            rows = np.arange(end - start)
            cols = np.arange(start, end)
            valid = cols < train_flat.shape[0]
            distances[rows[valid], cols[valid]] = np.inf
        nearest = np.partition(distances, kth=top_k - 1, axis=1)[:, :top_k]
        scores.append(nearest.mean(axis=1))
    return np.concatenate(scores)


def compute_scores(
    query_features: np.ndarray,
    train_features: np.ndarray,
    train_sources: List[str],
    args: argparse.Namespace,
    exclude_self_prefix: bool = False,
) -> np.ndarray:
    if args.score_method == "knn":
        return knn_scores(
            query_features,
            train_features,
            top_k=args.top_k,
            exclude_self_prefix=exclude_self_prefix,
        )
    if args.score_method in ("knn_domain_local_density", "domain_local_density_sum"):
        return domain_local_density_scores(
            query_features,
            train_features,
            train_sources,
            source_k=args.local_density_source_k,
            target_k=args.local_density_target_k,
            scale_mode=args.local_density_scale_mode,
            distance=args.local_density_distance,
            normalize_embedding=args.local_density_normalize_embedding,
            eps=args.score_eps,
            exclude_self_prefix=exclude_self_prefix,
        )
    raise ValueError(f"Unknown score method: {args.score_method}")


def write_two_column_csv(path: str, names: List[str], values: List[object]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for name, value in zip(names, values):
            writer.writerow([name, value])


def parse_layers(value: str) -> List[int]:
    layers = [int(item) for item in value.replace(",", " ").split()]
    if not layers:
        raise ValueError("At least one layer is required")
    return layers


def compute_machine_score_mean(
    model: nn.Module,
    config,
    device: torch.device,
    train_files: List[str],
    train_labels: List[int],
    train_sources: List[str],
    query_files: List[str],
    query_labels: List[int],
    layers: List[int],
    args: argparse.Namespace,
    compute_train_scores: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    layer_query_scores = []
    layer_train_scores = []
    for layer in layers:
        logger.info("extracting layer %d train embeddings", layer)
        train_features = extract_embeddings(
            model,
            train_files,
            train_labels,
            config,
            device,
            layer,
        )
        logger.info("extracting layer %d query embeddings", layer)
        query_features = extract_embeddings(
            model,
            query_files,
            query_labels,
            config,
            device,
            layer,
        )

        layer_query_scores.append(
            compute_scores(
                query_features,
                train_features,
                train_sources,
                args,
                exclude_self_prefix=False,
            )
        )
        if compute_train_scores:
            layer_train_scores.append(
                compute_scores(
                    train_features,
                    train_features,
                    train_sources,
                    args,
                    exclude_self_prefix=True,
                )
            )

    query_scores = np.mean(np.stack(layer_query_scores, axis=0), axis=0)
    if compute_train_scores:
        train_scores = np.mean(np.stack(layer_train_scores, axis=0), axis=0)
    else:
        train_scores = np.asarray([], dtype=np.float64)
    return query_scores, train_scores


def evaluate_metrics(
    labels: List[int],
    scores: np.ndarray,
    domains: List[str],
) -> Tuple[float, float, float, float, float, float, float]:
    gt = np.asarray(labels)
    auc = roc_auc_score(gt, scores)
    pauc = roc_auc_score(gt, scores, max_fpr=0.1)
    precision, recall, _ = precision_recall_curve(gt, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
    f1 = float(np.max(f1_scores))

    domain_array = np.asarray(domains)
    source_indices = np.where(domain_array == "source")[0]
    target_indices = np.where(domain_array == "target")[0]
    if len(source_indices) > 0 and len(np.unique(gt[source_indices])) > 1:
        source_auc = roc_auc_score(gt[source_indices], scores[source_indices])
        source_pauc = roc_auc_score(gt[source_indices], scores[source_indices], max_fpr=0.1)
    else:
        source_auc = 0.5
        source_pauc = 0.5
    if len(target_indices) > 0 and len(np.unique(gt[target_indices])) > 1:
        target_auc = roc_auc_score(gt[target_indices], scores[target_indices])
        target_pauc = roc_auc_score(gt[target_indices], scores[target_indices], max_fpr=0.1)
    else:
        target_auc = 0.5
        target_pauc = 0.5

    return (
        float(auc),
        float(pauc),
        float(f1),
        float(source_auc),
        float(source_pauc),
        float(target_auc),
        float(target_pauc),
    )


def aggregate_metrics(per_machine_metrics: List[Tuple[float, ...]]) -> Dict[str, float]:
    values = np.asarray(per_machine_metrics, dtype=np.float64)
    return {
        "auc": float(hmean(values[:, 0])),
        "pauc": float(hmean(values[:, 1])),
        "f1": float(hmean(values[:, 2])),
        "source_auc": float(hmean(values[:, 3])),
        "source_pauc": float(hmean(values[:, 4])),
        "target_auc": float(hmean(values[:, 5])),
        "target_pauc": float(hmean(values[:, 6])),
    }


def run_dev_check(
    model: nn.Module,
    config,
    device: torch.device,
    train_files: List[str],
    train_labels: List[int],
    train_machines: List[str],
    train_sources: List[str],
    dev_scp: str,
    layers: List[int],
    args: argparse.Namespace,
    output_dir: str,
) -> Dict[str, object]:
    dev_files, dev_labels, dev_machines, dev_sources, _ = parse_scp_with_section(dev_scp)
    per_machine = {}
    metric_tuples = []

    logger.info("Running dev check on %s", dev_scp)
    for machine_name in sorted(set(dev_machines)):
        logger.info("%s: dev-check scoring", machine_name)
        train_indices = [i for i, machine in enumerate(train_machines) if machine == machine_name]
        dev_indices = [i for i, machine in enumerate(dev_machines) if machine == machine_name]
        if not train_indices:
            raise RuntimeError(f"No training samples found for dev machine {machine_name}")

        scores, _ = compute_machine_score_mean(
            model=model,
            config=config,
            device=device,
            train_files=[train_files[i] for i in train_indices],
            train_labels=[train_labels[i] for i in train_indices],
            train_sources=[train_sources[i] for i in train_indices],
            query_files=[dev_files[i] for i in dev_indices],
            query_labels=[dev_labels[i] for i in dev_indices],
            layers=layers,
            args=args,
            compute_train_scores=False,
        )
        file_names = [os.path.basename(dev_files[i]) for i in dev_indices]
        score_path = os.path.join(output_dir, f"anomaly_score_{machine_name}_section_00_test.csv")
        write_two_column_csv(score_path, file_names, [f"{score:.12g}" for score in scores])
        metrics = evaluate_metrics(
            labels=[dev_labels[i] for i in dev_indices],
            scores=scores,
            domains=[dev_sources[i] for i in dev_indices],
        )
        metric_tuples.append(metrics)
        per_machine[machine_name] = {
            "auc": metrics[0],
            "pauc": metrics[1],
            "f1": metrics[2],
            "source_auc": metrics[3],
            "source_pauc": metrics[4],
            "target_auc": metrics[5],
            "target_pauc": metrics[6],
            "anomaly_score_file": os.path.basename(score_path),
        }
        logger.info(
            "%s dev metrics: AUC=%.4f pAUC=%.4f F1=%.4f source_AUC=%.4f "
            "source_pAUC=%.4f target_AUC=%.4f target_pAUC=%.4f",
            machine_name,
            *metrics,
        )

    aggregate = aggregate_metrics(metric_tuples)
    logger.info(
        "Dev aggregate: AUC=%.4f pAUC=%.4f F1=%.4f source_AUC=%.4f "
        "source_pAUC=%.4f target_AUC=%.4f target_pAUC=%.4f",
        aggregate["auc"],
        aggregate["pauc"],
        aggregate["f1"],
        aggregate["source_auc"],
        aggregate["source_pauc"],
        aggregate["target_auc"],
        aggregate["target_pauc"],
    )
    return {"aggregate": aggregate, "machines": per_machine}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create final DCASE submission files for EAT")
    parser.add_argument(
        "--config",
        default="configs/layer_sweep/eat_top1_ensemble/layers8_10_12_score_mean.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints_lr_sweep_train_scp/eat_base_full/"
            "eat_base_full_best_trainscp_lr4e-5_ep8/model_epoch3.pth"
        ),
    )
    parser.add_argument("--test-scp", default="/workspace/data/test_final.scp")
    parser.add_argument("--dev-scp", default=None)
    parser.add_argument("--train-scp", default=None)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--layers", default="8,10,12")
    parser.add_argument(
        "--layer-ensemble-mode",
        default="score_mean",
        choices=["score_mean"],
        help="Only score_mean is used for the selected best setting.",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--score-method",
        default="knn_domain_local_density",
        choices=["knn", "knn_domain_local_density", "domain_local_density_sum"],
    )
    parser.add_argument("--local-density-source-k", type=int, default=16)
    parser.add_argument("--local-density-target-k", type=int, default=9)
    parser.add_argument("--local-density-scale-mode", default="sum", choices=["sum", "mean"])
    parser.add_argument("--local-density-distance", default="l2", choices=["l2", "cosine"])
    parser.add_argument("--local-density-normalize-embedding", action="store_true")
    parser.add_argument("--score-eps", type=float, default=1e-8)
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--output-dir", default="final_submission_eat")
    parser.add_argument("--team-name", default="team_exp_diffpt_eat_layers8_10_12_score_mean")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.team_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(args.config)
    if args.train_scp is not None:
        config.validation.train_scp_path = args.train_scp
    if args.batch_size is not None:
        config.validation.batch_size = args.batch_size
    if args.num_workers is not None:
        config.validation.num_workers = args.num_workers

    train_scp = config.validation.train_scp_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_trained_model(config, args.checkpoint, device)

    train_files, train_labels, train_machines, train_sources, _ = parse_scp_with_section(train_scp)
    dev_result = None
    if args.dev_scp is not None:
        dev_result = run_dev_check(
            model=model,
            config=config,
            device=device,
            train_files=train_files,
            train_labels=train_labels,
            train_machines=train_machines,
            train_sources=train_sources,
            dev_scp=args.dev_scp,
            layers=layers,
            args=args,
            output_dir=output_dir,
        )

    if args.skip_final:
        if dev_result is not None:
            summary_path = os.path.join(output_dir, "dev_check_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(dev_result, f, indent=2)
            logger.info("Saved dev-check summary: %s", summary_path)
        logger.info("Skipping final submission generation because --skip-final is set")
        return 0

    test_files, test_labels, test_machines, _, test_sections = parse_scp_with_section(args.test_scp)

    summary: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "train_scp": train_scp,
        "test_scp": args.test_scp,
        "layers": layers,
        "layer_ensemble_mode": args.layer_ensemble_mode,
        "score_method": args.score_method,
        "top_k": args.top_k,
        "local_density_source_k": args.local_density_source_k,
        "local_density_target_k": args.local_density_target_k,
        "local_density_scale_mode": args.local_density_scale_mode,
        "local_density_distance": args.local_density_distance,
        "local_density_normalize_embedding": args.local_density_normalize_embedding,
        "threshold_percentile": args.threshold_percentile,
        "dev_check": dev_result,
        "machines": {},
    }

    for machine_name in sorted(set(test_machines)):
        train_indices = [i for i, machine in enumerate(train_machines) if machine == machine_name]
        test_indices = [i for i, machine in enumerate(test_machines) if machine == machine_name]
        if not train_indices:
            raise RuntimeError(f"No training samples found for test machine {machine_name}")

        train_files_machine = [train_files[i] for i in train_indices]
        train_labels_machine = [train_labels[i] for i in train_indices]
        train_sources_machine = [train_sources[i] for i in train_indices]
        test_files_machine = [test_files[i] for i in test_indices]
        test_labels_machine = [test_labels[i] for i in test_indices]

        logger.info("%s: final scoring", machine_name)
        test_scores, train_scores = compute_machine_score_mean(
            model=model,
            config=config,
            device=device,
            train_files=train_files_machine,
            train_labels=train_labels_machine,
            train_sources=train_sources_machine,
            query_files=test_files_machine,
            query_labels=test_labels_machine,
            layers=layers,
            args=args,
            compute_train_scores=True,
        )
        threshold = float(np.percentile(train_scores, q=args.threshold_percentile))
        decisions = (test_scores > threshold).astype(np.int64)

        sections = sorted(set(test_sections[i] for i in test_indices))
        if len(sections) != 1:
            raise RuntimeError(f"Expected one section for {machine_name}, got {sections}")
        section = sections[0]
        file_names = [os.path.basename(test_files[i]) for i in test_indices]

        anomaly_path = os.path.join(output_dir, f"anomaly_score_{machine_name}_{section}_test.csv")
        decision_path = os.path.join(output_dir, f"decision_result_{machine_name}_{section}_test.csv")
        write_two_column_csv(anomaly_path, file_names, [f"{score:.12g}" for score in test_scores])
        write_two_column_csv(decision_path, file_names, [str(int(value)) for value in decisions])

        summary["machines"][machine_name] = {
            "section": section,
            "test_count": len(test_indices),
            "train_count": len(train_indices),
            "source_train_count": int(sum(src == "source" for src in train_sources_machine)),
            "target_train_count": int(sum(src == "target" for src in train_sources_machine)),
            "threshold": threshold,
            "decision_positive_count": int(decisions.sum()),
            "score_min": float(np.min(test_scores)),
            "score_mean": float(np.mean(test_scores)),
            "score_max": float(np.max(test_scores)),
            "anomaly_score_file": os.path.basename(anomaly_path),
            "decision_result_file": os.path.basename(decision_path),
        }
        logger.info(
            "%s %s: test=%d train=%d threshold=%.6f decisions=%d",
            machine_name,
            section,
            len(test_indices),
            len(train_indices),
            threshold,
            int(decisions.sum()),
        )

    summary_path = os.path.join(output_dir, "submission_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in sorted(os.listdir(output_dir)):
            path = os.path.join(output_dir, file_name)
            if os.path.isfile(path) and file_name.endswith(".csv"):
                zf.write(path, arcname=file_name)

    logger.info("Saved submission directory: %s", output_dir)
    logger.info("Saved submission zip: %s", zip_path)
    logger.info("Saved summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
