"""
Create DCASE challenge submission files for test_final.scp.

The scoring path mirrors the best validation setting:
  top1 checkpoint, BEATs encoder layer 10, channel 1, norm=none, top_k=1,
  target-domain SMOTE with sampling_ratio=0.25 and k_neighbors=3.
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
from imblearn.over_sampling import SMOTE
from scipy.stats import hmean
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.compare_beats_layers import parse_scp
from scripts.compare_ch_fusion import extract_channel_embeddings
from scripts.inference import load_trained_model
from utils.config_loader import load_config


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_scp_with_section(path: str) -> Tuple[List[str], List[int], List[str], List[str], List[str]]:
    files, labels, machines, domains = parse_scp(path)
    sections = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            label = line.split()[0]
            parts = label.split("_")
            if len(parts) >= 3 and parts[1] == "section":
                sections.append(f"{parts[1]}_{parts[2]}")
            else:
                sections.append("section_00")
    if len(sections) != len(files):
        raise RuntimeError(f"Parsed section count mismatch for {path}")
    return files, labels, machines, domains, sections


def smote_target_ratio(
    train_features: np.ndarray,
    train_sources: List[str],
    sampling_ratio: float,
    k_neighbors: int,
    seed: int,
) -> Tuple[np.ndarray, List[str], int]:
    domain_array = np.asarray(train_sources)
    source_indices = np.where(domain_array == "source")[0]
    target_indices = np.where(domain_array == "target")[0]
    source_count = len(source_indices)
    target_count = len(target_indices)
    if source_count == 0 or target_count == 0:
        return train_features, list(train_sources), 0
    if k_neighbors >= target_count:
        raise ValueError(f"k_neighbors={k_neighbors} must be < target_count={target_count}")

    desired_target_count = max(target_count, int(round(source_count * sampling_ratio)))
    if desired_target_count <= target_count:
        return train_features, list(train_sources), 0

    y = np.zeros(len(train_features), dtype=np.int64)
    y[target_indices] = 1
    sampler = SMOTE(
        sampling_strategy={1: desired_target_count},
        k_neighbors=k_neighbors,
        random_state=seed,
    )
    x_resampled, y_resampled = sampler.fit_resample(train_features, y)
    synthetic_count = int(np.sum(y_resampled == 1) - target_count)
    resampled_sources = ["target" if label == 1 else "source" for label in y_resampled]
    return x_resampled, resampled_sources, synthetic_count


def knn_scores(
    query_features: np.ndarray,
    train_features: np.ndarray,
    top_k: int,
    exclude_self_prefix: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    scores = []
    for start in range(0, len(query_features), batch_size):
        end = min(start + batch_size, len(query_features))
        distances = pairwise_distances(query_features[start:end], train_features, metric="euclidean")
        if exclude_self_prefix:
            rows = np.arange(end - start)
            cols = np.arange(start, end)
            valid = cols < train_features.shape[0]
            distances[rows[valid], cols[valid]] = np.inf
        nearest = np.partition(distances, kth=top_k - 1, axis=1)[:, :top_k]
        scores.append(nearest.mean(axis=1))
    return np.concatenate(scores)


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


def density_distances(
    queries: np.ndarray,
    bank: np.ndarray,
    distance: str,
) -> np.ndarray:
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


def evaluate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    domains: List[str],
) -> Tuple[float, float, float, float, float, float, float]:
    labels = np.asarray(labels)
    auc = roc_auc_score(labels, scores)
    pauc = roc_auc_score(labels, scores, max_fpr=0.1)
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
    f1 = float(np.max(f1_scores))

    domain_array = np.asarray(domains)
    source_indices = np.where(domain_array == "source")[0]
    target_indices = np.where(domain_array == "target")[0]
    if len(source_indices) > 0 and len(np.unique(labels[source_indices])) > 1:
        source_auc = roc_auc_score(labels[source_indices], scores[source_indices])
        source_pauc = roc_auc_score(labels[source_indices], scores[source_indices], max_fpr=0.1)
    else:
        source_auc = 0.5
        source_pauc = 0.5
    if len(target_indices) > 0 and len(np.unique(labels[target_indices])) > 1:
        target_auc = roc_auc_score(labels[target_indices], scores[target_indices])
        target_pauc = roc_auc_score(labels[target_indices], scores[target_indices], max_fpr=0.1)
    else:
        target_auc = 0.5
        target_pauc = 0.5
    return auc, pauc, f1, source_auc, source_pauc, target_auc, target_pauc


def aggregate_metrics(per_machine_metrics: List[Tuple[float, float, float, float, float, float, float]]) -> Dict[str, float]:
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


def score_machine(
    machine_name: str,
    train_features: np.ndarray,
    train_machines: List[str],
    train_sources: List[str],
    query_features: np.ndarray,
    query_machines: List[str],
    args: argparse.Namespace,
) -> Tuple[List[int], np.ndarray, np.ndarray, float, int, int, int]:
    train_indices = [i for i, machine in enumerate(train_machines) if machine == machine_name]
    query_indices = [i for i, machine in enumerate(query_machines) if machine == machine_name]
    if not train_indices:
        raise RuntimeError(f"No training samples found for query machine {machine_name}")

    train_sources_machine = [train_sources[i] for i in train_indices]
    train_machine_features = train_features[train_indices]
    augmented_train, augmented_sources, synthetic_count = smote_target_ratio(
        train_machine_features,
        train_sources_machine,
        args.smote_sampling_ratio,
        args.smote_k_neighbors,
        args.seed,
    )

    query_scores = compute_scores(
        query_features[query_indices],
        augmented_train,
        augmented_sources,
        args,
        exclude_self_prefix=False,
    )
    train_scores = compute_scores(
        train_machine_features,
        augmented_train,
        augmented_sources,
        args,
        exclude_self_prefix=True,
    )
    threshold = float(np.percentile(train_scores, q=args.threshold_percentile))
    decisions = (query_scores > threshold).astype(np.int64)
    return query_indices, query_scores, decisions, threshold, len(train_indices), len(augmented_train), synthetic_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Create final DCASE submission files")
    parser.add_argument("--config", default="configs/beats_full.yaml")
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints_sweep_mixup_lr_train_scp_ep10/beats_full/"
            "beats_full_trainscp_lr1e-4_wd5e-3_mix035_ep10/model_epoch10.pth"
        ),
    )
    parser.add_argument("--test-scp", default="/workspace/data/test_final.scp")
    parser.add_argument(
        "--eval-scp",
        default=None,
        help="Optional labeled dev SCP to score with the same pipeline before creating submission.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run --eval-scp metrics and skip submission file creation.",
    )
    parser.add_argument("--train-scp", default=None)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--score-method",
        default="knn",
        choices=["knn", "knn_domain_local_density", "domain_local_density_sum"],
    )
    parser.add_argument("--local-density-source-k", type=int, default=16)
    parser.add_argument("--local-density-target-k", type=int, default=9)
    parser.add_argument("--local-density-scale-mode", default="sum", choices=["sum", "mean"])
    parser.add_argument("--local-density-distance", default="l2", choices=["l2", "cosine"])
    parser.add_argument("--local-density-normalize-embedding", action="store_true")
    parser.add_argument("--score-eps", type=float, default=1e-8)
    parser.add_argument("--smote-sampling-ratio", type=float, default=0.25)
    parser.add_argument("--smote-k-neighbors", type=int, default=3)
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="final_submission")
    parser.add_argument("--team-name", default="team_exp_beats_top1_smote")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

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
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    num_encoder_layers = len(base_model.beats.encoder.layers)
    if args.layer < 0 or args.layer > num_encoder_layers:
        raise ValueError(f"layer must be in 0..{num_encoder_layers}")

    train_files, train_labels, train_machines, train_sources, _ = parse_scp_with_section(train_scp)
    test_files, test_labels, test_machines, _, test_sections = parse_scp_with_section(args.test_scp)
    eval_files = eval_labels = eval_machines = eval_sources = eval_features = None
    if args.eval_scp is not None:
        eval_files, eval_labels, eval_machines, eval_sources, _ = parse_scp_with_section(args.eval_scp)

    logger.info("Extracting train embeddings: layer=%d ch=%d", args.layer, args.channel + 1)
    train_features, _ = extract_channel_embeddings(
        model,
        train_files,
        train_labels,
        config,
        device,
        args.layer,
        num_encoder_layers,
        args.channel,
    )
    if args.eval_scp is not None:
        logger.info("Extracting eval embeddings: %s layer=%d ch=%d", args.eval_scp, args.layer, args.channel + 1)
        eval_features, eval_labels_np = extract_channel_embeddings(
            model,
            eval_files,
            eval_labels,
            config,
            device,
            args.layer,
            num_encoder_layers,
            args.channel,
        )
        per_machine_metrics = []
        eval_rows = []
        for machine_name in sorted(set(eval_machines)):
            eval_indices, eval_scores, _, _, train_count, augmented_count, synthetic_count = score_machine(
                machine_name,
                train_features,
                train_machines,
                train_sources,
                eval_features,
                eval_machines,
                args,
            )
            eval_sections = sorted(set("section_00" for _ in eval_indices))
            if len(eval_sections) != 1:
                raise RuntimeError(f"Expected one eval section for {machine_name}, got {eval_sections}")
            eval_file_names = [os.path.basename(eval_files[i]) for i in eval_indices]
            eval_score_path = os.path.join(
                output_dir,
                f"anomaly_score_{machine_name}_{eval_sections[0]}_test.csv",
            )
            write_two_column_csv(eval_score_path, eval_file_names, [f"{score:.12g}" for score in eval_scores])
            labels = eval_labels_np[eval_indices]
            domains = [eval_sources[i] for i in eval_indices]
            metrics = evaluate_metrics(labels, eval_scores, domains)
            per_machine_metrics.append(metrics)
            row = {
                "machine": machine_name,
                "train_count": train_count,
                "augmented_train_count": augmented_count,
                "synthetic_count": synthetic_count,
                "auc": metrics[0],
                "pauc": metrics[1],
                "f1": metrics[2],
                "source_auc": metrics[3],
                "source_pauc": metrics[4],
                "target_auc": metrics[5],
                "target_pauc": metrics[6],
                "anomaly_score_file": os.path.basename(eval_score_path),
            }
            eval_rows.append(row)
            logger.info(
                "eval %s: AUC=%.6f pAUC=%.6f F1=%.6f synthetic=%d",
                machine_name,
                row["auc"],
                row["pauc"],
                row["f1"],
                synthetic_count,
            )
        eval_summary = aggregate_metrics(per_machine_metrics)
        logger.info(
            "eval aggregate: AUC=%.6f pAUC=%.6f F1=%.6f source_AUC=%.6f target_AUC=%.6f",
            eval_summary["auc"],
            eval_summary["pauc"],
            eval_summary["f1"],
            eval_summary["source_auc"],
            eval_summary["target_auc"],
        )
        eval_path = os.path.join(output_dir, "eval_metrics.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "eval_scp": args.eval_scp,
                    "aggregate": eval_summary,
                    "machines": eval_rows,
                },
                f,
                indent=2,
            )
        logger.info("Saved eval metrics: %s", eval_path)

    if args.eval_only:
        logger.info("Eval-only mode enabled; skipping submission files.")
        return 0

    logger.info("Extracting test embeddings: layer=%d ch=%d", args.layer, args.channel + 1)
    test_features, _ = extract_channel_embeddings(
        model,
        test_files,
        test_labels,
        config,
        device,
        args.layer,
        num_encoder_layers,
        args.channel,
    )

    summary: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "train_scp": train_scp,
        "eval_scp": args.eval_scp,
        "test_scp": args.test_scp,
        "layer": args.layer,
        "channel": args.channel,
        "top_k": args.top_k,
        "score_method": args.score_method,
        "norm": "none",
        "local_density_source_k": args.local_density_source_k,
        "local_density_target_k": args.local_density_target_k,
        "local_density_scale_mode": args.local_density_scale_mode,
        "local_density_distance": args.local_density_distance,
        "local_density_normalize_embedding": args.local_density_normalize_embedding,
        "smote_sampling_ratio": args.smote_sampling_ratio,
        "smote_k_neighbors": args.smote_k_neighbors,
        "threshold_percentile": args.threshold_percentile,
        "seed": args.seed,
        "machines": {},
    }

    for machine_name in sorted(set(test_machines)):
        test_indices, test_scores, decisions, threshold, train_count, augmented_count, synthetic_count = score_machine(
            machine_name,
            train_features,
            train_machines,
            train_sources,
            test_features,
            test_machines,
            args,
        )

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
            "train_count": train_count,
            "augmented_train_count": augmented_count,
            "synthetic_count": synthetic_count,
            "threshold": threshold,
            "decision_positive_count": int(decisions.sum()),
            "score_min": float(np.min(test_scores)),
            "score_mean": float(np.mean(test_scores)),
            "score_max": float(np.max(test_scores)),
            "anomaly_score_file": os.path.basename(anomaly_path),
            "decision_result_file": os.path.basename(decision_path),
        }
        logger.info(
            "%s %s: test=%d train=%d synthetic=%d threshold=%.6f decisions=%d",
            machine_name,
            section,
            len(test_indices),
            train_count,
            synthetic_count,
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
