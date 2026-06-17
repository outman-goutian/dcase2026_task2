#!/usr/bin/env python3
"""Fuse anomaly-score CSVs from BEATs, EAT/DiffPT and Omni submissions."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_INPUTS = {
    "beats": "final_submission/team_exp_beats_top1_smote_*",
    "diffpt": "final_submission_eat/team_exp_diffpt_eat_layers8_10_12_score_mean_*",
}

DEFAULT_WEIGHTS = {
    "beats": 0.55,
    "diffpt": 0.45,
}


def parse_weight_map(value: str | None, model_names: Sequence[str]) -> Dict[str, float]:
    if not value:
        weights = {name: DEFAULT_WEIGHTS.get(name, 1.0) for name in model_names}
    else:
        weights = {}
        for item in value.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"Invalid weight item {item!r}; expected name=value")
            name, raw_weight = item.split("=", 1)
            weights[name.strip()] = float(raw_weight)
        missing = [name for name in model_names if name not in weights]
        if missing:
            raise ValueError(f"Missing weights for: {', '.join(missing)}")

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Sum of weights must be positive")
    return {name: weight / total for name, weight in weights.items()}


def resolve_input_path(raw_path: str) -> Path:
    matches = sorted((Path(path) for path in glob.glob(raw_path)), key=lambda path: path.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return Path(raw_path).expanduser()


def parse_input_dirs(values: Sequence[str]) -> Dict[str, Path]:
    if not values:
        return {name: resolve_input_path(path) for name, path in DEFAULT_INPUTS.items()}
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid input {item!r}; expected name=path")
        name, raw_path = item.split("=", 1)
        result[name.strip()] = resolve_input_path(raw_path)
    return result


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_score_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    scores: List[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"{path} has invalid row: {row}")
            names.append(row[0])
            scores.append(float(row[1]))
    return names, np.asarray(scores, dtype=np.float64)


def parse_scp(path: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    by_machine: Dict[str, Dict[str, Dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid SCP line in {path}: {line}")
            label, wav_path = parts[0], parts[1]
            label_parts = label.split("_")
            machine = label_parts[0]
            domain = next((part for part in label_parts if part in ("source", "target")), "source")
            status = next((part for part in label_parts if part in ("normal", "anomaly")), "normal")
            wav_name = Path(wav_path).name
            by_machine.setdefault(machine, {})[wav_name] = {
                "label": 1 if status == "anomaly" else 0,
                "domain": domain,
            }
    return by_machine


def write_two_column_csv(path: Path, names: Sequence[str], values: Iterable[object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for name, value in zip(names, values):
            writer.writerow([name, value])


def machine_from_score_file(path: Path) -> str:
    name = path.name
    prefix = "anomaly_score_"
    suffix = "_section_00_test.csv"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected anomaly-score filename: {path}")
    return name[len(prefix) : -len(suffix)]


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)

    # Average tied ranks so equal scores stay equal.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks / float(len(scores) - 1)


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if math.isclose(lo, hi):
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - lo) / (hi - lo)


def zscore_normalize(scores: np.ndarray) -> np.ndarray:
    std = float(np.std(scores))
    if std <= 0:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - float(np.mean(scores))) / std


def normalize(scores: np.ndarray, method: str) -> np.ndarray:
    if method == "rank":
        return rank_normalize(scores)
    if method == "minmax":
        return minmax_normalize(scores)
    if method == "zscore":
        return zscore_normalize(scores)
    if method == "none":
        return scores.astype(np.float64, copy=True)
    raise ValueError(f"Unknown normalization method: {method}")


def load_submission_scores(input_dirs: Mapping[str, Path]) -> Dict[str, Dict[str, Tuple[List[str], np.ndarray]]]:
    loaded: Dict[str, Dict[str, Tuple[List[str], np.ndarray]]] = {}
    for model_name, input_dir in input_dirs.items():
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        model_scores = {}
        for path in sorted(input_dir.glob("anomaly_score_*_section_00_test.csv")):
            machine = machine_from_score_file(path)
            model_scores[machine] = read_score_csv(path)
        if not model_scores:
            raise FileNotFoundError(f"No anomaly_score CSVs found in {input_dir}")
        loaded[model_name] = model_scores
    return loaded


def validate_tune_scores(
    loaded: Mapping[str, Mapping[str, Tuple[List[str], np.ndarray]]],
    labels_by_machine: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> List[str]:
    machine_sets = [set(scores.keys()) for scores in loaded.values()]
    machines = sorted(set.intersection(*machine_sets).intersection(labels_by_machine.keys()))
    if not machines:
        raise ValueError("No common labeled machines across tune inputs and tune SCP")
    for machine in machines:
        expected_names = set(labels_by_machine[machine].keys())
        for model_name, model_scores in loaded.items():
            names, _ = model_scores[machine]
            missing = [name for name in names if name not in expected_names]
            if missing:
                raise ValueError(
                    f"{model_name}/{machine} has score names not found in tune SCP, "
                    f"first missing: {missing[0]}"
                )
    return machines


def fuse_scores(
    loaded: Mapping[str, Mapping[str, Tuple[List[str], np.ndarray]]],
    weights: Mapping[str, float],
    norm: str,
) -> Dict[str, Dict[str, object]]:
    machine_sets = [set(scores.keys()) for scores in loaded.values()]
    machines = sorted(set.intersection(*machine_sets))
    if not machines:
        raise ValueError("No common machines across input submissions")

    fused: Dict[str, Dict[str, object]] = {}
    for machine in machines:
        reference_names: List[str] | None = None
        fused_scores: np.ndarray | None = None
        components = {}
        for model_name, model_scores in loaded.items():
            names, raw_scores = model_scores[machine]
            if reference_names is None:
                reference_names = names
            elif names != reference_names:
                raise ValueError(f"File order mismatch for {machine} in model {model_name}")
            normalized = normalize(raw_scores, norm)
            components[model_name] = {
                "raw_min": float(np.min(raw_scores)),
                "raw_mean": float(np.mean(raw_scores)),
                "raw_max": float(np.max(raw_scores)),
                "normalized_min": float(np.min(normalized)),
                "normalized_mean": float(np.mean(normalized)),
                "normalized_max": float(np.max(normalized)),
            }
            contribution = normalized * weights[model_name]
            fused_scores = contribution if fused_scores is None else fused_scores + contribution

        assert reference_names is not None
        assert fused_scores is not None
        threshold = float(np.percentile(fused_scores, 90.0))
        decisions = (fused_scores > threshold).astype(np.int64)
        fused[machine] = {
            "names": reference_names,
            "scores": fused_scores,
            "decisions": decisions,
            "threshold": threshold,
            "decision_positive_count": int(np.sum(decisions)),
            "score_min": float(np.min(fused_scores)),
            "score_mean": float(np.mean(fused_scores)),
            "score_max": float(np.max(fused_scores)),
            "components": components,
        }
    return fused


def metric_hmean(values: Sequence[float]) -> float:
    if any(value <= 0 for value in values):
        return 0.0
    return float(len(values) / sum(1.0 / value for value in values))


def evaluate_binary_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    domains: Sequence[str],
) -> Tuple[float, float, float, float, float, float, float]:
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    auc = float(roc_auc_score(labels, scores))
    pauc = float(roc_auc_score(labels, scores, max_fpr=0.1))
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
    f1 = float(np.max(f1_scores))

    domain_array = np.asarray(domains)
    source_mask = domain_array == "source"
    target_mask = domain_array == "target"
    if np.any(source_mask) and len(np.unique(labels[source_mask])) > 1:
        source_auc = float(roc_auc_score(labels[source_mask], scores[source_mask]))
        source_pauc = float(roc_auc_score(labels[source_mask], scores[source_mask], max_fpr=0.1))
    else:
        source_auc = 0.5
        source_pauc = 0.5
    if np.any(target_mask) and len(np.unique(labels[target_mask])) > 1:
        target_auc = float(roc_auc_score(labels[target_mask], scores[target_mask]))
        target_pauc = float(roc_auc_score(labels[target_mask], scores[target_mask], max_fpr=0.1))
    else:
        target_auc = 0.5
        target_pauc = 0.5
    return auc, pauc, f1, source_auc, source_pauc, target_auc, target_pauc


def aggregate_metric_tuples(metric_tuples: Sequence[Tuple[float, ...]]) -> Dict[str, float]:
    values = np.asarray(metric_tuples, dtype=np.float64)
    return {
        "auc": metric_hmean(values[:, 0]),
        "pauc": metric_hmean(values[:, 1]),
        "f1": metric_hmean(values[:, 2]),
        "source_auc": metric_hmean(values[:, 3]),
        "source_pauc": metric_hmean(values[:, 4]),
        "target_auc": metric_hmean(values[:, 5]),
        "target_pauc": metric_hmean(values[:, 6]),
    }


def metric_score(aggregate: Mapping[str, float], metric_name: str) -> float:
    if metric_name == "official":
        return metric_hmean([aggregate["auc"], aggregate["pauc"]])
    return float(aggregate[metric_name])


def generate_weight_grid(model_names: Sequence[str], step: float) -> List[Dict[str, float]]:
    if step <= 0 or step > 1:
        raise ValueError("--weight-step must be in (0, 1]")
    scale = round(1.0 / step)
    if not math.isclose(scale * step, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("--weight-step must divide 1.0 exactly, e.g. 0.1, 0.05, 0.025")
    scale = int(scale)
    grids: List[Dict[str, float]] = []

    def rec(index: int, remaining: int, current: List[int]) -> None:
        if index == len(model_names) - 1:
            weights = current + [remaining]
            grids.append({name: weight / scale for name, weight in zip(model_names, weights)})
            return
        for weight in range(remaining + 1):
            rec(index + 1, remaining - weight, current + [weight])

    rec(0, scale, [])
    return grids


def evaluate_fusion_config(
    loaded: Mapping[str, Mapping[str, Tuple[List[str], np.ndarray]]],
    labels_by_machine: Mapping[str, Mapping[str, Mapping[str, object]]],
    machines: Sequence[str],
    weights: Mapping[str, float],
    norm: str,
) -> Dict[str, object]:
    per_machine = {}
    metric_tuples = []
    for machine in machines:
        names = None
        fused_scores = None
        for model_name, model_scores in loaded.items():
            model_names, raw_scores = model_scores[machine]
            if names is None:
                names = model_names
            elif names != model_names:
                raise ValueError(f"File order mismatch for tune machine {machine} in {model_name}")
            normalized = normalize(raw_scores, norm)
            contribution = normalized * weights[model_name]
            fused_scores = contribution if fused_scores is None else fused_scores + contribution
        assert names is not None
        assert fused_scores is not None
        labels = np.asarray([labels_by_machine[machine][name]["label"] for name in names], dtype=np.int64)
        domains = [str(labels_by_machine[machine][name]["domain"]) for name in names]
        metrics = evaluate_binary_scores(labels, fused_scores, domains)
        metric_tuples.append(metrics)
        per_machine[machine] = {
            "auc": metrics[0],
            "pauc": metrics[1],
            "f1": metrics[2],
            "source_auc": metrics[3],
            "source_pauc": metrics[4],
            "target_auc": metrics[5],
            "target_pauc": metrics[6],
        }
    aggregate = aggregate_metric_tuples(metric_tuples)
    return {
        "normalization": norm,
        "weights": dict(weights),
        "aggregate": aggregate,
        "machines": per_machine,
    }


def tune_fusion(
    tune_loaded: Mapping[str, Mapping[str, Tuple[List[str], np.ndarray]]],
    labels_by_machine: Mapping[str, Mapping[str, Mapping[str, object]]],
    norms: Sequence[str],
    weight_step: float,
    select_metric: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model_names = list(tune_loaded.keys())
    machines = validate_tune_scores(tune_loaded, labels_by_machine)
    candidates = []

    for norm in norms:
        for weights in generate_weight_grid(model_names, weight_step):
            result = evaluate_fusion_config(tune_loaded, labels_by_machine, machines, weights, norm)
            score = metric_score(result["aggregate"], select_metric)
            result["selection_metric"] = select_metric
            result["selection_score"] = score
            candidates.append(result)

    candidates.sort(
        key=lambda item: (
            item["selection_score"],
            item["aggregate"]["auc"],
            item["aggregate"]["pauc"],
            item["aggregate"]["f1"],
        ),
        reverse=True,
    )
    return candidates[0], candidates


def write_tuning_report(
    output_root: Path,
    best: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    top_k: int = 50,
) -> Tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_root / f"fusion_tuning_results_{timestamp}.csv"
    json_path = output_root / f"fusion_tuning_best_{timestamp}.json"
    model_names = list(best["weights"].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            "selection_metric",
            "selection_score",
            "normalization",
            *[f"w_{name}" for name in model_names],
            "auc",
            "pauc",
            "f1",
            "source_auc",
            "source_pauc",
            "target_auc",
            "target_pauc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(candidates[:top_k], start=1):
            aggregate = candidate["aggregate"]
            weights = candidate["weights"]
            row = {
                "rank": rank,
                "selection_metric": candidate["selection_metric"],
                "selection_score": candidate["selection_score"],
                "normalization": candidate["normalization"],
                "auc": aggregate["auc"],
                "pauc": aggregate["pauc"],
                "f1": aggregate["f1"],
                "source_auc": aggregate["source_auc"],
                "source_pauc": aggregate["source_pauc"],
                "target_auc": aggregate["target_auc"],
                "target_pauc": aggregate["target_pauc"],
            }
            row.update({f"w_{name}": weights[name] for name in model_names})
            writer.writerow(row)

    json_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def write_submission(
    fused: Mapping[str, Mapping[str, object]],
    output_root: Path,
    team_name: str,
    metadata: Mapping[str, object],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / f"{team_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = dict(metadata)
    summary["output_dir"] = str(output_dir)
    summary["machines"] = {}

    for machine, data in fused.items():
        names = data["names"]
        scores = data["scores"]
        decisions = data["decisions"]
        score_file = f"anomaly_score_{machine}_section_00_test.csv"
        decision_file = f"decision_result_{machine}_section_00_test.csv"
        write_two_column_csv(output_dir / score_file, names, [f"{score:.12g}" for score in scores])
        write_two_column_csv(output_dir / decision_file, names, [int(v) for v in decisions])
        summary["machines"][machine] = {
            "section": "section_00",
            "test_count": len(names),
            "threshold": data["threshold"],
            "decision_positive_count": data["decision_positive_count"],
            "score_min": data["score_min"],
            "score_mean": data["score_mean"],
            "score_max": data["score_max"],
            "anomaly_score_file": score_file,
            "decision_result_file": decision_file,
            "components": data["components"],
        }

    summary_path = output_dir / "submission_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            archive.write(path, arcname=path.name)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input submission as name=path. Defaults to latest local BEATs and EAT final submissions.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated model weights, e.g. beats=0.55,diffpt=0.45. Defaults to the best tuned weights.",
    )
    parser.add_argument("--norm", default="rank", choices=["rank", "minmax", "zscore", "none"])
    parser.add_argument(
        "--tune-input",
        action="append",
        default=[],
        help="Labeled test/dev score directory as name=path. Requires --tune-scp.",
    )
    parser.add_argument(
        "--tune-scp",
        default=None,
        help="Labeled SCP used to select the best normalization and weights, e.g. ../data/test_dev.scp.",
    )
    parser.add_argument(
        "--tune-norms",
        default="rank,minmax,zscore",
        help="Comma-separated normalization methods to search.",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.05,
        help="Grid step for non-negative weights that sum to 1.0.",
    )
    parser.add_argument(
        "--select-metric",
        default="official",
        choices=["official", "auc", "pauc", "f1", "source_auc", "source_pauc", "target_auc", "target_pauc"],
        help="Metric used to select the best tuning candidate. official=hmean(aggregate_auc, aggregate_pauc).",
    )
    parser.add_argument("--tuning-output-root", default="fusion_tuning")
    parser.add_argument("--output-root", default="final_submission_fusion")
    parser.add_argument("--team-name", default="team_exp_fusion_beats_diffpt_omni_rank_equal")
    args = parser.parse_args()

    input_dirs = parse_input_dirs(args.input)
    weights = parse_weight_map(args.weights, list(input_dirs.keys()))
    norm = args.norm
    tuning_metadata = None

    if args.tune_input or args.tune_scp:
        if not args.tune_input or not args.tune_scp:
            raise ValueError("--tune-input and --tune-scp must be provided together")
        tune_dirs = parse_input_dirs(args.tune_input)
        if set(tune_dirs.keys()) != set(input_dirs.keys()):
            raise ValueError("--tune-input model names must match --input model names")
        labels_by_machine = parse_scp(Path(args.tune_scp))
        tune_loaded = load_submission_scores(tune_dirs)
        best, candidates = tune_fusion(
            tune_loaded=tune_loaded,
            labels_by_machine=labels_by_machine,
            norms=parse_csv_list(args.tune_norms),
            weight_step=args.weight_step,
            select_metric=args.select_metric,
        )
        weights = best["weights"]
        norm = best["normalization"]
        report_csv, report_json = write_tuning_report(
            Path(args.tuning_output_root),
            best=best,
            candidates=candidates,
        )
        tuning_metadata = {
            "tune_scp": args.tune_scp,
            "tune_inputs": {name: str(path) for name, path in tune_dirs.items()},
            "tune_norms": parse_csv_list(args.tune_norms),
            "weight_step": args.weight_step,
            "select_metric": args.select_metric,
            "best": best,
            "report_csv": str(report_csv),
            "report_json": str(report_json),
        }
        print(f"selected_norm={norm}")
        print("selected_weights=" + ",".join(f"{name}={weight:.6g}" for name, weight in weights.items()))
        print(f"tuning_report_csv={report_csv}")
        print(f"tuning_report_json={report_json}")

    loaded = load_submission_scores(input_dirs)
    fused = fuse_scores(loaded, weights, norm)
    metadata = {
        "fusion_method": "weighted_score_average",
        "normalization": norm,
        "weights": weights,
        "decision_threshold": "per-machine 90th percentile of fused final-test scores",
        "inputs": {name: str(path) for name, path in input_dirs.items()},
    }
    if tuning_metadata is None:
        metadata["note"] = (
            "No --tune-scp/--tune-input was provided; this run uses the explicit/default "
            "normalization and weights."
        )
    else:
        metadata["tuning"] = tuning_metadata
    output_dir = write_submission(fused, Path(args.output_root), args.team_name, metadata)
    print(output_dir)
    print(output_dir.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
