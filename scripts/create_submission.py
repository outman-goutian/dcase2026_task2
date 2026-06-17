#!/usr/bin/env python3
"""Unified final submission entry point for EAT and BEATs."""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_submission(args: argparse.Namespace) -> int:
    if args.model == "eat":
        cmd = [
            sys.executable,
            "scripts/create_final_submission_eat.py",
            "--config",
            args.config or "configs/config_eat.yaml",
            "--checkpoint",
            args.checkpoint or "checkpoints/eat/checkpoint.pth",
            "--train-scp",
            args.train_scp,
            "--test-scp",
            args.test_scp,
            "--layers",
            "8,10,12",
            "--score-method",
            "knn_domain_local_density",
            "--local-density-source-k",
            "16",
            "--local-density-target-k",
            "9",
            "--local-density-scale-mode",
            "sum",
            "--local-density-distance",
            "l2",
            "--threshold-percentile",
            "90",
            "--output-dir",
            args.output_dir or "final_submission_eat",
            "--team-name",
            args.team_name or "team_exp_diffpt_eat_layers8_10_12_score_mean",
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        if getattr(args, "dev_scp", None):
            cmd.extend(["--dev-scp", args.dev_scp])
    else:
        cmd = [
            sys.executable,
            "scripts/create_final_submission_beats.py",
            "--config",
            args.config or "configs/config_beats.yaml",
            "--checkpoint",
            args.checkpoint or "checkpoints/beats/checkpoint.pth",
            "--train-scp",
            args.train_scp,
            "--test-scp",
            args.test_scp,
            "--layer",
            "10",
            "--channel",
            "0",
            "--score-method",
            "knn",
            "--top-k",
            "1",
            "--smote-sampling-ratio",
            "0.25",
            "--smote-k-neighbors",
            "3",
            "--output-dir",
            args.output_dir or "final_submission",
            "--team-name",
            args.team_name or "team_exp_beats_top1_smote",
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        if getattr(args, "dev_scp", None):
            cmd.extend(["--eval-scp", args.dev_scp])

    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Create DCASE final submission zip")
    parser.add_argument("--model", choices=["eat", "beats"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-scp", default="/workspace/data/train.scp")
    parser.add_argument("--test-scp", default="/workspace/data/test_final.scp")
    parser.add_argument("--dev-scp", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--team-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    return run_submission(args)


if __name__ == "__main__":
    raise SystemExit(main())
