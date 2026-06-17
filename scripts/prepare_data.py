#!/usr/bin/env python3
"""Prepare DCASE task2 SCP files or report missing data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEV_MACHINES = ["ToyCar", "ToyCarEmu", "bearingEmu", "fan", "gearboxEmu", "sliderEmu", "valveEmu"]
FINAL_MACHINES = ["BlowerDustCollector", "Sander", "SewingMachine", "ToothBrush", "ToyDrone"]


def has_machine_dirs(data_root: Path, machines: list[str], split: str) -> bool:
    return all((data_root / machine / split).is_dir() for machine in machines)


def run_gen(data_root: Path, output: Path, split: str, machines: list[str], path_prefix: str) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "data" / "gen_wav_scp.py"),
        str(data_root),
        str(output),
        split,
        "--path-prefix",
        path_prefix,
        "--machines",
        *machines,
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create train/dev/final SCP files")
    parser.add_argument("--data-root", default="/workspace/data", help="DCASE data root")
    parser.add_argument("--output-dir", default="/workspace/data", help="Where to write scp files")
    parser.add_argument("--path-prefix", default="/workspace/data", help="Path prefix written into scp")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing scp files")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.is_dir():
        print(
            f"Data root not found: {data_root}\n"
            "Please download and unzip the DCASE task2 dev/eval data first, or mount it to /workspace/data.",
            file=sys.stderr,
        )
        return 2

    targets = [
        ("train.scp", "train", DEV_MACHINES + FINAL_MACHINES),
        ("test_dev.scp", "test", DEV_MACHINES),
        ("test_final.scp", "test", FINAL_MACHINES),
    ]

    for filename, split, machines in targets:
        output = output_dir / filename
        if output.exists() and not args.overwrite:
            print(f"exists: {output}")
            continue
        if not has_machine_dirs(data_root, machines, split):
            missing = [m for m in machines if not (data_root / m / split).is_dir()]
            print(
                f"Cannot create {filename}; missing {split} directories under {data_root}: {', '.join(missing)}\n"
                "Please download/unzip the corresponding DCASE data before running this step.",
                file=sys.stderr,
            )
            return 2
        run_gen(data_root, output, split, machines, args.path_prefix)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
