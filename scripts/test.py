#!/usr/bin/env python3
"""Generate final test scores and decision CSV files."""

from __future__ import annotations

import argparse

from create_submission import run_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final test submission generation")
    parser.add_argument("--model", choices=["eat", "beats"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-scp", default="/workspace/data/train.scp")
    parser.add_argument("--test-scp", default="/workspace/data/test_final.scp")
    parser.add_argument("--dev-scp", default=None, help="Optional dev check before final generation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--team-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    return run_submission(args)


if __name__ == "__main__":
    raise SystemExit(main())
