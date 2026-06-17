#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
if [[ "${MODEL}" != "eat" && "${MODEL}" != "beats" ]]; then
    echo "Usage: $0 <eat|beats> [config_path]" >&2
    exit 2
fi

CONFIG="${2:-configs/config_${MODEL}.yaml}"
python3 scripts/train.py --config_file "${CONFIG}"
