#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
if [[ "${MODEL}" != "eat" && "${MODEL}" != "beats" ]]; then
    echo "Usage: $0 <eat|beats> [extra validate.py args...]" >&2
    exit 2
fi
shift || true

python3 scripts/validate.py --model "${MODEL}" "$@"
