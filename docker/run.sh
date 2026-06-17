#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-dcase2026-task2:latest}"
DATA_ROOT="${DATA_ROOT:-/kanas/asr/wangjunjie/plans/data}"

docker run --rm --gpus all --shm-size=16g \
    -v "$PWD":/workspace/dcase2026_task2 \
    -v "${DATA_ROOT}":/workspace/data \
    -w /workspace/dcase2026_task2 \
    "${IMAGE}" "$@"
