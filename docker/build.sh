#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-dcase2026-task2:latest}"
docker build -f docker/Dockerfile -t "${IMAGE}" .
