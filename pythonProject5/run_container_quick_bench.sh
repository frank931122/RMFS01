#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="rmfs01-bench:latest"

# Build image from repo root
cd "$(dirname "$0")/.."
docker build -f pythonProject5/Dockerfile -t "${IMAGE_TAG}" .

# Run quick bench inside container

docker run --rm \
  -v "$(pwd)":/workspace/RMFS01 \
  -w /workspace/RMFS01/pythonProject5 \
  -e PYTHONPATH=/workspace/RMFS01 \
  "${IMAGE_TAG}" \
  python main.py --quick-bench --prefix demo01 --bench-iters 200 --bench-seeds 0 --gammas 0 --bench-compare bench_baseline_demo01.json --bench-tol 0.001
