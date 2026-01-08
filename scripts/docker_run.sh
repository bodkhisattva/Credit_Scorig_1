#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="pd-model-api:latest"

dvc repro

docker build -t "$IMAGE_NAME" .

docker run --rm -p 8000:8000 \
  -e MODEL_PATH="models/credit_default_model.joblib" \
  "$IMAGE_NAME"