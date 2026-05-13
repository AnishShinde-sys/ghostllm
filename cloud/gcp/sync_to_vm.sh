#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-main-sunset-492903-m5}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-ghostllm-50k}"
RUN_ID="${RUN_ID:-full_50k_v1}"
TOTAL="${TOTAL:-50000}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-1}"
ACTIVE_SLOTS="${ACTIVE_SLOTS:-500}"
MAX_USER_BATCH="${MAX_USER_BATCH:-150}"
MAX_POKE_BATCH="${MAX_POKE_BATCH:-48}"
CONCURRENCY="${CONCURRENCY:-2}"
COMPOSE_PROFILES="${COMPOSE_PROFILES:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp_tar="$(mktemp -t ghostllm-src.XXXXXX.tar.gz)"
trap 'rm -f "${tmp_tar}"' EXIT

tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data/full_runs' \
  --exclude='cloud/cloud_data' \
  -czf "${tmp_tar}" \
  -C "${ROOT}" .

gcloud compute scp "${tmp_tar}" "${VM_NAME}:/tmp/ghostllm-src.tar.gz" \
  --project "${PROJECT}" \
  --zone "${ZONE}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "mkdir -p /opt/ghostllm/app && find /opt/ghostllm/app -mindepth 1 -maxdepth 1 ! -name cloud_data -exec rm -rf {} + && tar -xzf /tmp/ghostllm-src.tar.gz -C /opt/ghostllm/app && rm /tmp/ghostllm-src.tar.gz"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "cd /opt/ghostllm/app && RUN_ID='${RUN_ID}' TOTAL='${TOTAL}' GHOSTLLM_DATA_DIR='/opt/ghostllm-data' PARALLEL_SHARDS='${PARALLEL_SHARDS}' ACTIVE_SLOTS='${ACTIVE_SLOTS}' MAX_USER_BATCH='${MAX_USER_BATCH}' MAX_POKE_BATCH='${MAX_POKE_BATCH}' CONCURRENCY='${CONCURRENCY}' COMPOSE_PROFILES='${COMPOSE_PROFILES}' docker compose -f cloud/docker-compose.yml up -d --build"
