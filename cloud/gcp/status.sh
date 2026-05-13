#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-main-sunset-492903-m5}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-ghostllm-50k}"
RUN_ID="${RUN_ID:-full_50k_v4}"
TOTAL="${TOTAL:-50000}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "cd /opt/ghostllm/app && docker compose -f cloud/docker-compose.yml exec -T ghostllm-full-run python -B scripts/run_full_dataset.py --run-id '${RUN_ID}' --output-dir /data/ghostllm/runs --total '${TOTAL}' --status-only"
