#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-main-sunset-492903-m5}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-ghostllm-50k}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "cd /opt/ghostllm/app && docker compose -f cloud/docker-compose.yml logs -f"
