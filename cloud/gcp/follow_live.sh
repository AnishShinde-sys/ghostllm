#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-main-sunset-492903-m5}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-ghostllm-50k}"
RUN_ID="${RUN_ID:-full_50k_v4}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "python3 -B /opt/ghostllm/app/scripts/follow_jsonl_pretty.py /opt/ghostllm-data/${RUN_ID}/shards"
