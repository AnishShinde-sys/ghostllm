#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-main-sunset-492903-m5}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-ghostllm-50k}"
MACHINE_TYPE="${MACHINE_TYPE:-c3-standard-22}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-ssd}"
IMAGE_FAMILY="${IMAGE_FAMILY:-debian-12}"
IMAGE_PROJECT="${IMAGE_PROJECT:-debian-cloud}"

RUN_ID="${RUN_ID:-full_50k_v1}"
TOTAL="${TOTAL:-50000}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-1}"
ACTIVE_SLOTS="${ACTIVE_SLOTS:-500}"
MAX_USER_BATCH="${MAX_USER_BATCH:-150}"
MAX_POKE_BATCH="${MAX_POKE_BATCH:-48}"
CONCURRENCY="${CONCURRENCY:-2}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "${ROOT}/.env" ]]; then
  echo "ERROR: ${ROOT}/.env is required so the VM can call model APIs." >&2
  exit 1
fi

if [[ -z "$(gcloud auth list --filter=status:ACTIVE --format='value(account)')" ]]; then
  echo "ERROR: gcloud is not authenticated. Run: gcloud auth login" >&2
  exit 1
fi

if ! gcloud auth print-access-token --quiet >/dev/null 2>&1; then
  echo "ERROR: gcloud auth is expired. Run: gcloud auth login" >&2
  exit 1
fi

echo "Creating VM ${VM_NAME} in ${PROJECT}/${ZONE}"
echo "Machine: ${MACHINE_TYPE}, disk: ${BOOT_DISK_SIZE} ${BOOT_DISK_TYPE}"
echo "Run: ${RUN_ID}, total: ${TOTAL}"

gcloud compute instances create "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --image-family "${IMAGE_FAMILY}" \
  --image-project "${IMAGE_PROJECT}" \
  --boot-disk-size "${BOOT_DISK_SIZE}" \
  --boot-disk-type "${BOOT_DISK_TYPE}" \
  --scopes default \
  --tags ghostllm-runner

echo "Waiting for SSH..."
until gcloud compute ssh "${VM_NAME}" --project "${PROJECT}" --zone "${ZONE}" --command "true" >/dev/null 2>&1; do
  sleep 5
done

echo "Installing Docker and preparing persistent directories..."
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "bash -s" < "${ROOT}/cloud/gcp/install_docker_on_vm.sh"

tmp_tar="$(mktemp -t ghostllm-src.XXXXXX.tar.gz)"
trap 'rm -f "${tmp_tar}"' EXIT

echo "Packing repo..."
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data/full_runs' \
  --exclude='cloud/cloud_data' \
  -czf "${tmp_tar}" \
  -C "${ROOT}" .

echo "Uploading repo and .env..."
gcloud compute scp "${tmp_tar}" "${VM_NAME}:/tmp/ghostllm-src.tar.gz" \
  --project "${PROJECT}" \
  --zone "${ZONE}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "mkdir -p /opt/ghostllm/app && tar -xzf /tmp/ghostllm-src.tar.gz -C /opt/ghostllm/app && rm /tmp/ghostllm-src.tar.gz"

echo "Starting full dataset job..."
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "cd /opt/ghostllm/app && RUN_ID='${RUN_ID}' TOTAL='${TOTAL}' GHOSTLLM_DATA_DIR='/opt/ghostllm-data' PARALLEL_SHARDS='${PARALLEL_SHARDS}' ACTIVE_SLOTS='${ACTIVE_SLOTS}' MAX_USER_BATCH='${MAX_USER_BATCH}' MAX_POKE_BATCH='${MAX_POKE_BATCH}' CONCURRENCY='${CONCURRENCY}' docker compose -f cloud/docker-compose.yml up -d --build"

cat <<EOF

VM created and job started.

SSH:
  gcloud compute ssh ${VM_NAME} --project ${PROJECT} --zone ${ZONE}

Logs:
  gcloud compute ssh ${VM_NAME} --project ${PROJECT} --zone ${ZONE} --command "cd /opt/ghostllm/app && docker compose -f cloud/docker-compose.yml logs -f"

Status:
  gcloud compute ssh ${VM_NAME} --project ${PROJECT} --zone ${ZONE} --command "cd /opt/ghostllm/app && python -B scripts/run_full_dataset.py --run-id ${RUN_ID} --output-dir /opt/ghostllm-data --total ${TOTAL} --status-only"

Output:
  /opt/ghostllm-data/${RUN_ID}/dataset.jsonl
EOF
