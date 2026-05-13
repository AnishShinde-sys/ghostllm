#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-main-sunset-492903-m5}"
ZONE="${ZONE:-us-west1-a}"
VM_NAME="${VM_NAME:-ghostllm-50k}"
RUN_ID="${RUN_ID:-full_50k_v6}"

TINKER_CONTAINER_NAME="${TINKER_CONTAINER_NAME:-ghostllm-tinker-qwen3-poke-20k-turns}"
TINKER_IMAGE="${TINKER_IMAGE:-python:3.11-slim}"
TINKER_JOB_NAME="${TINKER_JOB_NAME:-qwen3_poke_voice_20k_turns}"
TINKER_MODEL="${TINKER_MODEL:-Qwen/Qwen3-8B}"
TINKER_RENDERER="${TINKER_RENDERER:-qwen3_disable_thinking}"
TINKER_LIMIT="${TINKER_LIMIT:-20000}"
TINKER_LR="${TINKER_LR:-2e-5}"
TINKER_BATCH_SIZE="${TINKER_BATCH_SIZE:-64}"
TINKER_LORA_RANK="${TINKER_LORA_RANK:-32}"
TINKER_TRAIN_ON="${TINKER_TRAIN_ON:-LAST_ASSISTANT_MESSAGE}"
TINKER_TEST_SIZE="${TINKER_TEST_SIZE:-500}"
TINKER_MAX_LENGTH="${TINKER_MAX_LENGTH:-4096}"
TINKER_SAVE_EVERY="${TINKER_SAVE_EVERY:-25}"
TINKER_EVAL_EVERY="${TINKER_EVAL_EVERY:-25}"
TINKER_INSTALL_DEPS="${TINKER_INSTALL_DEPS:-1}"
RESTART_TINKER="${RESTART_TINKER:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "mkdir -p /opt/ghostllm/app/scripts /opt/ghostllm/app/prompts"

gcloud compute scp \
  "${ROOT}/scripts/export_tinker_sft.py" \
  "${ROOT}/scripts/train_tinker_qwen.py" \
  "${ROOT}/prompts/conversation_only_poke_prompt.txt" \
  "${VM_NAME}:/tmp/" \
  --project "${PROJECT}" \
  --zone "${ZONE}"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "mv /tmp/export_tinker_sft.py /tmp/train_tinker_qwen.py /opt/ghostllm/app/scripts/ && mv /tmp/conversation_only_poke_prompt.txt /opt/ghostllm/app/prompts/"

remote_cmd=$(cat <<REMOTE
set -euo pipefail
cd /opt/ghostllm/app
TINKER_INSTALL_DEPS='${TINKER_INSTALL_DEPS}'

if ! grep -q '^TINKER_API_KEY=' .env; then
  echo 'TINKER_API_KEY is missing from /opt/ghostllm/app/.env' >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx '${TINKER_CONTAINER_NAME}'; then
  if [ '${RESTART_TINKER}' = '1' ]; then
    docker rm -f '${TINKER_CONTAINER_NAME}' >/dev/null
  else
    echo 'Tinker container already exists: ${TINKER_CONTAINER_NAME}'
    echo 'Use RESTART_TINKER=1 to replace it, or follow logs with:'
    echo 'docker logs -f ${TINKER_CONTAINER_NAME}'
    exit 0
  fi
fi

sudo mkdir -p "/opt/ghostllm-data/${RUN_ID}/tinker/${TINKER_JOB_NAME}"
sudo chown -R "\$(id -u):\$(id -g)" "/opt/ghostllm-data/${RUN_ID}/tinker"

docker run -d \
  --name '${TINKER_CONTAINER_NAME}' \
  --env-file /opt/ghostllm/app/.env \
  -e PYTHONUNBUFFERED=1 \
  -e TINKER_MODEL='${TINKER_MODEL}' \
  -e TINKER_RENDERER='${TINKER_RENDERER}' \
  -e TINKER_LR='${TINKER_LR}' \
  -e TINKER_BATCH_SIZE='${TINKER_BATCH_SIZE}' \
  -e TINKER_LORA_RANK='${TINKER_LORA_RANK}' \
  -e TINKER_TRAIN_ON='${TINKER_TRAIN_ON}' \
  -e TINKER_TEST_SIZE='${TINKER_TEST_SIZE}' \
  -e TINKER_MAX_LENGTH='${TINKER_MAX_LENGTH}' \
  -e TINKER_SAVE_EVERY='${TINKER_SAVE_EVERY}' \
  -e TINKER_EVAL_EVERY='${TINKER_EVAL_EVERY}' \
  -v /opt/ghostllm/app:/app \
  -v /opt/ghostllm-data:/data/ghostllm/runs \
  -w /app \
  -e TINKER_INSTALL_DEPS='${TINKER_INSTALL_DEPS}' \
  '${TINKER_IMAGE}' \
  sh -lc "if [ \"\${TINKER_INSTALL_DEPS}\" = '1' ]; then \
      apt-get update && apt-get install -y --no-install-recommends git && \
      python -m pip install --upgrade pip && \
      python -m pip install tinker-cookbook; \
    fi && \
    python -B scripts/export_tinker_sft.py \
      --shards '/data/ghostllm/runs/${RUN_ID}/shards' \
      --output '/data/ghostllm/runs/${RUN_ID}/tinker/${TINKER_JOB_NAME}/conversations.jsonl' \
      --limit '${TINKER_LIMIT}' \
      --wait \
      --poll-seconds 60 \
      --no-system \
      --balance-turns \
      --explode-assistant-turns \
      --strict-quality && \
    python -B scripts/train_tinker_qwen.py \
      --dataset '/data/ghostllm/runs/${RUN_ID}/tinker/${TINKER_JOB_NAME}/conversations.jsonl' \
      --log-path '/data/ghostllm/runs/${RUN_ID}/tinker/${TINKER_JOB_NAME}/logs' \
      --model '${TINKER_MODEL}' \
      --renderer '${TINKER_RENDERER}' \
      --learning-rate '${TINKER_LR}' \
      --batch-size '${TINKER_BATCH_SIZE}' \
      --lora-rank '${TINKER_LORA_RANK}' \
      --train-on '${TINKER_TRAIN_ON}' \
      --test-size '${TINKER_TEST_SIZE}' \
      --max-length '${TINKER_MAX_LENGTH}' \
      --save-every '${TINKER_SAVE_EVERY}' \
      --eval-every '${TINKER_EVAL_EVERY}' \
      --overwrite"

echo 'Started Tinker/Qwen pilot container: ${TINKER_CONTAINER_NAME}'
echo 'Follow logs:'
echo "gcloud compute ssh ${VM_NAME} --project ${PROJECT} --zone ${ZONE} --command 'docker logs -f ${TINKER_CONTAINER_NAME}'"
REMOTE
)

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT}" \
  --zone "${ZONE}" \
  --command "${remote_cmd}"
