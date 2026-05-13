#!/usr/bin/env bash
set -euo pipefail

cd /app

: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY is required}"

RUN_ID="${RUN_ID:-full_50k}"
TOTAL="${TOTAL:-50000}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/ghostllm/runs}"
PROFILES="${PROFILES:-balanced,utility,messy_human,context_rich,followup_realism}"
SHARDS="${SHARDS:-50}"
PARALLEL_SHARDS="${PARALLEL_SHARDS:-1}"
SEED_BASE="${SEED_BASE:-10000}"
ACTIVE_SLOTS="${ACTIVE_SLOTS:-500}"
MAX_USER_BATCH="${MAX_USER_BATCH:-150}"
MAX_POKE_BATCH="${MAX_POKE_BATCH:-48}"
CONCURRENCY="${CONCURRENCY:-2}"

mkdir -p "${OUTPUT_DIR}"

exec python -B scripts/run_full_dataset.py \
  --total "${TOTAL}" \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --profiles "${PROFILES}" \
  --shards "${SHARDS}" \
  --parallel-shards "${PARALLEL_SHARDS}" \
  --seed-base "${SEED_BASE}" \
  --active-slots "${ACTIVE_SLOTS}" \
  --max-user-batch "${MAX_USER_BATCH}" \
  --max-poke-batch "${MAX_POKE_BATCH}" \
  --concurrency "${CONCURRENCY}"
