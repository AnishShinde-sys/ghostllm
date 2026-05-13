# Cloud Full Run

Use a VM for this job, not a laptop. The VM needs internet access for model API
calls, but your laptop can be off after the container starts.

## What Persists

The full runner writes append-only shard files:

- `manifest.json`: planned run config
- `status.json`: current progress
- `shards/*.jsonl`: resumable shard outputs
- `logs/*.log`: per-shard logs
- `dataset.jsonl`: combined successful rows after completion
- `errors.jsonl`: failed rows, if any

Rerun with the same `RUN_ID` and data directory to resume.

## VM Setup

On a cloud VM with Docker installed:

```bash
git clone <this repo url> ghostllm
cd ghostllm
```

Create `.env` on the VM. Do not bake keys into the Docker image:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Start the job:

```bash
export RUN_ID=full_50k_v1
export TOTAL=50000
export GHOSTLLM_DATA_DIR=/opt/ghostllm-data
docker compose -f cloud/docker-compose.yml up -d --build
```

Follow logs:

```bash
docker compose -f cloud/docker-compose.yml logs -f
```

Do not paste `docker compose config` output anywhere public; it expands values
from `.env`.

Check persisted status:

```bash
python -B scripts/run_full_dataset.py \
  --run-id full_50k_v1 \
  --output-dir /opt/ghostllm-data \
  --total 50000 \
  --status-only
```

Resume after VM/container restart:

```bash
export RUN_ID=full_50k_v1
export TOTAL=50000
export GHOSTLLM_DATA_DIR=/opt/ghostllm-data
docker compose -f cloud/docker-compose.yml up -d
```

The important thing is keeping `RUN_ID` and `GHOSTLLM_DATA_DIR` the same.

## Throughput Knobs

Default cloud settings:

- `ACTIVE_SLOTS=500`
- `MAX_USER_BATCH=150`
- `MAX_POKE_BATCH=48`
- `CONCURRENCY=2`
- `PARALLEL_SHARDS=1`

Raise `PARALLEL_SHARDS` only if the model API rate limits allow it. The runner
already batches many independent trajectories per call inside each shard.
