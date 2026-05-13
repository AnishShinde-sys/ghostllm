# GCP High-Spec VM

This creates a GCP Compute Engine VM, uploads the current repo, installs Docker,
and starts the 50k dataset job in a restarting container. Your laptop can turn
off after the job starts.

Default target:

- Project: `main-sunset-492903-m5`
- Zone: `us-west1-a`
- VM: `ghostllm-50k`
- Machine: `c3-standard-22` (`22` vCPU, `88` GB RAM)
- Boot disk: `200GB` SSD
- Output dir on VM: `/opt/ghostllm-data/full_50k_v3`

The job is API-bound, so this is intentionally high CPU/RAM but no GPU.

## Before Creating

Reauthenticate GCP locally:

```bash
gcloud auth login
gcloud config set project main-sunset-492903-m5
```

Make sure `.env` exists locally with:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Create And Start

```bash
cd /Users/anishshinde/ghostllm
RUN_ID=full_50k_v3 TOTAL=50000 ./cloud/gcp/create_high_spec_vm.sh
```

Optional more aggressive run:

```bash
RUN_ID=full_50k_v3 TOTAL=50000 PARALLEL_SHARDS=2 ./cloud/gcp/create_high_spec_vm.sh
```

Use `PARALLEL_SHARDS=1` first unless rate limits are known to be high.

## Check Status

```bash
./cloud/gcp/status.sh
```

## Follow Logs

```bash
./cloud/gcp/logs.sh
```

## Optional ntfy.sh Progress Updates

Install the ntfy app, then subscribe to a long random topic name. Topic names on
the public `ntfy.sh` server are public to anyone who guesses the topic, so do
not use a simple topic or send secrets in messages.

```bash
NTFY_TOPIC=ghostllm-long-random-topic-name NOTIFY_EVERY_ROWS=1000 NOTIFY_INTERVAL_SECONDS=900 ./cloud/gcp/sync_to_vm.sh
```

Optional env vars:

```bash
NTFY_URL=https://ntfy.sh
NTFY_TITLE="ghostllm dataset"
NTFY_PRIORITY=default
NTFY_TAGS=hourglass
```

The notifier runs separately from generation and watches shard files under
`/opt/ghostllm-data`.

## Optional SMS Progress Updates

Add these to `.env` locally, then sync/restart the VM with the notify profile:

```bash
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM=+15551234567
TWILIO_TO=+15557654321
```

```bash
COMPOSE_PROFILES=notify NOTIFY_EVERY_ROWS=1000 NOTIFY_INTERVAL_SECONDS=900 ./cloud/gcp/sync_to_vm.sh
```

The notifier runs as a separate container and watches shard files under
`/opt/ghostllm-data`. It can be restarted without losing generation progress.

## Output

Final combined dataset:

```text
/opt/ghostllm-data/full_50k_v3/dataset.jsonl
```

Shard outputs and logs:

```text
/opt/ghostllm-data/full_50k_v3/shards/
/opt/ghostllm-data/full_50k_v3/logs/
```

Rerunning the same command with the same `RUN_ID` resumes from existing shard
files.
