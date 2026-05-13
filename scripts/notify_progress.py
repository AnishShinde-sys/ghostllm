"""Progress notifier for cloud dataset runs.

Polls shard JSONL files and sends SMS updates through Twilio when configured.
The notifier is separate from generation, so it can be restarted without
touching the dataset job.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=os.environ.get("RUN_ID", "full_50k_v1"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", ROOT / "data" / "full_runs")),
    )
    p.add_argument("--total", type=int, default=int(os.environ.get("TOTAL", "50000")))
    p.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.environ.get("NOTIFY_INTERVAL_SECONDS", "900")),
    )
    p.add_argument(
        "--every-rows",
        type=int,
        default=int(os.environ.get("NOTIFY_EVERY_ROWS", "1000")),
    )
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def run_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / args.run_id


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def count_file(path: Path) -> tuple[int, int]:
    successful = 0
    errors = 0
    if not path.exists():
        return successful, errors
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if row.get("error"):
                errors += 1
            else:
                successful += 1
    return successful, errors


def current_status(args: argparse.Namespace) -> dict[str, Any]:
    root = run_dir(args)
    successful = 0
    errors = 0
    rows = 0
    shards = sorted((root / "shards").glob("*.jsonl"))
    for shard in shards:
        good, bad = count_file(shard)
        successful += good
        errors += bad
        rows += good + bad

    pct = 100.0 * successful / max(1, args.total)
    return {
        "run_id": args.run_id,
        "target": args.total,
        "successful": successful,
        "rows": rows,
        "errors": errors,
        "pct": pct,
        "shard_files": len(shards),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def twilio_configured() -> bool:
    required = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM",
        "TWILIO_TO",
    ]
    return all(os.environ.get(name) for name in required)


def ntfy_configured() -> bool:
    return bool(os.environ.get("NTFY_TOPIC"))


def send_sms(body: str) -> None:
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    token = os.environ["TWILIO_AUTH_TOKEN"]
    from_number = os.environ["TWILIO_FROM"]
    to_number = os.environ["TWILIO_TO"]

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    payload = urllib.parse.urlencode(
        {"From": from_number, "To": to_number, "Body": body}
    ).encode("utf-8")
    auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"Twilio returned HTTP {resp.status}")


def send_ntfy(body: str, label: str) -> None:
    base_url = os.environ.get("NTFY_URL", "https://ntfy.sh").rstrip("/")
    topic = os.environ["NTFY_TOPIC"].strip("/")
    url = f"{base_url}/{urllib.parse.quote(topic)}"

    title = os.environ.get("NTFY_TITLE", "ghostllm dataset")
    priority = os.environ.get("NTFY_PRIORITY", "default")
    tags = os.environ.get("NTFY_TAGS", "hourglass")
    if label == "done":
        tags = os.environ.get("NTFY_DONE_TAGS", "white_check_mark")
        priority = os.environ.get("NTFY_DONE_PRIORITY", "high")

    headers = {
        "Title": title,
        "Priority": priority,
        "Tags": tags,
        "Content-Type": "text/plain; charset=utf-8",
    }

    token = os.environ.get("NTFY_TOKEN")
    user = os.environ.get("NTFY_USER")
    password = os.environ.get("NTFY_PASSWORD")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif user and password:
        auth = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {auth}"

    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"ntfy returned HTTP {resp.status}")


def format_message(status: dict[str, Any], label: str) -> str:
    return (
        f"ghostllm {label}: {status['successful']}/{status['target']} "
        f"({status['pct']:.1f}%), rows={status['rows']}, "
        f"errors={status['errors']}, shards_seen={status['shard_files']}"
    )


def should_send(
    args: argparse.Namespace, state: dict[str, Any], status: dict[str, Any]
) -> tuple[bool, str]:
    successful = int(status["successful"])
    target = int(status["target"])
    now = time.time()
    last_sent_rows = int(state.get("last_sent_rows", -1))
    last_sent_at = float(state.get("last_sent_at", 0.0))

    if not state.get("started_sent"):
        return True, "started"
    if successful >= target and not state.get("done_sent"):
        return True, "done"
    if successful >= last_sent_rows + args.every_rows:
        return True, "progress"
    if successful > last_sent_rows and now - last_sent_at >= args.interval_seconds:
        return True, "heartbeat"
    return False, ""


def notify_once(args: argparse.Namespace) -> bool:
    root = run_dir(args)
    state_path = root / "notify_state.json"
    state = read_json(state_path)
    status = current_status(args)
    write_json(root / "notify_status.json", status)

    send, label = should_send(args, state, status)
    if not send:
        print(format_message(status, "no-send"))
        return False

    message = format_message(status, label)
    print(message)
    delivered = False
    if ntfy_configured():
        send_ntfy(message, label)
        print("ntfy sent")
        delivered = True
    if twilio_configured():
        send_sms(message)
        print("sms sent")
        delivered = True
    if not delivered:
        print(
            "notifications disabled: set NTFY_TOPIC or "
            "TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM/TWILIO_TO"
        )

    state.update(
        {
            "last_sent_rows": status["successful"],
            "last_sent_at": time.time(),
            "last_label": label,
            "started_sent": bool(state.get("started_sent")) or label == "started",
            "done_sent": bool(state.get("done_sent")) or label == "done",
        }
    )
    write_json(state_path, state)
    return True


def main() -> int:
    args = parse_args()
    while True:
        try:
            notify_once(args)
        except Exception as e:
            print(f"notify error: {type(e).__name__}: {e}", flush=True)
        if args.once:
            return 0
        time.sleep(max(15, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
