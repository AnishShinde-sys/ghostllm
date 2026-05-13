"""Pretty-follow generated trajectory JSONL files."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="JSONL file or directory of shard JSONL files")
    p.add_argument("--from-start", action="store_true", help="print existing rows first")
    p.add_argument("--include-errors", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=1.0)
    return p.parse_args()


def paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("*.jsonl"))


def initial_offset(path: Path, from_start: bool) -> int:
    if from_start:
        return 0
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def render(row: dict[str, Any], source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(
        f"id={row.get('id')}  file={source.name}  "
        f"category={row.get('category')}  profile={row.get('user_prompt_profile')}"
    )
    if row.get("topic"):
        lines.append(f"topic: {row.get('topic')}")
    if row.get("error"):
        lines.append(f"ERROR: {row.get('error')}")
    lines.append("-" * 100)
    for turn in row.get("turns", []):
        role = "POKE" if turn.get("role") == "assistant" else "USER"
        content = str(turn.get("content", "")).strip()
        lines.append("")
        lines.append(f"{role}:")
        lines.append(content if content else "[empty]")
    return "\n".join(lines)


def read_new_rows(path: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                rows.append(
                    {
                        "id": None,
                        "error": f"invalid_json:{e}",
                        "turns": [{"role": "user", "content": line.strip()}],
                    }
                )
        return f.tell(), rows


def main() -> int:
    args = parse_args()
    offsets: dict[Path, int] = {}
    print(f"following {args.path}", flush=True)
    while True:
        for path in paths(args.path):
            if path not in offsets:
                offsets[path] = initial_offset(path, args.from_start)
            try:
                new_offset, rows = read_new_rows(path, offsets[path])
            except FileNotFoundError:
                continue
            offsets[path] = new_offset
            for row in rows:
                if row.get("error") and not args.include_errors:
                    continue
                print(render(row, path), flush=True)
        time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
