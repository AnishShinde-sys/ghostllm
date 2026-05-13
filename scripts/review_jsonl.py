"""Terminal review tool for generated trajectories.

Usage:
  python -B scripts/review_jsonl.py data/prompt_sweep/<run>/combined.jsonl

Keys:
  a  accept
  r  reject, then enter a short reason
  s  skip
  q  quit

Writes:
  <input>.reviews.jsonl
  <input>.accepted.jsonl
  <input>.rejected.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--start", type=int, default=1)
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clear() -> None:
    os.system("clear")


def render(row: dict[str, Any], ix: int, total: int) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(
        f"[{ix}/{total}] id={row.get('id')} profile={row.get('sweep_profile') or row.get('user_prompt_profile')}"
    )
    lines.append(f"category={row.get('category')} turns={len(row.get('turns', []))} error={row.get('error')}")
    lines.append(f"topic:    {row.get('topic')}")
    if row.get("scenario"):
        lines.append(f"scenario: {row.get('scenario')}")
    lines.append("-" * 100)
    for turn in row.get("turns", []):
        role = "POKE" if turn.get("role") == "assistant" else "USER"
        lines.append(f"\n{role}:")
        lines.append(str(turn.get("content", "")))
    lines.append("\n" + "-" * 100)
    lines.append("a=accept  r=reject  s=skip  q=quit")
    return "\n".join(lines)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        print(f"no rows: {args.input}")
        return 1

    reviews_path = args.input.with_suffix(args.input.suffix + ".reviews.jsonl")
    accepted_path = args.input.with_suffix(args.input.suffix + ".accepted.jsonl")
    rejected_path = args.input.with_suffix(args.input.suffix + ".rejected.jsonl")

    start = max(1, min(args.start, len(rows)))
    accepted = rejected = skipped = 0

    for ix in range(start, len(rows) + 1):
        row = rows[ix - 1]
        clear()
        print(render(row, ix, len(rows)))
        choice = input("> ").strip().lower()
        if choice == "q":
            break
        if choice == "a":
            review = {
                "id": row.get("id"),
                "decision": "accept",
                "profile": row.get("sweep_profile") or row.get("user_prompt_profile"),
                "category": row.get("category"),
                "note": "",
            }
            append_jsonl(reviews_path, review)
            append_jsonl(accepted_path, row)
            accepted += 1
            continue
        if choice == "r":
            note = input("reject reason> ").strip()
            review = {
                "id": row.get("id"),
                "decision": "reject",
                "profile": row.get("sweep_profile") or row.get("user_prompt_profile"),
                "category": row.get("category"),
                "note": note,
            }
            rejected_row = dict(row)
            rejected_row["review_note"] = note
            append_jsonl(reviews_path, review)
            append_jsonl(rejected_path, rejected_row)
            rejected += 1
            continue
        skipped += 1

    clear()
    print(f"reviews:  {reviews_path}")
    print(f"accepted: {accepted_path} ({accepted})")
    print(f"rejected: {rejected_path} ({rejected})")
    print(f"skipped:  {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
