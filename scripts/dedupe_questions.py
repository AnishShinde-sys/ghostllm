from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def key_for(question: str) -> str:
    key = question.lower().strip()
    key = re.sub(r"\s+", " ", key)
    key = re.sub(r"[\"'“”‘’]", "", key)
    key = re.sub(r"[?.!,;:]+$", "", key)
    return key


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    read = 0
    with args.input.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            read += 1
            row = json.loads(line)
            question = (row.get("question") or row.get("prompt") or "").strip()
            if not question:
                continue
            key = key_for(question)
            if key in seen:
                continue
            seen.add(key)
            row["question"] = question
            row["dedupe_key"] = key
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"input: {args.input}")
    print(f"out:   {args.out}")
    print(f"read={read} kept={kept} duplicates={max(0, read - kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
