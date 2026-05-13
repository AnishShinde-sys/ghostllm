from __future__ import annotations

import argparse
import json
from pathlib import Path

from dedupe_questions import key_for


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--existing", type=Path, required=True)
    p.add_argument("--humanllms", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    return p.parse_args()


def load_existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question = (row.get("question") or row.get("prompt") or "").strip()
            if question:
                keys.add(key_for(question))
    return keys


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = load_existing_keys(args.existing)
    kept = 0
    read = 0
    with args.humanllms.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            read += 1
            src_row = json.loads(line)
            question = (src_row.get("prompt") or src_row.get("question") or "").strip()
            if not question:
                continue
            key = key_for(question)
            if key in seen:
                continue
            seen.add(key)
            dst.write(
                json.dumps(
                    {
                        "id": f"humanllms-topup-{kept:05d}",
                        "prompt_type": "humanllms_dataset_topup",
                        "question": question,
                        "source": "humanllms_dpo_prompt_topup",
                        "source_question_id": src_row.get("id"),
                        "dedupe_key": key,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            if kept >= args.count:
                break
    print(f"existing: {args.existing}")
    print(f"humanllms: {args.humanllms}")
    print(f"out:      {args.out}")
    print(f"read={read} kept={kept}")
    if kept < args.count:
        raise SystemExit(f"only found {kept} new prompts, requested {args.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
