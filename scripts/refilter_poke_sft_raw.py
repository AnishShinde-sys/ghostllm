from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_poke_sft_answers import clean_answer, valid_answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("raw", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rejected-out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    rejected = 0
    seen: set[str] = set()
    with args.out.open("w", encoding="utf-8") as dst, args.rejected_out.open("w", encoding="utf-8") as rej:
        for path in args.raw:
            with path.open(encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    user = (row.get("user") or "").strip()
                    answer = clean_answer(row.get("answer") or row.get("raw") or "")
                    key = user.lower().strip()
                    if not user or key in seen:
                        continue
                    seen.add(key)
                    ok, reason = valid_answer(answer)
                    if not ok:
                        rejected += 1
                        rej.write(
                            json.dumps(
                                {"source": str(path), "user": user, "answer": answer, "reason": reason},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    dst.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": user},
                                    {"role": "assistant", "content": answer},
                                ],
                                "source": "humanllms_poke_amended_response_refiltered",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    kept += 1
                    if args.limit and kept >= args.limit:
                        break
            if args.limit and kept >= args.limit:
                break
    print(f"out: {args.out}")
    print(f"rejected: {args.rejected_out}")
    print(f"kept={kept} rejected={rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
