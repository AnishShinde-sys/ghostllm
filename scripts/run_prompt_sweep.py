"""Generate a small comparison set across fake-user prompt profiles.

Default: 5 profiles x 5 trajectories = 25 rows.
Outputs:
  data/prompt_sweep/<run_name>/<profile>.jsonl
  data/prompt_sweep/<run_name>/combined.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_PILOT = ROOT / "scripts" / "run_pilot.py"
PROFILES = ["balanced", "utility", "messy_human", "context_rich", "followup_realism"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-profile", type=int, default=5)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--active-slots", type=int, default=5)
    p.add_argument("--max-user-batch", type=int, default=5)
    p.add_argument("--max-poke-batch", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_name = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "data" / "prompt_sweep" / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_path = out_dir / "combined.jsonl"
    if combined_path.exists():
        combined_path.unlink()

    for ix, profile in enumerate(PROFILES):
        output = out_dir / f"{profile}.jsonl"
        cmd = [
            sys.executable,
            "-B",
            str(RUN_PILOT),
            "--n",
            str(args.n_per_profile),
            "--seed",
            str(args.seed + ix),
            "--output",
            str(output),
            "--no-resume",
            "--active-slots",
            str(args.active_slots),
            "--max-user-batch",
            str(args.max_user_batch),
            "--max-poke-batch",
            str(args.max_poke_batch),
            "--concurrency",
            str(args.concurrency),
            "--user-profile",
            profile,
        ]
        print("\n" + "=" * 90)
        print(f"profile={profile} output={output}")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

        with output.open("r", encoding="utf-8") as src, combined_path.open(
            "a", encoding="utf-8"
        ) as dst:
            for line in src:
                row = json.loads(line)
                row["sweep_profile"] = profile
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 90)
    print(f"combined: {combined_path}")
    print(f"review:   python -B scripts/review_jsonl.py {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
