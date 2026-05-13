"""Cloud-friendly full dataset runner.

This wraps run_pilot.py into resumable shards. Each shard writes its own JSONL
file, so killing/restarting the process only loses the in-flight API call. Rerun
with the same run id and output dir to resume.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RUN_PILOT = ROOT / "scripts" / "run_pilot.py"
DEFAULT_PROFILES = [
    "balanced",
    "utility",
    "messy_human",
    "context_rich",
    "followup_realism",
]


@dataclass(frozen=True)
class Shard:
    ix: int
    profile: str
    target: int
    seed: int
    output: Path
    log: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=50000)
    p.add_argument("--run-id", default=time.strftime("full_%Y%m%d_%H%M%S"))
    p.add_argument("--output-dir", type=Path, default=ROOT / "data" / "full_runs")
    p.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    p.add_argument("--shards", type=int, default=50)
    p.add_argument("--parallel-shards", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=10000)
    p.add_argument("--active-slots", type=int, default=500)
    p.add_argument("--max-user-batch", type=int, default=150)
    p.add_argument("--max-poke-batch", type=int, default=48)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status-only", action="store_true")
    p.add_argument("--combine-only", action="store_true")
    return p.parse_args()


def run_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / args.run_id


def parse_profiles(raw: str) -> list[str]:
    profiles = [p.strip() for p in raw.split(",") if p.strip()]
    if not profiles:
        raise SystemExit("profiles cannot be empty")
    return profiles


def split_counts(total: int, shards: int) -> list[int]:
    base, extra = divmod(total, shards)
    return [base + (1 if ix < extra else 0) for ix in range(shards)]


def plan_shards(args: argparse.Namespace) -> list[Shard]:
    profiles = parse_profiles(args.profiles)
    counts = split_counts(args.total, args.shards)
    root = run_dir(args)
    shard_dir = root / "shards"
    log_dir = root / "logs"
    out: list[Shard] = []
    for ix, target in enumerate(counts):
        profile = profiles[ix % len(profiles)]
        seed = args.seed_base + ix
        name = f"shard_{ix:04d}_{profile}"
        out.append(
            Shard(
                ix=ix,
                profile=profile,
                target=target,
                seed=seed,
                output=shard_dir / f"{name}.jsonl",
                log=log_dir / f"{name}.log",
            )
        )
    return out


def successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") and not row.get("error"):
                ids.add(str(row["id"]))
    return ids


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_manifest(args: argparse.Namespace, shards: list[Shard]) -> None:
    root = run_dir(args)
    write_json(
        root / "manifest.json",
        {
            "run_id": args.run_id,
            "total": args.total,
            "profiles": parse_profiles(args.profiles),
            "shards": args.shards,
            "parallel_shards": args.parallel_shards,
            "seed_base": args.seed_base,
            "active_slots": args.active_slots,
            "max_user_batch": args.max_user_batch,
            "max_poke_batch": args.max_poke_batch,
            "concurrency": args.concurrency,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "shard_plan": [
                {
                    **asdict(shard),
                    "output": str(shard.output),
                    "log": str(shard.log),
                }
                for shard in shards
            ],
        },
    )


def shard_status(shard: Shard) -> dict[str, Any]:
    good = len(successful_ids(shard.output))
    rows = line_count(shard.output)
    return {
        "ix": shard.ix,
        "profile": shard.profile,
        "target": shard.target,
        "successful": good,
        "rows": rows,
        "errors": max(0, rows - good),
        "done": good >= shard.target,
        "output": str(shard.output),
        "log": str(shard.log),
    }


def write_status(args: argparse.Namespace, shards: list[Shard]) -> dict[str, Any]:
    statuses = [shard_status(shard) for shard in shards]
    successful = sum(s["successful"] for s in statuses)
    rows = sum(s["rows"] for s in statuses)
    status = {
        "run_id": args.run_id,
        "target": args.total,
        "successful": successful,
        "rows": rows,
        "errors": max(0, rows - successful),
        "done_shards": sum(1 for s in statuses if s["done"]),
        "total_shards": len(shards),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shards": statuses,
    }
    write_json(run_dir(args) / "status.json", status)
    return status


def print_status(status: dict[str, Any]) -> None:
    target = max(1, int(status["target"]))
    pct = 100.0 * int(status["successful"]) / target
    print(
        f"run={status['run_id']} successful={status['successful']}/{status['target']} "
        f"({pct:.1f}%) rows={status['rows']} errors={status['errors']} "
        f"shards={status['done_shards']}/{status['total_shards']}"
    )


def shard_command(args: argparse.Namespace, shard: Shard) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(RUN_PILOT),
        "--n",
        str(shard.target),
        "--seed",
        str(shard.seed),
        "--output",
        str(shard.output),
        "--active-slots",
        str(args.active_slots),
        "--max-user-batch",
        str(args.max_user_batch),
        "--max-poke-batch",
        str(args.max_poke_batch),
        "--concurrency",
        str(args.concurrency),
        "--user-profile",
        shard.profile,
    ]


def run_shard(args: argparse.Namespace, shard: Shard) -> int:
    if len(successful_ids(shard.output)) >= shard.target:
        print(f"shard {shard.ix:04d} already complete: {shard.output}")
        return 0

    shard.output.parent.mkdir(parents=True, exist_ok=True)
    shard.log.parent.mkdir(parents=True, exist_ok=True)
    cmd = shard_command(args, shard)
    print(f"shard {shard.ix:04d} start profile={shard.profile} target={shard.target}")
    print(" ".join(cmd))

    with shard.log.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            prefixed = f"[{shard.ix:04d} {shard.profile}] {line}"
            print(prefixed, end="")
            log.write(line)
        return proc.wait()


def combine_outputs(args: argparse.Namespace, shards: list[Shard]) -> tuple[Path, Path]:
    root = run_dir(args)
    dataset_path = root / "dataset.jsonl"
    errors_path = root / "errors.jsonl"
    seen: set[str] = set()
    successful = 0
    errors = 0

    with dataset_path.open("w", encoding="utf-8") as dataset, errors_path.open(
        "w", encoding="utf-8"
    ) as error_file:
        for shard in shards:
            if not shard.output.exists():
                continue
            with shard.output.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        errors += 1
                        error_file.write(
                            json.dumps(
                                {
                                    "shard": shard.ix,
                                    "profile": shard.profile,
                                    "error": "invalid json",
                                    "line": line.strip(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    row["run_id"] = args.run_id
                    row["shard"] = shard.ix
                    row["shard_profile"] = shard.profile
                    row_id = str(row.get("id", ""))
                    if row.get("error"):
                        errors += 1
                        error_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        continue
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    successful += 1
                    dataset.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_json(
        root / "combined_status.json",
        {
            "run_id": args.run_id,
            "dataset": str(dataset_path),
            "errors": str(errors_path),
            "successful": successful,
            "errors_count": errors,
            "combined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return dataset_path, errors_path


def main() -> int:
    args = parse_args()
    if args.total <= 0:
        raise SystemExit("--total must be > 0")
    if args.shards <= 0:
        raise SystemExit("--shards must be > 0")
    if args.parallel_shards <= 0:
        raise SystemExit("--parallel-shards must be > 0")

    root = run_dir(args)
    root.mkdir(parents=True, exist_ok=True)
    shards = plan_shards(args)
    write_manifest(args, shards)

    if args.status_only:
        print_status(write_status(args, shards))
        return 0

    if args.dry_run:
        print(f"run_dir={root}")
        print(f"dataset={root / 'dataset.jsonl'}")
        for shard in shards[: min(10, len(shards))]:
            print(
                f"shard {shard.ix:04d}: profile={shard.profile} "
                f"target={shard.target} seed={shard.seed} output={shard.output}"
            )
        if len(shards) > 10:
            print(f"... {len(shards) - 10} more shards")
        print_status(write_status(args, shards))
        return 0

    if not args.combine_only:
        with ThreadPoolExecutor(max_workers=args.parallel_shards) as pool:
            futures = {pool.submit(run_shard, args, shard): shard for shard in shards}
            for fut in as_completed(futures):
                shard = futures[fut]
                code = fut.result()
                write_status(args, shards)
                if code != 0:
                    print(f"shard {shard.ix:04d} failed with exit code {code}")
                    return code

    status = write_status(args, shards)
    print_status(status)
    dataset, errors = combine_outputs(args, shards)
    print(f"dataset: {dataset}")
    print(f"errors:  {errors}")
    return 0 if status["successful"] >= args.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
