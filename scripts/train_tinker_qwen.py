"""Launch a Tinker LoRA SFT job for a Qwen chat model."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--log-path", type=Path, required=True)
    p.add_argument("--model", default=os.environ.get("TINKER_MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--renderer", default=os.environ.get("TINKER_RENDERER", "qwen3_disable_thinking"))
    p.add_argument("--learning-rate", type=float, default=float(os.environ.get("TINKER_LR", "5e-5")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("TINKER_BATCH_SIZE", "64")))
    p.add_argument("--lora-rank", type=int, default=int(os.environ.get("TINKER_LORA_RANK", "32")))
    p.add_argument("--train-on", default=os.environ.get("TINKER_TRAIN_ON", "LAST_ASSISTANT_MESSAGE"))
    p.add_argument("--num-epochs", type=int, default=int(os.environ.get("TINKER_NUM_EPOCHS", "1")))
    p.add_argument("--max-length", type=int, default=int(os.environ.get("TINKER_MAX_LENGTH", "4096")))
    p.add_argument("--test-size", type=int, default=int(os.environ.get("TINKER_TEST_SIZE", "500")))
    p.add_argument("--save-every", type=int, default=int(os.environ.get("TINKER_SAVE_EVERY", "25")))
    p.add_argument("--eval-every", type=int, default=int(os.environ.get("TINKER_EVAL_EVERY", "25")))
    p.add_argument("--max-steps", type=int, default=int(os.environ["TINKER_MAX_STEPS"]) if os.environ.get("TINKER_MAX_STEPS") else None)
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    p.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set")
    if not args.dataset.exists():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    if args.log_path.exists():
        if not args.overwrite:
            raise SystemExit(f"log path already exists, pass --overwrite to replace it: {args.log_path}")
        shutil.rmtree(args.log_path)
    args.log_path.mkdir(parents=True, exist_ok=True)

    try:
        from tinker_cookbook import renderers
        from tinker_cookbook.supervised import train
        from tinker_cookbook.supervised.data import FromConversationFileBuilder
        from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    except ImportError as exc:
        raise SystemExit(
            "Missing Tinker dependencies. Install with: pip install tinker-cookbook"
        ) from exc

    try:
        train_on_what = getattr(renderers.TrainOnWhat, args.train_on)
    except AttributeError as exc:
        valid = ", ".join(name for name in dir(renderers.TrainOnWhat) if name.isupper())
        raise SystemExit(f"unknown --train-on {args.train_on!r}; valid values: {valid}") from exc

    run_config = {
        "dataset": str(args.dataset),
        "log_path": str(args.log_path),
        "model": args.model,
        "renderer": args.renderer,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "lora_rank": args.lora_rank,
        "train_on": args.train_on,
        "num_epochs": args.num_epochs,
        "max_length": args.max_length,
        "test_size": args.test_size,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "max_steps": args.max_steps,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.log_path / "ghostllm_tinker_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dataset_builder = FromConversationFileBuilder(
        file_path=str(args.dataset),
        test_size=args.test_size,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model,
            renderer_name=args.renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            train_on_what=train_on_what,
        ),
    )
    config = train.Config(
        log_path=str(args.log_path),
        model_name=args.model,
        renderer_name=args.renderer,
        dataset_builder=dataset_builder,
        learning_rate=args.learning_rate,
        lr_schedule="linear",
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        infrequent_eval_every=0,
        max_steps=args.max_steps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    asyncio.run(train.main(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
