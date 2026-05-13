"""Sample a Tinker adapter with the Qwen3 chat renderer."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sampler-path")
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--renderer", default="qwen3_disable_thinking")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--prompts", nargs="+", required=True)
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set")

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.completers import TinkerMessageCompleter

    client = tinker.ServiceClient()
    if args.base_only:
        sampling_client = await client.create_sampling_client_async(base_model=args.base_model)
    else:
        if not args.sampler_path:
            raise SystemExit("--sampler-path is required unless --base-only is set")
        sampling_client = await client.create_sampling_client_async(model_path=args.sampler_path)
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.base_model)
    completer = TinkerMessageCompleter(
        sampling_client,
        renderer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    for i, prompt in enumerate(args.prompts, 1):
        response = await completer([{"role": "user", "content": prompt}])
        content = str(response.get("content", "")).strip()
        flags = {
            "contains_user_tag": "User:" in content,
            "contains_assistant_tag": "Assistant:" in content,
            "empty": not content,
        }
        print("=" * 80)
        print(f"case={i}")
        print(f"USER: {prompt}")
        print(f"ASSISTANT: {content}")
        print("FLAGS:", json.dumps(flags, sort_keys=True))
    return 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
