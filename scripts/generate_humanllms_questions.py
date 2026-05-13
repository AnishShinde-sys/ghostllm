from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from pathlib import Path
from urllib import request

from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "prompts" / "humanllms_exact"
DEFAULT_OUT = ROOT / "data" / "generated_questions" / "humanllms_style"

PROMPTS = {
    "general": PROMPT_DIR / "general_knowledge_questions_system.txt",
    "conversational": PROMPT_DIR / "conversational_questions_system.txt",
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "openai-compatible", "ollama"], default="openai")
    p.add_argument(
        "--model",
        default="accounts/fireworks/models/llama-v3p1-405b-instruct",
        help="For the HumanLLMs paper match, use accounts/fireworks/models/llama-v3p1-405b-instruct.",
    )
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--prompt", choices=sorted(PROMPTS), action="append", default=None)
    p.add_argument("--calls-per-prompt", type=int, default=1)
    p.add_argument("--questions-per-call", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def extract_questions(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            for key in ("questions", "messages", "prompts"):
                if isinstance(data.get(key), list):
                    return [str(x).strip() for x in data[key] if str(x).strip()]
    except json.JSONDecodeError:
        pass

    if '"questions"' in text and "[" in text and "]" in text:
        segment = text[text.find("[") + 1 : text.rfind("]")]
        candidates = segment.split('","')
        if len(candidates) > 1:
            cleaned = []
            for item in candidates:
                item = item.strip().strip(",").strip()
                item = item.removeprefix('"').removesuffix('"')
                item = item.removesuffix("”").strip()
                if item:
                    cleaned.append(item)
            return cleaned

    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            out.append(line)
    return out


def call_openai(args: argparse.Namespace, system: str, user: str) -> str:
    kwargs = {}
    if args.provider == "openai-compatible" and args.base_url:
        kwargs["base_url"] = args.base_url
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        kwargs["api_key"] = api_key
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=2200,
    )
    return (resp.choices[0].message.content or "").strip()


def call_ollama(args: argparse.Namespace, system: str, user: str) -> str:
    base = args.base_url or "http://localhost:11434"
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": args.temperature, "top_p": args.top_p},
        }
    ).encode("utf-8")
    req = request.Request(
        f"{base.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("message", {}).get("content") or "").strip()


def generate_call(args: argparse.Namespace, prompt_key: str, call_ix: int, system: str) -> tuple[str, int, str, list[str]]:
    user = (
        f"Produce exactly {args.questions_per_call} standalone user messages for the dataset. "
        'Return only valid JSON in this shape: {"questions":["..."]}. '
        "Do not include an intro, explanation, markdown, code fence, labels, or trailing text."
    )
    last_error: Exception | None = None
    for attempt in range(args.retries):
        try:
            if args.provider == "ollama":
                raw = call_ollama(args, system, user)
            else:
                raw = call_openai(args, system, user)
            return prompt_key, call_ix, raw, extract_questions(raw)
        except Exception as exc:  # Keep long batch runs alive across transient provider errors.
            last_error = exc
            if attempt + 1 >= args.retries:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"unreachable retry state: {last_error}")


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    prompt_keys = args.prompt or ["general", "conversational"]
    out = args.out or (DEFAULT_OUT / f"{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    jobs = [
        (prompt_key, call_ix, PROMPTS[prompt_key].read_text(encoding="utf-8"))
        for prompt_key in prompt_keys
        for call_ix in range(args.calls_per_prompt)
    ]
    rows_written = 0
    raw_path = out.with_suffix(".raw.jsonl")
    with out.open("w", encoding="utf-8") as dst, raw_path.open("w", encoding="utf-8") as raw_dst:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [
                executor.submit(generate_call, args, prompt_key, call_ix, system)
                for prompt_key, call_ix, system in jobs
            ]
            completed = 0
            for future in as_completed(futures):
                prompt_key, call_ix, raw, questions = future.result()
                completed += 1
                raw_dst.write(
                    json.dumps(
                        {
                            "prompt": prompt_key,
                            "call_ix": call_ix,
                            "model": args.model,
                            "raw": raw,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                for question_ix, question in enumerate(questions):
                    dst.write(
                        json.dumps(
                            {
                                "id": f"{prompt_key}-{call_ix:04d}-{question_ix:03d}",
                                "prompt_type": prompt_key,
                                "question": question,
                                "model": args.model,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "source": "humanllms_exact_prompt",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    rows_written += 1
                dst.flush()
                raw_dst.flush()
                print(
                    f"{completed}/{len(jobs)} {prompt_key} call={call_ix} "
                    f"parsed={len(questions)} rows={rows_written}",
                    flush=True,
                )

    print(f"questions: {out}")
    print(f"raw:       {raw_path}")
    print(f"rows:      {rows_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
