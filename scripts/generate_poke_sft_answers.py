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
PROMPT_DIR = ROOT / "prompts"
HUMANLLMS_RESPONSE_PROMPT = PROMPT_DIR / "humanllms_exact" / "human_like_response_system.txt"
DEFAULT_OUT_DIR = ROOT / "data" / "generated_sft" / "poke_humanllms_style"


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
    p.add_argument("questions", type=Path)
    p.add_argument("--provider", choices=["openai", "openai-compatible", "ollama"], default="openai-compatible")
    p.add_argument(
        "--model",
        default="accounts/fireworks/models/llama-v3-70b-instruct",
        help="For the HumanLLMs paper match, use accounts/fireworks/models/llama-v3-70b-instruct.",
    )
    p.add_argument("--base-url", default="https://api.fireworks.ai/inference/v1")
    p.add_argument("--api-key-env", default="FIREWORKS_API_KEY")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-output-tokens", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def build_system() -> str:
    return HUMANLLMS_RESPONSE_PROMPT.read_text(encoding="utf-8").strip()


def clean_answer(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("assistant", "answer", "response", "chosen"):
                if isinstance(data.get(key), str):
                    text = data[key].strip()
                    break
    except json.JSONDecodeError:
        pass
    text = re.sub(r"^\s*(assistant|poke)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def valid_answer(answer: str) -> tuple[bool, str]:
    lowered = answer.lower()
    bad_bits = [
        "<think>",
        "imagenew",
        "user:",
        "assistant:",
        "as an ai",
        "language model",
        "that's a great question",
        "what a great question",
        "i don't actually",
        "i do not actually",
        "i don't have",
        "i do not have",
        "i don’t have",
        "i don't really",
        "i do not really",
        "i don’t really",
        "don't really have",
        "do not really have",
        "i don't do",
        "i do not do",
        "i don’t do",
        "i don't need",
        "i do not need",
        "i don’t need",
        "i'm not really",
        "i am not really",
        "i don't unwind",
        "i do not unwind",
        "i'm just here",
        "i am just here",
        "i'm here to",
        "i am here to",
        "i can help you",
        "i can suggest",
        "just providing info",
        "providing info",
        "more about providing",
        "just helping",
        "no hobbies",
        "no animal adventures",
        "happy to chat about",
        "no reading list",
        "no opinions",
        "not really about",
        "not really for me",
        "not really, i'm",
        "not really, i’m",
        "not really, i am",
        "neither, just",
        "nothing recently",
        "not recall personal",
        "not remember personal",
        "if you want",
        "if you'd like",
        "if you need",
        "do you want me",
        "personal preferences",
        "personal experiences",
        "no personal",
        "none, i don't",
        "none, i do not",
        "none, i don’t",
        "i don't take",
        "i do not take",
        "i don’t take",
        "i don't travel",
        "i do not travel",
        "i don’t travel",
        "no secret talents",
        "no favorite",
        "no favourite",
        "no memories",
        "made to assist",
        "just chat",
        "no classes",
        "haven't met",
        "have not met",
        "no, haven't",
        "no, have not",
        "no, i haven't",
        "no, i have not",
        "honestly, i'm not",
        "honestly, i’m not",
        "that's hardcore",
        "my friends",
        "my family",
        "my childhood",
    ]
    for bit in bad_bits:
        if bit in lowered:
            return False, f"contains {bit}"
    if re.search(r"[\U0001F300-\U0001FAFF]", answer):
        return False, "emoji"
    if not answer:
        return False, "empty"
    if len(answer.split()) > 55:
        return False, "too_long"
    return True, "ok"


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def parse_batch_answers(text: str) -> dict[str, str] | None:
    cleaned = strip_json_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None

    items = data.get("answers") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return None

    out: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        answer = item.get("assistant") or item.get("answer") or item.get("response")
        if item_id is None or not isinstance(answer, str):
            continue
        out[str(item_id)] = answer.strip()
    return out or None


def batch_user_prompt(rows: list[dict]) -> str:
    items = [{"id": row["_local_id"], "user": row["question"]} for row in rows]
    return (
        "Write exactly one Poke-style assistant reply for each user message below.\n"
        "Return only a JSON object in this exact shape:\n"
        '{"answers":[{"id":"0","assistant":"short reply"}]}\n\n'
        "Do not include role labels, markdown, <think>, explanations, or extra keys.\n"
        "Keep each reply short, grounded in the user's text, and natural over text.\n\n"
        f"Messages:\n{json.dumps(items, ensure_ascii=False)}"
    )


def call_openai(args: argparse.Namespace, system: str, user: str, max_tokens: int | None = None) -> str:
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
        max_completion_tokens=max_tokens or args.max_output_tokens or 280,
    )
    return (resp.choices[0].message.content or "").strip()


def call_ollama(args: argparse.Namespace, system: str, user: str, max_tokens: int | None = None) -> str:
    base = args.base_url or "http://localhost:11434"
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                **({"num_predict": max_tokens or args.max_output_tokens} if (max_tokens or args.max_output_tokens) else {}),
            },
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


def answer_one(args: argparse.Namespace, system: str, row: dict) -> tuple[str, str]:
    raw = call_ollama(args, system, row["question"]) if args.provider == "ollama" else call_openai(args, system, row["question"])
    return raw, clean_answer(raw)


def answer_batch(args: argparse.Namespace, system: str, rows: list[dict]) -> list[tuple[dict, str, str]]:
    if len(rows) <= 1:
        raw, answer = answer_one(args, system, rows[0])
        return [(rows[0], raw, answer)]

    max_tokens = args.max_output_tokens or max(500, min(3500, 120 * len(rows)))
    prompt = batch_user_prompt(rows)
    raw = call_ollama(args, system, prompt, max_tokens=max_tokens) if args.provider == "ollama" else call_openai(args, system, prompt, max_tokens=max_tokens)
    parsed = parse_batch_answers(raw)
    if parsed and all(row["_local_id"] in parsed for row in rows):
        return [(row, raw, clean_answer(parsed[row["_local_id"]])) for row in rows]

    midpoint = len(rows) // 2
    return answer_batch(args, system, rows[:midpoint]) + answer_batch(args, system, rows[midpoint:])


def answer_batch_with_retries(args: argparse.Namespace, system: str, rows: list[dict]) -> list[tuple[dict, str, str]]:
    for attempt in range(args.retries):
        try:
            return answer_batch(args, system, rows)
        except Exception:
            if attempt + 1 >= args.retries:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable retry state")


def load_questions(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question = (row.get("question") or row.get("prompt") or "").strip()
            if question:
                row["question"] = question
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    system = build_system()
    questions = load_questions(args.questions, args.limit)
    out = args.out or (DEFAULT_OUT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    raw_out = out.with_suffix(".raw.jsonl")
    rejected_out = out.with_suffix(".rejected.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    rejected = 0
    with out.open("w", encoding="utf-8") as dst, raw_out.open("w", encoding="utf-8") as raw_dst, rejected_out.open("w", encoding="utf-8") as rej_dst:
        batch_size = max(1, args.batch_size)
        batches = []
        for start in range(0, len(questions), batch_size):
            batch = []
            for offset, row in enumerate(questions[start : start + batch_size]):
                row = dict(row)
                row["_ix"] = start + offset
                row["_local_id"] = str(start + offset)
                batch.append(row)
            batches.append(batch)

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [
                executor.submit(answer_batch_with_retries, args, system, batch)
                for batch in batches
            ]
            completed_batches = 0
            for future in as_completed(futures):
                completed_batches += 1
                for row, raw, answer in future.result():
                    ix = row["_ix"]
                    user = row["question"]
                    ok, reason = valid_answer(answer)
                    raw_dst.write(
                        json.dumps(
                            {
                                "ix": ix,
                                "user": user,
                                "raw": raw,
                                "answer": answer,
                                "ok": ok,
                                "reason": reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if not ok:
                        rejected += 1
                        rej_dst.write(
                            json.dumps(
                                {"ix": ix, "user": user, "answer": answer, "reason": reason},
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
                                "source_question_id": row.get("id"),
                                "source": "humanllms_poke_amended_response",
                                "answer_model": args.model,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    kept += 1
                    print(
                        f"batch={completed_batches}/{len(batches)} "
                        f"row={ix + 1}/{len(questions)} kept={kept} rejected={rejected}",
                        flush=True,
                    )

    print(f"sft:      {out}")
    print(f"raw:      {raw_out}")
    print(f"rejected: {rejected_out}")
    print(f"kept={kept} rejected={rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
