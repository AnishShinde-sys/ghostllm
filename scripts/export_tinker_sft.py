"""Export successful GhostLLM conversations to Tinker chat SFT JSONL.

Tinker Cookbook's chat SFT recipe expects one JSON object per line with a
`messages` list. This script converts the generator shard rows into that shape
and keeps only clean assistant-training examples.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


TURN_WEIGHTS = {
    4: 0.20,
    6: 0.30,
    8: 0.30,
    10: 0.067,
    12: 0.067,
    14: 0.066,
}

GENERIC_ONE_WORD = {"thanks", "ok", "okay", "yeah", "yep", "sure"}
WORD_RE = re.compile(r"[a-z0-9']+")
TRANSCRIPT_META_PATTERNS = (
    "you're repeating yourself",
    "you are repeating yourself",
    "you already said",
    "you said that twice",
    "you've said that twice",
    "you listed it twice",
    "you just asked me",
    "you asked me",
    "you're looping",
    "you are looping",
    "i asked you",
    "i asked what",
    "i'm asking you",
    "i'm waiting on you",
    "you asked what",
    "not me",
)
IDENTITY_LEAK_PATTERNS = (
    "deadpool",
    "marvel",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--system-prompt", type=Path)
    p.add_argument("--no-system", action="store_true")
    p.add_argument("--balance-turns", action="store_true", default=True)
    p.add_argument("--no-balance-turns", dest="balance_turns", action="store_false")
    p.add_argument("--min-turns", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=14)
    p.add_argument("--manifest", type=Path)
    p.add_argument(
        "--explode-assistant-turns",
        action="store_true",
        help=(
            "Write one training record per assistant turn, using the conversation "
            "prefix through that assistant message. Pair with train_on=LAST_ASSISTANT_MESSAGE."
        ),
    )
    p.add_argument(
        "--strict-quality",
        action="store_true",
        help="Also reject assistant/user overlap and repeated assistant phrasing.",
    )
    return p.parse_args()


def normalized_words(text: str) -> list[str]:
    text = text.replace("\\n", "\n")
    return WORD_RE.findall(text.lower())


def clean_content(text: Any) -> str:
    return str(text or "").replace("\\n", "\n").strip()


def longest_common_word_run(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    best = 0
    for aw in a:
        cur = [0] * (len(b) + 1)
        for j, bw in enumerate(b, 1):
            if aw == bw:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def content_words(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "for",
        "i",
        "if",
        "in",
        "is",
        "it",
        "like",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "with",
        "you",
        "your",
    }
    return {w for w in normalized_words(text) if w not in stop and len(w) > 2}


def too_echoed(user_text: str, assistant_text: str) -> bool:
    user_words = normalized_words(user_text)
    assistant_words = normalized_words(assistant_text)
    if not user_words or not assistant_words:
        return False
    if len(user_words) <= 3 and user_words == assistant_words[: len(user_words)]:
        return True
    if longest_common_word_run(user_words, assistant_words) >= 5:
        return True
    user_set = content_words(user_text)
    assistant_set = content_words(assistant_text)
    if len(user_set) < 4 or len(assistant_set) < 4:
        return False
    overlap = len(user_set & assistant_set)
    return overlap >= 5 and overlap / max(1, len(assistant_set)) >= 0.60


def assistant_repeats_itself(assistant_messages: list[str]) -> bool:
    previous: list[set[str]] = []
    previous_text: list[list[str]] = []
    for msg in assistant_messages:
        words = content_words(msg)
        norm = normalized_words(msg)
        if len(words) >= 4:
            for old in previous:
                overlap = len(words & old) / max(1, min(len(words), len(old)))
                if overlap >= 0.60:
                    return True
        if len(norm) >= 6:
            for old_norm in previous_text:
                if longest_common_word_run(norm, old_norm) >= 5:
                    return True
        previous.append(words)
        previous_text.append(norm)
    return False


def messages_repeat(messages: list[str], min_words: int = 4) -> bool:
    previous_words: list[set[str]] = []
    previous_text: list[list[str]] = []
    seen_exact: set[str] = set()
    for msg in messages:
        norm = normalized_words(msg)
        if len(norm) < min_words:
            continue
        exact = " ".join(norm)
        if exact in seen_exact:
            return True
        seen_exact.add(exact)

        words = content_words(msg)
        if len(words) >= min_words:
            for old in previous_words:
                overlap = len(words & old) / max(1, min(len(words), len(old)))
                if overlap >= 0.80:
                    return True

        if len(norm) >= 7:
            for old_norm in previous_text:
                if longest_common_word_run(norm, old_norm) >= 7:
                    return True

        previous_words.append(words)
        previous_text.append(norm)
    return False


def user_repeats_itself(user_messages: list[str]) -> bool:
    previous_text: list[list[str]] = []
    seen_exact: set[str] = set()
    for msg in user_messages:
        norm = normalized_words(msg)
        if len(norm) < 4:
            continue
        exact = " ".join(norm)
        if exact in seen_exact:
            return True
        seen_exact.add(exact)

        words = content_words(msg)
        if len(words) < 4:
            previous_text.append(norm)
            continue
        for old_norm in previous_text:
            if len(old_norm) < 4:
                continue
            common = longest_common_word_run(norm, old_norm)
            if common >= 9 and common / max(1, min(len(norm), len(old_norm))) >= 0.90:
                return True
        previous_text.append(norm)
    return False


def transcript_meta_reply(content: str) -> bool:
    low = content.lower()
    if any(pattern in low for pattern in TRANSCRIPT_META_PATTERNS):
        return True
    return False


def identity_leak(content: str) -> bool:
    low = content.lower()
    return any(pattern in low for pattern in IDENTITY_LEAK_PATTERNS)


def clean_row(
    row: dict[str, Any],
    min_turns: int,
    max_turns: int,
    strict_quality: bool,
) -> tuple[bool, str]:
    if row.get("error"):
        return False, "source_error"
    turns = row.get("turns")
    if not isinstance(turns, list):
        return False, "missing_turns"
    if len(turns) < min_turns or len(turns) > max_turns or len(turns) % 2:
        return False, "bad_turn_count"

    assistant_messages: list[str] = []
    user_messages: list[str] = []
    latest_user = ""
    for ix, turn in enumerate(turns):
        role = turn.get("role")
        content = clean_content(turn.get("content"))
        expected_role = "user" if ix % 2 == 0 else "assistant"
        if role != expected_role:
            return False, "bad_role_order"
        if not content:
            return False, "empty_content"
        if role == "user":
            latest_user = content
            user_messages.append(content)
            continue
        low = content.lower()
        if low in GENERIC_ONE_WORD or "[empty]" in low:
            return False, "generic_or_empty_assistant"
        if len(content.split()) < 2:
            return False, "too_short_assistant"
        if strict_quality and identity_leak(content):
            return False, "assistant_identity_leak"
        if strict_quality and transcript_meta_reply(content):
            return False, "transcript_meta_reply"
        if strict_quality and too_echoed(latest_user, content):
            return False, "assistant_echo"
        assistant_messages.append(content)

    if strict_quality:
        if user_repeats_itself(user_messages):
            return False, "user_self_repeat"
        if assistant_repeats_itself(assistant_messages) or messages_repeat(assistant_messages):
            return False, "assistant_self_repeat"
    return True, "ok"


def salvage_clean_prefix(
    row: dict[str, Any],
    min_turns: int,
    max_turns: int,
    strict_quality: bool,
) -> tuple[dict[str, Any], str] | None:
    if row.get("error"):
        return None
    turns = row.get("turns")
    if not isinstance(turns, list):
        return None
    original_turns = len(turns)
    start = min(max_turns, original_turns)
    if start % 2:
        start -= 1
    if start >= original_turns:
        start -= 2
    for turn_count in range(start, min_turns - 1, -2):
        candidate = dict(row)
        candidate["turns"] = turns[:turn_count]
        ok, reason = clean_row(candidate, min_turns, max_turns, strict_quality)
        if ok:
            candidate["_salvaged_from_turns"] = original_turns
            return candidate, reason
    return None


def iter_rows(shards: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(shards.glob("*.jsonl")):
        for line_no, line in enumerate(path.open("r", encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_source_file"] = path.name
            row["_source_line"] = line_no
            out.append(row)
    return out


def turn_caps(limit: int) -> dict[int, int]:
    raw = {turns: limit * weight for turns, weight in TURN_WEIGHTS.items()}
    caps = {turns: math.floor(value) for turns, value in raw.items()}
    remaining = limit - sum(caps.values())
    for turns, _value in sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True):
        if remaining <= 0:
            break
        caps[turns] += 1
        remaining -= 1
    return caps


def select_rows(rows: list[dict[str, Any]], limit: int, balance_turns: bool) -> list[dict[str, Any]]:
    if not balance_turns:
        return rows[:limit]

    caps = turn_caps(limit)
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    counts: Counter[int] = Counter()
    for row in rows:
        turn_count = len(row.get("turns", []))
        if counts[turn_count] >= caps.get(turn_count, 0):
            continue
        selected.append(row)
        used_ids.add(str(row.get("id")))
        counts[turn_count] += 1
        if len(selected) >= limit:
            return selected

    # If the still-growing run does not yet have enough long rows, fill the
    # pilot snapshot with otherwise clean examples rather than waiting forever.
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in used_ids:
            continue
        selected.append(row)
        used_ids.add(row_id)
        if len(selected) >= limit:
            break
    return selected


def tinker_record(row: dict[str, Any], system_prompt: str | None) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in row["turns"]:
        messages.append({"role": turn["role"], "content": clean_content(turn.get("content"))})
    return {
        "messages": messages,
        "source_id": row.get("id"),
        "category": row.get("category"),
        "topic": row.get("topic"),
        "turns": len(row.get("turns", [])),
        "salvaged_from_turns": row.get("_salvaged_from_turns"),
        "source_file": row.get("_source_file"),
        "source_line": row.get("_source_line"),
    }


def tinker_records(row: dict[str, Any], system_prompt: str | None, explode_assistant_turns: bool) -> list[dict[str, Any]]:
    if not explode_assistant_turns:
        return [tinker_record(row, system_prompt)]

    records: list[dict[str, Any]] = []
    turns = row["turns"]
    for ix, turn in enumerate(turns):
        if turn.get("role") != "assistant":
            continue
        prefix_row = dict(row)
        prefix_row["turns"] = turns[: ix + 1]
        record = tinker_record(prefix_row, system_prompt)
        source_id = row.get("id")
        record["source_id"] = f"{source_id}#assistant_turn_{ix}" if source_id is not None else None
        record["source_conversation_id"] = source_id
        record["target_turn_index"] = ix
        record["source_total_turns"] = len(turns)
        records.append(record)
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    system_prompt = None
    if not args.no_system:
        if not args.system_prompt:
            raise SystemExit("--system-prompt is required unless --no-system is set")
        system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()

    rejects: Counter[str] = Counter()
    salvaged: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    while True:
        rejects.clear()
        salvaged.clear()
        eligible.clear()
        for row in iter_rows(args.shards):
            ok, reason = clean_row(row, args.min_turns, args.max_turns, args.strict_quality)
            if ok:
                eligible.append(row)
                continue
            fixed = salvage_clean_prefix(row, args.min_turns, args.max_turns, args.strict_quality)
            if fixed:
                fixed_row, _fixed_reason = fixed
                eligible.append(fixed_row)
                salvaged[reason] += 1
                continue
            rejects[reason] += 1

        if len(eligible) >= args.limit or not args.wait:
            break
        print(
            f"waiting for {args.limit} clean rows: have {len(eligible)} "
            f"(rejects={sum(rejects.values())}); sleeping {args.poll_seconds}s",
            flush=True,
        )
        time.sleep(args.poll_seconds)

    if len(eligible) < args.limit:
        raise SystemExit(f"only {len(eligible)} clean rows available, need {args.limit}")

    selected = select_rows(eligible, args.limit, args.balance_turns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in selected:
            for record in tinker_records(row, system_prompt, args.explode_assistant_turns):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

    turns = Counter(len(row.get("turns", [])) for row in selected)
    categories = Counter(str(row.get("category")) for row in selected)
    manifest = {
        "output": str(args.output),
        "source_shards": str(args.shards),
        "limit": args.limit,
        "selected": len(selected),
        "records_written": records_written,
        "eligible": len(eligible),
        "balance_turns": args.balance_turns,
        "explode_assistant_turns": args.explode_assistant_turns,
        "include_system": bool(system_prompt),
        "strict_quality": args.strict_quality,
        "turns": dict(sorted(turns.items())),
        "categories": dict(categories.most_common()),
        "rejects": dict(rejects.most_common()),
        "salvaged": dict(salvaged.most_common()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    write_json(manifest_path, manifest)
    print(
        "exported {selected} rows / {records} records to {output}; turns={turns}; manifest={manifest}".format(
            selected=len(selected),
            records=records_written,
            output=args.output,
            turns=dict(sorted(turns.items())),
            manifest=manifest_path,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
