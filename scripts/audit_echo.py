"""Audit generated JSONL for Poke replies that echo the latest user message."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STOPWORDS = {
    "about",
    "absolutely",
    "actually",
    "again",
    "all",
    "also",
    "already",
    "always",
    "and",
    "anything",
    "apparently",
    "are",
    "basically",
    "because",
    "being",
    "but",
    "can",
    "cant",
    "completely",
    "could",
    "did",
    "didnt",
    "does",
    "doesnt",
    "doing",
    "dont",
    "even",
    "exactly",
    "feel",
    "feels",
    "for",
    "from",
    "gonna",
    "have",
    "how",
    "honestly",
    "its",
    "just",
    "kind",
    "know",
    "like",
    "literally",
    "maybe",
    "more",
    "mostly",
    "normal",
    "not",
    "okay",
    "one",
    "probably",
    "really",
    "right",
    "should",
    "something",
    "still",
    "than",
    "that",
    "thats",
    "them",
    "then",
    "there",
    "the",
    "thing",
    "things",
    "think",
    "this",
    "though",
    "too",
    "trying",
    "valid",
    "want",
    "wanted",
    "wants",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
    "your",
    "yeah",
}

REFLECTIVE_REPLY_LEADS = {
    "yeah",
    "yep",
    "exactly",
    "right",
    "totally",
    "basically",
    "thats",
    "that",
    "this",
    "it",
    "its",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument("--fail-on-issues", action="store_true")
    p.add_argument("--accepted-output", type=Path, default=None)
    p.add_argument("--rejected-output", type=Path, default=None)
    return p.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                row["_audit_source"] = str(path)
                row["_audit_line"] = line_no
                out.append(row)
    return out


def normalized_words(text: str) -> list[str]:
    return [
        raw.strip("'").replace("'", "")
        for raw in re.findall(r"[a-z0-9][a-z0-9']+", text.lower())
        if raw.strip("'")
    ]


def content_words(text: str) -> list[str]:
    words: list[str] = []
    for word in normalized_words(text):
        if word.endswith("s") and len(word) > 4:
            word = word[:-1]
        if len(word) >= 3 and word not in STOPWORDS:
            words.append(word)
    return words


def longest_common_run(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
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


def ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def has_reflective_lead(words: list[str]) -> bool:
    return bool(words and words[0] in REFLECTIVE_REPLY_LEADS)


def echo_reason(user: str, reply: str) -> str | None:
    user_norm = normalized_words(user)
    reply_norm = normalized_words(reply)
    if not user_norm or not reply_norm:
        return None
    if len(user_norm) <= 3 and user_norm == reply_norm[: len(user_norm)]:
        return "short_exact_echo"

    user_content = content_words(user)
    reply_content = content_words(reply)
    if longest_common_run(user_content, reply_content) >= 4:
        return "long_common_phrase"

    reflective_lead = has_reflective_lead(reply_norm)
    for n in (5, 4):
        if ngrams(user_norm, n) & ngrams(reply_norm, n):
            return f"exact_{n}gram"
    if reflective_lead and ngrams(user_content, 3) & ngrams(reply_content, 3):
        return "reflective_exact_3gram"

    shared = len(set(user_content) & set(reply_content))
    if not shared:
        return None
    user_coverage = shared / max(1, len(set(user_content)))
    reply_coverage = shared / max(1, len(set(reply_content)))
    agreement = reply_norm[0] in {"yeah", "yep", "exactly", "right", "totally"}
    if reflective_lead and shared >= 5 and reply_coverage >= 0.60:
        return "reflective_high_overlap"
    if agreement and shared >= 3 and reply_coverage >= 0.45:
        return "agreement_rephrase"
    if shared >= 5 and user_coverage >= 0.55 and reply_coverage >= 0.45:
        return "high_overlap"
    return None


def main() -> int:
    args = parse_args()
    checked = 0
    issues: list[dict[str, str]] = []
    all_rows: list[dict[str, Any]] = []
    for path in args.inputs:
        all_rows.extend(rows(path))
    rejected_ids: set[str] = set()
    for row in all_rows:
        latest_user = ""
        for turn_ix, turn in enumerate(row.get("turns", [])):
            role = turn.get("role")
            content = str(turn.get("content", ""))
            if role == "user":
                latest_user = content
                continue
            if role != "assistant" or not content:
                continue
            checked += 1
            reason = echo_reason(latest_user, content)
            if reason:
                rejected_ids.add(str(row.get("id")))
                issues.append(
                    {
                        "id": str(row.get("id")),
                        "source": str(row.get("_audit_source", "")),
                        "line": str(row.get("_audit_line", "")),
                        "turn": str(turn_ix),
                        "reason": reason,
                        "user": latest_user,
                        "assistant": content,
                    }
                )

    print(f"checked assistant turns: {checked}")
    print(f"echo issues: {len(issues)}")
    if checked:
        print(f"issue rate: {len(issues) / checked:.1%}")
    for issue in issues[: args.max_examples]:
        print("-" * 100)
        print(
            f"{issue['id']} {issue['source']}:{issue['line']} "
            f"turn={issue['turn']} reason={issue['reason']}"
        )
        print("USER:", issue["user"])
        print("POKE:", issue["assistant"])

    if args.accepted_output or args.rejected_output:
        if not args.accepted_output or not args.rejected_output:
            if len(args.inputs) != 1:
                raise SystemExit(
                    "--accepted-output and --rejected-output are required with multiple inputs"
                )
        first_input = args.inputs[0]
        accepted_output = args.accepted_output or first_input.with_suffix(
            first_input.suffix + ".no_echo.jsonl"
        )
        rejected_output = args.rejected_output or first_input.with_suffix(
            first_input.suffix + ".echo_rejected.jsonl"
        )
        accepted_output.parent.mkdir(parents=True, exist_ok=True)
        rejected_output.parent.mkdir(parents=True, exist_ok=True)
        accepted = rejected = 0
        with accepted_output.open("w", encoding="utf-8") as good, rejected_output.open(
            "w", encoding="utf-8"
        ) as bad:
            for row in all_rows:
                row_id = str(row.get("id"))
                row.pop("_audit_source", None)
                row.pop("_audit_line", None)
                if row_id in rejected_ids:
                    bad.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rejected += 1
                else:
                    good.write(json.dumps(row, ensure_ascii=False) + "\n")
                    accepted += 1
        print(f"accepted rows: {accepted_output} ({accepted})")
        print(f"rejected rows: {rejected_output} ({rejected})")
    return 1 if args.fail_on_issues and issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
