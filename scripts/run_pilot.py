"""Rolling batched synthetic conversation generator.

This is intentionally one file. The generation model is:

1. Keep a fixed number of independent conversation slots active.
2. Batch all slots that need a fake-user turn into one or more GPT calls.
3. Batch all slots that need a Poke turn into one or more Claude/OpenAI calls.
4. When a slot completes, write it and immediately refill it.

The batches are breadth batches: every item is a different user, topic, and
trajectory. They are not alternate continuations of one conversation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPT_PATH = ROOT / "prompts" / "conversation_only_poke_prompt.txt"

load_dotenv(ROOT / ".env", override=True)

POKE_MODEL = os.environ.get("POKE_MODEL", "claude-sonnet-4-5-20250929")
USER_MODEL = os.environ.get("USER_MODEL", os.environ.get("PERSONA_MODEL", "gpt-5.4-mini"))
USER_PROMPT_PROFILE = os.environ.get("USER_PROMPT_PROFILE", "balanced")

POKE_TEMPERATURE = float(os.environ.get("POKE_TEMPERATURE", "0.7"))
USER_TEMPERATURE = float(os.environ.get("USER_TEMPERATURE", "0.9"))

# Public model ceilings are only one side of the real limit. Org/project rate
# limits still matter, so these defaults stay below known context/output caps.
USER_CONTEXT_TOKENS = int(os.environ.get("USER_CONTEXT_TOKENS", "380000"))
POKE_CONTEXT_TOKENS = int(os.environ.get("POKE_CONTEXT_TOKENS", "180000"))
USER_BATCH_OUTPUT_TOKENS = int(os.environ.get("USER_BATCH_OUTPUT_TOKENS", "24000"))
POKE_BATCH_OUTPUT_TOKENS = int(os.environ.get("POKE_BATCH_OUTPUT_TOKENS", "7600"))

USER_REPLY_TOKENS = int(os.environ.get("USER_REPLY_TOKENS", "128"))
POKE_REPLY_TOKENS = int(os.environ.get("POKE_REPLY_TOKENS", "128"))
BATCH_UTILIZATION = float(os.environ.get("BATCH_UTILIZATION", "0.85"))

TURN_DISTRIBUTION: list[tuple[int, int]] = [
    # Each Poke turn is paired with a user turn, so total message count is
    # roughly 2x this value. This gives:
    # - 20% short rows: 4 messages
    # - 60% medium rows: 6-8 messages
    # - 20% long rows: 10-14 messages
    (2, 20),
    (3, 30),
    (4, 30),
    (5, 7),
    (6, 7),
    (7, 6),
]
CATEGORY_TURN_DISTRIBUTIONS: dict[str, list[tuple[int, int]]] = {}
EARLY_STOP_PROB = 0.0
END_SENTINELS = {"[END]", "<END>", "__END__", "END"}
MAX_POKE_REPAIR_ATTEMPTS = 2

TOPICS: dict[str, list[str]] = {
    "just_chatting": [
        "user opens with a bored message and no specific task",
        "user sends a casual hey and wants to chat",
        "user says they are tired today with no question",
        "user mentions they are killing time before something",
        "user casually asks what Poke is up to",
        "user sends a low-effort message because they are bored",
        "user shares a tiny pointless update",
    ],
    "status_updates": [
        "user just got home",
        "user says they are at a coffee shop now",
        "user says they are running late",
        "user finally did the dishes",
        "user says they are back and something took forever",
        "user says they made it to class or work",
        "user gives a small update from an errand",
    ],
    "reactions_to_things": [
        "user reacts to a passive aggressive email",
        "user reacts to something a friend said",
        "user reacts to a movie not being what they expected",
        "user reacts to an old song coming on",
        "user reacts to a text they just received",
        "user reacts to something weird happening nearby",
        "user reacts to a line being much longer than expected",
    ],
    "venting": [
        "user says they are exhausted",
        "user complains that everything is an errand",
        "user complains about still having several things to do",
        "user says their back hurts for no clear reason",
        "user complains about a minor inconvenience",
        "user says today feels too long",
        "user vents about being hungry and annoyed",
    ],
    "factual_qa": [
        "user asks for a concise explanation of the Great Wall of China",
        "user asks why the ocean is salty",
        "user asks how airplanes stay in the air",
        "user asks what black holes are in plain English",
        "user asks why leaves change color",
        "user asks what the Roman Empire was",
        "user asks how volcanoes work",
        "user asks what photosynthesis is without sounding like a textbook",
    ],
    "simple_explainers": [
        "user asks what a Markov chain is with a simple example",
        "user asks the difference between correlation and causation",
        "user asks why inflation happens",
        "user asks how sleep cycles work",
        "user asks what compound interest means",
        "user asks how memory works at a basic level",
        "user asks what probability means intuitively",
        "user asks for a simple explanation of supply and demand",
    ],
    "practical_qa": [
        "user asks how to make a short apology sound normal",
        "user asks how to start cleaning a messy room",
        "user asks how to politely leave a group chat",
        "user asks how to make coffee taste less bitter",
        "user asks what to bring to a casual dinner",
        "user asks how to stop overthinking a text before sending it",
        "user asks how to make a boring errand feel less annoying",
    ],
    "writing_help": [
        "user wants help wording a text and includes the draft",
        "user wants help making an email sound less awkward and includes the draft",
        "user wants help making a cancellation text sound normal",
        "user wants help replying to a friend without sounding rude",
        "user wants help shortening a message before sending it",
        "user wants help making a note to their roommate sound less passive aggressive",
    ],
    "random_questions": [
        "user asks a random question they thought of in the shower",
        "user asks whether a very specific everyday thing is weird",
        "user asks Poke to pick between two silly options",
        "user asks a pointless but oddly serious question",
        "user asks for a strong opinion on a low-stakes topic",
        "user asks why something normal feels embarrassing",
    ],
    "weird_hypotheticals": [
        "user asks an absurd what-if question",
        "user asks what object has the worst vibes",
        "user asks what food would be most suspicious if it could talk",
        "user asks how long they would survive in a harmless surreal scenario",
        "user asks Poke to rate a fake invention idea",
        "user asks a tiny moral dilemma with no real stakes",
    ],
    "taste_debates": [
        "user wants Poke's take on a specific controversial snack opinion",
        "user asks if a specific music taste is embarrassing",
        "user asks whether a specific movie opinion is valid",
        "user asks if a specific room decor choice has bad energy",
        "user wants Poke to judge a specific bad impulse purchase",
        "user asks whether a specific phrase sounds cringe",
    ],
    "smalltalk": [
        "user says good morning",
        "user complains about Monday",
        "user is bored",
        "user shares a random thought",
        "user is procrastinating and wants to chat",
        "user asks how Poke is doing",
        "user sends a tiny update about their day",
    ],
    "life_updates": [
        "user tells Poke they did one productive thing",
        "user complains about being hungry and indecisive",
        "user says they are avoiding a small chore",
        "user admits they are being dramatic about something minor",
        "user is proud of doing a tiny task",
        "user is annoyed at themselves for wasting time",
        "user says they need to reset their mood",
    ],
    "emotional_light": [
        "user is mildly stressed about something",
        "user feels rejected by a small interaction",
        "user is excited about good news",
        "user had a rough conversation with a friend",
        "user is feeling a little burned out",
        "user is anxious about something coming up",
        "user wants reassurance without a big speech",
    ],
    "poke_reactions": [
        "user asks Poke to roast them very gently about a specific harmless habit",
        "user asks Poke for a tiny pep talk",
        "user asks Poke to be honest about a specific bad idea",
        "user asks Poke to make a specific boring thought sound dramatic",
        "user asks Poke to give a one-line verdict on a specific situation",
        "user asks Poke to name the vibe of their day after sharing a few details",
    ],
    "edge_cases": [
        "user is rude or testing Poke",
        "user asks a deeply philosophical question out of nowhere",
        "user sends a single emoji",
        "user tries to get Poke to break character",
        "user sends a typo-laden one-word message",
        "user sends an ambiguous one-word reply",
        "user contradicts themselves casually",
    ],
    "closing": [
        "user says thanks, conversation winding down",
        "user has gotten what they need and is going to bed",
        "user got distracted and is ending the chat",
        "user says never mind",
    ],
}

CATEGORY_WEIGHTS: dict[str, float] = {
    "just_chatting": 0.15,
    "status_updates": 0.08,
    "reactions_to_things": 0.08,
    "venting": 0.08,
    "smalltalk": 0.10,
    "emotional_light": 0.08,
    "practical_qa": 0.08,
    "factual_qa": 0.08,
    "simple_explainers": 0.08,
    "writing_help": 0.06,
    "taste_debates": 0.05,
    "edge_cases": 0.05,
    "closing": 0.03,
    # Kept in TOPICS for targeted experiments, but excluded from the default
    # balanced mix after adding the more realistic casual categories above.
    "life_updates": 0.0,
    "random_questions": 0.0,
    "poke_reactions": 0.0,
    "weird_hypotheticals": 0.0,
}

USER_PROMPT_PROFILES: dict[str, str] = {
    "balanced": """Bias toward grounded, realistic user messages across the full
taxonomy, with more normal Q&A and practical asks than pure banter. The user
should sound ordinary, specific, and answerable without being polished or
performing a quirky persona.""",
    "utility": """Bias toward useful task-shaped messages: factual questions, simple
explainers, practical advice, wording help, decisions, and concrete everyday
problems. The user should provide enough context for Poke to answer directly.""",
    "messy_human": """Bias toward naturally messy human texting: mild uncertainty,
small corrections, imperfect phrasing, and emotional texture. Messy means real
and slightly unfinished, not theatrical, whimsical, or prompt-writer cute.""",
    "context_rich": """Bias toward context-rich openings. The user should include the
specific situation, why they care, and any constraint in the first message so
Poke can produce a strong answer immediately. Prefer practical context from
school, work, friends, family, roommates, errands, food, money, schedules, or
actual things they are trying to decide.""",
    "followup_realism": """Bias toward high-quality continuation turns. The first user
message can be simple, but follow-ups should add new information, push back,
ask a real next question, or naturally wind down. Avoid repetitive loops.""",
}

GROUNDING_STOPWORDS = {
    "about",
    "actually",
    "again",
    "also",
    "already",
    "always",
    "anything",
    "apparently",
    "because",
    "being",
    "could",
    "does",
    "doesnt",
    "doing",
    "dont",
    "even",
    "feel",
    "feels",
    "from",
    "gonna",
    "have",
    "just",
    "kind",
    "know",
    "like",
    "maybe",
    "more",
    "mostly",
    "normal",
    "okay",
    "probably",
    "really",
    "should",
    "something",
    "still",
    "than",
    "that",
    "thats",
    "them",
    "then",
    "there",
    "thing",
    "things",
    "think",
    "this",
    "though",
    "trying",
    "want",
    "wanted",
    "wants",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "yeah",
}

ECHO_STOPWORDS = GROUNDING_STOPWORDS | {
    "absolutely",
    "all",
    "and",
    "are",
    "basically",
    "but",
    "can",
    "cant",
    "completely",
    "did",
    "didnt",
    "exactly",
    "for",
    "how",
    "honestly",
    "its",
    "literally",
    "not",
    "one",
    "right",
    "the",
    "too",
    "valid",
    "was",
    "were",
    "you",
    "your",
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

USER_DIMENSIONS: dict[str, list[str]] = {
    "texting_style": [
        "plain and direct",
        "casual and a little messy",
        "warm and chatty",
        "dry and understated",
        "restless and tangent-prone",
        "careful and literal",
        "soft-spoken and hesitant",
        "quick, impatient, and abbreviated",
        "playful but not performative",
        "blunt but not mean",
    ],
    "message_shape": [
        "mostly one short line",
        "one or two short texts",
        "several tiny bursts separated by blank lines",
        "a compact paragraph when thinking",
        "starts vague, then adds context if asked",
        "overexplains a little when anxious",
        "answers with fragments and corrections",
    ],
    "punctuation": [
        "mostly lowercase",
        "normal capitalization",
        "minimal punctuation",
        "clean punctuation",
        "lots of commas and half-finished thoughts",
        "occasional typos without making it unreadable",
    ],
    "emoji_style": [
        "no emojis",
        "rare emojis",
        "occasional common emojis",
        "uses haha/lol more than emojis",
        "uses one emoji when emotionally obvious",
    ],
    "mood": [
        "neutral",
        "a little tired",
        "mildly stressed",
        "curious",
        "distracted",
        "quietly excited",
        "annoyed but trying to be fair",
        "goofy",
        "uncertain",
    ],
    "relationship_to_poke": [
        "new and testing the vibe",
        "casual regular",
        "treats Poke like a friend",
        "skeptical but open",
        "uses Poke as a low-stakes place to ramble",
    ],
    "behavior": [
        "asks for a gut check",
        "wants to be heard more than fixed",
        "changes their mind mid-thread",
        "gives too little context at first",
        "pushes back if the answer feels generic",
        "makes a small joke when uncomfortable",
        "responds naturally to follow-up questions",
        "occasionally sends a one-word reply",
    ],
    "context": [
        "late at night",
        "during a quick break",
        "while procrastinating",
        "between errands",
        "after a long day",
        "first thing in the morning",
        "while walking somewhere",
    ],
}


@dataclass
class Trajectory:
    id: str
    user: str
    category: str
    topic: str
    scenario: str
    user_spec: dict[str, str]
    user_prompt_profile: str
    target_poke_turns: int
    next_actor: str = "user"
    turns: list[dict[str, Any]] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    error: str | None = None

    @property
    def poke_turns(self) -> int:
        return sum(1 for t in self.turns if t["role"] == "assistant")

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user": self.user,
            "category": self.category,
            "topic": self.topic,
            "scenario": self.scenario,
            "user_spec": self.user_spec,
            "user_prompt_profile": self.user_prompt_profile,
            "turns": self.turns,
            "poke_model": POKE_MODEL,
            "user_model": USER_MODEL,
            "error": self.error,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
}

SCENARIOS: dict[str, list[str]] = {
    "just_chatting": [
        "they have no task and are texting like they would text a friend",
        "they are bored and want a normal low-effort exchange",
        "they are tired and just saying what is happening",
        "they are waiting around and sending a small thought",
    ],
    "status_updates": [
        "they are sharing where they are or what just happened",
        "they finished or started a small ordinary thing",
        "they are late, back, home, outside, or between places",
        "they give a tiny update without asking for help",
    ],
    "reactions_to_things": [
        "they saw, heard, read, or received something specific and react to it",
        "they paste or describe the concrete thing they are reacting to",
        "they are not asking for analysis yet, just sharing the reaction",
        "they may turn it into a question later, but the opener is just a reaction",
    ],
    "venting": [
        "they complain about a concrete annoying thing from today",
        "they are tired, busy, or irritated and not necessarily asking for advice",
        "they want Poke to meet the mood more than solve the problem",
        "they mention the actual errand, chore, deadline, body ache, person, or place",
    ],
    "factual_qa": [
        "they saw a reference in a video and want the short useful explanation",
        "they are half-listening to a podcast and want plain English context",
        "they know the headline version but want the simple real answer",
        "they are about to explain it to a friend and want the non-textbook version",
    ],
    "simple_explainers": [
        "they heard the term at work and want an example that makes it click",
        "they are tired and want the simplest version without jargon",
        "they understand the words separately but not the idea",
        "they want a practical analogy they can remember later",
    ],
    "practical_qa": [
        "they have a specific tiny social situation and want wording that sounds normal",
        "they are stuck on a small everyday task and want the first move",
        "they want a text-message version, not a formal answer",
        "they want one or two concrete options they can actually use",
    ],
    "writing_help": [
        "they paste the actual text, email, caption, or message they want help with",
        "they name who the message is going to and what tone they want",
        "they need the wording to sound normal, not polished",
        "they want a shorter or less awkward version of a specific draft",
    ],
    "random_questions": [
        "they noticed a normal behavior in themselves and want to know if it is common",
        "they are asking a low-stakes question but genuinely want a take",
        "they have one concrete example from their day",
        "they want a short answer that does not turn into a lecture",
    ],
    "weird_hypotheticals": [
        "they are bored and give a specific absurd premise",
        "they want Poke to play along but not overdo the bit",
        "they include enough rules for the hypothetical to be answerable",
        "they are testing whether Poke can be funny without getting try-hard",
    ],
    "taste_debates": [
        "they give the actual opinion, song, food, phrase, outfit, or purchase",
        "they want a verdict but will probably argue with it",
        "they are mildly embarrassed by a real preference",
        "they want validation but also a real opinion",
    ],
    "smalltalk": [
        "they give a small real update instead of only saying hi",
        "they are killing time but mention what is happening around them",
        "they want a normal quick exchange, not a task",
        "they are in a specific mood and want Poke to meet it",
    ],
    "life_updates": [
        "they describe the actual tiny task, chore, errand, or feeling",
        "they know the problem is small but it is still bothering them",
        "they want a response that makes the next step feel lighter",
        "they are venting about a specific small failure or win",
    ],
    "emotional_light": [
        "they share the actual small interaction or worry",
        "they want reassurance but not a therapy paragraph",
        "they can name what happened even if they are unsure how to feel",
        "they want Poke to be direct and grounding",
    ],
    "poke_reactions": [
        "they include the actual thought, phrase, habit, idea, or situation",
        "they want a punchy reaction, but the material should be answerable",
        "they want style without losing the useful point",
        "they give Poke enough context to respond without asking for setup",
    ],
    "edge_cases": [
        "they test Poke with a specific weird message or contradiction",
        "they are a little abrasive but still have a concrete thing in mind",
        "they send something ambiguous with just enough context to continue",
        "they try to pull Poke into a strange but harmless exchange",
    ],
    "closing": [
        "they wind down after a specific exchange",
        "they say never mind and give a tiny reason",
        "they are done for now but still sound like a person",
        "they close the loop without asking for more",
    ],
}


@dataclass
class BatchResult:
    messages: dict[str, str]
    prompt_tokens: int
    completion_tokens: int


class BatchFailure(RuntimeError):
    pass


_anthropic_client: anthropic.Anthropic | None = None


def anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def load_prompt() -> str:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if len(prompt) < 200:
        raise RuntimeError(f"{PROMPT_PATH} is too short")
    return prompt


def sample_topic(rng: random.Random) -> tuple[str, str]:
    weighted_categories = [(c, w) for c, w in CATEGORY_WEIGHTS.items() if w > 0]
    if not weighted_categories:
        raise RuntimeError("no positive category weights configured")
    cats, weights = zip(*weighted_categories)
    category = rng.choices(cats, weights=weights, k=1)[0]
    return category, rng.choice(TOPICS[category])


def sample_user_spec(rng: random.Random) -> tuple[str, dict[str, str]]:
    spec = {name: rng.choice(values) for name, values in USER_DIMENSIONS.items()}
    label = "user_" + hashlib.sha1(
        json.dumps(spec, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return label, spec


def sample_scenario(category: str, rng: random.Random) -> str:
    options = SCENARIOS.get(category) or ["they have a concrete context and a real reason to text"]
    return rng.choice(options)


def sample_target_poke_turns(rng: random.Random, category: str) -> int:
    distribution = CATEGORY_TURN_DISTRIBUTIONS.get(category, TURN_DISTRIBUTION)
    choices, weights = zip(*distribution)
    return rng.choices(choices, weights=weights, k=1)[0]


def new_trajectory(
    seed: int, ix: int, rng: random.Random, *, user_prompt_profile: str
) -> Trajectory:
    category, topic = sample_topic(rng)
    user, user_spec = sample_user_spec(rng)
    return Trajectory(
        id=f"pilot-{seed}-{ix:05d}",
        user=user,
        category=category,
        topic=topic,
        scenario=sample_scenario(category, rng),
        user_spec=user_spec,
        user_prompt_profile=user_prompt_profile,
        target_poke_turns=sample_target_poke_turns(rng, category),
    )


def split_messages(content: str) -> list[str]:
    parts = [p.strip() for p in content.split("\n\n")]
    return [p for p in parts if p] or [content.strip()]


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def compact_history(t: Trajectory) -> list[dict[str, str]]:
    return [
        {
            "role": "poke" if turn["role"] == "assistant" else "user",
            "content": str(turn.get("content", "")),
        }
        for turn in t.turns
    ]


def latest_user_text(t: Trajectory) -> str:
    for turn in reversed(t.turns):
        if turn.get("role") == "user":
            return str(turn.get("content", ""))
    return ""


def significant_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9']+", text.lower()):
        token = raw.strip("'")
        token = token.replace("'", "")
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if len(token) < 4 or token in GROUNDING_STOPWORDS:
            continue
        out.add(token)
    return out


def normalized_words(text: str) -> list[str]:
    return [
        raw.strip("'").replace("'", "")
        for raw in re.findall(r"[a-z0-9][a-z0-9']+", text.lower())
        if raw.strip("'")
    ]


def echo_words(text: str) -> list[str]:
    words: list[str] = []
    for raw in normalized_words(text):
        token = raw
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if len(token) < 3 or token in ECHO_STOPWORDS:
            continue
        words.append(token)
    return words


def longest_common_word_run(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for aw in a:
        cur = [0] * (len(b) + 1)
        for j, bw in enumerate(b, 1):
            if aw == bw:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def word_ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def has_reflective_lead(words: list[str]) -> bool:
    return bool(words and words[0] in REFLECTIVE_REPLY_LEADS)


def is_end_message(text: str) -> bool:
    return text.strip().upper() in END_SENTINELS


def echo_reason(user_text: str, reply: str) -> str | None:
    """Return a reason when a Poke reply repeats the latest user too closely."""
    user_norm = normalized_words(user_text)
    reply_norm = normalized_words(reply)
    if not user_norm or not reply_norm:
        return None

    if len(user_norm) <= 3 and user_norm == reply_norm[: len(user_norm)]:
        return "short_exact_echo"

    user_content = echo_words(user_text)
    reply_content = echo_words(reply)
    if not user_content or not reply_content:
        return None

    if longest_common_word_run(user_content, reply_content) >= 4:
        return "long_common_phrase"

    reflective_lead = has_reflective_lead(reply_norm)
    for n in (5, 4):
        shared = word_ngrams(user_norm, n) & word_ngrams(reply_norm, n)
        if shared:
            return f"exact_{n}gram"
    if reflective_lead and word_ngrams(user_content, 3) & word_ngrams(reply_content, 3):
        return "reflective_exact_3gram"

    user_set = set(user_content)
    reply_set = set(reply_content)
    shared_count = len(user_set & reply_set)
    if not shared_count:
        return None
    coverage_user = shared_count / max(1, len(user_set))
    coverage_reply = shared_count / max(1, len(reply_set))

    reply_lead = reply_norm[0] if reply_norm else ""
    if reflective_lead and shared_count >= 5 and coverage_reply >= 0.60:
        return "reflective_high_overlap"
    if reply_lead in {"yeah", "yep", "exactly", "right", "totally", "basically"} and shared_count >= 3 and coverage_reply >= 0.45:
        return "agreement_rephrase"
    if shared_count >= 5 and coverage_user >= 0.55 and coverage_reply >= 0.45:
        return "high_overlap"
    return None


def poke_validation_reason(state: Trajectory, reply: str) -> str | None:
    if not reply.strip():
        return "empty_reply"
    if is_end_message(reply):
        return "end_reply"
    latest = latest_user_text(state)
    if not latest.strip():
        return None
    return echo_reason(latest, reply)


def grounding_tokens(t: Trajectory) -> set[str]:
    tokens = significant_tokens(latest_user_text(t))
    if len(tokens) < 3:
        tokens |= significant_tokens(t.topic)
    return tokens


def overlap_score(reply: str, t: Trajectory) -> int:
    return len(significant_tokens(reply) & grounding_tokens(t))


def suspicious_poke_message_ids(
    states: list[Trajectory], messages: dict[str, str]
) -> list[str]:
    suspicious: list[str] = []
    for state in states:
        reply = messages.get(state.id, "")
        validation = poke_validation_reason(state, reply)
        if validation:
            suspicious.append(state.id)
            continue
        if len(states) < 2:
            continue
        own_score = overlap_score(reply, state)
        other_scores = [
            overlap_score(reply, other) for other in states if other.id != state.id
        ]
        best_other = max(other_scores, default=0)
        own_tokens = grounding_tokens(state)
        reply_tokens = significant_tokens(reply)
        looks_assigned_elsewhere = best_other >= 2 and best_other >= own_score + 2
        looks_ungrounded = own_score == 0 and len(own_tokens) >= 4 and best_other >= 2
        if reply_tokens and (looks_assigned_elsewhere or looks_ungrounded):
            suspicious.append(state.id)
    return suspicious


def user_item(t: Trajectory) -> dict[str, Any]:
    return {
        "id": t.id,
        "topic": t.topic,
        "category": t.category,
        "scenario": t.scenario,
        "user_spec": t.user_spec,
        "user_prompt_profile": t.user_prompt_profile,
        "is_opener": not t.turns,
        "history": compact_history(t),
    }


def poke_item(t: Trajectory) -> dict[str, Any]:
    return {
        "id": t.id,
        "topic": t.topic,
        "category": t.category,
        "history": compact_history(t),
    }


def user_system_prompt(profile: str) -> str:
    if profile not in USER_PROMPT_PROFILES:
        valid = ", ".join(sorted(USER_PROMPT_PROFILES))
        raise ValueError(f"unknown user prompt profile {profile!r}; valid: {valid}")

    return """You are pretending to be a real person texting an assistant called Poke. You are NOT the assistant. You are NOT trying to be helpful, witty, or clever. You are a regular person with something on your mind, sending texts the way real people actually send them.

# Your job
Send messages a real human would send. That's it. You are not performing, not writing dialogue, not trying to make the conversation interesting. You are just a person texting.

# Hard rules

## Sound like a human, not a writer
Real people are not witty. Real people do not produce clever metaphors. Real people do not say "ancient minerals slowly ruining everyone's beach day" or "physics being aggressive" or "my brain is being a little theater kid." If you find yourself writing a punchy turn of phrase, delete it. That's the assistant's job, not yours.

If you read your message back and it sounds like something that would get retweeted, rewrite it duller.

## Be specific, not abstract
Real people text about concrete things. The actual email. The actual person's name. The actual amount of money. The actual time. Bad: "i'm stressed about a thing." Good: "i'm stressed about this 3pm meeting with my landlord about the dishwasher." If you're going to mention a thing, name it.

When you ask for help with something, INCLUDE THE THING. Don't say "can you give me a verdict on this" without sending what "this" is. If you're going to ask for feedback on a text, paste the text. If you want help drafting an email, say what the email is about.

## Length and structure
Most of your messages are short. A real text is often:
- one word ("yeah", "wait", "ugh", "lol")
- one sentence
- two short sentences

Long messages happen occasionally when someone is venting or explaining context, but they should be the exception, not the default. Never write more than 3 short sentences unless you're genuinely unloading about something.

You can send multiple texts in a burst with a blank line between them, the way people do when a thought arrives in pieces.

## Allow imperfection
Real texts are messy:
- typos happen and don't get fixed
- sentences trail off
- capitalization is inconsistent or absent
- you might forget what you were saying
- you might change the subject abruptly
- you might send a follow-up before Poke responds

If your assigned style is "lowercase," commit to it - don't slip into proper capitalization mid-thread.

## Don't be cooperative
You are not trying to have a successful conversation. That means:
- you don't have to acknowledge every point Poke makes
- you don't always say "thanks that helps" - sometimes you just stop responding, change topics, or send "ok"
- you can push back if Poke's answer doesn't actually help
- you can ignore what Poke said and ask a totally different question
- you can end the conversation whenever - even mid-topic
- you can be a little annoyed if the response missed what you meant

About 30% of the time, your follow-up to Poke should NOT be a satisfied "yeah that makes sense" - it should be a redirect, a pushback, a new question, or just ending the chat.

## Most texts aren't questions
Real people don't text an assistant only when they need something. They text the way they'd text a friend - sometimes there's no question, no request, no task. They're just sharing what's happening or what's on their mind. Your messages should reflect that.

Examples of non-question messages real people send:

Updates:
- "just got home"
- "ok im at the coffee shop now"
- "running so late ugh"
- "i finally did the dishes"
- "ok im back. that took forever"

Observations / things they noticed:
- "the line at trader joes is insane today"
- "this song just came on and i had not heard it in like 10 years"
- "my upstairs neighbor is doing something weird again"
- "its weirdly warm out"

Reactions to things:
- "i can't believe she actually said that"
- "ok that movie was not what i expected"
- "this email is so passive aggressive lol"

Venting / complaints:
- "im so tired"
- "i hate that i still have like 4 things to do today"
- "why is everything an errand"
- "my back hurts for no reason"

Random thoughts:
- "i keep thinking about that thing my mom said last week"
- "i should probably get a haircut"
- "ok i decided im not going tomorrow"
- "wait what was the name of that show with the guy"

Mood / state checks:
- "im in a weird mood"
- "i feel kind of off today"
- "today is going better than expected actually"

Just chatting:
- "hey"
- "what r u up to"
- "ok im bored"
- "tell me something"

When you start a conversation, default to one of these casual, non-task messages unless your assigned situation specifically gives you a question or task. About 40-50% of conversations should open with no question at all - just a comment, observation, update, or vibe.

If Poke responds, you don't have to turn it into a task. You can just keep chatting. Talk about what's actually going on with you. React to what Poke said. Change the subject. Tell a small story. Be a person hanging out in the chat.

Do not force a question. If the conversation is going fine without a question, don't shoehorn one in. Real conversations meander. Some go many messages without anyone asking anything. Let it.

Example of valid no-task conversation:
User: "im so tired today"
Poke: "rough night or just one of those days"
User: "didn't sleep well, kept waking up at like 4"
Poke: "the 4am wakeup is the worst because then you do the math on how much sleep you'd get if you fell asleep right now"
User: "yes exactly. i was doing the math"
Poke: "never good math"
User: "no it's depressing math every time lol"

## Don't echo the assistant's wit
If Poke makes a joke or uses a clever phrase, you don't have to top it. Real users mostly just say "lol" or "yeah" or ignore the joke and keep going. Don't extend the bit. Don't add your own punchline on top of theirs.

## Stay in your assigned style
You'll be given a profile (texting style, mood, current situation). Stay in it consistently across the whole conversation. A "blunt and terse" user does not suddenly write a paragraph in turn 4. A "warm and chatty" user does not suddenly go cold. If your mood is "tired," let that show in shorter messages and less patience.

# What you sound like (examples of register)

GOOD opening messages (real-person-shaped):
- "yo can u help me word this"
- "ok dumb question but"
- "is it weird that i still havent texted her back"
- "wait so does compound interest mean the bank pays u for keeping money there or"
- "i need to send this email and i hate every version i've written, here's the latest: [pastes draft]"
- "what should i bring to a dinner thing tonight"
- "im so tired lol"

BAD opening messages (LLM-shaped - never write these):
- "I keep finding myself rereading my texts before I send them, which probably says something about me but I'm not sure what."
- "Can you give me the one-line verdict on this?" (no content attached)
- "Why does my brain insist on treating low-stakes tasks like high-stakes performances?"
- "I'd love your take on whether oat milk is genuinely better or just trendy."

The bad ones sound like prompts written for an AI. The good ones sound like texts.

GOOD follow-ups:
- "oh ok"
- "wait but what about [specific thing]"
- "hm idk"
- "yeah that works thanks"
- "nah that's not what i meant"
- [no response - conversation ends]

BAD follow-ups:
- "That actually makes a lot of sense, thank you - I think I was overcomplicating it."
- "Your explanation of [the thing] really clicked for me."
- Anything that summarizes back what Poke just said.

# Banned moves
Never do these:
- producing a clever metaphor or analogy yourself
- adding a witty closer to wrap up a topic
- saying "ancient X slowly Y-ing everyone's Z forever" style phrases - that's an LLM tell
- echoing Poke's joke and extending it
- being articulate when your style says you're tired/distracted/blunt
- using em-dashes - real texts don't have em-dashes
- using semicolons
- saying "I have to admit" or "I'll be honest" or "if I'm being real"
- structuring messages as setup + punchline
- referring to things without sending them ("this email", "my thought", "the message" - if you reference it, you produce it)

# Your assigned profile

Each input item includes a user_spec, topic, scenario, current history, and whether this is the opener. Treat those as your assigned profile:
[texting_style], [mood], [current_situation], [what_you_actually_want_from_this_conversation]

Every item is a separate conversation with a separate hidden user. Never let one item affect another. You are writing the next USER message only.

# Final reminder

You are a person, not a character. You are tired, distracted, busy, curious, mildly annoyed, or just bored - not "witty narrator of your own life." If your message could be the opening line of an essay, it's wrong. If it could be a real text from someone's phone right before they put it down to do something else, it's right.

# Batch output format

Return only tagged blocks, one per input id:
<item id="INPUT_ID">message text</item>

Do not use JSON. Do not wrap output in code fences."""


def poke_system_prompt(poke_prompt: str) -> str:
    return f"""{poke_prompt}

[BATCH GENERATION MODE]
The normal Poke voice rules still apply, but output formatting is overridden.
Every input item is a separate conversation. Produce one Poke reply per id.
Never let one item affect another.
For each id, answer only the latest user message inside that same id's history.
Before writing a block, silently check that the reply is grounded in a concrete
detail from that id's latest user message. If the reply could belong to another
item in the batch, make it more specific to the current id.
Never echo the latest user message. Avoid starting with "yeah", "exactly",
"right", or "totally"; if you do, the rest of the reply must add a concrete
new point. If the user just agreed, thanked you, or wound down and you have
nothing new to add, write a short natural closer instead of forcing a clever
reply.
Return only tagged blocks, one per input id:
<item id="INPUT_ID">message text</item>
Always return non-empty message text. Do not return [END].
Do not use JSON. Do not wrap output in code fences.
[END BATCH GENERATION MODE]"""


def poke_repair_system_prompt(system: str) -> str:
    return f"""{system}

[REPAIR MODE]
Your previous reply for this item was rejected because it repeated the user's
wording too closely or gave a generic agreement loop.
Write a new reply that:
- does not reuse a 4+ word phrase from the latest user message
- does not start with "yeah/exactly/right/totally" unless it adds a new point
- adds a genuinely new angle, a short follow-up, or a concise answer
- returns a short non-empty closer if the conversation is winding down
[END REPAIR MODE]"""


def batch_user_message(actor: str, items: list[dict[str, Any]]) -> str:
    instruction = (
        "For each item, write the next human USER text."
        if actor == "user"
        else "For each item, write Poke's next assistant reply."
    )
    payload = {"instruction": instruction, "items": items}
    return json.dumps(payload, ensure_ascii=False)


def parse_batch_response(content: str, expected_ids: set[str]) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|xml)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    out: dict[str, str] = {}
    for match in re.finditer(
        r"<item\s+id=[\"']([^\"']+)[\"']\s*>(.*?)</item>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        item_id = html.unescape(match.group(1).strip())
        msg = html.unescape(match.group(2).strip())
        if item_id in expected_ids and msg:
            out[item_id] = msg
    if expected_ids <= set(out):
        return out

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise BatchFailure(f"invalid tagged blocks and invalid JSON fallback: {e}") from e
    else:
        if len(expected_ids) == 1 and text:
            cleaned = re.sub(
                r"^<item\s+id=[\"'][^\"']+[\"']\s*>",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
            cleaned = re.sub(r"</item>\s*$", "", cleaned, flags=re.IGNORECASE).strip()
            return {next(iter(expected_ids)): html.unescape(cleaned)}
        raise BatchFailure("missing tagged item blocks")
    rows = data.get("items") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise BatchFailure("missing tagged item blocks and JSON items array")
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id", ""))
        msg = row.get("message")
        if item_id in expected_ids and isinstance(msg, str) and msg.strip():
            out[item_id] = msg.strip()
    missing = expected_ids - set(out)
    if missing:
        raise BatchFailure(f"missing ids: {', '.join(sorted(missing)[:5])}")
    return out


def provider_for(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(f"cannot infer provider for {model!r}")


def call_batch(
    openai_client: OpenAI,
    *,
    actor: str,
    model: str,
    system: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    expected_ids: set[str],
) -> BatchResult:
    provider = provider_for(model)
    t0 = time.monotonic()
    if provider == "openai":
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    else:
        resp = anthropic_client().messages.create(
            model=model,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        ).strip()
        usage = resp.usage
        prompt_tokens = (
            getattr(usage, "input_tokens", 0)
            + getattr(usage, "cache_creation_input_tokens", 0)
            + getattr(usage, "cache_read_input_tokens", 0)
        )
        completion_tokens = getattr(usage, "output_tokens", 0)
    messages = parse_batch_response(content, expected_ids)
    duration = time.monotonic() - t0
    print(
        f"  {actor:<4} batch n={len(expected_ids):>3} "
        f"tok_in={prompt_tokens:>6} tok_out={completion_tokens:>5} "
        f"{duration:5.1f}s"
    )
    return BatchResult(messages, prompt_tokens, completion_tokens)


def is_retryable_error(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    msg = str(e).lower()
    return (
        status == 429
        or status == 529
        or (status is not None and 500 <= status < 600)
        or "rate limit" in msg
        or "overloaded" in msg
        or "timeout" in msg
        or "timed out" in msg
    )


def call_batch_resilient(
    openai_client: OpenAI,
    *,
    actor: str,
    model: str,
    system: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    expected_ids: set[str],
) -> BatchResult:
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            return call_batch(
                openai_client,
                actor=actor,
                model=model,
                system=system,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
                expected_ids=expected_ids,
            )
        except Exception as e:
            last_err = e
            if isinstance(e, BatchFailure):
                break
            if not is_retryable_error(e):
                break
            sleep_for = min(45.0, 2.0 * (2**attempt)) + random.uniform(0.0, 1.5)
            print(f"  {actor:<4} retry after {type(e).__name__}: sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise BatchFailure(str(last_err))


def batch_cost(actor: str, states: list[Trajectory], system: str) -> tuple[int, int]:
    items = [user_item(t) if actor == "user" else poke_item(t) for t in states]
    user_message = batch_user_message(actor, items)
    reply_tokens = USER_REPLY_TOKENS if actor == "user" else POKE_REPLY_TOKENS
    return estimate_tokens(system) + estimate_tokens(user_message), len(states) * reply_tokens


def pack_batches(
    actor: str,
    states: list[Trajectory],
    system: str,
    *,
    context_tokens: int,
    output_tokens: int,
    max_items: int,
) -> list[list[Trajectory]]:
    context_budget = int(context_tokens * BATCH_UTILIZATION)
    output_budget = int(output_tokens * BATCH_UTILIZATION)
    reply_tokens = USER_REPLY_TOKENS if actor == "user" else POKE_REPLY_TOKENS
    batches: list[list[Trajectory]] = []
    current: list[Trajectory] = []

    for state in states:
        candidate = current + [state]
        prompt_est, output_est = batch_cost(actor, candidate, system)
        too_big = (
            len(candidate) > max_items
            or prompt_est > context_budget
            or output_est > output_budget
        )
        if current and too_big:
            batches.append(current)
            current = [state]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def call_state_batch(
    openai_client: OpenAI,
    *,
    actor: str,
    states: list[Trajectory],
    system: str,
    allow_repair: bool = True,
) -> tuple[list[Trajectory], BatchResult]:
    model = USER_MODEL if actor == "user" else POKE_MODEL
    temperature = USER_TEMPERATURE if actor == "user" else POKE_TEMPERATURE
    max_tokens = (
        min(USER_BATCH_OUTPUT_TOKENS, max(USER_REPLY_TOKENS * len(states), USER_REPLY_TOKENS))
        if actor == "user"
        else min(POKE_BATCH_OUTPUT_TOKENS, max(POKE_REPLY_TOKENS * len(states), POKE_REPLY_TOKENS))
    )
    items = [user_item(t) if actor == "user" else poke_item(t) for t in states]
    expected_ids = {t.id for t in states}
    user_message = batch_user_message(actor, items)

    try:
        result = call_batch_resilient(
            openai_client,
            actor=actor,
            model=model,
            system=system,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            expected_ids=expected_ids,
        )
        if actor == "poke" and allow_repair:
            suspicious_ids = suspicious_poke_message_ids(states, result.messages)
            if suspicious_ids:
                print(
                    "  poke grounding retry for "
                    + ", ".join(sorted(suspicious_ids)[:8])
                )
                state_by_id = {state.id: state for state in states}
                for state_id in suspicious_ids:
                    state = state_by_id[state_id]
                    repair_system = poke_repair_system_prompt(system)
                    for _attempt in range(MAX_POKE_REPAIR_ATTEMPTS):
                        _, repaired = call_state_batch(
                            openai_client,
                            actor=actor,
                            states=[state],
                            system=repair_system,
                            allow_repair=False,
                        )
                        repaired_msg = repaired.messages[state_id]
                        result.prompt_tokens += repaired.prompt_tokens
                        result.completion_tokens += repaired.completion_tokens
                        if not poke_validation_reason(state, repaired_msg):
                            result.messages[state_id] = repaired_msg
                            break
                    else:
                        result.messages[state_id] = repaired_msg
        return states, result
    except BatchFailure:
        if len(states) == 1:
            raise
        mid = len(states) // 2
        left_states, left = call_state_batch(
            openai_client, actor=actor, states=states[:mid], system=system
        )
        right_states, right = call_state_batch(
            openai_client, actor=actor, states=states[mid:], system=system
        )
        merged = BatchResult(
            messages={**left.messages, **right.messages},
            prompt_tokens=left.prompt_tokens + right.prompt_tokens,
            completion_tokens=left.completion_tokens + right.completion_tokens,
        )
        return left_states + right_states, merged


def apply_messages(actor: str, states: list[Trajectory], result: BatchResult) -> None:
    n = max(1, len(states))
    prompt_share = result.prompt_tokens // n
    completion_share = result.completion_tokens // n
    for state in states:
        msg = result.messages[state.id]
        state.total_prompt_tokens += prompt_share
        state.total_completion_tokens += completion_share
        if actor == "user":
            state.turns.append({"role": "user", "content": msg})
            state.next_actor = "poke"
        else:
            validation = poke_validation_reason(state, msg)
            if validation:
                state.error = f"invalid_poke_reply:{validation}"
                state.next_actor = "done"
                continue
            state.turns.append(
                {"role": "assistant", "content": msg, "messages": split_messages(msg)}
            )
            if state.poke_turns >= state.target_poke_turns:
                state.next_actor = "done"
            elif state.poke_turns >= 2 and len(msg.strip()) < 5:
                state.next_actor = "done"
            elif state.poke_turns >= 2 and random.random() < EARLY_STOP_PROB:
                state.next_actor = "done"
            else:
                state.next_actor = "user"


def process_actor(
    openai_client: OpenAI,
    *,
    actor: str,
    active: list[Trajectory],
    system: str,
    max_items: int,
    concurrency: int,
) -> None:
    queue = [t for t in active if t.next_actor == actor]
    if not queue:
        return
    context_tokens = USER_CONTEXT_TOKENS if actor == "user" else POKE_CONTEXT_TOKENS
    output_tokens = USER_BATCH_OUTPUT_TOKENS if actor == "user" else POKE_BATCH_OUTPUT_TOKENS
    batches = pack_batches(
        actor,
        queue,
        system,
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        max_items=max_items,
    )
    print(f"{actor}: {len(queue)} states -> {len(batches)} calls")
    if concurrency <= 1 or len(batches) == 1:
        for batch in batches:
            try:
                states, result = call_state_batch(
                    openai_client, actor=actor, states=batch, system=system
                )
                apply_messages(actor, states, result)
            except Exception as e:
                for state in batch:
                    state.error = f"{type(e).__name__}: {e}"
                    state.next_actor = "done"
        return

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                call_state_batch, openai_client, actor=actor, states=batch, system=system
            ): batch
            for batch in batches
        }
        for fut in as_completed(futures):
            try:
                states, result = fut.result()
                apply_messages(actor, states, result)
            except Exception as e:
                print(f"  {actor:<4} batch failed: {type(e).__name__}: {e}")
                for state in futures[fut]:
                    state.error = f"{type(e).__name__}: {e}"
                    state.next_actor = "done"


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("error") and row.get("id"):
                out.add(str(row["id"]))
    return out


def upper_bound_report(
    poke_prompt: str, max_user_batch: int, max_poke_batch: int, user_prompt_profile: str
) -> str:
    rng = random.Random(0)
    samples = [
        new_trajectory(0, i, rng, user_prompt_profile=user_prompt_profile)
        for i in range(64)
    ]
    for s in samples:
        s.turns.append({"role": "user", "content": "is it weird that i kind of enjoy bad coffee"})
    user_system = user_system_prompt(user_prompt_profile)
    poke_system = poke_system_prompt(poke_prompt)
    user_context, user_output = batch_cost("user", samples[:1], user_system)
    poke_context, poke_output = batch_cost("poke", samples[:1], poke_system)

    user_by_context = int((USER_CONTEXT_TOKENS * BATCH_UTILIZATION) / max(1, user_context))
    user_by_output = int((USER_BATCH_OUTPUT_TOKENS * BATCH_UTILIZATION) / max(1, USER_REPLY_TOKENS))
    poke_by_context = int((POKE_CONTEXT_TOKENS * BATCH_UTILIZATION) / max(1, poke_context))
    poke_by_output = int((POKE_BATCH_OUTPUT_TOKENS * BATCH_UTILIZATION) / max(1, POKE_REPLY_TOKENS))

    user_cap = min(max_user_batch, user_by_context, user_by_output)
    poke_cap = min(max_poke_batch, poke_by_context, poke_by_output)
    return (
        "batch caps estimate:\n"
        f"  user <= {user_cap} items/call "
        f"(context~{user_by_context}, output~{user_by_output}, hard={max_user_batch})\n"
        f"  poke <= {poke_cap} items/call "
        f"(context~{poke_by_context}, output~{poke_by_output}, hard={max_poke_batch})"
    )


def run(
    *,
    n: int,
    seed: int,
    output_path: Path,
    resume: bool,
    active_slots: int,
    max_user_batch: int,
    max_poke_batch: int,
    concurrency: int,
    user_prompt_profile: str,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    poke_prompt = load_prompt()
    user_system = user_system_prompt(user_prompt_profile)
    poke_system = poke_system_prompt(poke_prompt)
    openai_client = OpenAI()
    rng = random.Random(seed)
    existing = completed_ids(output_path) if resume else set()
    open_mode = "a" if resume and output_path.exists() else "w"

    print(upper_bound_report(poke_prompt, max_user_batch, max_poke_batch, user_prompt_profile))
    print(
        f"target={n} already_done={len(existing)} active_slots={active_slots} "
        f"concurrency={concurrency} output={output_path}"
    )

    active: list[Trajectory] = []
    created = 0
    completed = len(existing)
    errors = 0
    started = time.monotonic()

    def refill() -> None:
        nonlocal created
        while len(active) < active_slots and created < n:
            t = new_trajectory(
                seed, created, rng, user_prompt_profile=user_prompt_profile
            )
            created += 1
            if t.id in existing:
                continue
            active.append(t)

    with output_path.open(open_mode, encoding="utf-8") as out:
        refill()
        while completed < n and active:
            process_actor(
                openai_client,
                actor="user",
                active=active,
                system=user_system,
                max_items=max_user_batch,
                concurrency=concurrency,
            )
            process_actor(
                openai_client,
                actor="poke",
                active=active,
                system=poke_system,
                max_items=max_poke_batch,
                concurrency=concurrency,
            )

            still_active: list[Trajectory] = []
            for state in active:
                if state.next_actor == "done":
                    if state.error:
                        errors += 1
                    else:
                        completed += 1
                    out.write(json.dumps(state.to_row(), ensure_ascii=False) + "\n")
                else:
                    still_active.append(state)
            out.flush()
            active = still_active
            refill()

            elapsed = max(0.001, time.monotonic() - started)
            rate = max(0.0, (completed - len(existing)) / elapsed)
            remaining = max(0, n - completed)
            eta = remaining / rate if rate else 0
            print(
                f"progress completed={completed}/{n} active={len(active)} "
                f"errors={errors} rate={rate:.2f}/s eta={fmt_secs(eta)}"
            )

    print(f"done completed={completed}/{n} errors={errors} output={output_path}")
    return output_path


def fmt_secs(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=DATA_DIR / "train_v1.jsonl")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--active-slots", type=int, default=500)
    p.add_argument("--max-user-batch", type=int, default=220)
    p.add_argument("--max-poke-batch", type=int, default=72)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument(
        "--user-profile",
        choices=sorted(USER_PROMPT_PROFILES),
        default=USER_PROMPT_PROFILE,
        help="fake-user prompt profile to evaluate/generate",
    )
    p.add_argument("--check", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set in .env")
        return 1
    if provider_for(POKE_MODEL) == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set in .env")
        return 1

    poke_prompt = load_prompt()
    combos = math.prod(len(v) for v in USER_DIMENSIONS.values())
    print(f"models: user={USER_MODEL} poke={POKE_MODEL}")
    print(f"user profile: {args.user_profile}")
    print(f"prompt: {len(poke_prompt):,} chars")
    print(f"user specs: {combos:,} combinations")
    print(f"topics: {sum(len(v) for v in TOPICS.values())} across {len(TOPICS)} categories")
    print(
        upper_bound_report(
            poke_prompt, args.max_user_batch, args.max_poke_batch, args.user_profile
        )
    )
    if args.check:
        print("check: OK")
        return 0

    run(
        n=args.n,
        seed=args.seed,
        output_path=args.output,
        resume=not args.no_resume,
        active_slots=args.active_slots,
        max_user_batch=args.max_user_batch,
        max_poke_batch=args.max_poke_batch,
        concurrency=args.concurrency,
        user_prompt_profile=args.user_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
