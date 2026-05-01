"""Pure text helpers used by chunking, fallback extraction, and projection."""

from __future__ import annotations

import re
from typing import Any


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ENTITY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TEMPORAL_CUE_RE = re.compile(
    r"\b("
    r"today|tomorrow|yesterday|tonight|this morning|this evening|"
    r"last week|last month|last year|next week|next month|next year|"
    r"before|after|during|between|ago|currently|now|still|recently|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december"
    r")\b",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_PREFERENCE_RE = re.compile(r"\b(prefer|like|love|enjoy|avoid|favorite|favourite|usually|hate)\b", re.IGNORECASE)
_PLAN_RE = re.compile(r"\b(plan|planning|intend|going to|will|schedule|scheduled|want to)\b", re.IGNORECASE)
_STATE_RE = re.compile(r"\b(is|are|was|were|has|have|had|works as|lives in|currently|used to|now|still)\b", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_\.]+")
_PHONE_RE = re.compile(r"\+?\d[\d\-\(\) ]{6,}\d")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_LIST_INDEX_RE = re.compile(r"(?:^|\s)(\d+[\.\)])\s+\S+")


def normalize_entity_key(text: Any) -> str:
    return " ".join(_ENTITY_TOKEN_RE.findall(str(text or "").lower())).strip()


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(str(text or ""))]
    return [part for part in parts if part]


def extract_temporal_cues(text: str) -> list[str]:
    seen: list[str] = []
    for match in _TEMPORAL_CUE_RE.finditer(str(text or "")):
        cue = match.group(0).strip()
        if cue and cue.lower() not in {item.lower() for item in seen}:
            seen.append(cue)
    return seen


def extract_numeric_mentions(text: str) -> list[tuple[str, float | None, str]]:
    """Return `(raw, parsed_number, unit_suffix)` tuples."""

    source = str(text or "")
    out: list[tuple[str, float | None, str]] = []
    for match in _NUMERIC_RE.finditer(source):
        raw = match.group(0)
        suffix = source[match.end() : match.end() + 12].strip().split(" ", 1)[0]
        try:
            parsed = float(raw)
        except ValueError:
            parsed = None
        out.append((raw, parsed, suffix))
    return out


def classify_fact_kind(text: str) -> str:
    source = str(text or "")
    if _PREFERENCE_RE.search(source):
        return "preference"
    if _PLAN_RE.search(source):
        return "plan"
    if _STATE_RE.search(source):
        return "state"
    return "observation"


def classify_event_category(text: str) -> str:
    source = str(text or "").lower()
    rules = {
        "travel": ("flight", "trip", "hotel", "vacation", "travel", "visit"),
        "health": ("doctor", "vet", "workout", "yoga", "health", "sleep"),
        "food": ("restaurant", "meal", "bake", "recipe", "cook", "coffee", "pizza"),
        "work": ("job", "office", "meeting", "project", "interview", "report"),
        "study": ("study", "class", "course", "exam", "assignment", "workshop"),
        "shopping": ("buy", "bought", "order", "ordered", "purchase", "spent"),
    }
    for category, markers in rules.items():
        if any(marker in source for marker in markers):
            return category
    return "other"


def contains_exact_detail_token(text: str) -> bool:
    source = str(text or "")
    return bool(
        _HANDLE_RE.search(source)
        or _PHONE_RE.search(source)
        or _URL_RE.search(source)
        or _EMAIL_RE.search(source)
        or _LIST_INDEX_RE.search(source)
    )


def looks_generic_assistant_advice(text: str) -> bool:
    source = normalize_entity_key(text)
    generic_markers = (
        "the assistant recommended",
        "the assistant provided recommendations",
        "the assistant suggested",
        "the assistant advised",
        "the assistant listed",
        "the assistant gave tips",
        "the assistant shared strategies",
        "the assistant explained",
    )
    return any(marker in source for marker in generic_markers)
