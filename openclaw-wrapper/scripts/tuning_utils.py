#!/usr/bin/env python3
"""Retrieval tuning helpers for Daystrom DML OpenClaw integration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DOMAIN_HINTS = {
    "battlebot",
    "vanguard",
    "chassis",
    "weapon",
    "wheel",
    "armor",
    "blender",
    "usd",
    "usda",
    "export",
    "fallback",
    "anti-blob",
    "primitive",
    "locomotion",
    "rig",
    "glb",
    "manifest",
    "continuity",
    "compaction",
    "memory",
    "handoff",
    "operator",
    "blocker",
    "constraint",
    "milestone",
    "resume",
    "resumption",
    "task",
}

BLOCKER_EXPANSIONS = {
    "usd": ["usda", "wm.usd_export", "export_scene.usd", "out-usd", "usd_export_mode"],
    "export": ["converter", "usd-converter-cmd", "external_converter", "saved_usd", "glb"],
    "fallback": ["fallback_glb_only", "usd.fallback.json", "degraded", "recover", "manifest"],
}

NOISE_PATTERNS = [
    re.compile(r"\bheartbeat_summary\b", re.I),
    re.compile(r"\bfallback_trigger=none\b", re.I),
    re.compile(r"\bBuilder command failed\b", re.I),
    re.compile(r"\btelemetry score\b", re.I),
    re.compile(r"\battempts=\[", re.I),
]

WORD_RE = re.compile(r"[a-zA-Z0-9_.-]+")
CONTINUITY_HINTS = {
    "active",
    "task",
    "blocker",
    "next",
    "step",
    "constraint",
    "milestone",
    "continuity",
    "compaction",
    "resume",
    "resumption",
    "operator",
    "memory",
}


@dataclass(frozen=True)
class QueryIntent:
    terms: set[str]


def normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def smart_chunks(text: str, *, chunk_chars: int = 620, overlap: int = 90) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []

    blocks = [b.strip() for b in re.split(r"\n{2,}|(?=\{)|(?<=\})\n", clean) if b.strip()]
    chunks: list[str] = []
    for block in blocks:
        if len(block) <= chunk_chars:
            chunks.append(block)
            continue
        start = 0
        while start < len(block):
            end = min(start + chunk_chars, len(block))
            window = block[start:end]
            if end < len(block):
                cut = max(window.rfind(". "), window.rfind("\n"), window.rfind(", "))
                if cut > int(chunk_chars * 0.55):
                    end = start + cut + 1
                    window = block[start:end]
            chunks.append(window.strip())
            if end >= len(block):
                break
            start = max(start + 1, end - overlap)
    return [c for c in chunks if c]


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(text)}


def domain_focus_score(text: str) -> float:
    toks = _token_set(text)
    if not toks:
        return 0.0
    return len(toks & DOMAIN_HINTS) / max(1, min(len(DOMAIN_HINTS), len(toks)))


def noise_score(text: str) -> float:
    s = text.strip()
    if not s:
        return 1.0
    hit = sum(1 for p in NOISE_PATTERNS if p.search(s))
    braces = s.count("{") + s.count("}")
    structural = braces / max(1, len(s))
    return min(1.0, hit * 0.35 + structural * 12.0)


def continuity_focus_score(text: str) -> float:
    toks = _token_set(text)
    if not toks:
        return 0.0
    return len(toks & CONTINUITY_HINTS) / max(1, min(len(CONTINUITY_HINTS), len(toks)))


def should_keep_chunk(text: str, *, min_domain_focus: float = 0.02, max_noise: float = 0.72) -> bool:
    focus = domain_focus_score(text)
    continuity = continuity_focus_score(text)
    noise = noise_score(text)
    # Keep chunks with blocker/continuity structure even if somewhat noisy.
    toks = _token_set(text)
    if toks & set(BLOCKER_EXPANSIONS.keys()):
        return True
    if continuity >= 0.18 and any(token in toks for token in {"task", "blocker", "constraint", "milestone", "continuity"}):
        return True
    return max(focus, continuity) >= min_domain_focus and noise <= max_noise


def rewrite_query(query: str) -> str:
    q = query.strip()
    toks = _token_set(q)
    extra: list[str] = []
    for key, expansions in BLOCKER_EXPANSIONS.items():
        if key in toks:
            extra.extend(expansions)
    if not extra:
        return q
    return f"{q} | expansion: {' '.join(sorted(set(extra)))}"


def infer_intent_terms(query: str) -> QueryIntent:
    q = query.lower()
    terms = set()
    if "usd" in q or "export" in q:
        terms |= {"usd", "export", "glb", "fallback", "manifest"}
    if "fallback" in q:
        terms |= {"fallback", "recover", "manifest", "converter"}
    if "anti-blob" in q or "primitive" in q:
        terms |= {"anti-blob", "primitive", "chassis", "hard_surface_parts"}
    if "wheel" in q or "weapon" in q:
        terms |= {"wheel", "weapon", "mount", "layout", "locomotion"}
    if not terms:
        terms |= _token_set(query)
    return QueryIntent(terms=terms)


def relevance_score(text: str, intent: QueryIntent) -> float:
    toks = _token_set(text)
    if not toks or not intent.terms:
        return 0.0
    return len(toks & intent.terms) / len(intent.terms)


def continuity_handoff_summary(text: str, *, max_len: int = 220) -> str | None:
    clean = normalize_text(text)
    if not clean:
        return None

    def _compact(value: str, limit: int) -> str:
        value = " ".join(value.split()).rstrip(".")
        if len(value) <= limit:
            return value
        shortened = value[: max(0, limit - 1)].rstrip()
        cut = max(shortened.rfind(", "), shortened.rfind(" "))
        if cut >= max(10, limit // 2):
            shortened = shortened[:cut].rstrip()
        return shortened.rstrip(" ,;:") + "…"

    fields: list[tuple[str, str]] = []
    patterns = [
        ("task", re.compile(r"\bactive\s+task\s*:\s*(.+?)(?=(?:\.\s+[A-Z][a-z]+\s+\w+\s*:)|$)", re.I), 44),
        ("blocker", re.compile(r"\bcurrent\s+blocker\s*:\s*(.+?)(?=(?:\.\s+[A-Z][a-z]+\s+\w+\s*:)|$)", re.I), 54),
        ("next", re.compile(r"\bnext\s+step\s*:\s*(.+?)(?=(?:\.\s+[A-Z][a-z]+\s+\w+\s*:)|$)", re.I), 54),
        ("constraint", re.compile(r"\bkey\s+constraint\s*:\s*(.+?)(?=(?:\.\s+[A-Z][a-z]+\s+\w+\s*:)|$)", re.I), 42),
    ]
    for label, pattern, limit in patterns:
        match = pattern.search(clean)
        if not match:
            continue
        value = match.group(1).strip()
        if value:
            fields.append((label, _compact(value, limit)))

    if not fields:
        return None

    summary = " | ".join(f"{label}: {value}" for label, value in fields)
    if len(summary) <= max_len:
        return summary

    # Prefer retaining all labels with tighter per-field budgets over blunt tail truncation.
    tightened: list[tuple[str, str]] = []
    for label, value in fields:
        tighter_limit = 28 if label in {"task", "constraint"} else 36
        tightened.append((label, _compact(value, tighter_limit)))
    summary = " | ".join(f"{label}: {value}" for label, value in tightened)
    return summary[:max_len].rstrip()


def source_is_battlebot(path: Path) -> bool:
    p = str(path).lower()
    return "vlm-battlebot" in p or "battlebot" in p
