"""Filler-word vocabulary and matching helpers."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from .models import Cut, Word


# Used as the default `--fillers` value (and exposed for tests). Word matches
# go through `is_filler()`, which also recognizes arbitrary-length elongations
# of these stems (e.g. "ummmm", "uhhhhh", "ahhhh", "hmmmmm", "mmm").
DEFAULT_FILLERS: frozenset[str] = frozenset(
    {"um", "uh", "er", "erm", "ah", "hmm", "mhm", "mm", "uh-huh"}
)

# Stem -> regex pattern. The pattern matches the literal stem allowing
# any of its trailing letters to repeat (e.g. "u+m+" matches um/umm/ummmm).
_ELONGATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "um": re.compile(r"^u+m+$"),
    "uh": re.compile(r"^u+h+$"),
    "er": re.compile(r"^e+r+$"),
    "erm": re.compile(r"^e+r+m+$"),
    "ah": re.compile(r"^a+h+$"),
    "hmm": re.compile(r"^h+m+$"),
    "mhm": re.compile(r"^m+h+m*$"),
    "mm": re.compile(r"^m{2,}$"),
    "uh-huh": re.compile(r"^u+h+-h+u+h+$"),
}

_PUNCT_RE = re.compile(r"[^\w\-]+")


def normalize_word(text: str) -> str:
    """Lowercase and strip surrounding punctuation. Keeps internal hyphens."""
    return _PUNCT_RE.sub("", text.lower())


def is_filler(word: str, fillers: Iterable[str]) -> bool:
    """True if `word` (already normalized) matches the filler set, including
    arbitrary-length elongations (e.g. "uhhhhh" matches the "uh" stem)."""
    if word in fillers:
        return True
    for stem in fillers:
        pat = _ELONGATION_PATTERNS.get(stem)
        if pat is not None and pat.match(word):
            return True
    return False


def find_fillers(words: Sequence[Word], fillers: Iterable[str]) -> list[Cut]:
    """Return the time ranges of words that match the filler set."""
    filler_set = {f.lower() for f in fillers}
    return [
        Cut(w.start, w.end, w.text)
        for w in words
        if is_filler(normalize_word(w.text), filler_set)
    ]
