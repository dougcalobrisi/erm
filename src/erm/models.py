"""Core data types: a transcribed Word and a planned Cut."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass
class Cut:
    start: float
    end: float
    word: str

    def as_dict(self) -> dict:
        return asdict(self)
