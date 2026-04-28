"""Cut-list manipulation: merging close cuts and inverting to keep-ranges."""

from __future__ import annotations

from typing import Sequence

from .models import Cut


def merge_close_cuts(cuts: Sequence[Cut], min_gap_s: float = 0.10) -> list[Cut]:
    """Merge cuts whose between-cut gap is shorter than `min_gap_s`.

    A 40ms surviving fragment between two cuts gets eaten by the surrounding
    crossfades and produces an audible "blurp" — better to just collapse the
    two cuts into one. The merged cut takes the union of the spans and a
    label that reflects both (or the first one's label if they're identical).
    """
    if not cuts:
        return []
    sorted_cuts = sorted(cuts, key=lambda c: c.start)
    merged: list[Cut] = [sorted_cuts[0]]
    for c in sorted_cuts[1:]:
        last = merged[-1]
        if c.start - last.end < min_gap_s:
            label = last.word if last.word == c.word else f"{last.word}+{c.word}"
            merged[-1] = Cut(last.start, max(last.end, c.end), label)
        else:
            merged.append(c)
    return merged


def invert_to_keep_ranges(
    cuts: Sequence[Cut], total_duration: float
) -> list[tuple[float, float]]:
    """Return the complement of `cuts` over [0, total_duration].

    Overlapping or out-of-order cuts are merged. Empty keep-ranges (length 0)
    are dropped.
    """
    if total_duration <= 0:
        return []

    spans = sorted(
        (max(0.0, c.start), min(total_duration, c.end)) for c in cuts
    )
    spans = [(s, e) for s, e in spans if e > s]

    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in merged:
        if s > cursor:
            keep.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_duration:
        keep.append((cursor, total_duration))
    return keep
