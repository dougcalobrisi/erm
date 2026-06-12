"""Cut-list manipulation: merging close cuts and inverting to keep-ranges."""

from __future__ import annotations

from typing import Sequence

from .models import Cut, Word


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def pad_cuts(
    refined_cuts: Sequence[Cut],
    raw_cuts: Sequence[Cut],
    factor: float,
    min_pad_s: float,
    max_pad_s: float,
) -> list[Cut]:
    """Retain a fraction of the silence each cut snapped over, per cut.

    `refine_boundaries` emits one refined cut per raw cut, in input order, so
    `refined_cuts[i]` corresponds to `raw_cuts[i]`. The silence a cut snapped
    into lives between the raw (voiced) boundary and the refined (silence)
    boundary on each side:

        left_silence  = max(0, raw.start - refined.start)
        right_silence  = max(0, refined.end - raw.end)

    We move each refined endpoint back toward the voiced core by a *fraction*
    of that side's silence, clamped to ``[min_pad_s, max_pad_s]`` and never
    exceeding the silence that actually exists there — so a tight mid-sentence
    filler with no surrounding silence keeps butting its neighbors together,
    and padding can never add time that wasn't already inside the cut.

    Defensive guards: if the two lists aren't positionally 1:1 (length
    mismatch) the refined list is returned unchanged; if the pads would
    collapse or invert a cut, that cut is left unpadded so the filler is
    always removed.
    """
    if factor <= 0 or len(refined_cuts) != len(raw_cuts):
        return list(refined_cuts)

    padded: list[Cut] = []
    for refined, raw in zip(refined_cuts, raw_cuts):
        left_silence = max(0.0, raw.start - refined.start)
        right_silence = max(0.0, refined.end - raw.end)
        pad_left = min(left_silence, _clamp(factor * left_silence, min_pad_s, max_pad_s))
        pad_right = min(right_silence, _clamp(factor * right_silence, min_pad_s, max_pad_s))
        new_start = refined.start + pad_left
        new_end = refined.end - pad_right
        if new_end <= new_start:
            padded.append(refined)
        else:
            padded.append(Cut(new_start, new_end, refined.word))
    return padded


def inject_min_gaps(
    keep_ranges: Sequence[tuple[float, float]],
    words: Sequence[Word] | None,
    min_gap_s: float,
) -> list[tuple[str, float, float]]:
    """Build a render timeline that guarantees ``min_gap_s`` at every splice.

    Returns an ordered list of timeline items: ``("keep", start, end)`` for
    each keep range, with ``("gap", 0.0, duration)`` items interleaved at any
    splice where the *natural* surviving pause between the two flanking words
    falls short of ``min_gap_s``. For each splice between keep ``i`` and keep
    ``i+1``:

        prev_word_end   = max word.end <= keep[i].end   (else keep[i].end)
        next_word_start = min word.start >= keep[i+1].start (else keep[i+1].start)
        surviving_gap   = (keep[i].end - prev_word_end)
                        + (next_word_start - keep[i+1].start)

    When ``surviving_gap < min_gap_s`` a gap of exactly the shortfall is
    injected. With ``min_gap_s <= 0`` or no shortfall anywhere the result is
    just the keep ranges, a faithful superset of ``keep_ranges``.
    """
    timeline: list[tuple[str, float, float]] = []
    words = words or []
    count = len(keep_ranges)
    for i, (start, end) in enumerate(keep_ranges):
        timeline.append(("keep", start, end))
        if min_gap_s <= 0 or i == count - 1:
            continue
        left_edge = end
        right_edge = keep_ranges[i + 1][0]
        prev_word_end = max(
            (w.end for w in words if w.end <= left_edge), default=left_edge
        )
        next_word_start = min(
            (w.start for w in words if w.start >= right_edge), default=right_edge
        )
        surviving_gap = (left_edge - prev_word_end) + (next_word_start - right_edge)
        shortfall = min_gap_s - surviving_gap
        if shortfall > 0:
            timeline.append(("gap", 0.0, shortfall))
    return timeline
