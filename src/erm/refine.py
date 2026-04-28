"""Snap cut boundaries to nearby silence + zero-crossings."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .envelope import _rms_envelope, _snap_to_local_min, _snap_to_zero_crossing
from .models import Cut, Word


def _allowed_range(cut: Cut, words: Sequence[Word], total_duration: float
                   ) -> tuple[float, float]:
    """Range a cut may be expanded into without crossing a word boundary.

    - If `cut.start` falls inside a word, the refined start cannot go below
      that word's start (we'd eat the preceding phoneme).
    - If `cut.start` is in an inter-word gap, the refined start cannot go
      below the preceding word's end.
    - Symmetric logic for `cut.end`.

    Together these prevent `refine_boundaries`' energy-minimum search from
    snapping into a real word the cut wasn't supposed to touch.
    """
    lo = 0.0
    hi = total_duration
    for w in words:
        if w.start <= cut.start < w.end:
            lo = max(lo, w.start)
        elif w.end <= cut.start:
            lo = max(lo, w.end)
        if w.start < cut.end <= w.end:
            hi = min(hi, w.end)
        elif w.start >= cut.end:
            hi = min(hi, w.start)
    return lo, hi


def refine_boundaries(
    audio: np.ndarray,
    sr: int,
    cuts: Sequence[Cut],
    search_ms: float = 60.0,
    zc_search_ms: float = 5.0,
    win_ms: float = 10.0,
    words: Sequence[Word] | None = None,
    total_duration: float | None = None,
) -> list[Cut]:
    """Snap each cut endpoint to a local energy minimum, then a zero-crossing.

    Operates on mono float audio. If `audio` is multi-channel (shape (n, ch))
    the caller is responsible for mixing down — we treat the input as 1-D.

    When `words` is provided, refinement is clamped so it never crosses a
    word's start/end timestamp — preventing the energy-minimum search from
    extending a cut into a neighboring word the cut wasn't meant to touch.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    envelope, hop = _rms_envelope(audio, sr, win_ms=win_ms)

    search_frames = max(1, int(round(search_ms / win_ms)))
    zc_search_samples = max(1, int(round(sr * zc_search_ms / 1000.0)))

    duration = float(audio.size) / sr if total_duration is None else total_duration
    refined: list[Cut] = []
    for c in cuts:
        if words is not None:
            lo_s, hi_s = _allowed_range(c, words, duration)
        else:
            lo_s, hi_s = 0.0, duration
        lo_sample = max(0, int(round(lo_s * sr)))
        hi_sample = min(audio.size, int(round(hi_s * sr)))
        s_sample = int(round(c.start * sr))
        e_sample = int(round(c.end * sr))

        # Energy-minimum snap. Start prefers the earliest min in window
        # (leading edge of silence); end prefers the latest (trailing edge).
        s_frame = _snap_to_local_min(envelope, s_sample // hop, search_frames)
        e_frame = _snap_to_local_min(envelope, e_sample // hop, search_frames,
                                     prefer_late=True)
        s_sample = s_frame * hop
        # End snaps to the *end* of its frame so the cut covers the full
        # trailing silent frame, not just its onset.
        e_sample = (e_frame + 1) * hop

        s_sample = _snap_to_zero_crossing(audio, s_sample, zc_search_samples)
        e_sample = _snap_to_zero_crossing(audio, e_sample, zc_search_samples)

        s_sample = max(s_sample, lo_sample)
        e_sample = min(e_sample, hi_sample)

        if e_sample <= s_sample:
            # Refinement collapsed the cut; keep the original to avoid losing it.
            refined.append(c)
            continue
        refined.append(Cut(s_sample / sr, e_sample / sr, c.word))
    return refined
