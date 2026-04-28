"""Audio loading and quiet-region selection."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .models import Word


def load_audio_mono(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load any ffmpeg-readable audio file as mono float32 at `target_sr`."""
    import librosa  # heavy; lazy
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype(np.float32), int(sr)


def find_quiet_region(
    audio: np.ndarray,
    sr: int,
    words: Sequence[Word],
    min_length_s: float = 0.4,
    max_length_s: float = 1.5,
    win_ms: float = 10.0,
) -> tuple[float, float] | None:
    """Find a stretch of mostly-silent audio suitable as a room-tone sample.

    We need a region with no speech and only background noise (HVAC, mic
    hiss, room tone). The gap *before the first transcribed word* is usually
    the cleanest source — it's pre-roll silence with no speaker activity.
    Falls back to the gap after the last word if the leading gap is too
    short.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    total = float(audio.size) / sr

    sorted_words = sorted(words, key=lambda w: w.start)
    candidates: list[tuple[float, float]] = []
    if sorted_words:
        candidates.append((0.0, sorted_words[0].start))
        candidates.append((sorted_words[-1].end, total))
    else:
        candidates.append((0.0, total))

    # Trim 50ms off each side to avoid clipping the start of speech
    # or the tail of the previous word's silence-pad.
    pad = 0.05
    for start_s, end_s in candidates:
        if end_s - start_s < min_length_s + 2 * pad:
            continue
        s = start_s + pad
        e = min(end_s - pad, s + max_length_s)
        if e - s >= min_length_s:
            return (s, e)
    return None
