"""Acoustic feature checks (librosa-based, lazy-imported)."""

from __future__ import annotations

import numpy as np


def is_sustained_vowel(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    max_centroid_cv: float = 0.18,
    min_voiced_frac: float = 0.50,
) -> bool:
    """Return True if [start_s, end_s] looks acoustically like a sustained
    filler vowel ("uhhh", "ahhh", "ummm").

    Filler vowels have two distinguishing features compared to real word
    content: (a) the spectral energy stays in roughly the same place across
    the region (low spectral-centroid variation), and (b) most frames are
    voiced (ZCR in the voiced range, not silence or fricative noise).

    `max_centroid_cv` is the std/mean ratio of the spectral centroid; lower
    means more stable. `min_voiced_frac` is the fraction of frames whose
    zero-crossing rate is in the typical voiced-speech range.
    """
    import librosa  # heavy; lazy

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    s = max(0, int(start_s * sr))
    e = min(audio.size, int(end_s * sr))
    seg = audio[s:e]
    if seg.size < int(0.06 * sr):
        return False

    n_fft = 1024
    hop = max(1, int(0.020 * sr))
    if seg.size < n_fft:
        seg = np.pad(seg, (0, n_fft - seg.size), mode="constant")

    centroid = librosa.feature.spectral_centroid(
        y=seg, sr=sr, n_fft=n_fft, hop_length=hop,
    )[0]
    if centroid.size < 3:
        return False
    mean_c = float(centroid.mean())
    if mean_c <= 1e-6:
        return False
    cv = float(centroid.std() / mean_c)

    zcr = librosa.feature.zero_crossing_rate(
        y=seg, frame_length=n_fft, hop_length=hop,
    )[0]
    voiced = (zcr > 0.02) & (zcr < 0.20)
    voiced_frac = float(voiced.mean()) if voiced.size else 0.0

    return cv <= max_centroid_cv and voiced_frac >= min_voiced_frac
