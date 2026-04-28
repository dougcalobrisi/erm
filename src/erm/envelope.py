"""RMS envelope computation and snap-to-minimum / zero-crossing helpers.

Used by the audio detectors and by `refine_boundaries` to nudge cut endpoints
to acoustically natural splice points.
"""

from __future__ import annotations

import numpy as np


def _rms_envelope(audio: np.ndarray, sr: int, win_ms: float = 10.0) -> tuple[np.ndarray, int]:
    """Frame-based RMS energy. Returns (envelope, hop_samples).

    Frame size and hop are equal (non-overlapping ~win_ms windows). Mono only;
    callers should mix down stereo first.
    """
    hop = max(1, int(round(sr * win_ms / 1000.0)))
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32), hop
    n_frames = audio.size // hop
    if n_frames == 0:
        return np.array([float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))],
                        dtype=np.float32), hop
    trimmed = audio[: n_frames * hop].astype(np.float64).reshape(n_frames, hop)
    return np.sqrt((trimmed * trimmed).mean(axis=1)).astype(np.float32), hop


def _snap_to_local_min(
    envelope: np.ndarray, frame_idx: int, search_frames: int, prefer_late: bool = False
) -> int:
    """Index of the smallest envelope value within ±search_frames.

    Ties are broken by the earliest index, unless `prefer_late` — used for
    end-of-cut endpoints so the cut extends to the *trailing* edge of a silent
    region instead of its leading edge.
    """
    if envelope.size == 0:
        return frame_idx
    lo = max(0, frame_idx - search_frames)
    hi = min(envelope.size, frame_idx + search_frames + 1)
    if hi <= lo:
        return frame_idx
    window = envelope[lo:hi]
    if prefer_late:
        rev_idx = int(np.argmin(window[::-1]))
        return int(hi - 1 - rev_idx)
    return int(lo + np.argmin(window))


def _snap_to_zero_crossing(audio: np.ndarray, sample_idx: int, search_samples: int) -> int:
    """Snap to the nearest zero-crossing within ±search_samples.

    Falls back to the original index when no crossing is found in range.
    """
    n = audio.size
    if n == 0:
        return sample_idx
    sample_idx = int(np.clip(sample_idx, 0, n - 1))
    if audio[sample_idx] == 0:
        return sample_idx
    lo = max(1, sample_idx - search_samples)
    hi = min(n, sample_idx + search_samples + 1)
    if hi <= lo:
        return sample_idx
    window = audio[lo - 1:hi]
    signs = np.sign(window)
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs) != 0)[0]
    zeros = np.where(audio[lo:hi] == 0)[0]
    candidates = np.concatenate([crossings + lo, zeros + lo]) if zeros.size else crossings + lo
    if candidates.size == 0:
        return sample_idx
    return int(candidates[np.argmin(np.abs(candidates - sample_idx))])
