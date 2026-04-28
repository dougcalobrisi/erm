"""Audio-based filler detectors that go beyond Whisper's transcription.

These find fillers Whisper misses: dropped tokens in long silent gaps,
fillers fused into the timestamp of a neighboring word, and trailing
sustained-vowel fillers that flow continuously out of a real word.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .acoustic import is_sustained_vowel
from .envelope import _rms_envelope
from .fillers import normalize_word
from .models import Cut, Word


def expected_max_word_duration(word_text: str) -> float:
    """Conservative upper bound on how long a clean utterance of `word_text`
    should take, in seconds. Used to detect words whose timestamp is so much
    longer than the text justifies that the excess must be hidden filler.

    Tuned against typical English speech rates (~3 syllables/s, ~12-14 phonemes/s).
    Keep this generous so we never trim a fast-but-real word.
    """
    n = len(normalize_word(word_text))
    if n == 0:
        return 0.40
    # 0.12s/char + 0.18s base. So 'a' -> 0.30, 'as' -> 0.42, 'and' -> 0.54,
    # 'that' -> 0.66, 'session' -> 1.02, 'misunderstandings' -> 2.22.
    return 0.18 + 0.12 * n


def _voiced_runs_in_region(
    envelope: np.ndarray,
    hop: int,
    sr: int,
    region_start_s: float,
    region_end_s: float,
    threshold: float,
    min_s: float,
    max_s: float,
    bridge_frames: int,
    label: str,
) -> list[Cut]:
    """Return contiguous voiced runs within [region_start_s, region_end_s].

    Sub-threshold dips shorter than `bridge_frames` are bridged so a single
    drawn-out filler with mild amplitude flicker doesn't fragment.
    """
    f0 = max(0, int(region_start_s * sr / hop))
    f1 = min(envelope.size, int(region_end_s * sr / hop))
    if f1 <= f0:
        return []
    voiced = envelope[f0:f1] > threshold
    cuts: list[Cut] = []
    i = 0
    while i < voiced.size:
        if not voiced[i]:
            i += 1
            continue
        j = i + 1
        while j < voiced.size:
            if voiced[j]:
                j += 1
                continue
            k = j
            while k < voiced.size and not voiced[k] and (k - j) < bridge_frames:
                k += 1
            if k < voiced.size and voiced[k]:
                j = k
                continue
            break
        run_start_s = (f0 + i) * hop / sr
        run_end_s = (f0 + j) * hop / sr
        run_len = run_end_s - run_start_s
        if min_s <= run_len <= max_s:
            cuts.append(Cut(run_start_s, run_end_s, label))
        i = j
    return cuts


def detect_intraword_fillers(
    audio: np.ndarray,
    sr: int,
    words: Sequence[Word],
    min_word_s: float = 0.55,
    min_dip_ms: float = 50.0,
    min_voiced_s: float = 0.12,
    max_voiced_s: float = 1.50,
    silence_floor_db: float = -40.0,
    win_ms: float = 10.0,
    confirm_pitch: bool = True,
) -> list[Cut]:
    """Find fillers Whisper subsumes into a neighboring word's timestamp.

    Whisper often transcribes "in, uhhhhh" as a single token `'in'` whose
    end timestamp covers both the real word and the filler. A long word's
    interior typically looks like:

        [real word][dip][filler]                       — one filler
        [real word][dip][filler][dip][filler]          — multiple fillers
        [real word][dip][syllable][dip][syllable]...   — legitimate long word

    Strategy: split the word's interior into contiguous voiced runs separated
    by silence dips of at least `min_dip_ms`. Assume the *first* voiced run is
    the real word and emit a cut for every subsequent run that meets the
    voiced-duration thresholds. Words with no internal dip (single voiced run)
    are left alone — that's how legitimately-long words like "misunderstandings"
    avoid getting trimmed.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    envelope, hop = _rms_envelope(audio, sr, win_ms=win_ms)
    if envelope.size == 0:
        return []

    peak = float(envelope.max())
    if peak <= 0:
        return []
    threshold = peak * (10.0 ** (silence_floor_db / 20.0))
    min_dip_frames = max(1, int(round(min_dip_ms / win_ms)))

    cuts: list[Cut] = []
    for w in words:
        if w.end - w.start < min_word_s:
            continue
        f0 = max(0, int(w.start * sr / hop))
        f1 = min(envelope.size, int(w.end * sr / hop))
        if f1 - f0 < min_dip_frames * 2:
            continue
        voiced = envelope[f0:f1] > threshold

        runs: list[tuple[int, int]] = []
        i = 0
        n = voiced.size
        while i < n and not voiced[i]:
            i += 1
        while i < n:
            run_start = i
            while i < n:
                if voiced[i]:
                    i += 1
                    continue
                j = i
                while j < n and not voiced[j]:
                    j += 1
                if (j - i) >= min_dip_frames:
                    runs.append((run_start, i))
                    i = j
                    break
                i = j  # short dip — keep walking the same run
            else:
                runs.append((run_start, i))
                break

        if len(runs) <= 1:
            continue  # legit long word with no internal dip; leave alone

        # Sum-of-runs guard: if the total voiced time inside the word is no
        # larger than the word's expected duration, the multiple runs are
        # just natural phoneme structure (e.g. "sharing" splits into "shar"
        # + "ring" across a ~160ms dip) — not real word + hidden filler.
        expected_word = expected_max_word_duration(w.text)
        run_sum_s = sum(((re - rs) * hop / sr) for rs, re in runs)
        if run_sum_s <= expected_word * 1.2:
            continue

        run_seconds = [
            ((f0 + rs) * hop / sr, (f0 + re) * hop / sr) for rs, re in runs
        ]
        if confirm_pitch:
            non_vowel = [
                idx for idx, (s, e) in enumerate(run_seconds)
                if not is_sustained_vowel(audio, sr, s, e)
            ]
        else:
            non_vowel = list(range(len(runs)))

        if not non_vowel:
            continue

        expected = expected_word

        # Detect "structurally anomalous" words: a real word doesn't have
        # 200ms+ of silence in the middle of its bounds. If we find one,
        # any run *before* the big silence is suspicious — Whisper's start
        # boundary is probably engulfing a leading filler.
        big_dip_idx: int | None = None
        for k in range(len(runs) - 1):
            gap_frames = runs[k + 1][0] - runs[k][1]
            if gap_frames * win_ms / 1000.0 >= 0.20:
                big_dip_idx = k
                break
        if big_dip_idx is not None:
            non_vowel = [i for i in non_vowel if i > big_dip_idx]
            if not non_vowel:
                continue

        # Pick the non-vowel run whose duration is closest to (but not
        # wildly exceeding) `expected`.
        def _score(idx: int) -> float:
            s, e = run_seconds[idx]
            d = e - s
            if d > expected * 1.8:
                return abs(d - expected) * 4.0  # heavy penalty for overlong
            return abs(d - expected)

        real_idx = min(non_vowel, key=_score)

        for idx, (run_start, run_end) in enumerate(runs):
            if idx == real_idx:
                continue
            run_start_s = (f0 + run_start) * hop / sr
            run_end_s = (f0 + run_end) * hop / sr
            run_len = run_end_s - run_start_s
            if min_voiced_s <= run_len <= max_voiced_s:
                cuts.append(Cut(run_start_s, run_end_s, f"<in:{w.text}>"))
    return cuts


def detect_overlong_words(
    audio: np.ndarray,
    sr: int,
    words: Sequence[Word],
    excess_factor: float = 1.6,
    min_voiced_s: float = 0.12,
    max_voiced_s: float = 1.50,
    silence_floor_db: float = -40.0,
    win_ms: float = 10.0,
) -> list[Cut]:
    """Catch fillers in words whose interior has no detectable silence dip.

    `detect_intraword_fillers` requires a sub-threshold gap to split a word
    into "real word" and "filler" runs. When the filler runs continuously
    into the word with no breath, that detector fails. This pass uses the
    word's *expected* duration (from `expected_max_word_duration`) to flag
    the trailing portion of words whose timestamp is much longer than the
    text justifies, then scans that portion for voiced energy.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    envelope, hop = _rms_envelope(audio, sr, win_ms=win_ms)
    if envelope.size == 0:
        return []

    peak = float(envelope.max())
    if peak <= 0:
        return []
    threshold = peak * (10.0 ** (silence_floor_db / 20.0))
    bridge_frames = max(1, int(round(80.0 / win_ms)))

    cuts: list[Cut] = []
    for w in words:
        actual = w.end - w.start
        expected = expected_max_word_duration(w.text)
        if actual <= expected * excess_factor:
            continue
        trail_start = w.start + expected
        cuts.extend(_voiced_runs_in_region(
            envelope, hop, sr, trail_start, w.end,
            threshold=threshold, min_s=min_voiced_s, max_s=max_voiced_s,
            bridge_frames=bridge_frames, label=f"<long:{w.text}>",
        ))
    return cuts


def detect_gap_fillers(
    audio: np.ndarray,
    sr: int,
    words: Sequence[Word],
    total_duration: float,
    min_gap_s: float = 0.25,
    min_voiced_s: float = 0.10,
    max_voiced_s: float = 1.50,
    silence_floor_db: float = -40.0,
    win_ms: float = 10.0,
) -> list[Cut]:
    """Find filler regions Whisper dropped.

    Whisper aggressively cleans up disfluencies — even at large model sizes
    it often emits no token for "um"/"uh"/etc, leaving a long unexplained gap
    between transcribed words. This pass scans those gaps for voiced energy
    and treats any contiguous voiced region within them as a candidate filler.

    Boundaries are deliberately *loose* — the downstream `refine_boundaries`
    snap-to-silence pass tightens them to the actual filler edges.
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    envelope, hop = _rms_envelope(audio, sr, win_ms=win_ms)
    if envelope.size == 0:
        return []

    peak = float(envelope.max()) if envelope.size else 0.0
    if peak <= 0:
        return []
    threshold = peak * (10.0 ** (silence_floor_db / 20.0))

    sorted_words = sorted(words, key=lambda w: w.start)
    if not sorted_words:
        return []
    # Only scan gaps *between* transcribed words. Leading silence before the
    # first word and trailing silence after the last word are intro/outro,
    # not fillers — leave them alone.
    gap_bounds: list[tuple[float, float]] = []
    cursor = sorted_words[0].end
    for w in sorted_words[1:]:
        if w.start - cursor >= min_gap_s:
            gap_bounds.append((cursor, w.start))
        cursor = max(cursor, w.end)

    bridge_frames = max(1, int(round(80.0 / win_ms)))
    cuts: list[Cut] = []
    for g_start, g_end in gap_bounds:
        cuts.extend(_voiced_runs_in_region(
            envelope, hop, sr, g_start, g_end,
            threshold=threshold, min_s=min_voiced_s, max_s=max_voiced_s,
            bridge_frames=bridge_frames, label="<gap>",
        ))
    return cuts
