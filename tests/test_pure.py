"""Unit tests for the pure functions in erm.

These tests deliberately avoid importing faster-whisper or librosa so they
can run on a machine that hasn't downloaded a model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from erm import (
    Cut,
    Word,
    DEFAULT_FILLERS,
    find_fillers,
    invert_to_keep_ranges,
    is_filler,
    normalize_word,
    refine_boundaries,
)


# ---------- normalize_word -------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Um,", "um"),
        (" UH! ", "uh"),
        ("Hello.", "hello"),
        ('"hmm"', "hmm"),
        ("uh-huh", "uh-huh"),
        ("Don't", "dont"),
    ],
)
def test_normalize_word(raw, expected):
    assert normalize_word(raw) == expected


# ---------- find_fillers ---------------------------------------------------


def _w(text, start, end):
    return Word(text=text, start=start, end=end)


def test_find_fillers_basic():
    words = [
        _w("Hello", 0.0, 0.4),
        _w("um,", 0.4, 0.7),
        _w("world", 0.7, 1.1),
    ]
    cuts = find_fillers(words, DEFAULT_FILLERS)
    assert len(cuts) == 1
    assert cuts[0].start == pytest.approx(0.4)
    assert cuts[0].end == pytest.approx(0.7)
    assert cuts[0].word == "um,"


def test_find_fillers_none():
    words = [_w("All", 0.0, 0.3), _w("clean", 0.3, 0.7)]
    assert find_fillers(words, DEFAULT_FILLERS) == []


def test_find_fillers_empty_words():
    assert find_fillers([], DEFAULT_FILLERS) == []


def test_find_fillers_back_to_back():
    words = [
        _w("um", 0.0, 0.2),
        _w("uh", 0.2, 0.4),
        _w("yeah", 0.4, 0.7),
    ]
    cuts = find_fillers(words, DEFAULT_FILLERS)
    assert len(cuts) == 2
    assert [c.word for c in cuts] == ["um", "uh"]


def test_find_fillers_custom_set():
    words = [_w("like", 0.0, 0.2), _w("um", 0.2, 0.4)]
    cuts = find_fillers(words, {"like"})
    assert len(cuts) == 1
    assert cuts[0].word == "like"


def test_find_fillers_case_insensitive_punctuation():
    words = [_w("Um,", 0.0, 0.2), _w('"UH!"', 0.2, 0.4)]
    cuts = find_fillers(words, DEFAULT_FILLERS)
    assert len(cuts) == 2


@pytest.mark.parametrize(
    "word",
    ["um", "umm", "ummm", "ummmm",
     "uh", "uhh", "uhhh", "uhhhhh",
     "ah", "ahh", "ahhh", "ahhhhh",
     "er", "err", "erm", "erms",  # "erms" intentionally NOT a filler
     "hmm", "hmmm", "hmmmm",
     "mm", "mmm", "mmmm",
     "mhm", "mhmm", "mhmmm",
     "uh-huh", "uhh-huhh"],
)
def test_is_filler_elongations(word):
    if word == "erms":
        assert not is_filler(word, DEFAULT_FILLERS)
    else:
        assert is_filler(word, DEFAULT_FILLERS), word


def test_is_filler_rejects_real_words():
    for word in ["umbrella", "uhhuh", "ahead", "hum", "mum", "her", "errand"]:
        assert not is_filler(word, DEFAULT_FILLERS), word


def test_find_fillers_catches_long_elongations():
    words = [
        _w("So", 0.0, 0.2),
        _w("uhhhhh", 0.2, 0.9),
        _w("yeah", 0.9, 1.2),
    ]
    cuts = find_fillers(words, DEFAULT_FILLERS)
    assert len(cuts) == 1
    assert cuts[0].word == "uhhhhh"


# ---------- invert_to_keep_ranges -----------------------------------------


def test_invert_no_cuts_keeps_everything():
    keep = invert_to_keep_ranges([], total_duration=10.0)
    assert keep == [(0.0, 10.0)]


def test_invert_single_cut_in_middle():
    cuts = [Cut(2.0, 3.0, "um")]
    assert invert_to_keep_ranges(cuts, 5.0) == [(0.0, 2.0), (3.0, 5.0)]


def test_invert_cut_at_start():
    cuts = [Cut(0.0, 1.0, "um")]
    assert invert_to_keep_ranges(cuts, 5.0) == [(1.0, 5.0)]


def test_invert_cut_at_end():
    cuts = [Cut(4.0, 5.0, "um")]
    assert invert_to_keep_ranges(cuts, 5.0) == [(0.0, 4.0)]


def test_invert_full_duration_cut_is_empty():
    cuts = [Cut(0.0, 5.0, "um")]
    assert invert_to_keep_ranges(cuts, 5.0) == []


def test_invert_overlapping_cuts_merge():
    cuts = [Cut(1.0, 2.5, "um"), Cut(2.0, 3.0, "uh")]
    assert invert_to_keep_ranges(cuts, 5.0) == [(0.0, 1.0), (3.0, 5.0)]


def test_invert_back_to_back_cuts_merge():
    cuts = [Cut(1.0, 2.0, "um"), Cut(2.0, 3.0, "uh")]
    assert invert_to_keep_ranges(cuts, 5.0) == [(0.0, 1.0), (3.0, 5.0)]


def test_invert_unsorted_input_handled():
    cuts = [Cut(4.0, 4.5, "uh"), Cut(1.0, 2.0, "um")]
    assert invert_to_keep_ranges(cuts, 5.0) == [(0.0, 1.0), (2.0, 4.0), (4.5, 5.0)]


def test_invert_zero_duration_returns_empty():
    assert invert_to_keep_ranges([Cut(0.0, 1.0, "um")], 0.0) == []


# ---------- refine_boundaries ---------------------------------------------


def _make_signal(sr: int, sections: list[tuple[float, float, float]]) -> np.ndarray:
    """Build a deterministic test signal.

    Each section is (duration_s, frequency_hz, amplitude). Frequency 0 produces
    silence. Sections are concatenated.
    """
    parts = []
    for dur, freq, amp in sections:
        n = int(round(dur * sr))
        if freq == 0 or amp == 0:
            parts.append(np.zeros(n, dtype=np.float32))
        else:
            t = np.arange(n) / sr
            parts.append((amp * np.sin(2 * np.pi * freq * t)).astype(np.float32))
    return np.concatenate(parts)


def test_refine_snaps_into_silence_gap():
    """Whisper guesses a cut that lands inside the surrounding speech tones;
    refinement should pull it into the silent gap between them."""
    sr = 16_000
    # 0.0–0.30s tone (speech), 0.30–0.40s silence (filler placeholder),
    # 0.40–0.70s tone (speech). Whisper-reported filler: 0.27–0.43 (sloppy).
    audio = _make_signal(sr, [(0.30, 440.0, 0.5),
                              (0.10, 0.0, 0.0),
                              (0.30, 440.0, 0.5)])
    cuts = [Cut(0.27, 0.43, "um")]
    refined = refine_boundaries(audio, sr, cuts, search_ms=60.0)
    assert len(refined) == 1
    r = refined[0]
    # Both endpoints should land inside the silence gap [0.30, 0.40].
    assert 0.295 <= r.start <= 0.305, f"start={r.start}"
    assert 0.395 <= r.end <= 0.405, f"end={r.end}"


def test_refine_endpoints_land_on_zero_crossings():
    sr = 16_000
    audio = _make_signal(sr, [(0.30, 440.0, 0.5),
                              (0.10, 0.0, 0.0),
                              (0.30, 440.0, 0.5)])
    cuts = [Cut(0.27, 0.43, "um")]
    refined = refine_boundaries(audio, sr, cuts, search_ms=60.0)
    r = refined[0]
    s_idx = int(round(r.start * sr))
    e_idx = int(round(r.end * sr))
    # In the silent gap the samples are exactly zero — that counts as a crossing.
    assert abs(audio[s_idx]) < 1e-6
    assert abs(audio[e_idx]) < 1e-6


def test_refine_preserves_cut_when_collapsed():
    """If the search window can't find a valid arrangement we keep the original."""
    sr = 16_000
    audio = _make_signal(sr, [(0.5, 440.0, 0.5)])  # all speech, no silence
    cuts = [Cut(0.20, 0.30, "um")]
    refined = refine_boundaries(audio, sr, cuts, search_ms=10.0)
    assert len(refined) == 1
    # Just verify we still have a non-degenerate cut.
    assert refined[0].end > refined[0].start


def test_refine_handles_empty_cuts():
    sr = 16_000
    audio = np.zeros(sr, dtype=np.float32)
    assert refine_boundaries(audio, sr, [], search_ms=60.0) == []


def test_refine_handles_stereo_input():
    sr = 16_000
    mono = _make_signal(sr, [(0.30, 440.0, 0.5),
                             (0.10, 0.0, 0.0),
                             (0.30, 440.0, 0.5)])
    stereo = np.stack([mono, mono], axis=1)  # shape (n, 2)
    cuts = [Cut(0.27, 0.43, "um")]
    refined = refine_boundaries(stereo, sr, cuts, search_ms=60.0)
    assert len(refined) == 1
    assert 0.295 <= refined[0].start <= 0.305
