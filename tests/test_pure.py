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
    expected_max_word_duration,
    find_fillers,
    find_quiet_region,
    inject_min_gaps,
    invert_to_keep_ranges,
    is_filler,
    merge_close_cuts,
    normalize_word,
    pad_cuts,
    refine_boundaries,
)
from erm import ffmpeg_ops
from erm.ffmpeg_ops import _mute_filter, _splice_crossfade_s, render


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


# ---------- merge_close_cuts -----------------------------------------------


def test_merge_close_cuts_empty():
    assert merge_close_cuts([]) == []


def test_merge_close_cuts_far_apart_untouched():
    cuts = [Cut(0.0, 0.2, "um"), Cut(1.0, 1.2, "uh")]
    merged = merge_close_cuts(cuts, min_gap_s=0.10)
    assert len(merged) == 2
    assert merged == cuts


def test_merge_close_cuts_collapses_short_fragment():
    # 40ms surviving fragment between the two cuts -> merge.
    cuts = [Cut(0.0, 0.20, "um"), Cut(0.24, 0.40, "uh")]
    merged = merge_close_cuts(cuts, min_gap_s=0.10)
    assert len(merged) == 1
    assert merged[0].start == pytest.approx(0.0)
    assert merged[0].end == pytest.approx(0.40)
    assert merged[0].word == "um+uh"


def test_merge_close_cuts_identical_labels_kept_single():
    cuts = [Cut(0.0, 0.20, "um"), Cut(0.25, 0.40, "um")]
    merged = merge_close_cuts(cuts, min_gap_s=0.10)
    assert len(merged) == 1
    assert merged[0].word == "um"


def test_merge_close_cuts_gap_exactly_at_threshold_not_merged():
    # Gap == min_gap_s is NOT merged (strict <). 0.25/0.50 are exact in
    # binary float, so the boundary isn't blurred by rounding.
    cuts = [Cut(0.0, 0.25, "um"), Cut(0.50, 0.75, "uh")]
    merged = merge_close_cuts(cuts, min_gap_s=0.25)
    assert len(merged) == 2


def test_merge_close_cuts_union_when_second_nested():
    # Second cut ends before the first -> union keeps the larger end.
    cuts = [Cut(0.0, 0.50, "um"), Cut(0.05, 0.20, "uh")]
    merged = merge_close_cuts(cuts, min_gap_s=0.10)
    assert len(merged) == 1
    assert merged[0].end == pytest.approx(0.50)


def test_merge_close_cuts_sorts_unsorted_input():
    cuts = [Cut(1.0, 1.2, "uh"), Cut(0.0, 0.2, "um")]
    merged = merge_close_cuts(cuts, min_gap_s=0.10)
    assert [c.start for c in merged] == [0.0, 1.0]


def test_merge_close_cuts_chain_collapses_to_one():
    cuts = [Cut(0.0, 0.1, "a"), Cut(0.13, 0.2, "b"), Cut(0.23, 0.3, "c")]
    merged = merge_close_cuts(cuts, min_gap_s=0.10)
    assert len(merged) == 1
    assert merged[0].start == pytest.approx(0.0)
    assert merged[0].end == pytest.approx(0.30)


# ---------- expected_max_word_duration -------------------------------------


def test_expected_max_word_duration_empty_after_normalize():
    # Punctuation-only normalizes to "" -> the empty-word floor.
    assert expected_max_word_duration("...") == pytest.approx(0.40)


@pytest.mark.parametrize(
    "word,expected",
    [
        ("a", 0.30),       # 0.18 + 0.12*1
        ("as", 0.42),
        ("and", 0.54),
        ("that", 0.66),
        ("session", 1.02),
    ],
)
def test_expected_max_word_duration_formula(word, expected):
    assert expected_max_word_duration(word) == pytest.approx(expected)


def test_expected_max_word_duration_ignores_punctuation_in_count():
    # normalize strips the comma, so "and," counts as 3 chars like "and".
    assert expected_max_word_duration("and,") == pytest.approx(
        expected_max_word_duration("and")
    )


def test_expected_max_word_duration_monotonic_in_length():
    durations = [expected_max_word_duration("x" * n) for n in range(1, 8)]
    assert durations == sorted(durations)
    assert len(set(durations)) == len(durations)  # strictly increasing


# ---------- find_quiet_region ----------------------------------------------


def test_find_quiet_region_uses_leading_gap():
    sr = 16_000
    audio = np.zeros(2 * sr, dtype=np.float32)
    words = [_w("hello", 0.80, 1.20)]  # 0.80s of pre-roll silence
    region = find_quiet_region(audio, sr, words)
    assert region is not None
    start, end = region
    # 50ms pad trimmed off the front; region sits inside [0, 0.80].
    assert start == pytest.approx(0.05)
    assert end <= 0.80


def test_find_quiet_region_falls_back_to_trailing_gap():
    sr = 16_000
    total = 2.0
    audio = np.zeros(int(total * sr), dtype=np.float32)
    # Leading gap too short (0.10s), trailing gap is long.
    words = [_w("hi", 0.10, 0.30), _w("there", 0.30, 0.60)]
    region = find_quiet_region(audio, sr, words)
    assert region is not None
    start, end = region
    assert start >= 0.60  # after the last word
    assert end <= total


def test_find_quiet_region_returns_none_when_no_room():
    sr = 16_000
    audio = np.zeros(sr, dtype=np.float32)
    # Both gaps are shorter than min_length + padding.
    words = [_w("hi", 0.05, 0.50), _w("yo", 0.50, 0.95)]
    assert find_quiet_region(audio, sr, words) is None


def test_find_quiet_region_no_words_uses_whole_clip():
    sr = 16_000
    audio = np.zeros(2 * sr, dtype=np.float32)
    region = find_quiet_region(audio, sr, [])
    assert region is not None
    start, end = region
    assert start == pytest.approx(0.05)
    assert end - start <= 1.5  # capped at max_length_s


def test_find_quiet_region_caps_at_max_length():
    sr = 16_000
    audio = np.zeros(10 * sr, dtype=np.float32)
    words = [_w("late", 8.0, 8.5)]  # huge leading gap
    region = find_quiet_region(audio, sr, words, max_length_s=1.5)
    assert region is not None
    start, end = region
    assert end - start == pytest.approx(1.5)


# ---------- _splice_crossfade_s --------------------------------------------


def _cf(cut_s, prev_len=10.0, next_len=10.0, **kw):
    kw.setdefault("crossfade_ms", None)
    kw.setdefault("min_crossfade_ms", 40.0)
    kw.setdefault("max_crossfade_ms", 80.0)
    kw.setdefault("crossfade_factor", 0.10)
    return _splice_crossfade_s(cut_s, prev_len, next_len, **kw)


def test_splice_crossfade_scales_with_cut():
    # 600ms cut * 0.10 = 60ms, within [40, 80].
    assert _cf(0.60) == pytest.approx(0.060)


def test_splice_crossfade_floored_at_min():
    # 100ms cut * 0.10 = 10ms -> floored to 40ms.
    assert _cf(0.10) == pytest.approx(0.040)


def test_splice_crossfade_capped_at_max():
    # 2s cut * 0.10 = 200ms -> capped to 80ms.
    assert _cf(2.0) == pytest.approx(0.080)


def test_splice_crossfade_fixed_override_ignores_cut_and_bounds():
    cf = _cf(2.0, crossfade_ms=30.0)
    assert cf == pytest.approx(0.030)


def test_splice_crossfade_fixed_override_clamped_non_negative():
    assert _cf(0.5, crossfade_ms=-5.0) == pytest.approx(0.0)


def test_splice_crossfade_limited_by_short_fragment():
    # 60ms fade wanted, but the next fragment is only 0.08s -> half is 0.04s.
    assert _cf(0.60, next_len=0.08) == pytest.approx(0.040)


def test_splice_crossfade_word_room_clamp():
    # Fade would be 60ms, but only 10ms of room to the next word
    # -> 2 * 0.010 = 0.020s ceiling.
    cf = _cf(0.60, lhs_room=1.0, rhs_room=0.010)
    assert cf == pytest.approx(0.020)


def test_splice_crossfade_word_room_ignored_when_one_side_missing():
    # Both lhs_room and rhs_room must be present for the word clamp.
    cf = _splice_crossfade_s(
        0.60, 10.0, 10.0,
        crossfade_ms=None, min_crossfade_ms=40.0,
        max_crossfade_ms=80.0, crossfade_factor=0.10,
        lhs_room=None, rhs_room=0.010,
    )
    assert cf == pytest.approx(0.060)


# ---------- pad_cuts -------------------------------------------------------


def test_pad_cuts_factor_zero_is_noop():
    raw = [Cut(0.30, 0.40, "um")]
    refined = [Cut(0.25, 0.45, "um")]
    assert pad_cuts(refined, raw, 0.0, 0.0, 0.120) == refined


def test_pad_cuts_factor_one_shrinks_to_raw_core():
    # factor 1 retains ALL the snapped silence (capped by max_pad), so the
    # padded cut collapses back onto the raw voiced core.
    raw = [Cut(0.30, 0.40, "um")]
    refined = [Cut(0.25, 0.45, "um")]  # 50ms snapped on each side
    padded = pad_cuts(refined, raw, 1.0, 0.0, 0.120)
    assert padded[0].start == pytest.approx(0.30)
    assert padded[0].end == pytest.approx(0.40)
    assert padded[0].word == "um"


def test_pad_cuts_max_pad_clamps_retained_pause():
    raw = [Cut(0.30, 0.40, "um")]
    refined = [Cut(0.10, 0.60, "um")]  # 200ms snapped each side
    padded = pad_cuts(refined, raw, 1.0, 0.0, 0.050)  # cap 50ms per side
    assert padded[0].start == pytest.approx(0.15)  # 0.10 + 0.050
    assert padded[0].end == pytest.approx(0.55)     # 0.60 - 0.050


def test_pad_cuts_min_pad_does_not_exceed_available_silence():
    # min_pad floor is itself capped by the silence that exists, so a side
    # with only 20ms of silence retains at most 20ms even if min_pad is 50ms.
    raw = [Cut(0.30, 0.40, "um")]
    refined = [Cut(0.28, 0.42, "um")]  # 20ms snapped each side
    padded = pad_cuts(refined, raw, 0.10, 0.050, 0.120)
    assert padded[0].start == pytest.approx(0.30)  # retains all 20ms -> raw.start
    assert padded[0].end == pytest.approx(0.40)


def test_pad_cuts_asymmetric_silence():
    raw = [Cut(0.30, 0.40, "um")]
    refined = [Cut(0.20, 0.45, "um")]  # 100ms left, 50ms right
    padded = pad_cuts(refined, raw, 0.5, 0.0, 0.120)
    assert padded[0].start == pytest.approx(0.25)  # 0.20 + 0.5*0.10
    assert padded[0].end == pytest.approx(0.425)    # 0.45 - 0.5*0.05


def test_pad_cuts_zero_silence_edge_gets_no_padding():
    # A tight mid-sentence filler the refiner couldn't snap off — no silence
    # to retain, so the cut is unchanged and its neighbors still butt together.
    raw = [Cut(0.30, 0.40, "um")]
    refined = [Cut(0.30, 0.40, "um")]
    padded = pad_cuts(refined, raw, 1.0, 0.0, 0.120)
    assert padded[0].start == pytest.approx(0.30)
    assert padded[0].end == pytest.approx(0.40)


def test_pad_cuts_never_inverts_a_cut():
    # Pads big enough to cross would invert the cut; that cut is left as-is.
    raw = [Cut(0.30, 0.32, "um")]   # 20ms voiced core
    refined = [Cut(0.10, 0.50, "um")]
    padded = pad_cuts(refined, raw, 1.0, 0.300, 0.400)  # huge min_pad
    # min(left_sil=0.20, clamp(0.20,0.3,0.4)=0.3) = 0.20 -> start 0.30;
    # min(right_sil=0.18, clamp(0.18,0.3,0.4)=0.3) = 0.18 -> end 0.32.
    # That doesn't invert here, so verify the non-inverting result is sane.
    assert padded[0].end > padded[0].start


def test_pad_cuts_collapse_leaves_cut_unchanged():
    # Construct a genuine collapse: voiced core is zero-width, full retention
    # would push start past end, so the refined cut is kept verbatim.
    raw = [Cut(0.40, 0.40, "um")]
    refined = [Cut(0.30, 0.50, "um")]
    padded = pad_cuts(refined, raw, 1.0, 0.120, 0.120)  # 0.12 each side
    # start 0.30+0.10(=left_sil capped) ... left_sil=0.10 -> +0.10 = 0.40;
    # right_sil=0.10 -> -0.10 = 0.40 -> new_end == new_start -> kept unchanged.
    assert padded[0].start == pytest.approx(0.30)
    assert padded[0].end == pytest.approx(0.50)


def test_pad_cuts_length_mismatch_returns_input_unchanged():
    raw = [Cut(0.30, 0.40, "um")]
    refined = [Cut(0.25, 0.45, "um"), Cut(1.0, 1.2, "uh")]
    assert pad_cuts(refined, raw, 1.0, 0.0, 0.120) == refined


# ---------- inject_min_gaps ------------------------------------------------


def test_inject_min_gaps_below_floor_inserts_shortfall():
    # Splice with no surrounding silence (words hug both keep boundaries),
    # floor 0.15s -> inject exactly 0.15s.
    keep = [(0.0, 1.0), (1.0, 2.0)]
    words = [_w("a", 0.5, 1.0), _w("b", 1.0, 1.5)]
    timeline = inject_min_gaps(keep, words, 0.15)
    assert timeline == [
        ("keep", 0.0, 1.0),
        ("gap", 0.0, pytest.approx(0.15)),
        ("keep", 1.0, 2.0),
    ]


def test_inject_min_gaps_at_or_above_floor_no_insert():
    # 0.10s silence on the left + 0.10s on the right = 0.20s surviving > floor.
    keep = [(0.0, 1.0), (1.0, 2.0)]
    words = [_w("a", 0.5, 0.90), _w("b", 1.10, 1.5)]
    timeline = inject_min_gaps(keep, words, 0.15)
    assert timeline == [("keep", 0.0, 1.0), ("keep", 1.0, 2.0)]


def test_inject_min_gaps_zero_floor_keeps_only():
    keep = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    words = [_w("a", 0.5, 1.0), _w("b", 1.0, 1.5)]
    assert inject_min_gaps(keep, words, 0.0) == [
        ("keep", 0.0, 1.0), ("keep", 1.0, 2.0), ("keep", 2.0, 3.0),
    ]


def test_inject_min_gaps_missing_words_use_keep_boundaries():
    # No words -> surviving_gap defaults to 0 -> inject the full floor.
    keep = [(0.0, 1.0), (1.0, 2.0)]
    timeline = inject_min_gaps(keep, [], 0.20)
    assert timeline == [
        ("keep", 0.0, 1.0),
        ("gap", 0.0, pytest.approx(0.20)),
        ("keep", 1.0, 2.0),
    ]


def test_inject_min_gaps_partial_shortfall():
    # 0.05s natural pause, floor 0.15s -> inject the 0.10s shortfall only.
    keep = [(0.0, 1.0), (1.0, 2.0)]
    words = [_w("a", 0.5, 0.95), _w("b", 1.0, 1.5)]  # 0.05s left, 0 right
    timeline = inject_min_gaps(keep, words, 0.15)
    gaps = [item for item in timeline if item[0] == "gap"]
    assert len(gaps) == 1
    assert gaps[0][2] == pytest.approx(0.10)


def test_inject_min_gaps_first_and_last_splice():
    # Three keeps, two splices, both below floor -> two gaps interleaved.
    keep = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    words = [_w("a", 0.5, 1.0), _w("b", 1.0, 2.0), _w("c", 2.0, 2.5)]
    timeline = inject_min_gaps(keep, words, 0.10)
    kinds = [item[0] for item in timeline]
    assert kinds == ["keep", "gap", "keep", "gap", "keep"]


def test_inject_min_gaps_empty_keep_ranges():
    assert inject_min_gaps([], [], 0.15) == []


# ---------- _mute_filter ---------------------------------------------------


def test_mute_filter_single_range():
    assert _mute_filter([(1.0, 2.0)]) == (
        "volume=enable='between(t,1.000000,2.000000)':volume=0"
    )


def test_mute_filter_multiple_ranges_joined_with_plus():
    flt = _mute_filter([(1.0, 2.0), (3.5, 4.0)])
    assert flt == (
        "volume=enable='between(t,1.000000,2.000000)"
        "+between(t,3.500000,4.000000)':volume=0"
    )


def test_mute_filter_empty_is_blank():
    assert _mute_filter([]) == ""


# ---------- render word-room defaults --------------------------------------


def _capture_render_filter(monkeypatch, keep_ranges, words):
    """Run render() with ffmpeg stubbed; return the -filter_complex string."""
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr(ffmpeg_ops.subprocess, "run", _fake_run)
    render("in.wav", keep_ranges, "out.wav", words=words)
    cmd = captured["cmd"]
    return cmd[cmd.index("-filter_complex") + 1]


def test_render_keeps_crossfades_when_no_word_after_last_splice(monkeypatch):
    # A splice whose right side has no following word must not collapse the
    # fade to zero (which would force the whole render onto `concat`). With
    # keep-range-boundary defaults the fade survives and we still crossfade.
    keep_ranges = [(0.0, 1.0), (1.5, 2.0), (2.5, 3.0)]
    words = [_w("hello", 0.1, 0.9)]  # nothing at/after the later splices
    filter_complex = _capture_render_filter(monkeypatch, keep_ranges, words)
    assert "acrossfade" in filter_complex
    assert "concat" not in filter_complex


def test_render_falls_back_to_concat_when_a_real_word_hugs_the_splice(monkeypatch):
    # Sanity check the other direction: a word ending exactly at a splice
    # leaves zero room, so that fade is 0 and render uses concat. This pins
    # the behavior the default-fix is careful NOT to trigger spuriously.
    keep_ranges = [(0.0, 1.0), (1.5, 2.0)]
    words = [_w("hugs", 0.5, 1.0)]  # ends at the splice LHS -> lhs_room == 0
    filter_complex = _capture_render_filter(monkeypatch, keep_ranges, words)
    assert "concat" in filter_complex
    assert "acrossfade" not in filter_complex
