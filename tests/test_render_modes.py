"""Real-ffmpeg tests for the silence mode and min-gap injection render paths.

These drive `render_silenced` and `render(..., gap_inserts=...)` end-to-end
through ffmpeg on a synthesized PCM WAV and assert on the output duration via
`ffprobe`. They are skipped automatically when `ffmpeg`/`ffprobe` aren't on
PATH, so the fast pure suites stay runnable without them.
"""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from erm.ffmpeg_ops import ffprobe_duration, render, render_silenced


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="render-mode tests need ffmpeg + ffprobe on PATH",
)

SAMPLE_RATE = 16_000
CLIP_DURATION_S = 3.0


def _write_wav(path: Path, duration_s: float = CLIP_DURATION_S) -> None:
    """Synthesize a mono 16-bit WAV: a 220 Hz tone over a faint hiss."""
    n = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    samples = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    samples += np.random.default_rng(0).normal(0.0, 0.002, n)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())


def test_render_silenced_preserves_duration(tmp_path):
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    _write_wav(src)

    # Mute a mid-file span; the timeline (and total duration) must be untouched.
    render_silenced(src, [(1.0, 1.5)], out)

    assert out.exists()
    assert ffprobe_duration(out) == pytest.approx(CLIP_DURATION_S, abs=0.05)


def test_render_silenced_empty_ranges_transcodes_unchanged(tmp_path):
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    _write_wav(src)

    render_silenced(src, [], out)

    assert ffprobe_duration(out) == pytest.approx(CLIP_DURATION_S, abs=0.05)


def test_render_with_injected_gap_adds_its_duration(tmp_path):
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    _write_wav(src)

    # Keep two halves with a 0.4s cut between them; inject a 0.25s gap at the
    # splice. Output ~= kept_total (2.6s) + injected gap (0.25s).
    keep = [(0.0, 1.3), (1.7, 3.0)]
    kept_total = (1.3 - 0.0) + (3.0 - 1.7)
    render(src, keep, out, gap_inserts=[(0, 0.25)])

    assert ffprobe_duration(out) == pytest.approx(kept_total + 0.25, abs=0.05)


def test_render_gapless_is_shorter_than_source(tmp_path):
    # Sanity guard: the default (no gap_inserts) render still excises, so the
    # output is meaningfully shorter than the input.
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    _write_wav(src)

    keep = [(0.0, 1.3), (1.7, 3.0)]
    render(src, keep, out)

    out_dur = ffprobe_duration(out)
    assert out_dur < CLIP_DURATION_S - 0.2
    # Close to kept total (allowing for crossfade overlap shaving a little).
    assert out_dur == pytest.approx((1.3) + (3.0 - 1.7), abs=0.1)


def test_render_min_gap_floor_without_injection_still_renders(tmp_path):
    # A floor set but no gaps to inject routes through the gap-aware path (which
    # trims crossfades instead of injecting). It must still render a valid file
    # of roughly the kept-total duration.
    src = tmp_path / "in.wav"
    out = tmp_path / "out.wav"
    _write_wav(src)

    keep = [(0.0, 1.3), (1.7, 3.0)]
    render(src, keep, out, min_gap_s=0.05)

    assert out.exists()
    assert ffprobe_duration(out) == pytest.approx((1.3) + (3.0 - 1.7), abs=0.1)
