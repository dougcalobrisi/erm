"""Golden-path integration test for the `erm` dry-run pipeline.

This wires the CLI end-to-end through transcription -> filler detection ->
boundary refinement -> range inversion -> JSON output, using a tiny
synthesized WAV and a stubbed transcriber. No model download, no ffmpeg
render (``--dry-run``), and no room tone / denoise stages.

It is gated on librosa, which the audio loader uses to read the WAV. The
faster `test_pure`/`test_cli` suites stay importable without it.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa", reason="integration test needs the real audio loader")

from erm import Word, cli  # noqa: E402


SAMPLE_RATE = 16_000

# A short utterance: "hello <um> world" with the filler in the middle.
SCRIPT_WORDS = [
    Word(text="hello", start=0.20, end=0.60),
    Word(text="um", start=0.85, end=1.15),
    Word(text="world", start=1.45, end=1.95),
]
CLIP_DURATION_S = 2.30


def _write_wav(path: Path) -> None:
    """Synthesize a mono 16-bit WAV: a quiet hiss with voiced tones on words."""
    # Seeded so the noise floor is identical every run (no flakiness). The
    # 150 Hz voiced tone is what the *acoustic* detectors key off — but the
    # primary cut in this test comes from the stubbed transcriber's "um", not
    # the audio. `test_dry_run_without_acoustic_detectors_still_cuts_...`
    # pins that down: even with `--no-detect-gaps` the filler is still cut, so
    # this synthesized signal can't silently stop exercising the cut path if
    # detector thresholds are later retuned.
    samples = np.random.default_rng(0).normal(0.0, 0.003,
                                               int(CLIP_DURATION_S * SAMPLE_RATE))
    t = np.arange(samples.size) / SAMPLE_RATE
    for word in SCRIPT_WORDS:
        mask = (t >= word.start) & (t < word.end)
        samples[mask] += 0.25 * np.sin(2 * np.pi * 150.0 * t[mask])
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())


def _run_dry_run(tmp_path, monkeypatch, extra_args=()):
    wav_path = tmp_path / "clip.wav"
    json_path = tmp_path / "cuts.json"
    _write_wav(wav_path)

    def _fake_transcribe(path, **kwargs):
        return list(SCRIPT_WORDS), CLIP_DURATION_S

    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    rc = cli.main([
        str(wav_path),
        "--dry-run",
        "--no-room-tone",
        "--denoise", "none",
        "--json", str(json_path),
        *extra_args,
    ])
    payload = json.loads(json_path.read_text())
    return rc, payload


def test_dry_run_finds_the_filler_and_writes_json(tmp_path, monkeypatch):
    rc, payload = _run_dry_run(tmp_path, monkeypatch)

    assert rc == 0
    assert payload["input"].endswith("clip.wav")
    assert payload["duration_s"] == pytest.approx(CLIP_DURATION_S)

    # At least one cut, and one of them overlaps the "um" at 0.85-1.15s.
    cuts = payload["cuts"]
    assert cuts, "expected at least one cut for the spoken filler"
    um_start, um_end = 0.85, 1.15
    assert any(c["start"] < um_end and c["end"] > um_start for c in cuts), \
        f"no cut overlaps the um region; cuts={cuts}"

    # Time was saved and some audio survives.
    assert payload["time_saved_s"] > 0.0
    assert payload["keep_ranges"], "expected surviving keep ranges"


def test_dry_run_keep_ranges_cover_the_real_words(tmp_path, monkeypatch):
    _, payload = _run_dry_run(tmp_path, monkeypatch)
    keep = payload["keep_ranges"]

    def _kept(t: float) -> bool:
        return any(r["start"] <= t <= r["end"] for r in keep)

    # The two real words survive; the filler's center is removed.
    assert _kept(0.40), "hello should be kept"
    assert _kept(1.70), "world should be kept"
    assert not _kept(1.00), "the um (center 1.0s) should be cut out"


def test_dry_run_without_acoustic_detectors_still_cuts_transcribed_filler(
    tmp_path, monkeypatch
):
    # Isolate the transcribed-filler path: no gap/intra/overlong acoustic stage.
    rc, payload = _run_dry_run(
        tmp_path, monkeypatch, extra_args=("--no-detect-gaps",)
    )
    assert rc == 0
    assert payload["cuts"], "the transcribed 'um' alone should yield a cut"
