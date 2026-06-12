"""Tests for the CLI layer in erm.cli.

These exercise the pure parsing/dispatch logic only — argument parsing,
subcommand routing, and the small helpers — without invoking faster-whisper,
librosa, or ffmpeg. The heavy pipeline functions (`_cmd_remove`/`_cmd_validate`)
are monkeypatched so we can assert routing without running them.

The one exception is `test_cmd_remove_invalid_room_tone_source_*`, which drives
`_cmd_remove` end-to-end with every heavy dependency (transcribe, audio load,
denoise, render) stubbed, to cover the `--room-tone-source` error path and its
intermediate-file cleanup. It still avoids librosa/ffmpeg/whisper entirely.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from erm import Word, cli


# ---------- _parse_filler_set ----------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("um,uh", {"um", "uh"}),
        ("Um, UH ", {"um", "uh"}),
        ("um, um , uh", {"um", "uh"}),  # dedup + whitespace
        ("  ", set()),
        ("", set()),
        (",,um,,", {"um"}),  # empty fields dropped
    ],
)
def test_parse_filler_set(spec, expected):
    assert cli._parse_filler_set(spec) == expected


# ---------- _parse_room_tone_source ----------------------------------------


def test_parse_room_tone_source_valid():
    assert cli._parse_room_tone_source("0.05-1.4") == pytest.approx((0.05, 1.4))


@pytest.mark.parametrize(
    "bad",
    [
        "auto",          # not numeric
        "1.0",           # only one value
        "1.0-2.0-3.0",   # too many values
        "abc-def",       # non-numeric
        "",              # empty
        "-1.0-2.0",      # leading '-' -> 3 fields; negative offsets unsupported
        "0.5--1.0",      # negative end, same reason
        "2.0-1.0",       # end < start -> backwards segment
        "1.0-1.0",       # end == start -> empty segment
    ],
)
def test_parse_room_tone_source_invalid_raises(bad):
    with pytest.raises(ValueError):
        cli._parse_room_tone_source(bad)


# ---------- _timestamped ----------------------------------------------------


def test_timestamped_shape():
    out = cli._timestamped("/tmp/recording.m4a", "cleaned", "wav")
    # Sibling of the input, stem-suffix-YYYYMMDD-HHMMSS.ext
    assert str(out.parent) == "/tmp"
    assert re.fullmatch(r"recording-cleaned-\d{8}-\d{6}\.wav", out.name)


def test_timestamped_uses_input_stem_not_extension():
    out = cli._timestamped("clip.with.dots.wav", "cuts", "json")
    assert out.name.startswith("clip.with.dots-cuts-")
    assert out.suffix == ".json"


# ---------- remove-parser defaults -----------------------------------------


def test_remove_parser_defaults():
    args = cli._build_remove_parser().parse_args(["in.wav"])
    assert args.input == "in.wav"
    assert args.output is None
    assert args.model == "medium.en"
    assert args.device == "auto"
    assert args.compute_type == "auto"
    assert args.denoise == "hybrid"
    assert args.room_tone is True
    assert args.detect_gaps is True
    assert args.confirm_pitch is True
    assert args.dry_run is False
    assert args.crossfade_ms is None
    assert args.room_tone_source == "auto"


def test_remove_parser_boolean_optional_negation():
    args = cli._build_remove_parser().parse_args(
        ["in.wav", "--no-room-tone", "--no-detect-gaps", "--no-confirm-pitch"]
    )
    assert args.room_tone is False
    assert args.detect_gaps is False
    assert args.confirm_pitch is False


def test_remove_parser_typed_options():
    args = cli._build_remove_parser().parse_args(
        ["in.wav", "-o", "out.wav", "--device", "cpu",
         "--search-ms", "80", "--crossfade-ms", "30", "--dry-run"]
    )
    assert args.output == "out.wav"
    assert args.device == "cpu"
    assert args.search_ms == pytest.approx(80.0)
    assert args.crossfade_ms == pytest.approx(30.0)
    assert args.dry_run is True


def test_remove_parser_rejects_unknown_device():
    with pytest.raises(SystemExit):
        cli._build_remove_parser().parse_args(["in.wav", "--device", "tpu"])


def test_remove_parser_mode_and_spacing_defaults():
    args = cli._build_remove_parser().parse_args(["in.wav"])
    assert args.mode == "remove"
    assert args.pad_pause_factor == pytest.approx(0.0)
    assert args.pad_min_ms == pytest.approx(0.0)
    assert args.pad_max_ms == pytest.approx(120.0)
    assert args.min_gap_ms == pytest.approx(0.0)


def test_remove_parser_mode_and_spacing_set():
    args = cli._build_remove_parser().parse_args(
        ["in.wav", "--mode", "silence", "--pad-pause-factor", "0.5",
         "--pad-min-ms", "20", "--pad-max-ms", "200", "--min-gap-ms", "150"]
    )
    assert args.mode == "silence"
    assert args.pad_pause_factor == pytest.approx(0.5)
    assert args.pad_min_ms == pytest.approx(20.0)
    assert args.pad_max_ms == pytest.approx(200.0)
    assert args.min_gap_ms == pytest.approx(150.0)


def test_remove_parser_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        cli._build_remove_parser().parse_args(["in.wav", "--mode", "mute"])


# ---------- validate-parser ------------------------------------------------


def test_validate_parser_defaults():
    args = cli._build_validate_parser().parse_args(["in.wav", "out.wav"])
    assert args.input == "in.wav"
    assert args.output == "out.wav"
    assert args.cuts is None
    assert args.model == "medium.en"
    assert args.device == "auto"
    assert args.report is None


# ---------- main() subcommand routing --------------------------------------


@pytest.fixture
def captured_dispatch(monkeypatch):
    """Replace the two command handlers with recorders that capture args."""
    calls: dict[str, object] = {}

    def _fake_remove(args):
        calls["remove"] = args
        return 0

    def _fake_validate(args):
        calls["validate"] = args
        return 0

    monkeypatch.setattr(cli, "_cmd_remove", _fake_remove)
    monkeypatch.setattr(cli, "_cmd_validate", _fake_validate)
    return calls


def test_main_routes_bare_input_to_remove(captured_dispatch):
    assert cli.main(["song.wav"]) == 0
    assert "remove" in captured_dispatch
    assert "validate" not in captured_dispatch
    assert captured_dispatch["remove"].input == "song.wav"


def test_main_routes_explicit_remove_subcommand(captured_dispatch):
    assert cli.main(["remove", "song.wav"]) == 0
    assert "remove" in captured_dispatch
    # The "remove" token is stripped before parsing, not treated as input.
    assert captured_dispatch["remove"].input == "song.wav"


def test_main_routes_validate_subcommand(captured_dispatch):
    assert cli.main(["validate", "src.wav", "out.wav"]) == 0
    assert "validate" in captured_dispatch
    assert "remove" not in captured_dispatch
    assert captured_dispatch["validate"].input == "src.wav"
    assert captured_dispatch["validate"].output == "out.wav"


def test_main_remove_input_named_remove_is_disambiguated(captured_dispatch):
    # `remove` and `validate` are reserved as leading subcommand tokens: the
    # first one is always stripped before parsing. So to operate on a file
    # literally named "remove", the explicit subcommand must precede it.
    # This is the intended dispatch contract, not an accident of parsing.
    assert cli.main(["remove", "remove"]) == 0
    assert captured_dispatch["remove"].input == "remove"


def test_main_bare_reserved_word_has_no_input(captured_dispatch):
    # The flip side of the contract: `erm remove` with nothing after it is a
    # bare subcommand, NOT a request to process a file named "remove". The
    # stripped argv is empty, so the required `input` positional is missing
    # and argparse exits 2 — _cmd_remove is never reached.
    with pytest.raises(SystemExit):
        cli.main(["remove"])
    assert "remove" not in captured_dispatch


# ---------- _cmd_remove room-tone-source error path ------------------------


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Stub every heavy stage of `_cmd_remove` and record real intermediates.

    `transcribe` yields a single "um" so a cut survives to the render/room-tone
    stage; `denoise_to` and `render` write real files at their target paths so
    the error-path cleanup (`unlink`) is actually observable on disk.
    """
    created: dict[str, Path] = {}

    def _fake_transcribe(path, **kwargs):
        return [Word(text="um", start=0.40, end=0.70)], 1.0

    def _fake_load_audio_mono(path):
        return np.zeros(16_000, dtype=np.float32), 16_000

    def _fake_denoise_to(src, dst, **kwargs):
        Path(dst).write_bytes(b"denoised")
        created["denoised"] = Path(dst)

    def _fake_render(src, keep_ranges, dst, **kwargs):
        Path(dst).write_bytes(b"raw")
        created["render_target"] = Path(dst)

    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)
    monkeypatch.setattr(cli, "load_audio_mono", _fake_load_audio_mono)
    monkeypatch.setattr(cli, "denoise_to", _fake_denoise_to)
    monkeypatch.setattr(cli, "render", _fake_render)
    return created


def test_cmd_remove_invalid_room_tone_source_returns_2_and_cleans_up(
    tmp_path, stubbed_pipeline, capsys
):
    in_wav = tmp_path / "in.wav"
    in_wav.write_bytes(b"")  # only the path matters; transcribe is stubbed
    out_wav = tmp_path / "out.wav"

    rc = cli.main([
        str(in_wav), "-o", str(out_wav),
        "--denoise", "hybrid",       # creates a denoised intermediate to clean up
        "--no-detect-gaps",          # skip acoustic detectors
        "--room-tone-source", "not-a-range",  # -> ValueError -> exit 2
    ])

    assert rc == 2
    err = capsys.readouterr().err
    assert "invalid --room-tone-source" in err

    # Both intermediates were created, then removed on the error path; the
    # real output was never written.
    assert stubbed_pipeline["denoised"].exists() is False
    assert stubbed_pipeline["render_target"].exists() is False
    assert not out_wav.exists()
