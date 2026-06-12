"""Tests for the CLI layer in erm.cli.

These exercise the pure parsing/dispatch logic only — argument parsing,
subcommand routing, and the small helpers — without invoking faster-whisper,
librosa, or ffmpeg. The heavy pipeline functions (`_cmd_remove`/`_cmd_validate`)
are monkeypatched so we can assert routing without running them.
"""

from __future__ import annotations

import re

import pytest

from erm import cli


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
    # A first token of "remove" is consumed as the subcommand; the actual
    # filename follows. (Documents the current routing contract.)
    assert cli.main(["remove", "remove"]) == 0
    assert captured_dispatch["remove"].input == "remove"
