"""Real-ffmpeg tests for first-class video I/O.

These synthesize tiny clips with ffmpeg (`testsrc` + `sine`) and drive
`_cmd_remove` end to end, stubbing only the heavy Whisper transcribe so the
real ffmpeg render/mux paths run. They assert per-stream A/V duration parity
(within ~1 frame) across every splice path, the audio-only default, and format
inference. Skipped automatically when ffmpeg/ffprobe aren't on PATH.

The synthetic video uses the always-present `mpeg4` encoder (libx264 may be
absent in minimal ffmpeg builds), so duration-sync assertions never depend on a
particular codec.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from erm import cli
from erm.ffmpeg_ops import run_ffmpeg
from erm.models import Word
from erm.video import VideoInfo, audio_mux_args, mux_av, probe_video


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="video tests need ffmpeg + ffprobe on PATH",
)

FPS = 25
SR = 44_100
DURATION_S = 6.0
ONE_FRAME_MS = 1000.0 / FPS


def _make_av(path: Path, *, duration: float = DURATION_S, fps: int = FPS,
             with_audio: bool = True, vcodec: str = "mpeg4",
             audio_duration: float | None = None) -> None:
    """Synthesize a CFR `testsrc` clip (optionally + a sine tone) at `path`.

    `audio_duration` defaults to the video duration; pass a different value to
    mint a clip whose audio and video tracks aren't exactly equal-length (what a
    real recording often looks like), exercising the mux's `-shortest` clamp.
    """
    a_dur = duration if audio_duration is None else audio_duration
    cmd = ["ffmpeg", "-y", "-f", "lavfi",
           "-i", f"testsrc=duration={duration}:size=320x240:rate={fps}"]
    if with_audio:
        cmd += ["-f", "lavfi",
                "-i", f"sine=frequency=330:duration={a_dur}:sample_rate={SR}",
                "-map", "0:v", "-map", "1:a", "-c:a", "pcm_s16le"]
    else:
        cmd += ["-map", "0:v"]
    cmd += ["-c:v", vcodec, "-pix_fmt", "yuv420p", "-r", str(fps), str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _make_coverart_mp3(path: Path, duration: float = 2.0) -> None:
    """A sine mp3 with an attached cover image (a still, not motion video)."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1",
         "-map", "0:a", "-map", "1:v", "-disposition:v", "attached_pic",
         "-c:a", "libmp3lame", "-c:v", "mjpeg", str(path)],
        check=True, capture_output=True,
    )


def _stream_duration(path: Path, stream: str) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", "stream=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def _has_stream(path: Path, kind: str) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", kind,
         "-show_entries", "stream=index",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return bool(out)


# Words spanning the 6 s clip with several "um" fillers → multiple keeps/splices.
_WORDS = [
    Word("hello", 0.2, 0.6), Word("um", 1.0, 1.3), Word("world", 1.6, 2.0),
    Word("um", 2.6, 2.9), Word("this", 3.2, 3.6), Word("um", 4.0, 4.3),
    Word("is", 4.6, 4.9), Word("um", 5.2, 5.5), Word("test", 5.7, 5.95),
]
# Tightly-spaced words so removing "um" leaves sub-floor gaps → min-gap injects.
_TIGHT_WORDS = [
    Word("a", 0.2, 0.5), Word("um", 0.55, 0.85), Word("b", 0.9, 1.3),
    Word("c", 2.0, 2.4), Word("um", 2.45, 2.75), Word("d", 2.8, 3.2),
    Word("e", 4.0, 4.4), Word("um", 4.45, 4.75), Word("f", 4.8, 5.2),
]


def _run(monkeypatch, argv: list[str], words=_WORDS, duration: float = DURATION_S):
    monkeypatch.setattr(cli, "transcribe", lambda p, **k: (list(words), duration))
    return cli._cmd_remove(cli._build_remove_parser().parse_args(argv))


def _assert_av_synced(path: Path) -> None:
    assert _has_stream(path, "v"), "output lost its video stream"
    assert _has_stream(path, "a"), "output lost its audio stream"
    v = _stream_duration(path, "v:0")
    a = _stream_duration(path, "a:0")
    assert v is not None and a is not None
    # Within ~1 frame plus a small epsilon (audio is sample-exact, video frame-
    # quantized then conformed to the audio master).
    assert abs(v - a) * 1000.0 <= ONE_FRAME_MS + 5.0, (
        f"A/V drift {abs(v - a) * 1000:.1f}ms exceeds ~1 frame "
        f"(v={v:.3f} a={a:.3f})"
    )


# ----- probing -------------------------------------------------------------

def test_probe_video_detects_motion(tmp_path):
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    info = probe_video(clip)
    assert info.has_video and info.fps == pytest.approx(FPS, abs=0.01)
    assert info.width == 320 and info.height == 240


def test_probe_video_ignores_cover_art(tmp_path):
    if shutil.which("ffmpeg") and b"libmp3lame" not in subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True).stdout:
        pytest.skip("libmp3lame not available for cover-art fixture")
    song = tmp_path / "song.mp3"
    _make_coverart_mp3(song)
    assert probe_video(song).has_video is False


def test_probe_video_audio_only(tmp_path):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", str(wav)],
                   check=True, capture_output=True)
    assert probe_video(wav).has_video is False


# ----- audio-only default (backward compatibility) -------------------------

def test_video_input_without_flag_yields_audio_only_wav(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    out = tmp_path / "out.wav"
    rc = _run(monkeypatch, [str(clip), "-o", str(out), "--no-detect-gaps",
                            "--no-room-tone"])
    assert rc == 0
    assert _has_stream(out, "a") and not _has_stream(out, "v")


def test_default_output_extension_is_wav_without_video(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    monkeypatch.setattr(cli, "transcribe", lambda p, **k: (list(_WORDS), DURATION_S))
    args = cli._build_remove_parser().parse_args(
        [str(clip), "--no-detect-gaps", "--no-room-tone"])
    cli._cmd_remove(args)
    assert args.output.endswith(".wav")


# ----- silence mode --------------------------------------------------------

def test_silence_video_preserves_duration_and_keeps_streams(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    out = tmp_path / "out.mov"
    rc = _run(monkeypatch, [str(clip), "--video", "--mode", "silence",
                            "-o", str(out), "--no-detect-gaps", "--no-room-tone"])
    assert rc == 0
    _assert_av_synced(out)
    # Silence preserves the full timeline.
    assert _stream_duration(out, "v:0") == pytest.approx(DURATION_S, abs=0.06)


# ----- remove mode: every splice path --------------------------------------

@pytest.mark.parametrize("splice", ["crossfade", "cut"])
def test_remove_video_av_parity(tmp_path, monkeypatch, splice):
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    out = tmp_path / f"out_{splice}.mov"
    rc = _run(monkeypatch, [str(clip), "--video", "--video-splice", splice,
                            "--vcodec", "mpeg4", "-o", str(out),
                            "--no-detect-gaps", "--no-room-tone"])
    assert rc == 0
    _assert_av_synced(out)
    # Something was actually removed (output shorter than the source).
    assert _stream_duration(out, "v:0") < DURATION_S - 0.5


def test_remove_video_single_keep(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    out = tmp_path / "out_single.mov"
    # One trailing "um" → a single kept fragment (single-trim path).
    words = [Word("hello", 0.2, 1.0), Word("world", 1.2, 2.0),
             Word("um", 5.4, 5.9)]
    rc = _run(monkeypatch, [str(clip), "--video", "--vcodec", "mpeg4",
                            "-o", str(out), "--no-detect-gaps", "--no-room-tone"],
              words=words)
    assert rc == 0
    _assert_av_synced(out)


def test_remove_video_min_gap_plays_through(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    out = tmp_path / "out_mingap.mov"
    rc = _run(monkeypatch, [str(clip), "--video", "--min-gap-ms", "300",
                            "--vcodec", "mpeg4", "-o", str(out),
                            "--no-detect-gaps", "--no-room-tone"],
              words=_TIGHT_WORDS)
    assert rc == 0
    _assert_av_synced(out)


def test_remove_video_min_gap_exceeding_removed_span(tmp_path, monkeypatch):
    # An aggressive --min-gap-ms can request a longer injected pause than the
    # removed span actually holds. The played-through video then clamps its read
    # to the removed footage and clone-pads the rest, keeping each gap node
    # exactly the injected length — so A/V parity must still hold (the picture
    # never spills into the next kept fragment and never drifts).
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    out = tmp_path / "out_biggap.mov"
    rc = _run(monkeypatch, [str(clip), "--video", "--min-gap-ms", "900",
                            "--vcodec", "mpeg4", "-o", str(out),
                            "--no-detect-gaps", "--no-room-tone"],
              words=_TIGHT_WORDS)
    assert rc == 0
    _assert_av_synced(out)


# ----- format inference & guards -------------------------------------------

def test_output_extension_inferred_from_input(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    _make_av(clip)
    monkeypatch.setattr(cli, "transcribe", lambda p, **k: (list(_WORDS), DURATION_S))
    args = cli._build_remove_parser().parse_args(
        [str(clip), "--video", "--vcodec", "mpeg4",
         "--no-detect-gaps", "--no-room-tone"])
    cli._cmd_remove(args)
    assert args.output.endswith(".mp4")
    assert _has_stream(Path(args.output), "v")
    # mp4 re-encodes the PCM master to AAC (the one render path whose audio is
    # not stream-copied). AAC carries encoder-priming delay, so assert real A/V
    # parity here — not just stream presence — to catch priming drift that the
    # mov/copy paths can't surface.
    _assert_av_synced(Path(args.output))


def test_video_container_output_without_video_flag_errors(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    rc = _run(monkeypatch, [str(clip), "-o", str(tmp_path / "out.mp4"),
                            "--no-detect-gaps", "--no-room-tone"])
    assert rc == 2  # video container but no --video


# ----- codec selection (pure) ----------------------------------------------

def test_audio_mux_args_by_container():
    assert audio_mux_args(".mp4") == ["-c:a", "aac", "-b:a", "256k"]
    assert audio_mux_args(".webm")[:2] == ["-c:a", "libopus"]
    assert audio_mux_args(".mov") == ["-c:a", "copy"]
    assert audio_mux_args(".mkv") == ["-c:a", "copy"]


# ----- error surfacing -----------------------------------------------------

def test_run_ffmpeg_surfaces_stderr_on_failure(tmp_path):
    # A bogus input makes ffmpeg exit non-zero; run_ffmpeg must raise with the
    # diagnostic stderr tail rather than a bare CalledProcessError.
    missing = tmp_path / "does-not-exist.mp4"
    with pytest.raises(RuntimeError) as excinfo:
        run_ffmpeg(["ffmpeg", "-y", "-i", str(missing), str(tmp_path / "o.wav")])
    msg = str(excinfo.value)
    assert "ffmpeg failed" in msg
    # The ffmpeg stderr tail (mentioning the missing file) is carried through.
    assert "does-not-exist" in msg


# ----- silence-mode parity on unequal-length tracks ------------------------

def test_silence_video_parity_with_mismatched_tracks(tmp_path, monkeypatch):
    # A real recording's audio and video tracks rarely end at the exact same
    # instant. Silence mode stream-copies the picture (source video-track
    # length) and muxes our audio master onto it; the mux's `-shortest` must
    # clamp the native mismatch so A/V parity (~1 frame) still holds.
    clip = tmp_path / "clip.mov"
    _make_av(clip, duration=DURATION_S, audio_duration=DURATION_S + 0.3)
    out = tmp_path / "out.mov"
    rc = _run(monkeypatch, [str(clip), "--video", "--mode", "silence",
                            "-o", str(out), "--no-detect-gaps", "--no-room-tone"])
    assert rc == 0
    _assert_av_synced(out)


# ----- mux_av re-encode branch ---------------------------------------------

def test_mux_av_reencodes_video_when_vcodec_not_copy(tmp_path):
    # The remove path muxes with -c:v copy, but mux_av also supports re-encoding
    # the picture (general/reusable path). Exercise it directly with crf/preset.
    clip = tmp_path / "src.mov"
    _make_av(clip)
    audio = tmp_path / "master.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={DURATION_S}", str(audio)],
                   check=True, capture_output=True)
    out = tmp_path / "muxed.mov"
    mux_av(clip, audio, out, vcodec="mpeg4", crf=20.0, preset="ultrafast")
    assert _has_stream(out, "v") and _has_stream(out, "a")
    _assert_av_synced(out)


# ----- frame-rate guard ----------------------------------------------------

def test_remove_video_errors_when_fps_undeterminable(tmp_path, monkeypatch):
    # If the input reports a video stream but no usable frame rate, the render
    # can't build its CFR grid — fail cleanly (rc 1) instead of crashing deep in
    # ffmpeg, and don't leave the audio-master temp behind.
    clip = tmp_path / "clip.mov"
    _make_av(clip)
    monkeypatch.setattr(
        cli, "probe_video",
        lambda _p: VideoInfo(has_video=True, fps=None, width=320, height=240),
    )
    out = tmp_path / "out.mov"
    rc = _run(monkeypatch, [str(clip), "--video", "-o", str(out),
                            "--no-detect-gaps", "--no-room-tone"])
    assert rc == 1
    assert not out.exists()
    # No stray *-audiomaster-*.wav / *-analysis-*.wav temps left in the dir.
    leftovers = list(tmp_path.glob("*-audiomaster-*.wav")) \
        + list(tmp_path.glob("*-analysis-*.wav"))
    assert leftovers == [], f"temp files leaked: {leftovers}"


def test_render_failure_does_not_leak_temps(tmp_path, monkeypatch):
    # If an ffmpeg op raises mid-pipeline (here the audio render), the audio
    # master and the extracted analysis WAV must be cleaned on the way out, not
    # leaked. The exception still propagates (the CLI surfaces it upstream).
    clip = tmp_path / "clip.mov"
    _make_av(clip)

    def _boom(*a, **k):
        raise RuntimeError("ffmpeg failed (simulated render error)")

    monkeypatch.setattr(cli, "render", _boom)
    out = tmp_path / "out.mov"
    with pytest.raises(RuntimeError, match="simulated render error"):
        _run(monkeypatch, [str(clip), "--video", "--vcodec", "mpeg4",
                           "-o", str(out), "--no-detect-gaps", "--no-room-tone"])
    leftovers = list(tmp_path.glob("*-audiomaster-*.wav")) \
        + list(tmp_path.glob("*-analysis-*.wav"))
    assert leftovers == [], f"temp files leaked on render failure: {leftovers}"
