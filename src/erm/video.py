"""Video probing and (later) the video render + mux pipeline.

erm's edit timeline (keep ranges in seconds) is format-agnostic. This module
renders the *video* stream from that same timeline and muxes it with the
separately-rendered clean-PCM audio master, keeping A/V in sync by construction
(see `docs/render-pipeline.md`). Everything here shells out to the `ffmpeg` /
`ffprobe` CLIs already required by `ffmpeg_ops.py`; there is no Python video
dependency.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    """First video stream's properties, as probed by `probe_video`.

    `has_video` is False when the input has no video stream *or* only a still
    cover image (`attached_pic`, e.g. an mp3 thumbnail) — neither is motion
    video we should render. `fps` is the constant-frame-rate target we force the
    render to (from `avg_frame_rate`, which is VFR-safe; `r_frame_rate` can be a
    wildly high LCM on variable-rate inputs).
    """

    has_video: bool
    codec: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None
    sar: str | None = None


def _parse_rate(value: str) -> float | None:
    """Parse an ffprobe rational rate (``"30000/1001"``) into fps, or None."""
    value = value.strip()
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return float(value)
    except ValueError:
        return None


def probe_video(path: str | Path) -> VideoInfo:
    """Probe the first video stream of `path`.

    Mirrors `ffmpeg_ops._probe_audio_stream`'s parse style. A still cover image
    (`disposition.attached_pic=1`) is reported as `has_video=False` so an mp3's
    album art is never treated as motion video.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
         "pix_fmt,sample_aspect_ratio:stream_disposition=attached_pic",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout

    fields: dict[str, str] = {}
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()

    # No video stream at all → ffprobe printed nothing.
    if "codec_name" not in fields:
        return VideoInfo(has_video=False)

    # Cover art is a video stream but not motion video.
    if fields.get("DISPOSITION:attached_pic", "0") == "1":
        return VideoInfo(has_video=False)

    fps = _parse_rate(fields.get("avg_frame_rate", ""))
    if fps is None:
        fps = _parse_rate(fields.get("r_frame_rate", ""))

    def _int(name: str) -> int | None:
        try:
            return int(fields[name])
        except (KeyError, ValueError):
            return None

    return VideoInfo(
        has_video=True,
        codec=fields.get("codec_name"),
        fps=fps,
        width=_int("width"),
        height=_int("height"),
        pix_fmt=fields.get("pix_fmt"),
        sar=fields.get("sample_aspect_ratio"),
    )
