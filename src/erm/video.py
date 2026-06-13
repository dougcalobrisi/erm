"""Video probing and (later) the video render + mux pipeline.

erm's edit timeline (keep ranges in seconds) is format-agnostic. This module
renders the *video* stream from that same timeline and muxes it with the
separately-rendered clean-PCM audio master, keeping A/V in sync by construction
(see `docs/render-pipeline.md`). Everything here shells out to the `ffmpeg` /
`ffprobe` CLIs already required by `ffmpeg_ops.py`; there is no Python video
dependency.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg_ops import run_ffmpeg


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
    out = run_ffmpeg(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
         "pix_fmt,sample_aspect_ratio:stream_disposition=attached_pic",
         "-of", "default=noprint_wrappers=1", str(path)],
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


def render_video_keep_ranges(
    input_path: str | Path,
    keep_ranges: list[tuple[float, float]],
    fades: list[float],
    fr: float,
    output_path: str | Path,
    *,
    splice_style: str = "crossfade",
    vcodec: str = "libx264",
    crf: float = 18.0,
    preset: str = "medium",
    target_duration: float | None = None,
) -> None:
    """Render the picture for `keep_ranges`, mirroring the audio splice graph.

    The stream is first forced to constant frame rate (`fps={fr}`) so every
    downstream `trim`/`xfade` lands on a uniform frame grid — without this,
    variable-frame-rate input (phones, screen recorders) breaks the duration
    math. Then, mirroring `ffmpeg_ops.render`:

    - 1 keep → a single `trim` re-encode.
    - `crossfade` with all fades > 0 → per-fragment `trim`/`setpts`, chained
      `xfade=transition=fade:duration=dᵢ:offset=Oᵢ`. `dᵢ` is the frame-snapped
      fade shared with the audio `acrossfade`; `Oᵢ` is computed from the true
      float cumulative length (``Σprev_keeps − Σprev_fades``) so offset error
      never accumulates.
    - `cut`, or any zero fade → `concat` of all fragments (no overlap). This is
      the same all-or-nothing choice the audio path makes, so audio and video
      always pick the same structure and end at the same duration.

    `target_duration` conforms the final picture to an exact length (the audio
    master's sample-exact duration): the video is clone-padded if short and
    trimmed if long, so the two streams end frame-for-frame together. This
    absorbs the frame-quantized cut points of the `cut`/`concat` path and any
    residual from the `xfade` path.

    Audio is dropped (`-an`); it is muxed back from the clean PCM master by
    `mux_av`.
    """
    keep_ranges = list(keep_ranges)
    n = len(keep_ranges)
    if n == 0:
        raise ValueError("keep_ranges is empty — video would have no frames")

    def _conform(label_in: str) -> str:
        """Filter snippet forcing the stream to exactly `target_duration`."""
        if target_duration is None:
            return ""
        # Clone-pad by the full target (always ≥ any deficit, since the raw
        # stream is ≥ 0), then hard-trim to the target — exact length whether the
        # splice came out short or long.
        return (f"[{label_in}]tpad=stop_mode=clone:stop_duration={target_duration:.6f},"
                f"trim=end={target_duration:.6f},setpts=PTS-STARTPTS[outv]")

    tail = ["-c:v", vcodec, "-crf", f"{crf:g}", "-preset", preset,
            "-pix_fmt", "yuv420p", "-an", str(output_path)]

    if n == 1:
        s, e = keep_ranges[0]
        vf = f"fps={fr:.6f},trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS"
        if target_duration is not None:
            vf += (f",tpad=stop_mode=clone:stop_duration={target_duration:.6f},"
                   f"trim=end={target_duration:.6f},setpts=PTS-STARTPTS")
        cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vf", vf, *tail]
        run_ffmpeg(cmd)
        return

    parts: list[str] = [
        f"[0:v]fps={fr:.6f},format=yuv420p,split={n}"
        + "".join(f"[c{i}]" for i in range(n))
    ]
    for i, (s, e) in enumerate(keep_ranges):
        parts.append(
            f"[c{i}]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[k{i}]"
        )

    # Final spliced label is "outv" unless a conform pass renames it.
    spliced = "vraw" if target_duration is not None else "outv"
    if splice_style == "cut" or not all(d > 0 for d in fades):
        inputs = "".join(f"[k{i}]" for i in range(n))
        parts.append(f"{inputs}concat=n={n}:v=1:a=0[{spliced}]")
    else:
        prev = "k0"
        # True float cumulative length of the accumulated stream so far.
        cumulative = keep_ranges[0][1] - keep_ranges[0][0]
        for i in range(1, n):
            d = fades[i - 1]
            out = spliced if i == n - 1 else f"x{i}"
            offset = cumulative - d
            parts.append(
                f"[{prev}][k{i}]xfade=transition=fade:"
                f"duration={d:.6f}:offset={offset:.6f}[{out}]"
            )
            cumulative += (keep_ranges[i][1] - keep_ranges[i][0]) - d
            prev = out

    if target_duration is not None:
        parts.append(_conform(spliced))

    cmd = ["ffmpeg", "-y", "-i", str(input_path),
           "-filter_complex", ";".join(parts), "-map", "[outv]", *tail]
    run_ffmpeg(cmd)


def render_video_with_gaps(
    input_path: str | Path,
    keep_ranges: list[tuple[float, float]],
    gap_inserts: list[tuple[int, float]],
    fades: list[float],
    fr: float,
    output_path: str | Path,
    *,
    splice_style: str = "crossfade",
    vcodec: str = "libx264",
    crf: float = 18.0,
    preset: str = "medium",
    target_duration: float | None = None,
) -> None:
    """Min-gap video: mirror `ffmpeg_ops._render_with_gaps` node-for-node.

    The audio path injects `anullsrc` silence at tight splices to honor the gap
    floor; the picture **plays through** instead of freezing — each injected gap
    is filled with the *real removed footage* at that splice (a `trim` of the
    original starting where the kept fragment ended), muted, for the same
    frame-snapped gap duration. So the disfluency we cut still rolls under the
    injected pause and the motion never stalls.

    The graph folds left exactly like the audio: keep→keep joins `xfade`
    (crossfade, fade > 0) or `concat`; any join touching a gap uses `concat`, so
    the injected duration lands as-is. `target_duration` conforms the result to
    the audio master's length.
    """
    keep_ranges = list(keep_ranges)
    n_keep = len(keep_ranges)

    gaps_after: dict[int, list[float]] = defaultdict(list)
    for after_keep_index, duration in gap_inserts:
        gaps_after[after_keep_index].append(float(duration))
    n_gap = sum(len(v) for v in gaps_after.values())
    n_split = n_keep + n_gap

    parts: list[str] = [
        f"[0:v]fps={fr:.6f},format=yuv420p,split={n_split}"
        + "".join(f"[src{i}]" for i in range(n_split))
    ]

    # Keep nodes: trim each kept fragment from a CFR copy of the source.
    src = 0
    lengths: dict[str, float] = {}
    for i, (s, e) in enumerate(keep_ranges):
        parts.append(f"[src{src}]trim=start={s:.6f}:end={e:.6f},"
                     f"setpts=PTS-STARTPTS[k{i}]")
        lengths[f"k{i}"] = e - s
        src += 1

    # Gap nodes: the removed footage right after keep i, played through muted.
    gap_label: dict[tuple[int, int], str] = {}
    for i in range(n_keep):
        for j, dur in enumerate(gaps_after.get(i, [])):
            # Stack consecutive gaps after the same keep along the removed span.
            prior = sum(gaps_after[i][:j])
            gstart = keep_ranges[i][1] + prior
            label = f"g{i}_{j}"
            parts.append(f"[src{src}]trim=start={gstart:.6f}:duration={dur:.6f},"
                         f"setpts=PTS-STARTPTS[{label}]")
            lengths[label] = dur
            gap_label[(i, j)] = label
            src += 1

    # Node sequence: each keep, then any gaps spliced after it (audio's order).
    nodes: list[tuple[str, str, int | None]] = []
    for i in range(n_keep):
        nodes.append(("keep", f"k{i}", i))
        for j in range(len(gaps_after.get(i, []))):
            nodes.append(("gap", gap_label[(i, j)], None))

    final = "vraw" if target_duration is not None else "outv"
    if len(nodes) == 1:
        # Degenerate (single keep, no gaps) — just pass it through.
        parts.append(f"[{nodes[0][1]}]setpts=PTS-STARTPTS[{final}]")
    else:
        prev = nodes[0][1]
        cumulative = lengths[prev]  # true float length of the fold so far
        for idx in range(1, len(nodes)):
            kind, label, keep_index = nodes[idx]
            prev_kind = nodes[idx - 1][0]
            out = final if idx == len(nodes) - 1 else f"m{idx}"
            do_xfade = (
                prev_kind == "keep" and kind == "keep"
                and splice_style != "cut" and fades[keep_index - 1] > 0
            )
            if do_xfade:
                d = fades[keep_index - 1]
                offset = cumulative - d
                parts.append(
                    f"[{prev}][{label}]xfade=transition=fade:"
                    f"duration={d:.6f}:offset={offset:.6f}[{out}]"
                )
                cumulative += lengths[label] - d
            else:
                parts.append(f"[{prev}][{label}]concat=n=2:v=1:a=0[{out}]")
                cumulative += lengths[label]
            prev = out

    if target_duration is not None:
        parts.append(
            f"[vraw]tpad=stop_mode=clone:stop_duration={target_duration:.6f},"
            f"trim=end={target_duration:.6f},setpts=PTS-STARTPTS[outv]"
        )

    cmd = ["ffmpeg", "-y", "-i", str(input_path),
           "-filter_complex", ";".join(parts), "-map", "[outv]",
           "-c:v", vcodec, "-crf", f"{crf:g}", "-preset", preset,
           "-pix_fmt", "yuv420p", "-an", str(output_path)]
    run_ffmpeg(cmd)


def stream_duration(path: str | Path, stream: str = "v:0") -> float | None:
    """Duration (s) of a single stream (e.g. ``"v:0"``/``"a:0"``), or None.

    Reads the *stream* duration, not the container's `format=duration`: silence
    mode conforms the audio master to the picture's own length, and the validate
    A/V-parity check compares the two streams' individual durations. Returns None
    when ffprobe reports no parseable duration (e.g. ``N/A``).
    """
    out = run_ffmpeg(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", "stream=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def video_stream_duration(path: str | Path) -> float | None:
    """Duration (s) of the first video stream (``v:0``), or None if unreadable."""
    return stream_duration(path, "v:0")


def conform_audio_to_duration(audio_path: str | Path, output_path: str | Path,
                              target_s: float) -> None:
    """Pad/trim `audio_path` to exactly `target_s` seconds, written as PCM.

    ``apad`` appends silence and ``atrim`` caps the length, so the result is
    exactly `target_s` whether the input was shorter or longer than the target.

    This is silence mode's A/V-parity mechanism: the picture is stream-copied at
    the *source's* video-track duration (untouched, lossless), so the audio
    master is conformed to that exact length and the two streams end
    frame-for-frame. (Remove mode does the inverse — it conforms the *picture*
    to the audio master — so it never needs this.)
    """
    cmd = ["ffmpeg", "-y", "-i", str(audio_path),
           "-af", f"apad,atrim=end={target_s:.6f}",
           "-c:a", "pcm_s16le", str(output_path)]
    run_ffmpeg(cmd)


def audio_mux_args(output_ext: str) -> list[str]:
    """ffmpeg `-c:a …` args for muxing the clean PCM master into `output_ext`.

    The audio pipeline produces a clean `pcm_s16le` master; this picks how to
    store it per container, preferring **no re-encode** where the container holds
    PCM natively (mov/mkv/avi → ``-c:a copy``). mp4 has no universally-supported
    lossless audio, so it gets transparent-for-speech AAC 256k; webm is
    Opus-only.
    """
    ext = output_ext.lower().lstrip(".")
    if ext in ("mp4", "m4v"):
        return ["-c:a", "aac", "-b:a", "256k"]
    if ext == "webm":
        return ["-c:a", "libopus", "-b:a", "160k"]
    # mov / mkv / avi hold PCM natively — copy the master losslessly.
    return ["-c:a", "copy"]


def mux_av(video_path: str | Path, audio_path: str | Path,
           output_path: str | Path, *, vcodec: str = "copy",
           crf: float | None = None, preset: str | None = None) -> None:
    """Mux one video stream + the audio master into `output_path`.

    Takes `v:0` from `video_path` and `a:0` from `audio_path` (the clean PCM
    master). `vcodec="copy"` stream-copies the picture (silence mode — frame
    accurate, zero quality loss); pass a real encoder (e.g. ``libx264``) with
    `crf`/`preset` when the video was re-encoded upstream. Audio codec is chosen
    by the output container (see `audio_mux_args`).

    `-shortest` ends the output when the first stream ends, bounding A/V drift to
    ≤1 frame. The remove path already conforms the picture to the audio master's
    exact length, so this is a no-op safety net there; it is the real guarantee
    for **silence mode**, where the picture is stream-copied at the *source's*
    video-track duration, which can differ from the audio master by more than a
    frame on real files whose v/a tracks aren't exactly equal-length.
    """
    ext = Path(output_path).suffix
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", vcodec]
    if vcodec != "copy":
        if crf is not None:
            cmd += ["-crf", f"{crf:g}"]
        if preset is not None:
            cmd += ["-preset", preset]
    cmd += audio_mux_args(ext)
    if ext.lower() in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-shortest", str(output_path)]
    run_ffmpeg(cmd)
