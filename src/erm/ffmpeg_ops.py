"""ffmpeg / ffprobe wrappers: probe, segment extraction, denoise, render."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .models import Word


def ffprobe_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _probe_audio_stream(path: str | Path) -> tuple[int, int]:
    """Return ``(sample_rate, channels)`` of the first audio stream.

    Used to mint injected-silence sources (`anullsrc`) that match the kept
    audio's format exactly, so ffmpeg's `concat` filter — which requires a
    uniform sample rate and channel layout across its inputs — joins them
    without resampling the real audio.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    fields: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    return int(fields["sample_rate"]), int(fields["channels"])


def gap_channel_layout(path: str | Path) -> str:
    """Return the `anullsrc` ``channel_layout`` name for min-gap injection.

    Only mono and stereo have an unambiguous layout name to mint matching
    injected-silence (`anullsrc`) sources; for anything else, raise `ValueError`
    rather than mislabel a multichannel mix as stereo and silently corrupt the
    output. The CLI calls this up front (before transcription) so a `--min-gap-ms`
    run on an unsupported input fails fast instead of after the whole pipeline.
    """
    _, channels = _probe_audio_stream(path)
    layout = {1: "mono", 2: "stereo"}.get(channels)
    if layout is None:
        raise ValueError(
            f"min-gap injection supports mono/stereo input only; got {channels} "
            "channels. Re-run without --min-gap-ms, or downmix the input first."
        )
    return layout


def has_video_stream(path: str | Path) -> bool:
    """Return True if `path` has a real (non-cover-art) video stream.

    ``ffprobe`` reports a still image embedded as cover art (e.g. an mp3's
    album thumbnail) as a video stream with ``disposition.attached_pic=1``;
    that is not motion video, so it's excluded. Used to decide whether the
    input must be routed through :func:`extract_audio_wav` before any
    librosa-based analysis (which falls back to a slow, deprecated decoder on
    video containers).
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream_disposition=attached_pic",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    # One `DISPOSITION:attached_pic=<0|1>` line per video stream. A real motion
    # stream has it 0; cover art has it 1. True iff any stream is real video.
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "DISPOSITION:attached_pic" and value.strip() == "0":
            return True
    return False


def extract_audio_wav(input_path: str | Path, output_path: str | Path,
                      *, sample_rate: int = 16_000, channels: int = 1) -> None:
    """Decode `input_path`'s audio to a mono PCM WAV via ffmpeg.

    `-vn` drops any video stream, so this works on video containers (mp4/mov)
    that librosa's soundfile backend can't open. The result is the canonical
    analysis input: 16 kHz mono `pcm_s16le` — exactly what the transcriber and
    the numpy detectors downsample to anyway, so feeding them this WAV loses
    nothing while avoiding librosa's deprecated `audioread` fallback.
    """
    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vn",
           "-ac", str(channels), "-ar", str(sample_rate),
           "-c:a", "pcm_s16le", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_segment(input_path: str | Path, start_s: float, end_s: float,
                    output_path: str | Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(input_path),
           "-ss", f"{start_s:.6f}", "-to", f"{end_s:.6f}",
           "-c:a", "pcm_s16le", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def denoise_to(input_path: str | Path, output_path: str | Path,
               nr: float = 12.0, nf: float = -25.0) -> None:
    """Run ffmpeg's afftdn denoiser on `input_path`, writing PCM to `output_path`.

    `nr` is the noise reduction in dB (higher = more aggressive). `nf` is the
    noise floor in dB. Defaults are gentle — strong enough to flatten room
    tone and HVAC hiss without obviously processing the speech.
    """
    cmd = ["ffmpeg", "-y", "-i", str(input_path),
           "-af", f"afftdn=nr={nr}:nf={nf}",
           "-c:a", "pcm_s16le", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def overlay_room_tone(audio_path: str | Path, tone_path: str | Path,
                      output_path: str | Path, level_db: float = -12.0) -> None:
    """Mix a looped room-tone sample under `audio_path` and write to `output_path`.

    The tone loops indefinitely and is attenuated by `level_db` dB so it sits
    below the speech as an ambient floor. We use `amix=duration=first` so the
    output length matches `audio_path` exactly — the tone is truncated to the
    main audio's duration.
    """
    gain = 10.0 ** (level_db / 20.0)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-stream_loop", "-1", "-i", str(tone_path),
        "-filter_complex",
        f"[1:a]volume={gain:.6f}[tone];"
        f"[0:a][tone]amix=inputs=2:duration=first:dropout_transition=0[out]",
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _mute_filter(mute_ranges: Sequence[tuple[float, float]]) -> str:
    """Build an ffmpeg `volume` filter that silences every `mute_ranges` span.

    Produces ``volume=enable='between(t,s1,e1)+between(t,s2,e2)+...':volume=0``
    — a single timeline-gated pass that zeroes the audio inside each span and
    leaves everything else byte-identical. An empty `mute_ranges` yields an
    empty string (no filter needed).
    """
    if not mute_ranges:
        return ""
    spans = "+".join(f"between(t,{s:.6f},{e:.6f})" for s, e in mute_ranges)
    return f"volume=enable='{spans}':volume=0"


def render_silenced(
    input_path: str | Path,
    mute_ranges: Sequence[tuple[float, float]],
    output_path: str | Path,
) -> None:
    """Mute `mute_ranges` in place, preserving the input's exact duration.

    Unlike `render`, this never excises anything — it gates the audio to zero
    inside each span via a single `volume` pass, so the timeline (and any A/V
    sync, multi-track alignment, or caption timing keyed to it) is untouched.
    The muted holes are filled with the natural floor by the room-tone overlay
    step. Empty `mute_ranges` ⇒ a straight transcode of the input.
    """
    mute_filter = _mute_filter(mute_ranges)
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    if mute_filter:
        cmd += ["-af", mute_filter]
    cmd += ["-c:a", "pcm_s16le", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _splice_crossfade_s(
    cut_s: float,
    prev_len: float,
    next_len: float,
    *,
    crossfade_ms: float | None,
    min_crossfade_ms: float,
    max_crossfade_ms: float,
    crossfade_factor: float,
    lhs_room: float | None = None,
    rhs_room: float | None = None,
) -> float:
    """Per-splice crossfade length (seconds) for one splice. See `render`.

    When `crossfade_ms` is given it's a fixed override; otherwise the fade
    scales with the cut as ``cut_ms * crossfade_factor`` clamped to
    ``[min_crossfade_ms, max_crossfade_ms]``. The result is then capped to
    half of each surrounding fragment (so a fade can't exceed the audio it
    has to live in) and, when `lhs_room`/`rhs_room` are supplied (the
    distance from the splice back to the nearest real word on each side), to
    twice that room — a fade reaches ~half its length into each side, so
    ``2 * room`` keeps it from attenuating a real word. Never negative.
    """
    if crossfade_ms is not None:
        cf = max(0.0, crossfade_ms) / 1000.0
    else:
        cf_ms = min(max_crossfade_ms,
                    max(min_crossfade_ms, cut_s * 1000.0 * crossfade_factor))
        cf = cf_ms / 1000.0
    cf = min(cf, prev_len / 2, next_len / 2)
    if lhs_room is not None and rhs_room is not None:
        cf = min(cf, 2 * lhs_room, 2 * rhs_room)
    return max(0.0, cf)


def _keep_fades(
    keep_ranges: Sequence[tuple[float, float]],
    words: Sequence[Word] | None,
    *,
    crossfade_ms: float | None,
    min_crossfade_ms: float,
    max_crossfade_ms: float,
    crossfade_factor: float,
    min_gap_s: float = 0.0,
    snap_fps: float | None = None,
) -> list[float]:
    """Per-splice crossfade lengths for each keep→keep join.

    The per-splice fade computation shared by `render`'s default path and
    `_render_with_gaps` (the same `_splice_crossfade_s` scaling and word-room
    clamp), returning a list of length ``len(keep_ranges) - 1`` where entry
    ``i`` is the fade for the join between keep ``i`` and keep ``i+1``.

    When `min_gap_s > 0`, each fade is additionally clamped so it can't pull the
    two flanking words below the gap floor. A gapless `acrossfade` overlaps the
    survivors by `fade`, eating that much out of the silence between the words,
    so the audible gap is ``surviving_gap - fade``; capping ``fade`` at
    ``surviving_gap - min_gap_s`` keeps that gap ≥ the floor. ``surviving_gap``
    is ``lhs_room + rhs_room`` — the same per-side silence already measured for
    the word-protection clamp, and the same quantity `inject_min_gaps` uses, so
    the two enforcement paths agree: splices below the floor get silence
    *injected* (a `concat`, no overlap), splices just above it get their
    crossfade *trimmed* here. With `min_gap_s == 0` (every default run) the
    clamp is skipped and the fades are byte-for-byte the prior values.

    When `snap_fps` is set (a video render), each fade is rounded to a whole
    video frame (``round(fade·fps)/fps``). The identical snapped list feeds both
    the audio `acrossfade` and the video `xfade`, so both streams shorten by the
    same amount at every splice and stay in sync by construction. `snap_fps=None`
    (every audio-only run) leaves the fades untouched — byte-identical output.
    """
    fades: list[float] = []
    for i in range(1, len(keep_ranges)):
        cut_s = keep_ranges[i][0] - keep_ranges[i - 1][1]
        prev_len = keep_ranges[i - 1][1] - keep_ranges[i - 1][0]
        next_len = keep_ranges[i][1] - keep_ranges[i][0]
        lhs_room = rhs_room = None
        if words is not None:
            splice_lhs = keep_ranges[i - 1][1]
            splice_rhs = keep_ranges[i][0]
            prev_word_end = max(
                (w.end for w in words if w.end <= splice_lhs),
                default=keep_ranges[i - 1][0],
            )
            next_word_start = min(
                (w.start for w in words if w.start >= splice_rhs),
                default=keep_ranges[i][1],
            )
            lhs_room = splice_lhs - prev_word_end
            rhs_room = next_word_start - splice_rhs
        fade = _splice_crossfade_s(
            cut_s, prev_len, next_len,
            crossfade_ms=crossfade_ms,
            min_crossfade_ms=min_crossfade_ms,
            max_crossfade_ms=max_crossfade_ms,
            crossfade_factor=crossfade_factor,
            lhs_room=lhs_room, rhs_room=rhs_room,
        )
        if min_gap_s > 0 and lhs_room is not None and rhs_room is not None:
            surviving_gap = lhs_room + rhs_room
            fade = min(fade, max(0.0, surviving_gap - min_gap_s))
        if snap_fps:
            # Snap to whole frames. ffmpeg's `xfade` corrupts a chained graph
            # when a transition is a single frame, so floor any positive fade at
            # two frames (still a valid audio crossfade — ≥2 frames is ~67-80ms
            # at 24-30 fps, within the normal crossfade range). A fade that
            # rounds to zero stays zero (that splice hard-cuts on both streams).
            frames = round(fade * snap_fps)
            if frames == 1:
                frames = 2
            fade = frames / snap_fps
        fades.append(fade)
    return fades


def _render_with_gaps(
    input_path: str | Path,
    keep_ranges: Sequence[tuple[float, float]],
    output_path: str | Path,
    gap_inserts: Sequence[tuple[int, float]],
    *,
    crossfade_ms: float | None,
    min_crossfade_ms: float,
    max_crossfade_ms: float,
    crossfade_factor: float,
    words: Sequence[Word] | None,
    min_gap_s: float = 0.0,
    fades: Sequence[float] | None = None,
) -> None:
    """Render keeps with injected silent gaps, as a linear filtergraph fold.

    Each `gap_inserts` item ``(after_keep_index, duration)`` places a silent
    segment at the splice following that keep. The graph is folded left to
    right: keep→keep joins reuse the existing per-splice `acrossfade` (or
    `concat` when that fade would be zero); any join touching an injected gap
    uses `concat` so the injected duration lands exactly. Injected silence is
    an `anullsrc` matched to the input's sample rate / channel layout (the
    room-tone overlay later fills it with the natural floor).

    `min_gap_s` is forwarded to `_keep_fades` so the surviving (un-injected)
    crossfades are trimmed to keep their flanking words at or above the floor —
    the splices that *were* injected already honor it exactly via `concat`.
    """
    fades = list(fades) if fades is not None else _keep_fades(
        keep_ranges, words,
        crossfade_ms=crossfade_ms,
        min_crossfade_ms=min_crossfade_ms,
        max_crossfade_ms=max_crossfade_ms,
        crossfade_factor=crossfade_factor,
        min_gap_s=min_gap_s,
    )
    sample_rate, _ = _probe_audio_stream(input_path)
    # The injected `anullsrc` must match the real audio's channel layout so
    # `concat` joins them without a mismatch (the CLI validates this up front).
    layout = gap_channel_layout(input_path)

    parts: list[str] = []
    for i, (s, e) in enumerate(keep_ranges):
        parts.append(
            f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[k{i}]"
        )

    gaps_after: dict[int, list[float]] = defaultdict(list)
    for after_keep_index, duration in gap_inserts:
        gaps_after[after_keep_index].append(float(duration))

    # Node sequence: each keep, followed by any silent gaps spliced after it.
    nodes: list[tuple[str, str, int | None]] = []
    for i in range(len(keep_ranges)):
        nodes.append(("keep", f"k{i}", i))
        for j, duration in enumerate(gaps_after.get(i, [])):
            gap_label = f"g{i}_{j}"
            parts.append(
                f"anullsrc=channel_layout={layout}:sample_rate={sample_rate},"
                f"atrim=duration={duration:.6f},asetpts=PTS-STARTPTS[{gap_label}]"
            )
            nodes.append(("gap", gap_label, None))

    if len(nodes) == 1:
        map_label = nodes[0][1]
    else:
        prev_label = nodes[0][1]
        for n in range(1, len(nodes)):
            kind, label, keep_index = nodes[n]
            out_label = "out" if n == len(nodes) - 1 else f"m{n}"
            prev_kind = nodes[n - 1][0]
            if prev_kind == "keep" and kind == "keep":
                fade = fades[keep_index - 1]
                if fade > 0:
                    parts.append(
                        f"[{prev_label}][{label}]"
                        f"acrossfade=d={fade:.6f}:c1=tri:c2=tri[{out_label}]"
                    )
                else:
                    parts.append(
                        f"[{prev_label}][{label}]concat=n=2:v=0:a=1[{out_label}]"
                    )
            else:
                parts.append(
                    f"[{prev_label}][{label}]concat=n=2:v=0:a=1[{out_label}]"
                )
            prev_label = out_label
        map_label = "out"

    filter_complex = ";".join(parts)
    cmd = ["ffmpeg", "-y", "-i", str(input_path),
           "-filter_complex", filter_complex,
           "-map", f"[{map_label}]", "-c:a", "pcm_s16le", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def render(
    input_path: str | Path,
    keep_ranges: Sequence[tuple[float, float]],
    output_path: str | Path,
    crossfade_ms: float | None = None,
    min_crossfade_ms: float = 50.0,
    max_crossfade_ms: float = 120.0,
    crossfade_factor: float = 0.15,
    words: Sequence[Word] | None = None,
    gap_inserts: Sequence[tuple[int, float]] | None = None,
    min_gap_s: float = 0.0,
    fades: Sequence[float] | None = None,
) -> None:
    """Render `keep_ranges` from `input_path` to `output_path` via ffmpeg.

    Uses `atrim` + `acrossfade` so each splice gets an equal-power crossfade.
    The fade length scales with the cut size at that splice — longer cuts
    splice across audio that differs more in pitch/energy and need a longer
    fade to mask the transition. Per-splice formula:

        fade = clamp(min_crossfade_ms, cut_ms * crossfade_factor, max_crossfade_ms)

    Pass `crossfade_ms` to override with a single fixed length (legacy /
    A/B testing); when None, the per-splice scaling is used.

    `gap_inserts` (a list of ``(after_keep_index, duration_s)``) injects silent
    gaps at specific splices to honor a minimum-gap floor. `min_gap_s` is that
    floor; when it is set, the gap-aware path runs even with no injections so
    the surviving crossfades are trimmed not to pull words below the floor (see
    `_keep_fades`). With both unset/zero (every default run) the verbatim
    default render path below runs and the output is byte-identical.

    `fades` overrides the per-splice crossfade lengths (length
    ``len(keep_ranges) - 1``) instead of computing them internally. A video
    render passes the same frame-snapped list to both this audio render and the
    video render so the two streams shorten identically at every splice.
    """
    if not keep_ranges:
        raise ValueError("keep_ranges is empty — output would have no audio")

    # Route through the gap-aware path when a gap was injected, or when a floor
    # is set and there is at least one splice to trim. A single keep has no
    # splices, so the floor is moot there and the fast path below still applies.
    if gap_inserts or (min_gap_s > 0 and len(keep_ranges) > 1):
        _render_with_gaps(
            input_path, keep_ranges, output_path, gap_inserts or [],
            crossfade_ms=crossfade_ms,
            min_crossfade_ms=min_crossfade_ms,
            max_crossfade_ms=max_crossfade_ms,
            crossfade_factor=crossfade_factor,
            words=words,
            min_gap_s=min_gap_s,
            fades=fades,
        )
        return

    if len(keep_ranges) == 1:
        s, e = keep_ranges[0]
        cmd = ["ffmpeg", "-y", "-i", str(input_path),
               "-ss", f"{s:.6f}", "-to", f"{e:.6f}",
               "-c:a", "pcm_s16le", str(output_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        return

    # Per-splice crossfade lengths. The word-aware clamp inside `_keep_fades`
    # measures the room back to the nearest real word on each side so a fade
    # never attenuates one; when a side has no word (e.g. a splice past the
    # last word) it falls back to that fragment's own boundary, imposing
    # nothing beyond the fragment-length cap.
    fades_s = list(fades) if fades is not None else _keep_fades(
        keep_ranges, words,
        crossfade_ms=crossfade_ms,
        min_crossfade_ms=min_crossfade_ms,
        max_crossfade_ms=max_crossfade_ms,
        crossfade_factor=crossfade_factor,
    )

    parts: list[str] = []
    for i, (s, e) in enumerate(keep_ranges):
        parts.append(
            f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}]"
        )

    if all(cf > 0 for cf in fades_s):
        prev = "a0"
        for i in range(1, len(keep_ranges)):
            cf = fades_s[i - 1]
            out_label = f"x{i}" if i < len(keep_ranges) - 1 else "out"
            parts.append(
                f"[{prev}][a{i}]acrossfade=d={cf:.6f}:c1=tri:c2=tri[{out_label}]"
            )
            prev = out_label
    else:
        concat_inputs = "".join(f"[a{i}]" for i in range(len(keep_ranges)))
        parts.append(
            f"{concat_inputs}concat=n={len(keep_ranges)}:v=0:a=1[out]"
        )

    filter_complex = ";".join(parts)
    cmd = ["ffmpeg", "-y", "-i", str(input_path),
           "-filter_complex", filter_complex,
           "-map", "[out]", "-c:a", "pcm_s16le", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)
