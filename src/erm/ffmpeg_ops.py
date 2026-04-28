"""ffmpeg / ffprobe wrappers: probe, segment extraction, denoise, render."""

from __future__ import annotations

import subprocess
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


def render(
    input_path: str | Path,
    keep_ranges: Sequence[tuple[float, float]],
    output_path: str | Path,
    crossfade_ms: float | None = None,
    min_crossfade_ms: float = 40.0,
    max_crossfade_ms: float = 80.0,
    crossfade_factor: float = 0.10,
    words: Sequence[Word] | None = None,
) -> None:
    """Render `keep_ranges` from `input_path` to `output_path` via ffmpeg.

    Uses `atrim` + `acrossfade` so each splice gets an equal-power crossfade.
    The fade length scales with the cut size at that splice — longer cuts
    splice across audio that differs more in pitch/energy and need a longer
    fade to mask the transition. Per-splice formula:

        fade = clamp(min_crossfade_ms, cut_ms * crossfade_factor, max_crossfade_ms)

    Pass `crossfade_ms` to override with a single fixed length (legacy /
    A/B testing); when None, the per-splice scaling is used.
    """
    if not keep_ranges:
        raise ValueError("keep_ranges is empty — output would have no audio")

    if len(keep_ranges) == 1:
        s, e = keep_ranges[0]
        cmd = ["ffmpeg", "-y", "-i", str(input_path),
               "-ss", f"{s:.6f}", "-to", f"{e:.6f}",
               "-c:a", "pcm_s16le", str(output_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        return

    fades_s: list[float] = []
    for i in range(1, len(keep_ranges)):
        cut_s = keep_ranges[i][0] - keep_ranges[i - 1][1]
        if crossfade_ms is not None:
            cf = max(0.0, crossfade_ms) / 1000.0
        else:
            cf_ms = min(max_crossfade_ms, max(min_crossfade_ms, cut_s * 1000.0 * crossfade_factor))
            cf = cf_ms / 1000.0
        prev_len = keep_ranges[i - 1][1] - keep_ranges[i - 1][0]
        next_len = keep_ranges[i][1] - keep_ranges[i][0]
        cf = min(cf, prev_len / 2, next_len / 2)

        # Word-aware clamp: a crossfade extends ~cf/2 into the audio on
        # *each* side of the splice. Keep that half-width from reaching
        # back across a word boundary, so the fade never attenuates a
        # real word.
        if words is not None:
            splice_lhs = keep_ranges[i - 1][1]
            splice_rhs = keep_ranges[i][0]
            prev_word_end = max(
                (w.end for w in words if w.end <= splice_lhs), default=0.0
            )
            next_word_start = min(
                (w.start for w in words if w.start >= splice_rhs), default=splice_rhs,
            )
            lhs_room = splice_lhs - prev_word_end
            rhs_room = next_word_start - splice_rhs
            cf = min(cf, 2 * lhs_room, 2 * rhs_room)

        fades_s.append(max(0.0, cf))

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
