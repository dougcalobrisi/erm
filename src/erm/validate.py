"""Validate a rendered output: duration math + no-filler-survives invariant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .asr import transcribe
from .ffmpeg_ops import ffprobe_duration
from .fillers import DEFAULT_FILLERS, is_filler, normalize_word
from .video import probe_video, stream_duration


def validate_output(
    input_path: str | Path,
    output_path: str | Path,
    cuts_path: str | Path | None,
    model_name: str = "large-v3",
    fillers: Iterable[str] = DEFAULT_FILLERS,
    duration_tolerance_ms: float = 50.0,
    device: str = "auto",
    compute_type: str = "auto",
) -> dict:
    """Run the deterministic checks described in the plan. Returns a report dict.

    The report has shape:
        {"ok": bool, "checks": {name: {"ok": bool, "detail": str, ...}}, ...}
    """
    report: dict = {"checks": {}}

    try:
        out_dur = ffprobe_duration(output_path)
        report["checks"]["container_sanity"] = {
            "ok": True, "output_duration_s": out_dur,
        }
    except Exception as exc:  # pragma: no cover - exercised only on real ffprobe failures
        report["checks"]["container_sanity"] = {"ok": False, "detail": str(exc)}
        report["ok"] = False
        return report

    in_dur = ffprobe_duration(input_path)
    report["input_duration_s"] = in_dur
    report["output_duration_s"] = out_dur

    cuts_total = 0.0
    mode = "remove"
    injected = 0.0
    if cuts_path is not None:
        with open(cuts_path) as f:
            cuts_data = json.load(f)
        cuts_total = sum(
            float(c["end"]) - float(c["start"]) for c in cuts_data.get("cuts", [])
        )
        mode = cuts_data.get("mode", "remove")
        injected = float(cuts_data.get("injected_gap_s", 0.0))

    # In `silence` mode nothing is excised — the cuts are muted in place, so
    # the output keeps the input's full duration. In `remove` mode the cuts are
    # spliced out (shrinking the timeline) and any injected min-gap silence is
    # added back.
    if mode == "silence":
        expected = in_dur
    else:
        expected = in_dur - cuts_total + injected
    duration_ok = abs(out_dur - expected) <= duration_tolerance_ms / 1000.0
    report["checks"]["duration_math"] = {
        "ok": duration_ok,
        "mode": mode,
        "expected_s": expected,
        "actual_s": out_dur,
        "delta_ms": (out_dur - expected) * 1000.0,
        "tolerance_ms": duration_tolerance_ms,
    }

    # A/V parity (video outputs only): the picture and the audio must end within
    # ~1 frame of each other. Audio is sample-exact; the video is frame-quantized
    # and conformed to the audio, so the bar is one frame plus a small epsilon.
    out_video = probe_video(output_path)
    if out_video.has_video:
        video_dur = stream_duration(output_path, "v:0")
        audio_dur = stream_duration(output_path, "a:0")
        fps = out_video.fps or 30.0
        av_tolerance_s = 1.0 / fps + 0.005
        if video_dur is None or audio_dur is None:
            av_ok = False
            delta_ms = None
        else:
            delta_ms = (video_dur - audio_dur) * 1000.0
            av_ok = abs(video_dur - audio_dur) <= av_tolerance_s
        report["checks"]["av_sync"] = {
            "ok": av_ok,
            "video_duration_s": video_dur,
            "audio_duration_s": audio_dur,
            "delta_ms": delta_ms,
            "tolerance_ms": av_tolerance_s * 1000.0,
        }

    words, _ = transcribe(
        output_path, model_name=model_name,
        device=device, compute_type=compute_type,
    )
    filler_set = {f.lower() for f in fillers}
    surviving = [
        {"text": w.text, "start": w.start, "end": w.end}
        for w in words if is_filler(normalize_word(w.text), filler_set)
    ]
    report["checks"]["no_filler_invariant"] = {
        "ok": len(surviving) == 0,
        "surviving_count": len(surviving),
        "surviving": surviving[:10],
    }

    report["ok"] = all(c["ok"] for c in report["checks"].values())
    return report
