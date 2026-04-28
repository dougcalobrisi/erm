"""Validate a rendered output: duration math + no-filler-survives invariant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .asr import transcribe
from .ffmpeg_ops import ffprobe_duration
from .fillers import DEFAULT_FILLERS, is_filler, normalize_word


def validate_output(
    input_path: str | Path,
    output_path: str | Path,
    cuts_path: str | Path | None,
    model_name: str = "medium.en",
    fillers: Iterable[str] = DEFAULT_FILLERS,
    duration_tolerance_ms: float = 50.0,
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
    if cuts_path is not None:
        with open(cuts_path) as f:
            cuts_data = json.load(f)
        cuts_total = sum(
            float(c["end"]) - float(c["start"]) for c in cuts_data.get("cuts", [])
        )
    expected = in_dur - cuts_total
    duration_ok = abs(out_dur - expected) <= duration_tolerance_ms / 1000.0
    report["checks"]["duration_math"] = {
        "ok": duration_ok,
        "expected_s": expected,
        "actual_s": out_dur,
        "delta_ms": (out_dur - expected) * 1000.0,
        "tolerance_ms": duration_tolerance_ms,
    }

    words, _ = transcribe(output_path, model_name=model_name)
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
