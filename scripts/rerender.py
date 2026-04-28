"""Re-render an audio file from an existing cuts JSON, varying crossfade params.

Saves the cost of re-transcribing when you just want to A/B different
crossfade settings against an already-decided cut list.

Usage:
    # Fixed crossfade lengths:
    python scripts/rerender.py <input> <cuts.json> fixed <ms1,ms2,...>
    # Adaptive (floor:ceiling:factor) profiles:
    python scripts/rerender.py <input> <cuts.json> adaptive <f1:c1:k1>,<f2:c2:k2>,...

Each variant produces a sibling output. Filenames encode the parameters so
you can tell them apart in the file browser.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from erm import invert_to_keep_ranges, merge_close_cuts, render, Cut


def main() -> int:
    if len(sys.argv) < 5 or sys.argv[3] not in {"fixed", "adaptive"}:
        print(__doc__, file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1])
    cuts_json = Path(sys.argv[2])
    mode = sys.argv[3]
    spec = sys.argv[4]

    data = json.loads(cuts_json.read_text())
    cuts = [Cut(c["start"], c["end"], c["word"]) for c in data["cuts"]]
    duration = float(data["duration_s"])
    cuts = merge_close_cuts(cuts, min_gap_s=0.12)
    keep = invert_to_keep_ranges(cuts, duration)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if mode == "fixed":
        for ms_str in spec.split(","):
            ms = float(ms_str)
            out = input_path.with_name(f"{input_path.stem}-cf{int(ms)}-{stamp}.wav")
            print(f"rendering {out.name} (fixed crossfade={ms}ms)...")
            render(input_path, keep, out, crossfade_ms=ms)
    else:
        for prof in spec.split(","):
            floor, ceiling, factor = (float(x) for x in prof.split(":"))
            out = input_path.with_name(
                f"{input_path.stem}-cf{int(floor)}-{int(ceiling)}-{factor:g}-{stamp}.wav"
            )
            print(f"rendering {out.name} "
                  f"(floor={floor} ceiling={ceiling} factor={factor})...")
            render(input_path, keep, out, crossfade_ms=None,
                   min_crossfade_ms=floor, max_crossfade_ms=ceiling,
                   crossfade_factor=factor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
