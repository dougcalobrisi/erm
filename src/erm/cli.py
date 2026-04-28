"""Command-line interface: `erm` and `erm validate`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .asr import transcribe
from .audio import find_quiet_region, load_audio_mono
from .detect import (
    detect_gap_fillers,
    detect_intraword_fillers,
    detect_overlong_words,
)
from .acoustic import is_sustained_vowel
from .ffmpeg_ops import denoise_to, extract_segment, overlay_room_tone, render
from .fillers import DEFAULT_FILLERS, find_fillers
from .models import Cut
from .ranges import invert_to_keep_ranges, merge_close_cuts
from .refine import refine_boundaries
from .validate import validate_output


def _build_remove_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="erm",
        description="Strip disfluencies from spoken audio.",
    )
    p.add_argument("input", help="Input audio file.")
    p.add_argument("-o", "--output", help="Output audio file (.wav).")
    p.add_argument("--model", default="medium.en",
                   help="faster-whisper model (default: medium.en).")
    p.add_argument("--fillers", default=",".join(sorted(DEFAULT_FILLERS)),
                   help="Comma-separated filler word list.")
    p.add_argument("--search-ms", type=float, default=60.0)
    p.add_argument("--crossfade-ms", type=float, default=None,
                   help="Fixed crossfade length for every splice. When omitted "
                        "(default), each splice scales with its cut length.")
    p.add_argument("--min-crossfade-ms", type=float, default=50.0,
                   help="Floor for the per-splice crossfade scaling.")
    p.add_argument("--max-crossfade-ms", type=float, default=120.0,
                   help="Ceiling for the per-splice crossfade scaling.")
    p.add_argument("--crossfade-factor", type=float, default=0.15,
                   help="Per-splice crossfade = cut_length * factor, "
                        "clamped to [min, max]. Higher = smoother but blurrier.")
    p.add_argument("--merge-gap-ms", type=float, default=120.0,
                   help="Merge two cuts whose surviving fragment is shorter "
                        "than this (the fragment would otherwise be eaten "
                        "by the surrounding crossfades and audibly blurp).")
    p.add_argument("--denoise", choices=("none", "pre", "post", "hybrid"),
                   default="hybrid",
                   help="Background-noise handling. "
                        "'none': leave audio alone. "
                        "'pre': denoise input, then cut. Cleanest splices, "
                        "but detection is less sensitive on denoised audio. "
                        "'post': cut the original, then denoise the output. "
                        "Same detection sensitivity as 'none', but the noise "
                        "floor mismatch at each splice is smoothed afterward. "
                        "'hybrid' (default): detect on the original (full "
                        "sensitivity, all real fillers caught), render cuts "
                        "from the denoised copy (clean splices). Best of both.")
    p.add_argument("--denoise-nr", type=float, default=12.0,
                   help="ffmpeg afftdn noise-reduction strength (dB).")
    p.add_argument("--denoise-nf", type=float, default=-25.0,
                   help="ffmpeg afftdn noise floor (dB).")
    p.add_argument("--room-tone", dest="room_tone",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Sample a quiet region of the *original* recording "
                        "and lay it under the output as a constant ambient "
                        "undertone. Masks splice discontinuities by ensuring "
                        "the noise floor is identical everywhere. Especially "
                        "useful with --denoise (which strips room tone) — "
                        "this puts a bit of natural room tone back, "
                        "consistently. Default on.")
    p.add_argument("--room-tone-level-db", type=float, default=-12.0,
                   help="Attenuation applied to the looped room-tone sample "
                        "before mixing under the speech. Lower = quieter. "
                        "Around -12 to -20 dB is usually right.")
    p.add_argument("--room-tone-source", default="auto",
                   help="Either 'auto' (find a quiet stretch automatically) "
                        "or 'START-END' in seconds (e.g. '0.05-1.4').")
    p.add_argument("--detect-gaps", dest="detect_gaps",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Also cut voiced regions in long inter-word gaps "
                        "(catches fillers Whisper drops). Default on.")
    p.add_argument("--gap-min-ms", type=float, default=350.0,
                   help="Min inter-word gap to scan (ms). Below this, the "
                        "pause is too short to plausibly hide a filler.")
    p.add_argument("--gap-min-voiced-ms", type=float, default=100.0)
    p.add_argument("--gap-max-voiced-ms", type=float, default=1500.0)
    p.add_argument("--intraword-min-ms", type=float, default=550.0,
                   help="Min word duration to scan for hidden trailing "
                        "fillers Whisper subsumed into the word's bounds.")
    p.add_argument("--confirm-pitch", dest="confirm_pitch",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Confirm aggressive overlong-word candidates by "
                        "checking they look like sustained filler vowels "
                        "(stable spectral centroid + voiced ZCR). "
                        "Drops cuts that fall on real speech. Default on.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", dest="json_out",
                   help="Write cut list JSON to this path.")
    return p


def _build_validate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="erm validate",
        description="Validate a rendered output against its source.",
    )
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--cuts", help="Cut list JSON written by `remove`.")
    p.add_argument("--model", default="medium.en")
    p.add_argument("--report", help="Write report JSON to this path.")
    return p


def _timestamped(input_path: str | Path, suffix: str, ext: str) -> Path:
    """Build a sibling output path: {stem}-{suffix}-{YYYYMMDD-HHMMSS}.{ext}.

    Lives next to the input so tooling that pairs source/output (e.g. the
    `validate` subcommand) finds them together.
    """
    p = Path(input_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return p.with_name(f"{p.stem}-{suffix}-{stamp}.{ext}")


def _cmd_remove(args: argparse.Namespace) -> int:
    fillers = {f.strip().lower() for f in args.fillers.split(",") if f.strip()}

    if not args.output and not args.dry_run:
        args.output = str(_timestamped(args.input, "cleaned", "wav"))
        print(f"      output: {args.output}", file=sys.stderr)
    if not args.json_out:
        args.json_out = str(_timestamped(args.input, "cuts", "json"))
        print(f"      cuts:   {args.json_out}", file=sys.stderr)

    # Denoise stages produce two virtual inputs:
    #   `analysis_input` — what transcribe + audio detectors see
    #   `render_input`   — what ffmpeg cuts from
    # `none`:   both = original
    # `pre`:    both = denoised  (cleanest splices, but detection less sensitive
    #                             because denoising flattens the energy/pitch
    #                             signals our detectors rely on)
    # `post`:   both = original; output is denoised at the end
    # `hybrid`: analysis on original (full detection sensitivity), render from
    #           denoised (clean splices). Best filler coverage AND clean splices.
    analysis_input = args.input
    render_input = args.input
    denoised_path: Path | None = None
    if args.denoise in ("pre", "hybrid"):
        denoised_path = _timestamped(args.input, "denoised", "wav")
        print(f"[0/4] denoising input -> {denoised_path}", file=sys.stderr)
        denoise_to(args.input, denoised_path,
                   nr=args.denoise_nr, nf=args.denoise_nf)
        if args.denoise == "pre":
            analysis_input = str(denoised_path)
            render_input = str(denoised_path)
        else:  # hybrid
            render_input = str(denoised_path)

    print(f"[1/4] transcribing with {args.model}...", file=sys.stderr)
    words, duration = transcribe(analysis_input, model_name=args.model)

    word_cuts = find_fillers(words, fillers)
    print(f"[2/4] found {len(word_cuts)} transcribed filler(s) in {duration:.2f}s",
          file=sys.stderr)

    audio = None
    sr = 0
    gap_cuts: list[Cut] = []
    intra_cuts: list[Cut] = []
    if args.detect_gaps:
        audio, sr = load_audio_mono(analysis_input)
        gap_cuts = detect_gap_fillers(
            audio, sr, words, duration,
            min_gap_s=args.gap_min_ms / 1000.0,
            min_voiced_s=args.gap_min_voiced_ms / 1000.0,
            max_voiced_s=args.gap_max_voiced_ms / 1000.0,
        )
        intra_cuts = detect_intraword_fillers(
            audio, sr, words,
            min_word_s=args.intraword_min_ms / 1000.0,
            min_voiced_s=args.gap_min_voiced_ms / 1000.0,
            max_voiced_s=args.gap_max_voiced_ms / 1000.0,
            confirm_pitch=args.confirm_pitch,
        )
        long_cuts = detect_overlong_words(
            audio, sr, words,
            min_voiced_s=args.gap_min_voiced_ms / 1000.0,
            max_voiced_s=args.gap_max_voiced_ms / 1000.0,
        )
        long_cuts_pre = len(long_cuts)
        if args.confirm_pitch:
            long_cuts = [
                c for c in long_cuts
                if is_sustained_vowel(audio, sr, c.start, c.end)
            ]
        print(f"      detected {len(gap_cuts)} gap + {len(intra_cuts)} intra "
              f"+ {len(long_cuts)}/{long_cuts_pre} overlong "
              f"(pitch-confirmed) candidate(s)", file=sys.stderr)
    else:
        long_cuts = []

    raw_cuts = sorted(word_cuts + gap_cuts + intra_cuts + long_cuts,
                      key=lambda c: c.start)
    if raw_cuts:
        print("[3/4] refining cut boundaries...", file=sys.stderr)
        if audio is None:
            audio, sr = load_audio_mono(analysis_input)
        cuts = refine_boundaries(
            audio, sr, raw_cuts, search_ms=args.search_ms,
            words=words, total_duration=duration,
        )
    else:
        cuts = []

    cuts = merge_close_cuts(cuts, min_gap_s=args.merge_gap_ms / 1000.0)
    keep = invert_to_keep_ranges(cuts, duration)
    saved = sum(c.end - c.start for c in cuts)

    cuts_payload = {
        "input": str(args.input),
        "duration_s": duration,
        "cuts": [c.as_dict() for c in cuts],
        "keep_ranges": [{"start": s, "end": e} for s, e in keep],
        "time_saved_s": saved,
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(cuts_payload, indent=2))
        print(f"      wrote cut list to {args.json_out}", file=sys.stderr)

    if args.dry_run:
        print(json.dumps(cuts_payload, indent=2))
        if denoised_path is not None:
            Path(denoised_path).unlink(missing_ok=True)
        return 0

    if not keep:
        print("error: no audio left after removing fillers", file=sys.stderr)
        if denoised_path is not None:
            Path(denoised_path).unlink(missing_ok=True)
        return 1

    print(f"[4/4] rendering {args.output} ({saved:.2f}s removed)", file=sys.stderr)
    needs_post_denoise = args.denoise == "post"
    needs_room_tone = args.room_tone

    render_target = args.output
    if needs_post_denoise or needs_room_tone:
        render_target = str(_timestamped(args.input, "raw", "wav"))

    render(render_input, keep, render_target,
           crossfade_ms=args.crossfade_ms,
           min_crossfade_ms=args.min_crossfade_ms,
           max_crossfade_ms=args.max_crossfade_ms,
           crossfade_factor=args.crossfade_factor,
           words=words)

    current = render_target
    if needs_post_denoise:
        print(f"      denoising output...", file=sys.stderr)
        next_target = (args.output if not needs_room_tone
                       else str(_timestamped(args.input, "denoised-out", "wav")))
        denoise_to(current, next_target,
                   nr=args.denoise_nr, nf=args.denoise_nf)
        if current != args.output:
            Path(current).unlink(missing_ok=True)
        current = next_target

    if needs_room_tone:
        # Always sample the room tone from the *original* — that's what has
        # the real ambient character. Denoising would strip it.
        if args.room_tone_source == "auto":
            if audio is None:
                audio, sr = load_audio_mono(args.input)
            region = find_quiet_region(audio, sr, words)
            if region is None:
                print("      room tone: no quiet region found — skipping",
                      file=sys.stderr)
                if current != args.output:
                    Path(args.output).unlink(missing_ok=True)
                    Path(current).rename(args.output)
                if denoised_path is not None:
                    Path(denoised_path).unlink(missing_ok=True)
                return 0
            tone_start, tone_end = region
        else:
            try:
                ts, te = (float(x) for x in args.room_tone_source.split("-"))
            except ValueError:
                print(f"error: invalid --room-tone-source {args.room_tone_source!r}",
                      file=sys.stderr)
                if current != args.output:
                    Path(current).unlink(missing_ok=True)
                if denoised_path is not None:
                    Path(denoised_path).unlink(missing_ok=True)
                return 2
            tone_start, tone_end = ts, te
        print(f"      room tone: {tone_start:.2f}-{tone_end:.2f}s "
              f"({(tone_end-tone_start)*1000:.0f}ms) "
              f"@ {args.room_tone_level_db:.1f}dB", file=sys.stderr)
        tone_path = _timestamped(args.input, "tone", "wav")
        extract_segment(args.input, tone_start, tone_end, tone_path)
        overlay_room_tone(current, tone_path, args.output,
                          level_db=args.room_tone_level_db)
        Path(tone_path).unlink(missing_ok=True)
        if current != args.output:
            Path(current).unlink(missing_ok=True)

    if denoised_path is not None:
        Path(denoised_path).unlink(missing_ok=True)

    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    if not args.report:
        args.report = str(_timestamped(args.output, "validate", "json"))
        print(f"      report: {args.report}", file=sys.stderr)
    report = validate_output(
        args.input, args.output, args.cuts, model_name=args.model,
    )
    text = json.dumps(report, indent=2)
    print(text)
    Path(args.report).write_text(text)
    return 0 if report.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "validate":
        return _cmd_validate(_build_validate_parser().parse_args(raw[1:]))
    if raw and raw[0] == "remove":
        raw = raw[1:]
    return _cmd_remove(_build_remove_parser().parse_args(raw))
