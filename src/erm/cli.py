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
from .ffmpeg_ops import (
    denoise_to,
    extract_audio_wav,
    extract_segment,
    gap_channel_layout,
    has_video_stream,
    overlay_room_tone,
    render,
    render_silenced,
)
from .fillers import DEFAULT_FILLERS, find_fillers
from .models import Cut
from .ranges import (
    inject_min_gaps,
    invert_to_keep_ranges,
    merge_close_cuts,
    pad_cuts,
)
from .refine import refine_boundaries
from .validate import validate_output
from .video import VideoInfo, probe_video


def _build_remove_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="erm",
        description="Strip disfluencies from spoken audio.",
    )
    p.add_argument("input", help="Input audio file.")
    p.add_argument("-o", "--output", help="Output audio file (.wav).")
    p.add_argument("--model", default="large-v3",
                   help="faster-whisper model (default: large-v3).")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Compute device for transcription. 'auto' (default) "
                        "uses the GPU when available and silently falls back to "
                        "CPU if the CUDA runtime libraries can't be loaded. "
                        "Force 'cpu' to skip the GPU entirely.")
    p.add_argument("--compute-type", dest="compute_type", default="auto",
                   help="faster-whisper compute type (e.g. int8, float16). "
                        "'auto' (default) lets the backend choose.")
    p.add_argument("--fillers", default=",".join(sorted(DEFAULT_FILLERS)),
                   help="Comma-separated filler word list. Replaces the "
                        "built-in default set entirely; use --add-fillers to "
                        "extend the defaults instead.")
    p.add_argument("--add-fillers", dest="add_fillers", default="",
                   help="Comma-separated words to add on top of --fillers "
                        "(e.g. 'basically,like'). Convenient for keeping the "
                        "defaults and adding a few of your own verbal tics. "
                        "Note: custom words match verbatim only — automatic "
                        "elongation (ummmm -> um) applies to built-in stems.")
    p.add_argument("--remove-fillers", dest="remove_fillers", default="",
                   help="Comma-separated words to drop from the set after "
                        "--fillers/--add-fillers are applied (e.g. 'ah' if it "
                        "over-matches). Removal wins over additions. Emptying "
                        "the set disables pass-1 word matching entirely (the "
                        "gap and intra-word detectors still run).")
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
    p.add_argument("--mode", choices=("remove", "silence"), default="remove",
                   help="How to apply cuts. 'remove' (default): excise each "
                        "cut and splice the survivors together (timeline "
                        "shrinks). 'silence': mute each cut span in place, "
                        "preserving the original duration exactly (keeps A/V "
                        "sync, multi-track alignment, and caption timing). The "
                        "room-tone overlay fills the muted holes with the "
                        "natural floor.")
    p.add_argument("--video", action="store_true",
                   help="Render the picture too, keeping A/V in sync, and write "
                        "a video output whose container is inferred from the "
                        "input (mp4->mp4, mov->mov...). Default OFF: every input, "
                        "including a video file, produces the cleaned audio as "
                        ".wav (the common 'pull the audio out of this video' "
                        "case). The flags below only apply with --video.")
    p.add_argument("--video-splice", dest="video_splice",
                   choices=("crossfade", "cut"), default="crossfade",
                   help="--video only. How to join kept fragments visually. "
                        "'crossfade' (default): proportional dissolve matching "
                        "the audio crossfade at each splice. 'cut': hard jump "
                        "cuts (audio is hard-cut too, declicked, so A/V can't "
                        "drift).")
    p.add_argument("--vcodec", default="libx264",
                   help="--video only. Video encoder for re-encoded output "
                        "(remove mode). Default libx264.")
    p.add_argument("--crf", type=float, default=18.0,
                   help="--video only. x264/x265 quality (lower = better/larger). "
                        "Default 18 (visually lossless).")
    p.add_argument("--preset", default="medium",
                   help="--video only. Encoder speed/efficiency preset. "
                        "Default medium.")
    p.add_argument("--pad-pause-factor", dest="pad_pause_factor", type=float,
                   default=0.0,
                   help="remove mode only. Retain this fraction of the silence "
                        "each cut snapped over, so tight splices keep a little "
                        "breathing room. 0 (default) removes the whole cut. "
                        "Never adds time beyond the silence already in the cut.")
    p.add_argument("--pad-min-ms", dest="pad_min_ms", type=float, default=0.0,
                   help="Lower clamp on the retained pause per side (ms).")
    p.add_argument("--pad-max-ms", dest="pad_max_ms", type=float, default=120.0,
                   help="Upper clamp on the retained pause per side (ms).")
    p.add_argument("--min-gap-ms", dest="min_gap_ms", type=float, default=0.0,
                   help="remove mode only. Guarantee at least this much gap "
                        "between the words flanking each splice, injecting "
                        "silence when the natural pause is shorter. 0 (default) "
                        "injects nothing. Adds a little duration when it "
                        "engages; the room-tone overlay fills the injected "
                        "silence with the natural floor.")
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
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="Compute device for transcription (see `erm remove --help`).")
    p.add_argument("--compute-type", dest="compute_type", default="auto",
                   help="faster-whisper compute type (e.g. int8, float16).")
    p.add_argument("--report", help="Write report JSON to this path.")
    return p


def _parse_filler_set(spec: str) -> set[str]:
    """Parse a comma-separated filler list into a normalized set.

    Whitespace is stripped, words are lowercased, blanks dropped, and
    duplicates collapsed (so ``"Um, um , UH"`` becomes ``{"um", "uh"}``).
    """
    return {word.strip().lower() for word in spec.split(",") if word.strip()}


def _resolve_filler_set(base: str, add: str, remove: str) -> set[str]:
    """Compose the effective filler set from the three CLI specs.

    ``base`` (``--fillers``) defines the set, ``add`` (``--add-fillers``)
    unions words on top, and ``remove`` (``--remove-fillers``) subtracts.
    Removal is applied last, so it wins over additions and a word present in
    both ``add`` and ``remove`` ends up excluded.
    """
    fillers = _parse_filler_set(base)
    fillers |= _parse_filler_set(add)
    fillers -= _parse_filler_set(remove)
    return fillers


def _parse_room_tone_source(value: str) -> tuple[float, float]:
    """Parse a ``'START-END'`` seconds spec into a ``(start, end)`` pair.

    Raises ``ValueError`` if the spec isn't exactly two dash-separated
    numbers (the caller turns that into a usage error with exit code 2).
    Negative offsets are rejected as a side effect: ``-`` is the field
    separator, so ``"-1.0-2.0"`` splits into three parts and fails. Room
    tone is sampled from real audio time, which always starts at 0, so
    there's no valid negative spec to support.

    A non-increasing range (``end <= start``) is also rejected: it would
    extract an empty or backwards segment downstream, which ffmpeg turns
    into a confusing failure far from the user's typo.
    """
    start_s, end_s = (float(part) for part in value.split("-"))
    if start_s < 0 or end_s <= start_s:
        raise ValueError(f"room-tone range must be 0 <= start < end: {value!r}")
    return start_s, end_s


def _timestamped(input_path: str | Path, suffix: str, ext: str) -> Path:
    """Build a sibling output path: {stem}-{suffix}-{YYYYMMDD-HHMMSS}.{ext}.

    Lives next to the input so tooling that pairs source/output (e.g. the
    `validate` subcommand) finds them together.
    """
    p = Path(input_path)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return p.with_name(f"{p.stem}-{suffix}-{stamp}.{ext}")


def _cmd_remove(args: argparse.Namespace) -> int:
    fillers = _resolve_filler_set(args.fillers, args.add_fillers,
                                  args.remove_fillers)

    # Validate spacing knobs up front so a bad combination fails immediately
    # rather than after the (slow) transcribe/refine pass.
    if args.pad_pause_factor < 0:
        print("error: --pad-pause-factor must be >= 0", file=sys.stderr)
        return 2
    if args.pad_min_ms < 0 or args.pad_max_ms < 0:
        print("error: --pad-min-ms / --pad-max-ms must be >= 0", file=sys.stderr)
        return 2
    if args.pad_min_ms > args.pad_max_ms:
        print(f"error: --pad-min-ms ({args.pad_min_ms:g}) cannot exceed "
              f"--pad-max-ms ({args.pad_max_ms:g})", file=sys.stderr)
        return 2
    if args.min_gap_ms < 0:
        print("error: --min-gap-ms must be >= 0", file=sys.stderr)
        return 2
    # Min-gap injection mints mono/stereo-only silence sources. Probe the input
    # now (channel count survives denoising) so an unsupported file fails fast
    # with a clean message instead of a traceback at the final render step. A
    # dry run never renders, so the limitation doesn't apply there.
    if args.mode == "remove" and args.min_gap_ms > 0 and not args.dry_run:
        try:
            gap_channel_layout(args.input)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    # The spacing knobs only shape the splices that `remove` mode creates;
    # `silence` mode makes no splices, so they're inert there. Warn rather than
    # error so a caller flipping --mode on an existing command line isn't broken.
    if args.mode == "silence":
        ignored = [
            flag
            for flag, value in (
                ("--pad-pause-factor", args.pad_pause_factor),
                ("--min-gap-ms", args.min_gap_ms),
            )
            if value > 0
        ]
        if ignored:
            print(f"warning: {' / '.join(ignored)} ignored in --mode silence "
                  "(they only shape remove-mode splices)", file=sys.stderr)

    # ----- Video output gating ------------------------------------------------
    # Video is opt-in (--video). Without it, every input — including a video
    # file — yields the cleaned audio as .wav (today's behavior, unchanged, and
    # the common "pull the audio out of this video" case). With --video and a
    # real video stream present, the output container is inferred from the input.
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
    video_info = probe_video(args.input) if args.video else VideoInfo(has_video=False)
    render_video = args.video and video_info.has_video
    if args.video and not video_info.has_video:
        print("warning: --video given but the input has no motion-video stream; "
              "writing audio-only output", file=sys.stderr)

    # Warn about video-only knobs passed without --video (inert), mirroring the
    # silence-mode spacing-knob warning above.
    if not args.video:
        vid_ignored = [
            flag
            for flag, changed in (
                ("--video-splice", args.video_splice != "crossfade"),
                ("--vcodec", args.vcodec != "libx264"),
                ("--crf", args.crf != 18.0),
                ("--preset", args.preset != "medium"),
            )
            if changed
        ]
        if vid_ignored:
            print(f"warning: {' / '.join(vid_ignored)} ignored without --video",
                  file=sys.stderr)

    # `-o`'s extension always wins, but a video container without --video is a
    # footgun (we'd silently write audio into a .mp4); reject it. Conversely a
    # non-video `-o` with --video can't hold a picture, so fall back to audio.
    if args.output:
        out_is_video_container = Path(args.output).suffix.lower() in VIDEO_EXTS
        if out_is_video_container and not args.video:
            print(f"error: -o {args.output} names a video container but --video "
                  "was not given. Add --video to render the picture, or choose a "
                  ".wav output.", file=sys.stderr)
            return 2
        if render_video and not out_is_video_container:
            print(f"warning: -o {args.output} is not a video container; "
                  "writing audio-only", file=sys.stderr)
            render_video = False

    output_ext = (Path(args.input).suffix.lstrip(".").lower() or "mp4") \
        if render_video else "wav"
    if not args.output and not args.dry_run:
        args.output = str(_timestamped(args.input, "cleaned", output_ext))
        print(f"      output: {args.output}", file=sys.stderr)
    if not args.json_out:
        args.json_out = str(_timestamped(args.input, "cuts", "json"))
        print(f"      cuts:   {args.json_out}", file=sys.stderr)

    # When the input carries a real (non-cover-art) video stream, decode its
    # audio to a temp 16 kHz mono WAV up front. Every librosa-based analysis
    # below — the numpy gap/intra/overlong detectors and room-tone sampling —
    # then reads plain PCM instead of falling back to librosa's slow, deprecated
    # audioread/ffmpeg decoder on the video container. Audio-only inputs are
    # passed through untouched (byte-identical behavior). This is analysis-only:
    # rendering and denoising still operate on the full-quality `args.input`.
    original_audio = args.input
    analysis_wav: Path | None = None
    if has_video_stream(args.input):
        analysis_wav = _timestamped(args.input, "analysis", "wav")
        print(f"      extracting audio for analysis -> {analysis_wav}",
              file=sys.stderr)
        extract_audio_wav(args.input, analysis_wav)
        original_audio = str(analysis_wav)

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
    analysis_input = original_audio
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

    def _cleanup_temps() -> None:
        """Remove every intermediate temp file this run may have created."""
        for temp in (analysis_wav, denoised_path):
            if temp is not None:
                Path(temp).unlink(missing_ok=True)

    print(f"[1/4] transcribing with {args.model}...", file=sys.stderr)
    words, duration = transcribe(
        analysis_input, model_name=args.model,
        device=args.device, compute_type=args.compute_type,
    )

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
        # Pause-proportional padding retains a fraction of the silence each
        # cut snapped over (remove mode only — silence mode preserves timing
        # already). Applied before merge_close_cuts, while cuts is still 1:1
        # with raw_cuts (the invariant pad_cuts relies on).
        if args.mode == "remove" and args.pad_pause_factor > 0:
            cuts = pad_cuts(
                cuts, raw_cuts, args.pad_pause_factor,
                args.pad_min_ms / 1000.0, args.pad_max_ms / 1000.0,
            )
    else:
        cuts = []

    cuts = merge_close_cuts(cuts, min_gap_s=args.merge_gap_ms / 1000.0)
    keep = invert_to_keep_ranges(cuts, duration)
    saved = sum(c.end - c.start for c in cuts)

    # Minimum-gap floor (remove mode only): inject silence at any splice whose
    # natural surviving pause is below the floor. gap_inserts feeds render();
    # injected is the total added duration (subtracted from the net savings).
    gap_inserts: list[tuple[int, float]] | None = None
    injected = 0.0
    if args.mode == "remove" and args.min_gap_ms > 0 and keep:
        timeline = inject_min_gaps(keep, words, args.min_gap_ms / 1000.0)
        gap_inserts = []
        keep_index = -1
        for kind, _start, duration in timeline:
            if kind == "keep":
                keep_index += 1
            else:
                gap_inserts.append((keep_index, duration))
        injected = sum(duration for _, duration in gap_inserts)

    cuts_payload = {
        "input": str(args.input),
        "duration_s": duration,
        "mode": args.mode,
        "cuts": [c.as_dict() for c in cuts],
        "keep_ranges": [{"start": s, "end": e} for s, e in keep],
    }
    if args.mode == "silence":
        # Nothing is excised; the cuts are muted in place. No net time saved,
        # but report how much audio was muted.
        cuts_payload["injected_gap_s"] = 0.0
        cuts_payload["muted_s"] = saved
        cuts_payload["time_saved_s"] = 0.0
    else:
        cuts_payload["injected_gap_s"] = injected
        cuts_payload["time_saved_s"] = saved - injected

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(cuts_payload, indent=2))
        print(f"      wrote cut list to {args.json_out}", file=sys.stderr)

    if args.dry_run:
        print(json.dumps(cuts_payload, indent=2))
        _cleanup_temps()
        return 0

    # The empty-output guard only applies to remove mode (where an empty keep
    # list means nothing survives). silence mode mutes in place, so even an
    # all-cut clip still produces a full-duration (muted) output.
    if args.mode == "remove" and not keep:
        print("error: no audio left after removing fillers", file=sys.stderr)
        _cleanup_temps()
        return 1

    # TODO(phase 2-4): wire the video render + mux here. Until then, fail loudly
    # rather than silently emit audio into a video-named output.
    if render_video:
        print("error: --video rendering is not wired yet (landing in a "
              "follow-up phase)", file=sys.stderr)
        _cleanup_temps()
        return 1

    needs_post_denoise = args.denoise == "post"
    needs_room_tone = args.room_tone

    # Warn when the holes (muted spans / injected gaps) would be bare digital
    # silence rather than the natural room-tone floor. Muting zeroes the span and
    # denoise only *reduces* signal (never fills a zeroed hole), so room tone is
    # the only thing that backfills the floor — warn whenever it's off, in any
    # denoise mode.
    if args.mode == "silence" and not needs_room_tone:
        print("warning: --mode silence with --no-room-tone — muted holes will be "
              "digital silence, not a natural floor", file=sys.stderr)
    if args.mode == "remove" and injected > 0 and not needs_room_tone:
        print("warning: --min-gap-ms injects silence at tight splices; without "
              "room tone those gaps will be digital silence", file=sys.stderr)

    if args.mode == "silence":
        print(f"[4/4] rendering {args.output} ({saved:.2f}s muted)",
              file=sys.stderr)
    else:
        print(f"[4/4] rendering {args.output} ({saved:.2f}s removed"
              + (f", {injected:.2f}s gap injected)" if injected > 0 else ")"),
              file=sys.stderr)

    render_target = args.output
    if needs_post_denoise or needs_room_tone:
        render_target = str(_timestamped(args.input, "raw", "wav"))

    if args.mode == "silence":
        render_silenced(render_input, [(c.start, c.end) for c in cuts],
                        render_target)
    else:
        render(render_input, keep, render_target,
               crossfade_ms=args.crossfade_ms,
               min_crossfade_ms=args.min_crossfade_ms,
               max_crossfade_ms=args.max_crossfade_ms,
               crossfade_factor=args.crossfade_factor,
               words=words,
               gap_inserts=gap_inserts,
               min_gap_s=args.min_gap_ms / 1000.0)

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
                audio, sr = load_audio_mono(original_audio)
            region = find_quiet_region(audio, sr, words)
            if region is None:
                print("      room tone: no quiet region found — skipping",
                      file=sys.stderr)
                if current != args.output:
                    Path(args.output).unlink(missing_ok=True)
                    Path(current).rename(args.output)
                _cleanup_temps()
                return 0
            tone_start, tone_end = region
        else:
            try:
                tone_start, tone_end = _parse_room_tone_source(args.room_tone_source)
            except ValueError:
                print(f"error: invalid --room-tone-source {args.room_tone_source!r}",
                      file=sys.stderr)
                if current != args.output:
                    Path(current).unlink(missing_ok=True)
                _cleanup_temps()
                return 2
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

    _cleanup_temps()

    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    if not args.report:
        args.report = str(_timestamped(args.output, "validate", "json"))
        print(f"      report: {args.report}", file=sys.stderr)
    report = validate_output(
        args.input, args.output, args.cuts, model_name=args.model,
        device=args.device, compute_type=args.compute_type,
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
