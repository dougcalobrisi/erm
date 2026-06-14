# FAQ

Short answers with pointers to the deeper docs.

## Does anything leave my machine?

No. `erm` is fully local — it transcribes with
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) and cuts with
`ffmpeg`, both on your own machine. No API keys, no uploads.

## Do I need a GPU?

No. Transcription runs on CPU by default. If you have an NVIDIA GPU with the
CUDA 12 runtime installed, `--device auto` (the default) will use it and
otherwise fall back to CPU with a warning. See
[Installation → Transcription device](installation.md#transcription-device-gpu-vs-cpu).

## How do I see what would be cut before rendering?

Run with `--dry-run` — it transcribes, detects, and prints (or writes, with
`--json`) the cut list without rendering audio. The recommended loop is
**`--dry-run` → read the cut list → render**, tuning detection between passes.
See [Tuning & workflow](usage.md).

## A real word got cut. How do I make detection less aggressive?

Most over-cutting comes from the gap / overlong / intra-word detectors. Turn the
aggressive passes down or off (`--no-detect-gaps`, `--confirm-pitch`), widen the
thresholds, or drop a specific stem with `--remove-fillers`. Symptom-to-knob
mapping lives in [Troubleshooting](troubleshooting.md).

## A filler survived the pass. What now?

Whisper sometimes hides a filler inside a real word or drops it entirely; the
audio-domain detectors exist to catch those. Make detection more aggressive
(lower `--gap-min-ms`, enable gap detection) or add the word with
`--add-fillers`. See [Troubleshooting](troubleshooting.md) and
[Detection internals](detection.md).

## Can I keep the video, not just the audio?

Yes. A video input emits cleaned **audio only** by default. Pass `--video` to
render a synced picture too — A/V stays in sync by construction. See
[Working with video](video.md).

## Can I keep the original duration (for captions / multitrack sync)?

Yes — use `--mode silence`. Instead of excising cuts, it mutes each span in
place, preserving the exact timeline so caption timestamps and multi-track
alignment stay intact. The default `--mode remove` shortens the timeline. See
[Render pipeline](render-pipeline.md).

## What input formats are supported?

Anything `ffmpeg` can decode — `.wav`, `.mp3`, `.m4a`, `.flac`, and video
containers (`.mp4`, `.mov`, …). Output defaults to `.wav` for audio; with
`--video` the container is inferred from the input (an `-o` extension
overrides).

## Why is `large-v3` the default model?

It catches more fillers and produces tighter word boundaries than the smaller
models, which improves detection quality out of the box. For faster, lower-
accuracy runs use `--model medium.en` or `--model small.en`.

## How do I check the output is correct?

`erm validate input output --cuts cuts.json` runs container sanity, per-mode
duration math, a no-filler-survives invariant, and (for video outputs) an
A/V-sync check. See the [CLI reference](cli-reference.md#erm-validate).
