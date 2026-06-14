# erm

**Strip disfluencies — `um`, `uh`, `er`, `erm`, `ah`, `hmm`, `mhm`, `mm`,
`uh-huh`, plus any-length elongations like `ummmm` / `uhhhhh` — from recordings
of English speech.**

`erm` is a local command-line tool. It transcribes your audio with
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) to get word-level
timestamps, runs three audio-domain detectors that catch fillers Whisper hides,
and uses ffmpeg to make the cuts. Each splice is snapped to a local energy
minimum and zero-crossing, optionally crossfaded, and laid over a constant loop
of the recording's own room tone so the noise floor stays uniform across edits.

Nothing leaves your machine — no API keys, no uploads.

For the story behind it, see the introductory blog post,
[*erm: a local CLI that strips ums, uhs, and erms from speech*](https://doug.sh/posts/erm-a-local-cli-that-strips-ums-uhs-and-erms-from-speech/).
The package is on [PyPI](https://pypi.org/project/erm/) and the source is on
[GitHub](https://github.com/dougcalobrisi/erm).

## Quick start

Requires Python 3.11+ and `ffmpeg` / `ffprobe` on your `PATH`. With
[`uv`](https://docs.astral.sh/uv/) you can run it with no install:

```sh
uvx erm input.wav
```

See [Installation](installation.md) for the venv / dev-install paths and GPU
setup. Then the common loop:

```sh
# Remove fillers — output and cut-list are auto-named next to the input.
erm input.wav

# Inspect what would be cut, without rendering anything.
erm input.wav --dry-run

# Render explicitly, then validate the result against the source.
erm input.wav -o cleaned.wav
erm validate input.wav cleaned.wav --cuts cuts.json
```

The recommended loop is **`--dry-run` → read the cut list → render** — see the
[tuning & workflow guide](usage.md) for how to use it well.

!!! tip "GPU is optional"
    Transcription runs on CPU by default and needs no extra setup. With
    `--device auto` (the default), `erm` will use an NVIDIA GPU if the CUDA
    runtime is present and otherwise fall back to CPU automatically. The full
    flag list lives in the [CLI reference](cli-reference.md); GPU setup is in
    [Installation](installation.md#transcription-device-gpu-vs-cpu).

## Where to go next

### Usage guides — for anyone running `erm`

- **[Tuning & workflow](usage.md)** — deciding and iterating: which `--mode`,
  which `--denoise`, how aggressive to detect, and the `--dry-run` →
  read-the-cuts → render loop for efficient tuning.
- **[Recipes](recipes.md)** — copy-paste command lines for common jobs
  (podcast, caption-safe video, multitrack, noisy room, fastest pass, …).
- **[Working with video](video.md)** — pulling clean audio out of a video vs.
  rendering a synced picture with `--video`: the mode / splice interactions and
  the min-gap "plays through" behavior.
- **[Troubleshooting](troubleshooting.md)** — symptom → knob: describe a bad
  result, find the fix.

### Internals — how the pipeline is shaped, and why

Maintainer-facing design docs. Start with the **[architecture
overview](architecture.md)** for the end-to-end pipeline map, and keep
**[concepts & glossary](concepts.md)** handy for the shared vocabulary and the
signal-processing theory (RMS envelope, silence floor, zero-crossing splicing,
equal-power crossfades). Then each doc below covers one stage in depth, in
pipeline order:

- **[Detection](detection.md)** — the four-pass filler pipeline (word-list,
  gap, intra-word, overlong), the shared RMS-envelope substrate, and the
  sustained-vowel pitch confirmation that guards the aggressive detectors.
- **[Render pipeline](render-pipeline.md)** — turning cuts into audio:
  boundary refinement, close-cut merging, crossfade scaling, the `remove` vs
  `silence` modes, and the `--pad-pause-factor` / `--min-gap-ms` spacing knobs.
- **[Video render & A/V sync](video-render.md)** — the `--video` path:
  decoupled render + mux, sync by construction (CFR + frame-snapped shared
  fades), the tail conform, min-gap "plays through", codecs, and pixel format.
- **[Denoise & room tone](denoise-and-room-tone.md)** — the
  none/pre/post/hybrid denoise routing and the room-tone overlay that gives the
  output a single uniform noise floor.
- **[Transcription](transcription.md)** — the Whisper front end: the verbatim
  prompt that makes filler detection possible, and the CUDA → CPU device
  fallback.

For the full flag list and defaults, see the
[CLI reference](cli-reference.md) — generated directly from `erm`'s parser, so
it always matches the installed version.
