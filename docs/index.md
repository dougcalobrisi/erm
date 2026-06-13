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

## Quick start

Requires Python 3.11+ and `ffmpeg` / `ffprobe` on your `PATH`.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

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
    runtime is present and otherwise fall back to CPU automatically. Full flag
    list and GPU setup live in the
    [README](https://github.com/dougcalobrisi/erm#readme).

## Where to go next

### Usage guides — for anyone running `erm`

- **[Tuning & workflow](usage.md)** — deciding and iterating: which `--mode`,
  which `--denoise`, how aggressive to detect, and the `--dry-run` →
  read-the-cuts → render loop for efficient tuning.
- **[Recipes](recipes.md)** — copy-paste command lines for common jobs
  (podcast, caption-safe video, multitrack, noisy room, fastest pass, …).
- **[Troubleshooting](troubleshooting.md)** — symptom → knob: describe a bad
  result, find the fix.

### Internals — how the pipeline is shaped, and why

Maintainer-facing design docs. The pipeline runs in this order; each doc covers
one stage:

- **[Detection](detection.md)** — the four-pass filler pipeline (word-list,
  gap, intra-word, overlong), the shared RMS-envelope substrate, and the
  sustained-vowel pitch confirmation that guards the aggressive detectors.
- **[Render pipeline](render-pipeline.md)** — turning cuts into audio:
  boundary refinement, close-cut merging, crossfade scaling, the `remove` vs
  `silence` modes, and the `--pad-pause-factor` / `--min-gap-ms` spacing knobs.
- **[Denoise & room tone](denoise-and-room-tone.md)** — the
  none/pre/post/hybrid denoise routing and the room-tone overlay that gives the
  output a single uniform noise floor.
- **[Transcription](transcription.md)** — the Whisper front end: the verbatim
  prompt that makes filler detection possible, and the CUDA → CPU device
  fallback.

For the full flag list and defaults, the
[README](https://github.com/dougcalobrisi/erm#readme) remains the single source
of truth for the CLI surface.
