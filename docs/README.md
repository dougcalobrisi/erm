# `erm` docs

Two layers of documentation live here. For the flag list and defaults, the
top-level [README](https://github.com/dougcalobrisi/erm#readme) remains the
single source of truth for the CLI surface.

## Usage guides (start here)

Task-oriented, for anyone running `erm`:

- **[usage.md](usage.md)** — deciding and iterating: which `--mode`, which
  `--denoise`, how aggressive to detect, and the `--dry-run` → read-the-cuts →
  render loop for efficient tuning.
- **[recipes.md](recipes.md)** — copy-paste command lines for common jobs
  (podcast, caption-safe video, multitrack, noisy room, fastest pass, …).
- **[troubleshooting.md](troubleshooting.md)** — symptom → knob: describe a bad
  result, find the fix.

## Internals (design docs)

Maintainer-facing, explaining *why* the pipeline is shaped the way it is. The
pipeline runs in this order; each doc covers one stage:

- **[detection.md](detection.md)** — the four-pass filler pipeline (word-list,
  gap, intra-word, overlong), the shared RMS-envelope substrate, and the
  sustained-vowel pitch confirmation that guards the aggressive detectors.
- **[render-pipeline.md](render-pipeline.md)** — turning cuts into audio:
  boundary refinement, close-cut merging, crossfade scaling, the `remove` vs
  `silence` modes, and the `--pad-pause-factor` / `--min-gap-ms` spacing knobs.
- **[denoise-and-room-tone.md](denoise-and-room-tone.md)** — the
  none/pre/post/hybrid denoise routing and the room-tone overlay that gives the
  output a single uniform noise floor.
- **[transcription.md](transcription.md)** — the Whisper front end: the verbatim
  prompt that makes filler detection possible, and the CUDA → CPU device
  fallback.
