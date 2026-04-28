# erm

Local CLI that strips disfluencies (`um`, `uh`, `er`, `erm`, `ah`, `hmm`, `mhm`,
`mm`, `uh-huh`, plus any-length elongations like `ummmm` / `uhhhhh`) from
recordings of English speech.

It uses [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (running
the `medium.en` Whisper model by default — override with `--model`) for
word-level timestamps, three audio-domain detectors that catch fillers Whisper
hides, and ffmpeg for the cuts. Each splice is snapped to a local energy
minimum and zero-crossing, optionally crossfaded with a length that scales
with the cut size, and laid over a constant looped sample of the recording's
own room tone so the noise floor stays uniform across edits.

## Install

Requires Python 3.11+ and `ffmpeg` / `ffprobe` on `PATH`.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Usage

```sh
# Remove fillers; output and cut-list paths are auto-generated next to the input.
erm input.wav

# Specify output explicitly.
erm input.wav -o cleaned.wav

# Inspect what would be cut without rendering.
erm input.wav --dry-run

# Validate a rendered output against its source.
erm validate input.wav cleaned.wav --cuts cuts.json
```

When `-o` / `--json` are omitted, output paths are written next to the input as
`{stem}-cleaned-{YYYYMMDD-HHMMSS}.wav` and `{stem}-cuts-{YYYYMMDD-HHMMSS}.json`.

## How it works

1. **Transcribe.** `faster-whisper` runs with `word_timestamps=True` and a
   verbatim-bias `initial_prompt` so it emits filler tokens instead of
   silently cleaning them up.
2. **Detect.** Four passes produce candidate cut ranges:
   - **Word-list match** — words whose normalized text is in `--fillers`,
     including arbitrary-length elongations (e.g. `ummmm` matches the `um`
     stem).
   - **Gap fillers** — voiced regions in inter-word gaps longer than
     `--gap-min-ms`. Catches fillers Whisper drops entirely.
   - **Intra-word fillers** — long words whose interior splits across a
     silence dip into multiple voiced runs. The non-vowel run whose duration
     best matches the word's expected duration is treated as the real word;
     siblings become cuts. Catches `"in, uhhhhh"` that Whisper rolls into one
     `'in'` token.
   - **Overlong words** — words much longer than `expected_max_word_duration`
     for their text. The trailing portion is scanned for voiced runs.
     Optionally pitch-confirmed (`--confirm-pitch`) by checking the cut
     region looks like a sustained filler vowel (stable spectral centroid,
     voiced ZCR), so we don't trim slow-but-real speech.
3. **Refine.** Each cut endpoint snaps to a local RMS-energy minimum within
   ±`--search-ms`, then to the nearest zero-crossing. Refinement is clamped
   so it never crosses a neighboring word's timestamp.
4. **Merge.** Cuts whose surviving fragment would be shorter than
   `--merge-gap-ms` are collapsed into one — a 40ms surviving fragment
   between two cuts gets eaten by the surrounding crossfades and would
   otherwise blurp.
5. **Render.** ffmpeg `atrim` + `acrossfade` renders the kept segments. Each
   splice's crossfade length scales with that splice's cut size:
   `clamp(min, cut_ms * factor, max)`. Crossfades are also clamped so they
   never reach back across a real word boundary.
6. **Room tone (optional, on by default).** A quiet region of the *original*
   recording is sampled and looped under the output at `--room-tone-level-db`.
   This keeps the noise floor identical everywhere, masking the residual
   noise-floor mismatch at each splice.

## Denoising

`--denoise` picks how ffmpeg's `afftdn` denoiser is used:

| Mode     | Detection sees | ffmpeg cuts from         | Notes |
|----------|----------------|--------------------------|-------|
| `none`   | original       | original                 | No denoising. |
| `pre`    | denoised       | denoised                 | Cleanest splices, but detection less sensitive (denoising flattens energy/pitch signals). |
| `post`   | original       | original; output denoised at end | Full detection sensitivity; splice noise-floor mismatch smoothed afterward. |
| `hybrid` (default) | original | denoised               | Full detection sensitivity *and* clean splices. Recommended. |

Tune with `--denoise-nr` (reduction strength dB) and `--denoise-nf` (noise
floor dB).

## Flags

### Detection

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `medium.en` | Any faster-whisper model. `small.en` faster; `large-v3` more accurate. |
| `--fillers` | `ah,er,erm,hmm,mhm,mm,uh,uh-huh,um` | Comma-separated stems. Elongations matched dynamically. |
| `--detect-gaps` / `--no-detect-gaps` | on | Run gap + intra-word + overlong detectors. |
| `--gap-min-ms` | `350` | Minimum inter-word gap to scan for fillers. |
| `--gap-min-voiced-ms` / `--gap-max-voiced-ms` | `100` / `1500` | Voiced-run length bounds. |
| `--intraword-min-ms` | `550` | Minimum word length to scan internally. |
| `--confirm-pitch` / `--no-confirm-pitch` | on | Drop overlong/intra candidates that don't look like sustained filler vowels. |

### Cuts and splices

| Flag | Default | Notes |
|------|---------|-------|
| `--search-ms` | `60` | How far each endpoint may slide to find a local energy minimum. |
| `--crossfade-ms` | *(unset)* | Force a fixed crossfade length for every splice. When unset, per-splice scaling is used. |
| `--min-crossfade-ms` / `--max-crossfade-ms` | `50` / `120` | Floor and ceiling for the per-splice crossfade scaling. |
| `--crossfade-factor` | `0.15` | `cut_ms * factor`, clamped to `[min, max]`. Higher = smoother but blurrier. |
| `--merge-gap-ms` | `120` | Merge two cuts whose surviving fragment would be shorter than this. |

### Audio cleanup

| Flag | Default | Notes |
|------|---------|-------|
| `--denoise` | `hybrid` | `none` / `pre` / `post` / `hybrid` (see table above). |
| `--denoise-nr` | `12.0` | `afftdn` noise reduction (dB). |
| `--denoise-nf` | `-25.0` | `afftdn` noise floor (dB). |
| `--room-tone` / `--no-room-tone` | on | Loop a quiet sample of the original under the output. |
| `--room-tone-level-db` | `-12.0` | Attenuation applied to the looped tone. `-12` to `-20` is usually right. |
| `--room-tone-source` | `auto` | `auto` finds a quiet region; otherwise `START-END` in seconds (e.g. `0.05-1.4`). |

### Output

| Flag | Default | Notes |
|------|---------|-------|
| `-o`, `--output` | auto-named next to input | Output `.wav` path. |
| `--json PATH` | auto-named next to input | Cut list JSON. |
| `--dry-run` | off | Print the cut list and exit; no audio rendered. |

## `validate` subcommand

```sh
erm validate input.wav cleaned.wav --cuts cuts.json
```

Runs three deterministic checks:

- **Container sanity** — `ffprobe` reads the output without errors.
- **Duration math** — `output_duration ≈ input_duration - sum(cut lengths)`,
  within 50ms.
- **No-filler invariant** — re-transcribe the output; assert no token in the
  filler set survives.

Writes a JSON report to `--report PATH` (or auto-named next to the output)
and exits non-zero if any check fails.

## Tests

```sh
pytest
```

The pure helpers (`find_fillers`, `invert_to_keep_ranges`,
`refine_boundaries`, `merge_close_cuts`, `expected_max_word_duration`,
`_voiced_runs_in_region`, …) run without faster-whisper or librosa imported.
Heavy deps are imported lazily inside `transcribe`, `render`,
`load_audio_mono`, and `is_sustained_vowel`.

## Out of scope

- Removing `like`, `you know`, `I mean` — too risky for meaning.
- Languages other than English.
- Real-time / streaming.
