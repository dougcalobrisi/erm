# erm

[![Docs](https://img.shields.io/badge/docs-erm-blue)](https://dougcalobrisi.github.io/erm/)

Local CLI that strips disfluencies (`um`, `uh`, `er`, `erm`, `ah`, `hmm`, `mhm`,
`mm`, `uh-huh`, plus any-length elongations like `ummmm` / `uhhhhh`) from
recordings of English speech.

It uses [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (running
the `large-v3` Whisper model by default — override with `--model`) for
word-level timestamps, three audio-domain detectors that catch fillers Whisper
hides, and ffmpeg for the cuts. Each splice is snapped to a local energy
minimum and zero-crossing, optionally crossfaded with a length that scales
with the cut size, and laid over a constant looped sample of the recording's
own room tone so the noise floor stays uniform across edits.

> **Full docs at [dougcalobrisi.github.io/erm](https://dougcalobrisi.github.io/erm/)**
> (source in [`docs/`](docs/index.md)): usage guides for getting good
> results — [tuning & workflow](docs/usage.md), [recipes](docs/recipes.md),
> [troubleshooting](docs/troubleshooting.md) — plus maintainer-facing design docs
> on the detection passes, render pipeline, denoise/room-tone, and transcription.

## Install

Requires Python 3.11+ and `ffmpeg` / `ffprobe` on `PATH`.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Transcription device (GPU vs CPU)

Transcription runs on CPU by default and needs no extra setup. If you have an
NVIDIA GPU, faster-whisper can use it — but only when the CUDA runtime libraries
(`libcublas`, `libcudnn`) are installed. A machine with an NVIDIA GPU and driver
but no CUDA runtime is the common case that produces:

```
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

`erm` handles this automatically: with the default `--device auto`, if the GPU
can't be loaded it prints a warning and falls back to CPU, so transcription
still completes. You have two ways to make it explicit:

- **Force CPU** (no warning, skips the GPU probe): `erm input.wav --device cpu`
- **Enable the GPU** by installing the CUDA wheels into the same environment:

  ```sh
  pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
  ```

  faster-whisper's CUDA backend needs CUDA 12 / cuDNN 9. See the
  [faster-whisper GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu)
  for details.

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

# Pull just the cleaned AUDIO out of a video (default — no picture rendered).
erm talk.mp4            # -> talk-cleaned-*.wav

# Keep the PICTURE: render a synced video output (container inferred from input).
erm talk.mp4 --video    # -> talk-cleaned-*.mp4, audio + video in sync
```

When `-o` / `--json` are omitted, output paths are written next to the input as
`{stem}-cleaned-{YYYYMMDD-HHMMSS}.wav` and `{stem}-cuts-{YYYYMMDD-HHMMSS}.json`.

### Video

A video input works like any other recording, and **by default `erm` emits the
cleaned audio only** (`.wav`) — the common "pull the audio out of this video"
case, and identical to how `erm` has always behaved. Pass `--video` to render
the picture too:

- The output **container is inferred from the input** (mp4→mp4, mov→mov…); an
  `-o` extension always overrides.
- Audio and video stay **in sync by construction** (within ~1 frame): both
  streams are rendered from the same edit timeline with the same frame-snapped
  crossfades, and the picture is conformed to the audio's exact length.
- `--mode silence` stream-copies the picture untouched (lossless, frame-exact).
  `--mode remove` re-encodes the spliced picture; `--video-splice crossfade`
  (default) dissolves each splice, `--video-splice cut` makes hard jump cuts.
- `--min-gap-ms` with `--video` **plays the removed footage through** the
  injected pause (muted) rather than freezing the frame.
- Audio is stored losslessly where the container allows (PCM in mov/mkv); mp4
  gets AAC 256k, webm gets Opus.

See [Modes](#modes) and the
[video render & A/V sync design doc](docs/video-render.md) for the A/V-sync
derivation.

## Use inside AI coding agents

erm ships agent guidance so an AI assistant can install, run, and tune it for you.

**Claude Code / Cowork (plugin):**

```
/plugin marketplace add dougcalobrisi/erm
/plugin install erm@erm
```

This adds two skills: **`erm`** (install + clean a file, asks which mode/recipe
fits) and **`erm-tune`** (diagnoses a bad result and maps the symptom to the
right knob). Ask in plain language — "install erm", "clean up this podcast", "the
splices sound smeared" — and the right skill triggers.

**Other agents (Codex, Copilot, OpenCode, Cursor, Gemini CLI, pi.dev):**

- A root [`AGENTS.md`](AGENTS.md) gives install/use/tune guidance, read
  automatically by these tools.
- The skills in [`skills/`](skills/) are open-format
  [Agent Skills](https://agentskills.io). Copy them into your agent's skills
  directory — `~/.claude/skills/` (Claude, OpenCode) or `~/.agents/skills/`
  (Codex, OpenCode, pi.dev). Codex reads only `.agents/skills/`.

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
5. **Render.** In `remove` mode (default), ffmpeg `atrim` + `acrossfade`
   renders the kept segments. Each splice's crossfade length scales with that
   splice's cut size: `clamp(min, cut_ms * factor, max)`. Crossfades are also
   clamped so they never reach back across a real word boundary. `--mode
   silence` instead mutes the cut spans in place, preserving the original
   duration (see [Modes](#modes)).
6. **Room tone (optional, on by default).** A quiet region of the *original*
   recording is sampled and looped under the output at `--room-tone-level-db`.
   This keeps the noise floor identical everywhere, masking the residual
   noise-floor mismatch at each splice.

## Modes

`--mode` chooses how detected cuts are applied to the audio:

| Mode | Timeline | What happens |
|------|----------|--------------|
| `remove` (default) | shrinks | Each cut span is excised and the survivors are spliced together with crossfades. |
| `silence` | preserved | Each cut span is muted *in place* (a single ffmpeg `volume` pass); the output keeps the input's exact duration. |

Use `silence` when timing must be preserved — A/V sync, multi-track alignment
(you can't excise one mic without de-syncing the others), or caption/transcript
timestamps. It removes the *sound* of the filler but leaves a hole of the
original length.

**`silence` depends on a floor in the hole.** The muted spans are filled by the
room-tone overlay so the noise floor stays uniform. Muting zeroes the span and
denoising only *reduces* signal (it never backfills a zeroed hole), so room tone
is the only thing that restores a floor — with `--mode silence --no-room-tone`
the holes are bare digital silence (an audible "drop out") in *any* denoise mode,
and `erm` warns whenever room tone is off. Keep room tone on (the default) for
natural-sounding mutes.

`silence` mode ignores `--pad-pause-factor` and `--min-gap-ms` — those only
shape the splices that `remove` mode creates, and `silence` makes no splices.

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
| `--model` | `large-v3` | Any faster-whisper model. `medium.en` / `small.en` faster but less accurate. |
| `--device` | `auto` | `auto` / `cpu` / `cuda`. `auto` uses the GPU when available and falls back to CPU if the CUDA runtime can't be loaded (see [Transcription device](#transcription-device-gpu-vs-cpu)). |
| `--compute-type` | `auto` | faster-whisper compute type (e.g. `int8`, `float16`). `auto` lets the backend choose. |
| `--fillers` | `ah,er,erm,hmm,mhm,mm,uh,uh-huh,um` | Comma-separated stems; **replaces** the default set. Elongations matched dynamically. |
| `--add-fillers` | _(empty)_ | Comma-separated words to add on top of `--fillers` (e.g. `basically,like`). Keeps the defaults; matches verbatim only. |
| `--remove-fillers` | _(empty)_ | Comma-separated words to drop after add/replace. Removal wins over additions. |
| `--detect-gaps` / `--no-detect-gaps` | on | Run gap + intra-word + overlong detectors. |
| `--gap-min-ms` | `350` | Minimum inter-word gap to scan for fillers. |
| `--gap-min-voiced-ms` / `--gap-max-voiced-ms` | `100` / `1500` | Voiced-run length bounds. |
| `--intraword-min-ms` | `550` | Minimum word length to scan internally. |
| `--confirm-pitch` / `--no-confirm-pitch` | on | Drop overlong/intra candidates that don't look like sustained filler vowels. |

### Cuts and splices

Two independent knobs control the spacing left behind by a `remove`-mode cut —
they compose but do different things:

- **`--pad-pause-factor`** retains a *fraction* of the silence that already
  existed inside a cut (the bit `refine` snapped over). It's context-aware and
  **never adds time**: a tight mid-sentence "um" with no surrounding silence
  gets ~0 padding and its flanking words still butt together. Bounded per side
  by `--pad-min-ms` / `--pad-max-ms` and by the silence that actually exists.
- **`--min-gap-ms`** *guarantees* at least N ms between the two words flanking a
  cut, **injecting** silence at the splice when the natural pause is below N.
  This is what fixes "words too close after cutting an um." It adds a little
  duration when it engages. The injected silence is filled by the room-tone
  overlay (bare digital silence if room tone is off — `erm` warns).

The factor shapes how much existing pause survives; the floor puts a hard
minimum under it.

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `remove` | `remove` (excise + splice) or `silence` (mute in place, duration preserved). See [Modes](#modes). |
| `--search-ms` | `60` | How far each endpoint may slide to find a local energy minimum. |
| `--crossfade-ms` | *(unset)* | Force a fixed crossfade length for every splice. When unset, per-splice scaling is used. |
| `--min-crossfade-ms` / `--max-crossfade-ms` | `50` / `120` | Floor and ceiling for the per-splice crossfade scaling. |
| `--crossfade-factor` | `0.15` | `cut_ms * factor`, clamped to `[min, max]`. Higher = smoother but blurrier. |
| `--merge-gap-ms` | `120` | Merge two cuts whose surviving fragment would be shorter than this. |
| `--pad-pause-factor` | `0.0` | (`remove` mode) Fraction of each cut's snapped silence to retain. `0` removes the whole cut. Never adds time beyond the cut's own silence. |
| `--pad-min-ms` / `--pad-max-ms` | `0` / `120` | Lower/upper clamp on the retained pause, per side (ms). |
| `--min-gap-ms` | `0.0` | (`remove` mode) Guarantee at least this much gap between the words flanking each splice, injecting silence when the natural pause is shorter. `0` injects nothing. Mono/stereo input only (the injected silence must match the channel layout). |

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
| `-o`, `--output` | auto-named next to input | Output path; its extension overrides the inferred container. |
| `--json PATH` | auto-named next to input | Cut list JSON. |
| `--dry-run` | off | Print the cut list and exit; no audio rendered. |

### Video

| Flag | Default | Notes |
|------|---------|-------|
| `--video` | off (audio-only `.wav`) | Render the picture too; output container inferred from the input. |
| `--video-splice` | `crossfade` | `crossfade` dissolves each splice; `cut` makes hard jump cuts. |
| `--vcodec` | `libx264` | Encoder for the re-encoded picture (remove mode). |
| `--crf` | `18` | x264/x265 quality (lower = better/larger). |
| `--preset` | `medium` | Encoder speed/efficiency preset. |

Without `--video`, the picture flags warn and are ignored. Naming an `-o` video
container without `--video` is an error (add `--video` or choose a `.wav` path).

## `validate` subcommand

```sh
erm validate input.wav cleaned.wav --cuts cuts.json
```

Runs three deterministic checks:

- **Container sanity** — `ffprobe` reads the output without errors.
- **Duration math** — within 50ms of the per-mode expectation, read from the
  cut-list JSON's `mode` and `injected_gap_s` fields (both default to
  `remove` / `0.0` when absent, so older cut lists validate unchanged):
  - `remove`: `output ≈ input − sum(cut lengths) + injected_gap_s`.
  - `silence`: `output ≈ input` (nothing is excised; cuts are muted in place).
- **A/V sync** (video outputs only) — the video and audio streams end within
  ~1 frame of each other.
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

The suite is split into:

- `test_pure.py` — pure logic, no heavy imports: filler matching, range
  inversion, boundary refinement, close-cut merging (`merge_close_cuts`),
  the per-word duration bound (`expected_max_word_duration`), room-tone
  region selection (`find_quiet_region`), the per-splice crossfade
  clamp (`_splice_crossfade_s`), pause-proportional padding (`pad_cuts`),
  min-gap injection (`inject_min_gaps`), and the mute filter (`_mute_filter`).
- `test_render_modes.py` — real-ffmpeg checks of `render_silenced` (duration
  preserved) and `render(..., gap_inserts=...)` (injected gap lands exactly).
  Skipped automatically when `ffmpeg`/`ffprobe` aren't on PATH.
- `test_asr_fallback.py` — the CUDA → CPU fallback in `transcribe`, with
  faster-whisper mocked.
- `test_cli.py` — argument parsing, defaults, and `main()` subcommand
  routing (`remove` / `validate` / bare-input). The pipeline handlers are
  monkeypatched, so nothing heavy runs.
- `test_integration.py` — a golden-path `--dry-run` over a synthesized WAV
  with a stubbed transcriber, wiring transcription → filler detection →
  refinement → range inversion → JSON. Gated on `librosa` (the audio
  loader); skipped automatically if it isn't installed.

## Out of scope

- Removing `like`, `you know`, `I mean` — too risky for meaning.
- Languages other than English.
- Real-time / streaming.
