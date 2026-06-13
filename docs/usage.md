# Usage guide: deciding and iterating

A practical guide to running `erm` well: which settings to pick up front, and how
to tune efficiently. For the full flag list and defaults see the top-level
[README](../README.md); for *why* each knob works the way it does, follow the
links into the internals docs. If you have a bad result and want the fix, jump
straight to [troubleshooting.md](troubleshooting.md); for ready-made command
lines, see [recipes.md](recipes.md).

## The basic run

```sh
erm input.wav
```

That transcribes, detects fillers, and renders a cleaned `.wav` plus a cut-list
JSON, both auto-named next to the input. Most recordings need nothing more. The
sections below are for when the default isn't quite right — and the **iterate
loop** at the end is how you find out cheaply.

## Decision: which `--mode`?

**Pick `remove` (the default) unless something downstream depends on the
timeline staying the same length.**

| You're producing… | Use | Why |
|-------------------|-----|-----|
| A podcast, audiobook, audio-only clip | `remove` | The filler and its time disappear; the recording gets shorter and tighter. |
| Video / screencast with captions or A/V sync | `silence` | The filler's *sound* is muted but its slot stays, so caption timestamps and lip-sync don't drift. |
| One mic of a multitrack session | `silence` | Excising one stem de-syncs it from the others; muting keeps every track aligned. |

`silence` mode leans on the room-tone overlay to fill the muted holes with the
natural floor — keep room tone on (the default). Details:
[render-pipeline.md → Part 2](render-pipeline.md#part-2--mode-remove-vs-silence).

## Decision: which `--denoise`?

**Stay on `hybrid` (the default) unless you have a specific reason not to.** It
detects fillers on the original audio (full sensitivity) but cuts from a denoised
copy (clean splices) — the best of both.

- `none` — your audio is already clean, or you denoise in another tool and don't
  want `erm` touching the noise floor.
- `post` — you want full detection sensitivity but would rather denoise the
  finished output as one pass instead of cutting from a denoised copy.
- `pre` — rarely the right call: denoising *before* detection flattens the
  energy/pitch cues the acoustic detectors rely on, so it misses real fillers.

Why these tradeoffs exist:
[denoise-and-room-tone.md](denoise-and-room-tone.md).

## Decision: detection aggressiveness

Three knobs move sensitivity together. The defaults are tuned to catch a lot
without trimming real speech:

- `--model` — the biggest lever. `medium.en` (default) is a good balance;
  `large-v3` catches more fillers and gives tighter word boundaries (slower);
  `small.en` is faster but coarser.
- `--detect-gaps` (on) — runs the three audio detectors that catch fillers
  Whisper drops or fuses into a word. Turning it off leaves only the word-list
  match.
- `--confirm-pitch` (on) — the safety net: it makes the aggressive
  overlong/intra-word detectors prove a cut looks like a sustained filler vowel
  before trimming. **Keep it on** unless trailing drawls are slipping through and
  you've accepted the risk of clipping slow speech.

What each detector does and what governs it: [detection.md](detection.md).

## The iterate loop

Rendering is the slow part (transcription + ffmpeg). Tuning by re-rendering is
painful — so don't. Iterate on the **dry run**, which does everything except
render:

```sh
erm input.wav --dry-run
```

This prints (and writes) the cut-list JSON without producing audio. Inspect it,
adjust flags, re-run `--dry-run`, and only render once you're happy.

### Reading the cut list

The JSON's top-level fields tell you the shape of the edit: `mode`,
`time_saved_s` (net seconds removed in `remove` mode), `injected_gap_s` (silence
added by `--min-gap-ms`), and `muted_s` (in `silence` mode). The interesting part
for tuning is the **per-cut label** — it tells you *which detector* fired, which
in turn tells you *which knob* governs that cut:

| Label in JSON | Detector | If this cut is wrong, reach for… |
|---------------|----------|----------------------------------|
| the literal word (`"um"`, `"uhhh"`) | word-list match | `--fillers` |
| `<gap>` | gap filler (voiced energy in a silence) | `--gap-min-ms`, `--gap-min-voiced-ms` / `--gap-max-voiced-ms` |
| `<in:WORD>` | intra-word (filler fused inside `WORD`) | `--intraword-min-ms`, `--confirm-pitch` |
| `<long:WORD>` | overlong (`WORD` ran much longer than its text) | `--confirm-pitch` |

So if a real word keeps getting trimmed, find its cut in the JSON, read the
label, and you know exactly which detector to rein in. The mechanism behind each
label is in [detection.md](detection.md).

### Confirm the final render

Once rendered, validate the output against the source:

```sh
erm validate input.wav input-cleaned-*.wav --cuts input-cuts-*.json
```

This runs three deterministic checks — container sanity, duration math (per
mode), and a re-transcribe that asserts no filler survived. See the README's
[`validate`](../README.md#validate-subcommand) section. It's the cheap
confidence check before you ship the file.

## Where to go next

- A symptom to fix → [troubleshooting.md](troubleshooting.md)
- A ready-made command for a common job → [recipes.md](recipes.md)
- The mechanism behind a knob → the internals docs ([detection](detection.md),
  [render pipeline](render-pipeline.md),
  [denoise & room tone](denoise-and-room-tone.md),
  [transcription](transcription.md))
