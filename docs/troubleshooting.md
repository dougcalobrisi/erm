# Troubleshooting: symptom → knob

Describe the bad result, find the fix. Each entry names the likely cause, the
knob(s) to turn, and which way. For the mechanism behind a fix, follow the link;
for the tuning workflow that makes diagnosis cheap, see
[usage.md → the iterate loop](usage.md#the-iterate-loop).

**Diagnose on a `--dry-run` first.** Most "wrong cut" problems are visible in the
cut-list JSON without rendering — and the per-cut **label** tells you which
detector fired (see the [label table](usage.md#reading-the-cut-list)), which
narrows the fix immediately.

## Fillers still audible / not removed

The detector never flagged it, or it flagged it too conservatively. Check the
`--dry-run` JSON: is there a cut near the filler at all?

- **No cut there** → detection missed it. The default `--model large-v3` is
  already the most accurate; lower `--gap-min-ms` (default 350) so shorter pauses
  get scanned; make sure `--detect-gaps` is on.
- **A cut there but too short** → it's being trimmed conservatively. For trailing
  "uhhh" drawls that survive, `--no-confirm-pitch` lets the overlong detector cut
  without acoustic confirmation — but it raises the risk of clipping slow real
  speech, so use it with care. Background: [detection.md](detection.md).

## Real words getting clipped or removed

Detection is too aggressive. Read the label on the offending cut:

- `<long:WORD>` or `<in:WORD>` → keep `--confirm-pitch` **on** (it exists to
  protect real speech); raise `--intraword-min-ms` (default 550) so shorter words
  aren't scanned internally.
- `<gap>` → raise `--gap-min-ms` so brief natural pauses aren't treated as
  fillers.
- a literal word → your `--fillers` list is too broad; remove that word.

See [detection.md → the guards](detection.md#pass-3--intra-word-fillers-detectpydetect_intraword_fillers).

## Splices sound clipped / too abrupt

The cut left no breathing room and the neighboring words butt together. Raise
`--pad-pause-factor` (try 0.3–0.5) to keep a fraction of the cut's own silence,
and/or set `--min-gap-ms` (try 80–120) to guarantee a minimum gap. These two
differ — [render-pipeline.md → Part 3](render-pipeline.md#part-3--two-distinct-spacing-knobs-dont-conflate).

## Words run together right after a cut

Set a `--min-gap-ms` floor — it injects silence at any splice whose natural pause
fell below the floor. [render-pipeline.md → Part 4](render-pipeline.md#part-4--min-gap-injection-mechanism).

## Splices sound smeared / blurry

The crossfade is too long for the material. Lower `--crossfade-factor` (default
0.15) or `--max-crossfade-ms` (default 120). [render-pipeline.md → Part 1](render-pipeline.md#part-1--cuts-to-splice-points).

## Audible click or pop at a splice

An endpoint didn't land on silence / a zero crossing. Raise `--search-ms`
(default 60) to give the refiner more room to find a local energy minimum, and
keep room tone on to mask the seam. [render-pipeline.md → refinement](render-pipeline.md#refinement-refinepyrefine_boundaries).

## Noise floor "pumps" or drops out at edits

The floor isn't uniform across splices. Keep `--room-tone` on (it lays a constant
floor everywhere); adjust `--room-tone-level-db` (default −12; −12 to −20 is the
usual range); or switch to `--denoise post` to smooth the whole output's floor in
one pass. [denoise-and-room-tone.md](denoise-and-room-tone.md).

## `silence`-mode output has dead-sounding holes

The muted spans are bare digital zero because room tone is off. Re-enable
`--room-tone` — `silence` mode depends on it to fill the holes with the natural
floor (this is the warning `erm` prints). [render-pipeline.md → the silence ↔ room-tone dependency](render-pipeline.md#the-silence--room-tone-floor-dependency).

## `--min-gap-ms` errors on a multichannel file

Min-gap injection mints silence that must match the channel layout, and only
mono/stereo have an unambiguous one. Downmix the input to stereo first, or drop
`--min-gap-ms`. [render-pipeline.md → Part 4](render-pipeline.md#part-4--min-gap-injection-mechanism).

## GPU / CUDA error or warning

`erm` picked the GPU but the CUDA runtime libraries aren't installed. With
`--device auto` (default) it warns and falls back to CPU automatically. To
silence it, pass `--device cpu`; to actually use the GPU, install the CUDA wheels
(README's [Transcription device](https://github.com/dougcalobrisi/erm#transcription-device-gpu-vs-cpu)).
Details: [transcription.md → CUDA fallback](transcription.md#cuda--cpu-fallback-asr_is_recoverable_cuda_error).

## Too slow

Use a smaller `--model` (`small.en`); add `--device cuda` if you have the CUDA
runtime; set `--compute-type int8` for faster, lower-precision inference. See the
[fastest-pass recipe](recipes.md#fastest-possible-pass).

## Output over- or under-denoised

Tune the denoiser: `--denoise-nr` (reduction strength dB, default 12) up for more
aggressive cleanup, or `--denoise-nf` (noise floor dB, default −25). If denoising
is doing more harm than good, switch `--denoise` mode (`none`/`post`). See
[usage.md → which denoise](usage.md#decision-which---denoise).
