# Recipes: command lines for common jobs

Copy-paste starting points for the situations that come up most. Each one notes
*why* those flags. For the reasoning behind a setting, follow the links into
[usage.md](usage.md) and the internals docs; for full flag defaults, see the
[README](https://github.com/dougcalobrisi/erm#readme).

Output and cut-list paths are auto-named next to the input unless you pass `-o` /
`--json`, so the commands below omit them.

## Quick clean — start here

```sh
erm input.wav
```

The default pipeline: `remove` mode, `hybrid` denoise, room tone on, all
detectors on. Most recordings need nothing else.

## Inspect before committing

```sh
erm input.wav --dry-run
```

Prints the cut list (and writes the JSON) without rendering — the fast way to see
*what* would be cut and tune flags before paying for a render. This is the core
of the [iterate loop](usage.md#the-iterate-loop).

## Conversational podcast that keeps its breathing room

```sh
erm input.wav --pad-pause-factor 0.5 --min-gap-ms 100
```

Removes the filler but doesn't slam the neighboring words together: `--pad-pause-factor`
keeps half of the pause that was already inside the cut, and `--min-gap-ms`
guarantees at least 100 ms between the flanking words. Natural-sounding for
speech. How the two differ:
[render-pipeline.md → Part 3](render-pipeline.md#part-3--two-distinct-spacing-knobs-dont-conflate).

## Video / screencast that must keep caption & A/V timing

```sh
erm input.wav --mode silence
```

Mutes each filler in place instead of excising it, so the output is the exact
same length — caption timestamps and lip-sync don't drift. Leave room tone on; it
fills the muted holes with the natural floor. To render the **picture** too (not
just the audio), add `--video` — see [working with video](video.md). Background:
[usage.md → which mode](usage.md#decision-which---mode).

## Multitrack stem you'll re-mix against other tracks

```sh
erm mic2.wav --mode silence
```

Same reason: excising one stem would de-sync it from the other mics. `silence`
keeps every track frame-aligned.

## Noisy room

```sh
erm input.wav --denoise-nr 18 --denoise-nf -30
```

Keeps the default `hybrid` denoise but pushes it harder (`--denoise-nr` is
reduction strength in dB, `--denoise-nf` the noise floor). If the floor still
shifts at edits, lower `--room-tone-level-db` toward `-18`. Mechanism:
[denoise-and-room-tone.md](denoise-and-room-tone.md).

## Already-clean studio capture

```sh
erm input.wav --denoise none --no-room-tone
```

Minimal-touch pass: no denoising, no added floor. Use only when the recording is
genuinely clean — without room tone, splices and any `silence`/`min-gap` holes
get no floor fill.

## Fastest possible pass

```sh
erm input.wav --model small.en --device cuda --compute-type int8
```

Smaller model + GPU + 8-bit compute. Noticeably faster; slightly coarser word
boundaries and filler coverage. Drop `--device cuda` if you don't have the CUDA
runtime installed (see [transcription.md](transcription.md)).

## Maximum filler coverage

```sh
erm input.wav --gap-min-ms 250
```

`large-v3` (the default) already catches the most fillers and lands the tightest
boundaries; the lower gap threshold scans shorter pauses for dropped fillers.
Slower, more thorough.

## Custom filler vocabulary

Three flags shape pass 1's word list. They compose in order — `--fillers`
defines the set, `--add-fillers` unions on top, `--remove-fillers` subtracts
(removal wins):

```sh
# Replace the built-in list entirely.
erm input.wav --fillers "um,uh,er,like"

# Keep the defaults and add your own verbal tics — the common case.
erm input.wav --add-fillers "basically,like,you-know"

# Keep the defaults but drop one that over-matches your voice.
erm input.wav --remove-fillers "ah"
```

Reach for `--add-fillers` instead of `--fillers` when you just want the
defaults plus a few words: `--fillers` makes you re-type the whole built-in
list, and forgetting a stem silently drops it.

Caveat: automatic elongation matching (`ummmm` → `um`) only applies to the
**built-in** stems — a custom word like `basically` matches verbatim only. See
[detection.md → pass 1](detection.md#pass-1--word-list-match-fillerspy).

---

Got a result that's *wrong* rather than a job to set up? Use
[troubleshooting.md](troubleshooting.md).
