---
name: erm-tune
description: >-
  Diagnose and tune erm's output quality. Use when an erm run is imperfect or
  the user wants to adjust settings — fillers still audible, real words clipped,
  splices click or sound smeared/blurry, noise floor pumps or is audible, words
  run together, detection too aggressive or missing fillers, or questions about
  crossfade, pause spacing, denoise, room tone, models, or detection thresholds.
  For first-time install or a basic clean, use the erm skill instead.
allowed-tools: Bash
---

# erm — tune and troubleshoot

`erm` exposes ~40 flags that cluster into five knob groups. Tune by **symptom**,
change **one cluster at a time**, and re-check with `--dry-run` + `validate`.

## Resolving documentation

Resolve detail in this order (broadest compatibility last):

1. **`erm --help`** — definitive flag names, defaults, and units.
2. **Public docs:** https://dougcalobrisi.github.io/erm/ — `troubleshooting`,
   `detection`, `render-pipeline`, `denoise-and-room-tone`.
3. **Bundled docs** (Claude plugin only): `${CLAUDE_PLUGIN_ROOT}/docs/*.md`;
   flag defaults in `${CLAUDE_PLUGIN_ROOT}/src/erm/cli.py`.

Never guess values — read one of the above before recommending a setting.

## 1. Diagnose first — ask what's wrong

Use AskUserQuestion to pin the symptom (each maps to a different cluster), unless
the user already described it:

- Fillers still audible / missed → **detection**
- Real words clipped or chopped → **detection** (too aggressive) / **refinement**
- Splices click, pop, or sound smeared/blurry → **crossfade** / **refinement**
- Noise floor pumps, audible level changes at edits → **denoise / room tone**
- Words run together with no breath → **splice spacing**
- Too slow → **detection** (`--model`/`--device`)

Then read the **troubleshooting** doc for the symptom→knob fix.

## 2. The five knob clusters

Read the linked doc page for good-value ranges before changing anything.

1. **Detection aggressiveness** (what gets cut) — `--model` (biggest lever),
   `--detect-gaps`, `--confirm-pitch`, `--gap-min-ms`, `--gap-min-voiced-ms`,
   `--gap-max-voiced-ms`, `--intraword-min-ms`, `--fillers`. → `detection` doc.
2. **Refinement / merge** (clean splice points) — `--search-ms`,
   `--merge-gap-ms`. → `render-pipeline` doc.
3. **Splice spacing** (remove mode breathing room) — `--pad-pause-factor`,
   `--pad-min-ms`, `--pad-max-ms`, `--min-gap-ms`. → `render-pipeline` doc.
4. **Crossfade** (splice smoothness) — `--crossfade-factor`,
   `--min-crossfade-ms`, `--max-crossfade-ms`, `--crossfade-ms`. → `render-pipeline` doc.
5. **Denoise / room tone** (uniform floor) — `--denoise none|pre|post|hybrid`,
   `--denoise-nr`, `--denoise-nf`, `--room-tone`/`--no-room-tone`,
   `--room-tone-level-db`, `--room-tone-source`. → `denoise-and-room-tone` doc.

## 3. Iterate safely

1. Tune **detection** against the cut list first: `erm IN.wav --dry-run` and
   inspect the JSON — cheaper than re-rendering audio.
2. Change **one cluster**, re-render, and `erm validate IN.wav OUT.wav`.
3. Compare against the previous output before changing anything else.

### Optional: parallel A/B (enhancement)

When several settings are plausible, you may render a few variants with different
knob values in parallel, `validate` each, and compare — instead of serial
guess-and-check. Keep variants labeled by the flag that changed.
