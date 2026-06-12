# Render modes & pause spacing

This documents the `--mode {remove,silence}` switch and the two `remove`-mode
spacing knobs (`--pad-pause-factor`, `--min-gap-ms`) added in 0.3.0. Detection,
refinement, denoise, and room-tone are shared across modes — only the
post-`cuts` render differs.

## Why

The original pipeline had exactly one render strategy: detect fillers, excise
their spans, and splice the survivors with crossfades. Two needs weren't met:

1. **Preserve timing.** Excising shifts everything after the first cut earlier,
   which breaks A/V sync, multi-track alignment (one mic can't be excised
   without de-syncing the others), and caption/transcript timestamps.
2. **Splice rhythm.** Excising a filler plus the small surrounding silence that
   `refine_boundaries` snaps into the cut can make tight splices feel clipped;
   and because `acrossfade` overlaps the surviving segments, flanking words can
   end up *tighter* than the original.

All three additions default to the exact prior behavior, so existing usage —
and the rendered `.wav` bytes — are unchanged.

## Mode: `remove` vs `silence`

- **`remove`** (default): `invert_to_keep_ranges` + `render` (crossfade
  splices). The timeline shrinks by the cut total (minus crossfade overlap,
  plus any injected min-gap silence).
- **`silence`**: each cut span is muted in place via a single ffmpeg `volume`
  pass (`_mute_filter` → `render_silenced`). Duration is preserved exactly. Cuts
  are already refined onto silence/zero-crossings, so binary gating is
  click-free.

### The silence ↔ room-tone floor dependency

A muted hole is digital zero. On its own that's an audible drop-out against the
recording's noise floor. The existing room-tone overlay (on by default) lays a
constant sample of the recording's own room tone under the whole output,
filling the muted holes with the natural floor — the same mechanism that masks
splice discontinuities in `remove` mode. `silence` mode therefore *relies* on a
floor being present. Denoising can't substitute: it only *reduces* signal, so it
never backfills a zeroed hole. Room tone is the only thing that restores a floor,
so `erm` warns whenever `--mode silence` is combined with `--no-room-tone`,
regardless of the `--denoise` setting.

## The 1:1 refine invariant padding relies on

`refine_boundaries` emits **one cut per input cut, in input order** — including
the collapse path (`e_sample <= s_sample`) that re-appends the original. So the
refined list is positionally 1:1 with the `raw_cuts` passed in. `pad_cuts` uses
this to find each cut's voiced core (the raw boundary) versus the silence the
refiner snapped over (the refined boundary) without threading any extra state:

```
left_silence  = max(0, raw.start - refined.start)
right_silence = max(0, refined.end - raw.end)
```

`pad_cuts` defends the invariant anyway: if the two lists aren't the same length
it returns the refined list unchanged, and if padding would collapse/invert a
cut it leaves that cut unpadded (so the filler is always removed). Padding is
applied **before** `merge_close_cuts`, while the lists are still aligned.

## Two distinct spacing knobs (don't conflate)

- **Proportional padding (`--pad-pause-factor`)** retains a *fraction* of the
  silence already inside a cut. Per side: `min(silence, clamp(factor * silence,
  pad_min, pad_max))`. Context-aware, **never adds time** (capped by the silence
  that exists), so a tight mid-sentence "um" with no surrounding silence gets ~0
  padding. `factor = 0` (default) ⇒ the whole cut is removed.
- **Minimum-gap floor (`--min-gap-ms`)** *guarantees* ≥ N ms between the two
  words flanking a cut, **injecting** silence at the splice when the natural
  pause is below N. It adds a little duration when it engages. `min_gap_ms = 0`
  (default) ⇒ nothing injected.

They compose: `factor` shapes how much existing pause survives; `min-gap` puts a
hard floor under it.

## Min-gap injection mechanism

After `invert_to_keep_ranges`, `inject_min_gaps` walks each splice between keep
range `i` and `i+1`:

```
prev_word_end   = max word.end  <= keep[i].end     (else keep[i].end)
next_word_start = min word.start >= keep[i+1].start (else keep[i+1].start)
surviving_gap   = (keep[i].end - prev_word_end) + (next_word_start - keep[i+1].start)
if surviving_gap < min_gap_s: inject (min_gap_s - surviving_gap) of silence here
```

It returns an ordered **render timeline** of `("keep", start, end)` items
interleaved with `("gap", 0.0, duration)` items. The CLI converts that into the
`gap_inserts` list (`(after_keep_index, duration)`) that `render` consumes.

`render` builds the injected path as a **linear fold**: each keep becomes an
`atrim`; each injected gap becomes an `anullsrc` matched to the input's sample
rate and channel layout (so `concat` joins it without resampling the real
audio). Keep→keep joins reuse the existing per-splice `acrossfade` (or `concat`
when that fade would be zero); any join touching a gap uses `concat`, which
makes the injected duration exact. Injected silence is bare silence, **not**
room tone — the room-tone overlay fills it with the natural floor afterward,
exactly like the `silence`-mode holes.

The default render path is gated behind `if gap_inserts or (min_gap_s > 0 and
len(keep_ranges) > 1)` and is otherwise **untouched** — when no gap is injected
*and* no floor is set (every existing caller and every default run), the verbatim
original code runs, producing byte-identical output.

The injected `anullsrc` needs an unambiguous `channel_layout` name to match the
real audio, so min-gap injection supports **mono/stereo input only**
(`gap_channel_layout`). The CLI probes the input up front and rejects anything
else with a clean error before the (slow) transcribe pass, rather than failing
at the final render step.

### Honoring the floor on gapless joins too

A `concat` join lands the injected silence exactly, but a gapless `acrossfade`
join *overlaps* the survivors by `fade`, eating that much out of the silence
between the flanking words — so a splice whose natural pause was just above the
floor could finish a few ms under it. `_keep_fades` closes this: whenever a
floor is set it caps each surviving fade at `surviving_gap - min_gap_s`, where
`surviving_gap = lhs_room + rhs_room` is the same per-side silence it already
measures for the word-protection clamp (and the same quantity `inject_min_gaps`
compares against). The two enforcement paths therefore agree — splices *below*
the floor get silence **injected** (`concat`, exact), splices *just above* it
get their crossfade **trimmed** — so the floor holds at every splice, not only
the injected ones. Because the floor (`min_gap_s > 0`) also routes the render
through the gap-aware per-join path, a fade trimmed to zero degrades to a single
`concat` for that one join instead of disabling crossfades everywhere.

## Cut-list JSON & validation

The cut list gains two fields:

- `"mode"`: `"remove"` or `"silence"`.
- `"injected_gap_s"`: total injected min-gap silence (`0.0` unless
  `--min-gap-ms` engaged).

In `remove` mode `time_saved_s` becomes the **net** `saved - injected_gap_s`. In
`silence` mode `time_saved_s` is `0.0` and a `"muted_s"` total is added. All
other fields keep their prior values, so a default run's `time_saved_s` still
equals the raw cut total.

`validate_output` reads `mode` and `injected_gap_s` (defaulting to `"remove"` /
`0.0` when absent, so older cut lists validate unchanged) and applies the
matching duration expectation:

- `remove`: `output ≈ input − sum(cut lengths) + injected_gap_s`.
- `silence`: `output ≈ input`.

The assumed mode is surfaced in the `duration_math` check detail.
