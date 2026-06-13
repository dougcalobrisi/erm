# Render pipeline: boundaries → splices → modes → spacing

Once the four detectors (see [detection.md](detection.md)) have produced a list
of raw cut spans, the render pipeline turns them into audio. In order:

```
raw_cuts → refine_boundaries → pad_cuts → merge_close_cuts
         → invert_to_keep_ranges → inject_min_gaps → render / render_silenced
```

This doc covers that whole layer: boundary refinement, close-cut merging,
crossfade scaling, the two render **modes** (`remove`/`silence`), and the two
`remove`-mode spacing knobs (`--pad-pause-factor`, `--min-gap-ms`). Detection,
denoise, and room tone are shared across modes — only the post-`cuts` render
differs. (This doc absorbs the former `modes-and-padding.md`.)

Everything here defaults to the exact original behavior: with default flags the
rendered `.wav` bytes are unchanged from before any of these knobs existed.

---

# Part 1 — Cuts to splice points

## Refinement (`refine.py:refine_boundaries`)

A detector's cut boundary is approximate — it lands somewhere in the filler, not
necessarily on a clean splice point. Splicing mid-vowel clicks. `refine_boundaries`
snaps each endpoint, in two stages, to a place that splices silently:

1. **Energy minimum** within `±search_ms` (`--search-ms`, 60 ms). The start
   endpoint prefers the *earliest* minimum in its window (the leading edge of the
   silence); the end endpoint prefers the *latest* via `prefer_late=True`
   (`refine.py:81–83`), so the cut extends across the whole silent trough rather
   than stopping at its onset. The end also snaps to the *end* of its frame
   (`(e_frame + 1) * hop`, `refine.py:87`) so it covers the full trailing silent
   frame.
2. **Zero-crossing** within `±zc_search_ms` (5 ms, `refine.py:89–90`). Splicing
   at a zero crossing avoids a step discontinuity (a click) even when the energy
   minimum isn't exactly zero.

### The word-safe clamp

`_allowed_range` (`refine.py:13`) bounds how far an endpoint may slide so the
energy search can't wander into a neighboring real word: if `cut.start` is inside
a word it can't drop below that word's start; if it's in a gap it can't drop
below the preceding word's end (symmetric for `cut.end`). This is why `words` is
threaded into `refine_boundaries`.

### The collapse guard and the 1:1 invariant

If refinement inverts a cut (`e_sample <= s_sample`), the original cut is
re-appended unchanged (`refine.py:95–98`) — a cut is **never lost** to
refinement.

Because of that guard, `refine_boundaries` emits **exactly one cut per input
cut, in input order** — including the collapse path. The refined list is
positionally 1:1 with `raw_cuts`. `pad_cuts` (Part 3) depends on this invariant
to recover how much silence each cut snapped over, by comparing
`refined_cuts[i]` against `raw_cuts[i]` without threading any extra state.

## Merging close cuts (`ranges.py:merge_close_cuts`)

Two cuts separated by a tiny surviving fragment are collapsed into one when the
gap between them is below `--merge-gap-ms` (120 ms). A ~40 ms fragment left
between two cuts gets eaten by the crossfades on both sides and produces an
audible "blurp"; merging avoids it (`ranges.py:10`). The merged cut takes the
union of the spans and a label reflecting both (`a+b`, or just the shared label).

Merging runs **after** padding, while order still matters only for adjacency
(padding needs the 1:1 alignment first; see Part 3).

## Inverting to keep-ranges (`ranges.py:invert_to_keep_ranges`)

`remove` mode renders what *survives*, so the cut list is inverted to its
complement over `[0, duration]` (`ranges.py:32`). Overlapping/out-of-order cuts
are merged and zero-length keeps dropped here — this is what makes overlapping
detector output harmless.

## Crossfade scaling (`ffmpeg_ops.py:_splice_crossfade_s`, `_keep_fades`)

Each `remove`-mode splice gets an equal-power (`tri`) crossfade. The length
**scales with the cut**, not the surrounding words:

```
fade = clamp(min_crossfade_ms, cut_ms * crossfade_factor, max_crossfade_ms)
```

with CLI defaults `--min-crossfade-ms 50`, `--max-crossfade-ms 120`,
`--crossfade-factor 0.15`. The rationale: a longer cut splices across audio that
differs more in pitch/energy and needs a longer fade to mask the seam.
`--crossfade-ms` overrides the scaling with one fixed length (legacy / A/B
testing).

The scaled value is then clamped in layers (`_splice_crossfade_s`,
`ffmpeg_ops.py:144`):

1. **Half-fragment cap** — `min(cf, prev_len/2, next_len/2)`. A fade can't be
   longer than half the audio it has to live in.
2. **Word-protection cap** — `min(cf, 2*lhs_room, 2*rhs_room)`, where `room` is
   the distance from the splice back to the nearest real word on each side
   (measured in `_keep_fades`, `ffmpeg_ops.py:217–226`). A crossfade reaches
   ~half its length into each side, so capping at `2*room` keeps it from
   attenuating a real word. When a side has no word (e.g. a splice past the last
   word) the room falls back to the fragment boundary, imposing nothing beyond
   the half-fragment cap.

`_keep_fades` (`ffmpeg_ops.py:179`) is the shared per-splice fade computation
used by both the default render path and the gap-aware path. It also applies the
min-gap floor trim — see Part 5.

---

# Part 2 — Mode: `remove` vs `silence`

`--mode` chooses how the cuts are applied:

- **`remove`** (default): `invert_to_keep_ranges` + `render` (crossfade splices).
  The timeline shrinks by the cut total (minus crossfade overlap, plus any
  injected min-gap silence).
- **`silence`**: each cut span is muted in place via a single ffmpeg `volume`
  pass (`_mute_filter` → `render_silenced`, `ffmpeg_ops.py:109`, `:123`).
  Duration is preserved exactly. Cuts are already refined onto
  silence/zero-crossings, so binary gating is click-free.

Use `silence` when timing must be preserved — A/V sync, multi-track alignment
(you can't excise one mic without de-syncing the others), or caption/transcript
timestamps. It removes the *sound* of the filler but leaves a hole of the
original length.

### The silence ↔ room-tone floor dependency

A muted hole is digital zero. On its own that's an audible drop-out against the
recording's noise floor. The room-tone overlay (on by default) lays a constant
sample of the recording's own room tone under the whole output, filling the
muted holes with the natural floor — the same mechanism that masks splice
discontinuities in `remove` mode. `silence` mode therefore *relies* on a floor
being present. Denoising can't substitute: it only *reduces* signal, so it never
backfills a zeroed hole. Room tone is the only thing that restores a floor, so
`erm` warns whenever `--mode silence` is combined with `--no-room-tone`,
regardless of the `--denoise` setting (`cli.py:419–421`). See
[denoise-and-room-tone.md](denoise-and-room-tone.md) for the overlay mechanism.

`silence` mode makes no splices, so it ignores `--pad-pause-factor` and
`--min-gap-ms` and warns if you pass them (`cli.py:243–254`).

---

# Part 3 — Two distinct spacing knobs (don't conflate)

Both are `remove`-mode only and compose, but they do different things:

- **Proportional padding (`--pad-pause-factor`)** retains a *fraction* of the
  silence already inside a cut. Per side: `min(silence, clamp(factor * silence,
  pad_min, pad_max))`. Context-aware, **never adds time** (capped by the silence
  that exists), so a tight mid-sentence "um" with no surrounding silence gets ~0
  padding. `factor = 0` (default) ⇒ the whole cut is removed.
- **Minimum-gap floor (`--min-gap-ms`)** *guarantees* ≥ N ms between the two
  words flanking a cut, **injecting** silence at the splice when the natural
  pause is below N. It adds a little duration when it engages. `min_gap_ms = 0`
  (default) ⇒ nothing injected.

`factor` shapes how much existing pause survives; `min-gap` puts a hard floor
under it.

## How padding uses the 1:1 invariant (`ranges.py:pad_cuts`)

`pad_cuts` (`ranges.py:70`) finds each cut's voiced core (the raw boundary)
versus the silence the refiner snapped over (the refined boundary), using the
1:1 alignment from Part 1:

```
left_silence  = max(0, raw.start - refined.start)
right_silence = max(0, refined.end - raw.end)
```

It moves each refined endpoint back toward the voiced core by `clamp(factor *
silence, pad_min, pad_max)`, never exceeding the silence that exists there.

`pad_cuts` defends the invariant anyway: if the two lists aren't the same length
it returns the refined list unchanged, and if padding would collapse/invert a
cut it leaves that cut unpadded (so the filler is always removed). Padding is
applied **before** `merge_close_cuts` (`cli.py:347–351`), while the lists are
still aligned.

---

# Part 4 — Min-gap injection mechanism

After `invert_to_keep_ranges`, `inject_min_gaps` (`ranges.py:116`) walks each
splice between keep range `i` and `i+1`:

```
prev_word_end   = max word.end  <= keep[i].end     (else keep[i].end)
next_word_start = min word.start >= keep[i+1].start (else keep[i+1].start)
surviving_gap   = (keep[i].end - prev_word_end) + (next_word_start - keep[i+1].start)
if surviving_gap < min_gap_s: inject (min_gap_s - surviving_gap) of silence here
```

It returns an ordered **render timeline** of `("keep", start, end)` items
interleaved with `("gap", 0.0, duration)` items. The CLI (`cli.py:364–373`)
converts that into the `gap_inserts` list (`(after_keep_index, duration)`) that
`render` consumes.

`render` (`ffmpeg_ops.py:337`) builds the injected path (`_render_with_gaps`,
`:242`) as a **linear fold**: each keep becomes an `atrim`; each injected gap
becomes an `anullsrc` matched to the input's sample rate and channel layout (so
`concat` joins it without resampling the real audio). Keep→keep joins reuse the
per-splice `acrossfade` (or `concat` when that fade would be zero); any join
touching a gap uses `concat`, which makes the injected duration exact. Injected
silence is bare silence, **not** room tone — the room-tone overlay fills it with
the natural floor afterward, exactly like the `silence`-mode holes.

Because both joins flanking an injected gap (`keep→gap` and `gap→keep`) are
`concat`, a splice that gets a gap injected **loses its crossfade** — the gap
replaces the overlap rather than being faded into. That's fine: cuts are already
refined onto silence/zero-crossings (so the hard `concat` boundary is click-free)
and the room-tone overlay masks the floor across it. A given splice is smoothed
*either* by a crossfade (no injection) *or* separated by injected silence — never
both.

The default render path is gated behind `if gap_inserts or (min_gap_s > 0 and
len(keep_ranges) > 1)` (`ffmpeg_ops.py:374`) and is otherwise **untouched** —
when no gap is injected *and* no floor is set (every existing caller and every
default run), the verbatim original code runs, producing byte-identical output.

The injected `anullsrc` needs an unambiguous `channel_layout` name to match the
real audio, so min-gap injection supports **mono/stereo input only**
(`gap_channel_layout`, `ffmpeg_ops.py:44`). The CLI probes the input up front and
rejects anything else with a clean error *before* the slow transcribe pass
(`cli.py:233–238`), rather than failing at the final render step.

---

# Part 5 — Honoring the floor on gapless joins too

A `concat` join lands the injected silence exactly, but a gapless `acrossfade`
join *overlaps* the survivors by `fade`, eating that much out of the silence
between the flanking words — so a splice whose natural pause was just above the
floor could finish a few ms under it. `_keep_fades` (`ffmpeg_ops.py:235–237`)
closes this: whenever a floor is set it caps each surviving fade at
`surviving_gap - min_gap_s`, where `surviving_gap = lhs_room + rhs_room` is the
same per-side silence it already measures for the word-protection clamp (Part 1)
and the same quantity `inject_min_gaps` compares against.

The two enforcement paths therefore agree — splices *below* the floor get
silence **injected** (`concat`, exact), splices *just above* it get their
crossfade **trimmed** — so the floor holds at every splice, not only the injected
ones. Because the floor (`min_gap_s > 0`) also routes the render through the
gap-aware per-join path, a fade trimmed to zero degrades to a single `concat`
for that one join instead of disabling crossfades everywhere.

---

# Part 6 — Cut-list JSON & validation

The cut list (`cli.py:375–390`) gains two fields:

- `"mode"`: `"remove"` or `"silence"`.
- `"injected_gap_s"`: total injected min-gap silence (`0.0` unless `--min-gap-ms`
  engaged).

In `remove` mode `time_saved_s` becomes the **net** `saved - injected_gap_s`. In
`silence` mode `time_saved_s` is `0.0` and a `"muted_s"` total is added. All
other fields keep their prior values, so a default run's `time_saved_s` still
equals the raw cut total.

`validate_output` reads `mode` and `injected_gap_s` (defaulting to `"remove"` /
`0.0` when absent, so older cut lists validate unchanged) and applies the
matching duration expectation:

- `remove`: `output ≈ input − sum(cut lengths) + injected_gap_s`.
- `silence`: `output ≈ input`.

The assumed mode is surfaced in the `duration_math` check detail. See the
README's `validate` section for the end-user view.

---

# Part 7 — Video render & A/V sync (`video.py`)

The edit timeline (keep-ranges + per-splice fades + injected gaps) is
format-agnostic. With `--video`, `erm` renders the **picture** from that same
timeline and muxes it onto the separately-rendered clean-PCM audio master. The
audio path is untouched for audio-only runs; all video logic lives in
`video.py`.

## Why decoupled render + mux (not one process)

The audio pipeline is multi-pass — splice → optional `afftdn` denoise → optional
room-tone `amix`, each its own ffmpeg run — so it can't share a single
filtergraph with the video. Instead the audio finishes to a temp WAV master and
the video is rendered in one `filter_complex` pass, then `mux_av` combines them.
Sync does **not** rely on muxing tricks; it's built in at the timeline level
(below) plus a final conform.

## Sync by construction: CFR + frame-snapped fades

Two facts collide: `atrim`/`acrossfade` are **sample-accurate**, but video
`trim`/`xfade` land on **frame boundaries**. Three measures keep the streams
together:

1. **Force CFR.** `render_video_keep_ranges` puts `fps=FR` at the head of the
   graph (`FR` from the input's `avg_frame_rate`, VFR-safe). Without this,
   variable-frame-rate input breaks the duration math non-deterministically.
2. **Frame-snapped, shared fades.** `_keep_fades(..., snap_fps=FR)` rounds each
   crossfade to a whole frame, and the CLI passes the *same* list to both the
   audio `acrossfade` (via `render(fades=…)`) and the video `xfade`. Both
   streams therefore shorten by an identical amount at every splice, so the
   `Σkeeps − Σfades (+ Σgaps)` total holds for each. A positive fade is floored
   at **two frames** — a one-frame `xfade` corrupts a chained filtergraph.
3. **Float-cumulative xfade offsets.** Each `xfade` offset is `Oᵢ = (true float
   cumulative length) − dᵢ`, computed from the exact timeline rather than summed
   rounded values, so per-fragment frame-quantization re-aligns at every splice
   instead of accumulating.

`cut` splices `concat` **both** streams (zero fades), so neither overlaps and
neither drifts; the audio is the same hard-cut concat the audio path already
uses when a fade is zero.

**All-or-nothing crossfade.** Both renderers crossfade only when *every* splice
fade is positive (`render`'s `all(cf > 0)`, the video's `not all(d > 0)`); if any
one fade is zero they both fall back to `concat` for the whole stream. So a
single splice whose snapped fade rounds to zero turns the entire render into hard
cuts on both streams. The two-frame floor on positive fades makes a true zero
rare (it needs a fade that rounds to 0 frames outright), and crucially audio and
video make the *same* choice, so A/V parity always holds — but the visual result
can flip from dissolves to jump cuts at that threshold.

## The tail conform

`concat` (the `cut` path) has no fade to absorb the video's frame-quantized cut
points, so its total can sit a frame or two off. `render_video_keep_ranges`
takes a `target_duration` (the audio master's sample-exact length) and appends
`tpad=stop_mode=clone:stop_duration=…,trim=end=target` — clone-padding a short
picture and trimming a long one to exactly the target. The downstream `trim`
caps the stream, so the large `stop_duration` never actually generates more than
`target` worth of frames. Net A/V parity: **≤ 1 frame**, checked by
`validate_output`'s `av_sync` check (`|video_dur − audio_dur| ≤ 1/FR`).

## Min-gap "plays through" (`render_video_with_gaps`)

Mirrors `_render_with_gaps` node-for-node: keep nodes via `trim`, keep→keep
joins `xfade`/`concat`, gap-adjacent joins `concat`. Where the audio injects
`anullsrc` silence, the video injects the **real removed footage** at that
splice (a `trim` of the original starting where the kept fragment ended), muted —
so the excised disfluency rolls under the injected pause instead of the frame
freezing. Injected gap durations are frame-snapped (CLI side) so audio and video
inject identical lengths.

## Codec by container (`audio_mux_args`)

The pipeline produces a clean PCM master; the mux preserves it where the
container allows — `-c:a copy` (PCM) into mov/mkv/avi, **AAC 256k** for mp4
(no universal lossless), **Opus 160k** for webm. The picture is `-c:v copy`'d
through the mux (silence mode copies the source untouched; remove mode copies
the already-encoded splice), never re-encoded twice.

The final mux adds `-shortest`, ending the output when the first stream ends.
In **remove** mode the picture is already conformed to the audio master's exact
length, so this is a no-op safety net; in **silence** mode the picture is
stream-copied at the *source's* video-track duration, which on a real file need
not exactly equal the audio-track duration — `-shortest` clamps that native
mismatch so the A/V-parity guarantee (≤1 frame) holds in silence mode too.

## Pixel format: forced `yuv420p`

The re-encoded picture (remove mode) is forced to `-pix_fmt yuv420p` for maximal
player/container compatibility. A source in 4:2:2 or 4:4:4 is therefore
chroma-subsampled to 4:2:0 on output — a loss beyond `--crf`, but invisible for
the talking-head/screen-recording footage `erm` targets. Silence mode never
re-encodes the picture (`-c:v copy`), so it preserves the source pixel format
exactly.
