# Detection: the four-pass filler pipeline

This documents how `erm` decides *what* to cut — the four detectors that run
before refinement and rendering. The README's "How it works" step 2 summarizes
them; this is the why-and-how for maintainers. The flags named here
(`--fillers`, `--detect-gaps`, `--gap-min-ms`, `--gap-min-voiced-ms`,
`--gap-max-voiced-ms`, `--intraword-min-ms`, `--confirm-pitch`) are documented
for end users in the README's Detection table.

## Why four passes

Whisper is the starting point, not the answer. It emits word-level timestamps,
but it was trained to produce *readable* transcripts, so it disposes of
disfluencies in three different ways — and each way needs a different detector:

1. It transcribes the filler as a token (`"um"`, `"uhhhh"`). → **word-list match**
2. It drops the filler entirely, leaving an unexplained silent gap between two
   real words. → **gap detector**
3. It folds the filler into a neighboring word's timestamp, either as an
   interior run separated by a breath (`"in, uhhhh"` → one `'in'` token) or as a
   seamless trailing vowel with no breath at all. → **intra-word** and
   **overlong** detectors

The four detectors run in `cli.py:_cmd_remove` (lines 293–333), each emits a
list of `Cut` spans, and the union is sorted by start time into `raw_cuts`
(`cli.py:333`). Overlaps are harmless — `invert_to_keep_ranges` merges them
later (see [render-pipeline.md](render-pipeline.md)).

## The shared acoustic substrate

The three audio detectors (everything except the word-list match) share one
primitive: a frame-based RMS energy envelope, `_rms_envelope` in
`envelope.py:12`. Frames are non-overlapping `win_ms` (10 ms) windows; the
return is `(envelope, hop_samples)`. For the *why* behind RMS framing and the
dB-relative threshold below, see
[Concepts → RMS energy envelope](concepts.md#rms-energy-envelope).

"Voiced" is defined relative to the recording's own loudness, not an absolute
dB value:

```
peak      = envelope.max()
threshold = peak * 10**(silence_floor_db / 20)   # silence_floor_db = -40 dB
```

So a frame counts as voiced if it's within 40 dB of the loudest frame in the
file (`detect.py:124`, `:254`, `:303`). This auto-scales to quiet and loud
recordings alike — there are no hand-tuned per-detector thresholds.

`_voiced_runs_in_region` (`detect.py:36`) walks the envelope inside a time
window and returns contiguous voiced runs whose length is in
`[min_s, max_s]`. Its one subtlety is **dip bridging**: a sub-threshold dip
shorter than `bridge_frames` (80 ms, `detect.py:255`, `:318`) does *not* end a
run. A single drawn-out "ummmm" flickers in amplitude; without bridging it would
fragment into several tiny runs and either fall below `min_voiced_ms` or splice
badly. Bridging keeps one filler as one run.

## Pass 1 — word-list match (`fillers.py`)

`find_fillers` (`fillers.py:52`) keeps any word whose normalized text
(`normalize_word` — lowercase, strip surrounding punctuation, keep internal
hyphens) `is_filler` against the `--fillers` set. The set defaults to
`DEFAULT_FILLERS` (`fillers.py:14`): `um, uh, er, erm, ah, hmm, mhm, mm,
uh-huh`.

The non-obvious part is **elongations**. `is_filler` (`fillers.py:40`) first
checks set membership, then falls back to a per-stem regex in
`_ELONGATION_PATTERNS` (`fillers.py:20`). `um` is backed by `^u+m+$`, so
`um`/`umm`/`ummmm` all match; `uh` by `^u+h+$`; etc. This is why you can't get
elongation coverage for a *custom* filler you pass via `--fillers` unless a
stem pattern exists for it — a word with no pattern only matches verbatim.

The word list is composed from three flags (`cli.py:_resolve_filler_set`):
`--fillers` defines the set (defaulting to `DEFAULT_FILLERS`), `--add-fillers`
unions words on top, and `--remove-fillers` subtracts — removal applied last, so
it wins over additions. Prefer `--add-fillers "basically"` /
`--remove-fillers "ah"` to adjust the defaults; `--fillers` replaces the whole
set and means re-typing every stem.

This pass is purely textual; it never touches the audio.

## Pass 2 — gap fillers (`detect.py:detect_gap_fillers`)

Scans the silent gaps *between* transcribed words for voiced energy Whisper
didn't account for. Only gaps `>= min_gap_s` (`--gap-min-ms`, default 350 ms)
are examined (`detect.py:314`) — a shorter pause can't plausibly hide a filler.

Crucially, it only scans gaps *between* words. Leading silence before the first
word and trailing silence after the last word are intro/outro, not fillers, and
are skipped (`detect.py:308–316`). Each gap is handed to `_voiced_runs_in_region`
with the shared threshold; surviving runs become `<gap>`-labeled cuts.

Boundaries here are deliberately loose — the downstream `refine_boundaries` pass
tightens every cut to the actual silence edges, so the gap detector only has to
find *roughly* where the filler is.

## Pass 3 — intra-word fillers (`detect.py:detect_intraword_fillers`)

Targets words at least `min_word_s` long (`--intraword-min-ms`, default 550 ms,
`detect.py:129`) and splits each one's interior into voiced runs separated by
silence dips of at least `min_dip_ms` (50 ms, `detect.py:125`). The model of a
suspect word is:

```
[real word][dip][filler]                     — one filler
[real word][dip][filler][dip][filler]        — multiple fillers
[real word][dip][syllable][dip][syllable]    — legitimately long word (leave alone)
```

A word with a single voiced run (no interior dip) is left alone outright
(`detect.py:160`) — that's how `"misunderstandings"` survives.

Three guards keep this from eating real speech:

- **Sum-of-runs guard** (`detect.py:163–170`). If the total voiced time across
  all runs is `<= expected_max_word_duration(text) * 1.2`, the runs are just
  natural phoneme structure (e.g. `"sharing"` splitting `shar`/`ring` across a
  breath), not word + filler. Skip.
- **Pitch confirmation** (`detect.py:175–181`, when `--confirm-pitch` on).
  Classifies each run with `is_sustained_vowel` (below) and only considers the
  *non-vowel* runs as candidate "real word" anchors — a sustained vowel is
  filler, not the word.
- **Structural anomaly** (`detect.py:188–201`). A real word doesn't contain
  200 ms+ of interior silence. If one is found, every run *before* that big gap
  is discarded as a candidate real-word anchor — Whisper's start boundary has
  probably engulfed a *leading* filler.

After the guards, the run whose duration is closest to the word's expected
duration is chosen as the real word (`_score`, `detect.py:205–212`, with a ×4
penalty for runs more than 1.8× expected so an overlong run never wins), and
every *other* run that meets the voiced-length bounds becomes an `<in:WORD>`
cut.

## Pass 4 — overlong words (`detect.py:detect_overlong_words`)

The intra-word detector needs an interior silence dip to split on. When a filler
flows continuously out of a word with no breath, there's no dip — so this pass
uses *duration* instead.

`expected_max_word_duration` (`detect.py:20`) is a deliberately generous upper
bound on how long a clean utterance should take: `0.18 + 0.12 * n` seconds for
`n` normalized characters (empty → 0.40). So `a` → 0.30 s, `and` → 0.54 s,
`session` → 1.02 s. A word longer than `expected * excess_factor` (1.6,
`detect.py:261`) is suspect; its trailing portion from `start + expected` to
`end` is scanned by `_voiced_runs_in_region` for `<long:WORD>` cuts.

This pass is the most aggressive, so it's the most gated by pitch confirmation.

## Pitch confirmation (`acoustic.py:is_sustained_vowel`)

`--confirm-pitch` (default on) guards the two acoustic detectors that infer
fillers from word *shape* (intra-word and overlong), where the risk of trimming
slow-but-real speech is highest. It does **not** gate the word-list or gap
passes — a transcribed `"um"` or a voiced blob in a 400 ms silent gap needs no
acoustic second opinion. For the overlong pass the confirmation is applied in
the CLI itself (`cli.py:322–326`), not inside the detector.

`is_sustained_vowel` (`acoustic.py:8`) returns true when a region looks like a
held filler vowel, using two librosa features:

- **Spectral-centroid stability.** A held vowel keeps its formants in roughly
  one place, so the spectral centroid barely moves. The coefficient of variation
  (std/mean) must be `<= max_centroid_cv` (0.18). Real speech moves its
  articulators and fails this.
- **Voiced fraction.** At least `min_voiced_frac` (0.50) of frames must have a
  zero-crossing rate in `(0.02, 0.20)` — the voiced-speech band, excluding
  silence (too few crossings) and fricatives/noise (too many).

Regions shorter than 60 ms return false outright (`acoustic.py:35`) — too short
to measure. librosa is lazy-imported here because it's heavy.

## Labels & flow

Each detector tags its cuts so the cut-list JSON shows provenance: the raw word
text (pass 1), `<gap>` (pass 2), `<in:WORD>` (pass 3), `<long:WORD>` (pass 4).
The combined `raw_cuts` list then flows into refinement and rendering, covered
in [render-pipeline.md](render-pipeline.md). The original audio those detectors
see depends on `--denoise`; see
[denoise-and-room-tone.md](denoise-and-room-tone.md).
