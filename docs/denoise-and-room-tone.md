# Denoise & room tone: the noise-floor layers

`erm` touches the recording's background noise in two independent places: an
optional `afftdn` denoise pass (`--denoise`), and a room-tone overlay
(`--room-tone`, on by default). They solve opposite halves of the same problem —
denoising *removes* floor, room tone *adds a uniform one back* — and they're
designed to be used together. The README has the user-facing tables; this is the
why for maintainers.

## Denoise: one filter, four routings (`cli.py:_cmd_remove`)

The denoiser itself is trivial — `denoise_to` (`ffmpeg_ops.py:71`) runs ffmpeg's
`afftdn=nr=<nr>:nf=<nf>` (`--denoise-nr` 12 dB, `--denoise-nf` −25 dB). The
interesting part is *which copy of the audio gets denoised, and when*. The
pipeline has three distinct audio roles:

- **`analysis_input`** — what `transcribe` and the audio detectors see.
- **`render_input`** — what ffmpeg actually cuts from.
- the **final output** — optionally denoised at the very end.

`--denoise` picks how those three are wired (`cli.py:263–285`, `:411`,
`:452–460`):

| Mode | analysis | render | output | Why |
|------|----------|--------|--------|-----|
| `none` | original | original | as-rendered | No denoising at all. |
| `pre` | denoised | denoised | denoised | Cleanest splices, but **detection is weaker** — denoising flattens the very energy/pitch signals the acoustic detectors rely on. |
| `post` | original | original | denoised | Full detection sensitivity; the per-splice noise-floor mismatch is smoothed *after* rendering. |
| `hybrid` (default) | original | denoised | denoised | Full-sensitivity detection on the original **and** clean cuts from the denoised copy. |

### Why `hybrid` is the default

The acoustic detectors ([detection.md](detection.md)) find fillers by reading
the RMS energy envelope and spectral centroid. `afftdn` smooths exactly those
features, so detecting *on* denoised audio misses real fillers — that's the cost
of `pre`. But splicing the *original* leaves a noticeable noise-floor jump at
each seam. `hybrid` gets both: it detects on the pristine original (so the
detectors see every cue) but cuts from the denoised copy (so the splices are
clean). The detector timestamps line up because denoising doesn't move audio in
time — it only attenuates noise.

`post` is the alternative when you'd rather not denoise the kept speech at all
until the end (it renders from the original, then denoises the whole output as
one pass — `cli.py:452–460`, writing through an intermediate `raw` file).

## Room tone: a uniform floor under everything

### Finding the sample (`audio.py:find_quiet_region`)

Room tone is a short loop of the recording's *own* background noise. The cleanest
source is the **pre-roll** — the gap before the first transcribed word, which has
no speaker activity. `find_quiet_region` (`audio.py:20`) builds two candidates
(pre-roll, then post-roll after the last word), trims 50 ms off each edge to
avoid clipping speech, and returns the first that yields a window of at least
`min_length_s` (0.4 s), capped at `max_length_s` (1.5 s). If neither is long
enough it returns `None` and the CLI skips the overlay with a notice
(`cli.py:469–471`).

`--room-tone-source` overrides `auto` with an explicit `START-END` seconds spec
(`_parse_room_tone_source`, `cli.py:180`), which rejects negative and
non-increasing ranges so a typo fails cleanly instead of confusing ffmpeg later.

### The overlay (`ffmpeg_ops.py:overlay_room_tone`)

The sample is looped indefinitely (`-stream_loop -1`), attenuated by
`--room-tone-level-db` (−12 dB; −12 to −20 is usually right), and mixed under the
audio with `amix=inputs=2:duration=first` so the output length matches the main
audio exactly — the tone is truncated, never extends it (`ffmpeg_ops.py:85`).

### The key invariant: always sampled from the original

Room tone is extracted from the **original** input, never the denoised copy
(`cli.py:462–467`, `extract_segment(args.input, …)`). Denoising would strip the
exact ambient character we're trying to reintroduce, so sampling the denoised
version would defeat the purpose. This holds in every denoise mode.

### Room tone's three jobs

The same constant floor does three things at once:

1. **Masks splice discontinuities** in `remove` mode — the residual noise-floor
   mismatch at each seam disappears under a floor that's identical everywhere.
2. **Fills `silence`-mode muted holes** — a muted span is digital zero; room
   tone backfills the natural floor so it doesn't read as a drop-out.
3. **Fills injected min-gap silence** — `--min-gap-ms` injects bare `anullsrc`
   silence, which room tone then covers with the natural floor.

Jobs 2 and 3 are why room tone is *required*, not just nice, when those features
are used: denoising can only reduce signal, so nothing else can backfill a zeroed
hole. `erm` warns when `--mode silence` or `--min-gap-ms` is combined with
`--no-room-tone` (`cli.py:419–424`). The floor-dependency reasoning lives in
full in [render-pipeline.md](render-pipeline.md).
