# Video render & A/V sync (`video.py`)

This is the last stage of the pipeline, and it runs only with `--video`. The
audio path that produces the cleaned master is covered in
[render pipeline](render-pipeline.md); for the user-facing flag guide see
[working with video](video.md).

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

The injected pause is `min_gap_s − surviving_pause`, bounded by `--min-gap-ms`
rather than by how much footage was cut at that splice — so an aggressive floor
over a short filler can ask for a longer pause than the removed span holds.
Reading that straight would spill the played-through footage into the *next*
kept fragment (you'd glimpse upcoming content under the pause), so the read is
**capped at the removed span** and clone-padded (the last removed frame freezes)
for any remainder. The gap node stays exactly the injected length — A/V parity
is untouched — but never shows frames belonging to a kept fragment.

## Codec by container (`audio_mux_args`)

The pipeline produces a clean PCM master; the mux preserves it where the
container allows — `-c:a copy` (PCM) into mov/mkv/avi, **AAC 256k** for mp4
(no universal lossless), **Opus 160k** for webm. The picture is `-c:v copy`'d
through the mux (silence mode copies the source untouched; remove mode copies
the already-encoded splice), never re-encoded twice.

**Encoder priming.** mov/mkv/avi copy the PCM master sample-for-sample, so audio
starts at exactly t=0. mp4 (AAC) and webm (Opus) re-encode it, and lossy
encoders prepend priming/pre-skip samples — on the order of ~20 ms for AAC. Both
streams still *start* at t=0 (the picture is rendered from the same timeline with
no leading offset) and the priming sits well inside the one-frame `av_sync`
tolerance, so parity holds; the mp4 path is covered by a real-ffmpeg parity test
(not just a stream-presence check) precisely because it is the one render path
whose audio is not stream-copied.

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

## `--crf` / `--preset` only reach encoders that honor them

`-crf` and `-preset` are gated **independently**, because support doesn't come as
a pair:

- `-crf` (constant quality): the x264/x265 family **plus** `libvpx-vp9`,
  `libaom-av1`, and `libsvtav1`.
- `-preset`: the x264/x265 family **and** `libsvtav1` (numeric speed preset).
  `libvpx-vp9`/`libaom-av1` have no `-preset` (they steer speed via
  `-deadline`/`-cpu-used`), so it must not be passed there.

`_crf_preset_args(vcodec, crf, preset)` consults `_CRF_ENCODERS` and
`_PRESET_ENCODERS` separately and emits only the flag(s) the chosen encoder
accepts — so `--vcodec libvpx-vp9 --crf 30` correctly passes `-crf 30` while
omitting `-preset`, and every render/mux command stays clean regardless of
`--vcodec`. For an encoder that honors neither (`mpeg4`, hardware encoders,
`copy`), both are dropped; when the user *explicitly* set a value that gets
dropped, the CLI prints a warning (rather than letting it vanish), and you should
reach for that encoder's own quality knob instead.

> A single combined allowlist used to gate both flags together, which silently
> dropped `--crf` for VP9/AV1 (encoders that *do* support it) — a user's quality
> setting was ignored with no effect. The split allowlists fix that.
