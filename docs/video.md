# Working with video

`erm` was built for audio, but it handles video inputs too — either pulling the
clean audio out, or rendering a tightened picture that stays in sync. This page
is the user-facing guide; for the A/V-sync mechanism see the internals doc,
[video render & A/V sync](video-render.md).

## Audio-only or keep the picture?

Feed `erm` a video file and **by default you get the cleaned audio only**
(`.wav`) — the fast path when you just want the audio, and identical to how
`erm` has always behaved. Add `--video` to render a synced video output
(container inferred from the input):

| You want… | Run | Result |
|-----------|-----|--------|
| Just the audio from a video | `erm talk.mp4` | `talk-cleaned-*.wav`, no picture |
| A tightened video, fillers cut | `erm talk.mp4 --video` | re-encoded video, crossfaded splices, A/V in sync |
| Same length, captions intact | `erm talk.mp4 --video --mode silence` | picture stream-copied (lossless), fillers muted in place |
| Hard jump cuts | `erm talk.mp4 --video --video-splice cut` | concat splices on both streams |

## Container and codecs

- The output **container is inferred from the input** (mp4→mp4, mov→mov…); an
  `-o` extension always overrides. Naming an `-o` video container *without*
  `--video` is an error — add `--video`, or choose a `.wav` path.
- Audio is stored **losslessly where the container allows** (PCM in mov/mkv/avi).
  mp4 gets AAC 256k, webm gets Opus 160k — there's no universal lossless audio
  codec for those containers.
- Without `--video`, the picture-related flags warn and are ignored.

## Splice style: crossfade vs. cut

`--video-splice crossfade` (the default) dissolves each splice so the join is
soft; `--video-splice cut` makes hard jump cuts on both streams. The audio and
picture always make the **same** choice at every splice, so they can't drift
apart regardless of style.

## Keeping captions and lip-sync aligned

If something downstream depends on the timeline staying the same length —
caption timestamps, lip-sync, a multitrack edit — use `--mode silence`. The
filler's *sound* is muted but its slot stays, and with `--video` the picture is
**stream-copied untouched** (lossless, frame-exact). See
[which `--mode`](usage.md#decision-which---mode) for the full decision.

## Min-gap plays the footage through

With `--video --min-gap-ms N`, the injected pause **plays the removed footage
through** (muted) rather than freezing the frame — the excised disfluency rolls
under the pause instead of a frozen still. Audio and video inject identical,
frame-snapped lengths, so sync holds within ~1 frame. The mechanism is in
[video render & A/V sync → min-gap plays through](video-render.md#min-gap-plays-through-render_video_with_gaps).

## Where to go next

- The full flag list and defaults → the
  [README](https://github.com/dougcalobrisi/erm#readme).
- A copy-paste command for a common job → [recipes.md](recipes.md).
- How sync is guaranteed (CFR, frame-snapped fades, the tail conform) →
  [video render & A/V sync](video-render.md).
