# Transcription: Whisper, the verbatim prompt, and device fallback

Transcription is the front of the pipeline: `faster-whisper` produces the
word-level timestamps every detector builds on. The logic lives in
`asr.py:transcribe` (`asr.py:34`). The README's "Transcription device" section
covers the *user-facing* install/GPU story; this is the maintainer's view of the
three non-obvious choices in this module.

## Model, device, compute type

`transcribe(path, model_name="large-v3", verbatim=True, device="auto",
compute_type="auto")` maps directly to the CLI's `--model`, `--device`, and
`--compute-type`. `faster_whisper.WhisperModel` is **lazy-imported** inside the
function (`asr.py:51`) — it's heavy, and the pure-logic test suite must import
the rest of the package without paying for it (see the README's Tests section).

`large-v3` is the default for best accuracy; `medium.en` is a faster balance and
`small.en` faster still. The model choice matters for detection quality:
a larger model produces tighter word boundaries, which directly improves the
intra-word and overlong detectors (they reason about word duration). `device` and
`compute_type` are passed straight through to `WhisperModel`.

## The verbatim prompt (`asr.py:VERBATIM_PROMPT`)

Whisper was trained to produce *readable* transcripts, so by default it silently
cleans up disfluencies — which would leave the word-list detector
([detection.md](detection.md), pass 1) with nothing to match. Two settings on the
`model.transcribe` call (`asr.py:57–62`) bias it the other way:

- `initial_prompt=VERBATIM_PROMPT` — a short primer (`"Um, uh, er, erm, ah, hmm.
  … Verbatim transcription including all filler words and disfluencies."`) that
  conditions the model to *emit* fillers as tokens.
- `condition_on_previous_text=False` — otherwise each segment is conditioned on
  the model's own prior output, which dilutes the prompt's influence as
  transcription proceeds.

Together these make the cheap, exact word-list detector viable; the three
acoustic detectors are the safety net for fillers Whisper still drops or fuses
despite the prompt.

## CUDA → CPU fallback (`asr._is_recoverable_cuda_error`)

The common failure mode for `faster-whisper` on a GPU box is a machine with an
NVIDIA GPU and driver but **no CUDA runtime wheels** (`libcublas`, `libcudnn`).
With `device="auto"` the backend picks the GPU, then raises a bare `RuntimeError`
— and it does so **lazily, on the first `encode()`**, not at model construction
(`asr.py:64–65`). That's why the entire segment iteration is wrapped inside the
`_run` closure: the error can't be caught around the constructor alone.

The recovery (`asr.py:77–91`):

- `_run(device)` is attempted first.
- On `RuntimeError`, the message is matched against `_RECOVERABLE_CUDA_MARKERS`
  (`"cublas"`, `"cudnn"`, `"cuda"`, `asr.py:26`). These substrings cover the
  observed cublas/cudnn load failures and most other GPU-side CUDA errors
  (driver too old, OOM).
- The retry on CPU happens **only when `device == "auto"`** — i.e. the user let
  `erm` choose. It prints a warning to stderr and re-runs on CPU.
- An explicit `--device cuda` is treated as a demand: the real error is
  re-raised, never silently downgraded.

So `auto` always completes (GPU when it works, CPU otherwise, with a warning),
`cpu` skips the GPU probe entirely, and `cuda` surfaces failures. The README's
"Transcription device" section documents how to install the CUDA wheels to make
the GPU path actually load.
