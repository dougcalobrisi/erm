"""faster-whisper transcription (lazy-imported)."""

from __future__ import annotations

import sys
from pathlib import Path

from .models import Word


VERBATIM_PROMPT = (
    "Um, uh, er, erm, ah, hmm. Like, you know, I mean, sort of. "
    "Verbatim transcription including all filler words and disfluencies."
)


# Substrings that mark a CUDA error we can recover from by retrying on CPU.
# faster-whisper (via ctranslate2) raises a bare RuntimeError when `device="auto"`
# picks the GPU but the CUDA wheels (libcublas, libcudnn) aren't installed — a very
# common setup on machines that have an NVIDIA GPU + driver but no CUDA runtime.
# This intentionally also matches other GPU-side CUDA failures (driver too old,
# out of memory): under `device="auto"` the user didn't demand the GPU, so any
# CUDA failure is better handled by silently retrying on CPU than by crashing.
# `cublas`/`cudnn` are substrings of the `lib*` names, and most messages also
# contain `cuda`, so this short list covers the observed failure strings.
_RECOVERABLE_CUDA_MARKERS = ("cublas", "cudnn", "cuda")


def _is_recoverable_cuda_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RECOVERABLE_CUDA_MARKERS)


def transcribe(
    path: str | Path,
    model_name: str = "large-v3",
    verbatim: bool = True,
    device: str = "auto",
    compute_type: str = "auto",
) -> tuple[list[Word], float]:
    """Transcribe `path` with faster-whisper. Returns (words, duration_seconds).

    `verbatim=True` passes an `initial_prompt` that biases Whisper toward
    keeping disfluencies, which it normally cleans up silently.

    `device` is passed straight to faster-whisper ("auto", "cpu", or "cuda").
    When it's "auto" and the CUDA runtime libraries can't be loaded (e.g. an
    NVIDIA GPU is present but the CUDA wheels aren't installed), we transparently
    fall back to CPU instead of crashing.
    """
    from faster_whisper import WhisperModel  # heavy; lazy

    def _run(resolved_device: str) -> tuple[list[Word], float]:
        model = WhisperModel(
            model_name, device=resolved_device, compute_type=compute_type
        )
        segments, info = model.transcribe(
            str(path),
            word_timestamps=True,
            initial_prompt=VERBATIM_PROMPT if verbatim else None,
            condition_on_previous_text=False,  # otherwise the prompt gets diluted
        )
        words: list[Word] = []
        # The CUDA load error surfaces lazily here, on the first encode() — not
        # at model construction — so the whole iteration must stay inside _run.
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                words.append(
                    Word(text=w.word.strip(), start=float(w.start), end=float(w.end))
                )
        return words, float(info.duration)

    try:
        return _run(device)
    except RuntimeError as exc:
        # Only auto-recover when the user let us pick the device. If they asked
        # for "cuda" explicitly, surface the real error.
        if device == "auto" and _is_recoverable_cuda_error(exc):
            print(
                f"warning: GPU transcription failed ({exc}); falling back to CPU. "
                "Pass --device cpu to silence this, or see the README's "
                "\"Transcription device\" section for installing the CUDA "
                "runtime libraries.",
                file=sys.stderr,
            )
            return _run("cpu")
        raise
