"""faster-whisper transcription (lazy-imported)."""

from __future__ import annotations

from pathlib import Path

from .models import Word


VERBATIM_PROMPT = (
    "Um, uh, er, erm, ah, hmm. Like, you know, I mean, sort of. "
    "Verbatim transcription including all filler words and disfluencies."
)


def transcribe(
    path: str | Path,
    model_name: str = "medium.en",
    verbatim: bool = True,
) -> tuple[list[Word], float]:
    """Transcribe `path` with faster-whisper. Returns (words, duration_seconds).

    `verbatim=True` passes an `initial_prompt` that biases Whisper toward
    keeping disfluencies, which it normally cleans up silently.
    """
    from faster_whisper import WhisperModel  # heavy; lazy

    model = WhisperModel(model_name, device="auto", compute_type="auto")
    segments, info = model.transcribe(
        str(path),
        word_timestamps=True,
        initial_prompt=VERBATIM_PROMPT if verbatim else None,
        condition_on_previous_text=False,  # otherwise the prompt gets diluted
    )
    words: list[Word] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            words.append(Word(text=w.word.strip(), start=float(w.start), end=float(w.end)))
    return words, float(info.duration)
