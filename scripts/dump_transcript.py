"""Print every word the model emits, with timestamps and a normalized form.

Useful for diagnosing why find_fillers came up empty: maybe the model
transcribed "um" as "uhm", "hmm" as "mhmm", or dropped them altogether.
"""

from __future__ import annotations

import sys

from erm import DEFAULT_FILLERS, normalize_word, transcribe


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: dump_transcript.py <audio> [model]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "small.en"
    words, duration = transcribe(path, model_name=model)
    print(f"# {model}: {len(words)} words in {duration:.2f}s\n")
    for w in words:
        norm = normalize_word(w.text)
        flag = "  <-- FILLER" if norm in DEFAULT_FILLERS else ""
        print(f"{w.start:7.2f} {w.end:7.2f}  {w.text!r:20}  norm={norm!r:15}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
