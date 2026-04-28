"""erm: strip disfluencies from spoken audio.

The pure-helper modules (`fillers`, `ranges`, `refine`, `envelope`, `models`)
depend only on numpy + stdlib so the unit tests can run without
faster-whisper or librosa installed. Heavy deps (`librosa`,
`faster_whisper`) are imported lazily inside the functions that need them.
"""

__version__ = "0.1.1"

from .acoustic import is_sustained_vowel
from .asr import VERBATIM_PROMPT, transcribe
from .audio import find_quiet_region, load_audio_mono
from .cli import main
from .detect import (
    detect_gap_fillers,
    detect_intraword_fillers,
    detect_overlong_words,
    expected_max_word_duration,
)
from .ffmpeg_ops import (
    denoise_to,
    extract_segment,
    ffprobe_duration,
    overlay_room_tone,
    render,
)
from .fillers import DEFAULT_FILLERS, find_fillers, is_filler, normalize_word
from .models import Cut, Word
from .ranges import invert_to_keep_ranges, merge_close_cuts
from .refine import refine_boundaries
from .validate import validate_output

__all__ = [
    "__version__",
    "Cut",
    "DEFAULT_FILLERS",
    "VERBATIM_PROMPT",
    "Word",
    "denoise_to",
    "detect_gap_fillers",
    "detect_intraword_fillers",
    "detect_overlong_words",
    "expected_max_word_duration",
    "extract_segment",
    "ffprobe_duration",
    "find_fillers",
    "find_quiet_region",
    "invert_to_keep_ranges",
    "is_filler",
    "is_sustained_vowel",
    "load_audio_mono",
    "main",
    "merge_close_cuts",
    "normalize_word",
    "overlay_room_tone",
    "refine_boundaries",
    "render",
    "transcribe",
    "validate_output",
]
