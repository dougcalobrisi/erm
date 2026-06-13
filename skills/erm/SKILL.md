---
name: erm
description: >-
  Install and run erm, the local CLI that removes filler words / disfluencies
  (um, uh, er, erm, ah, hmm, mhm, mm, uh-huh and elongations) from spoken-audio
  recordings. Use when the user wants to install or set up erm, clean up a
  recording/podcast/voiceover, strip "ums" and "uhs" from audio, or asks which
  erm command to run. For fixing imperfect output or adjusting knobs, use the
  erm-tune skill instead.
allowed-tools: Bash, Read, AskUserQuestion
---

# erm — install and use

`erm` strips disfluencies from English speech audio. It transcribes with
faster-whisper, runs extra audio-domain detectors for fillers Whisper hides,
and splices with ffmpeg (energy-snapped, crossfaded, room-tone-matched).

## Resolving documentation

When you need authoritative detail, resolve it in this order (each works in more
environments than the last):

1. **`erm --help`** and **`erm validate --help`** — definitive flags and defaults; works once installed.
2. **Public docs:** https://dougcalobrisi.github.io/erm/ — `usage`, `recipes`, `troubleshooting`, etc.
3. **Bundled docs** (Claude Code/Cowork plugin only): `${CLAUDE_PLUGIN_ROOT}/docs/*.md` and the
   source of truth for flag defaults, `${CLAUDE_PLUGIN_ROOT}/src/erm/cli.py`.

Never guess flag names or defaults — read one of the above.

## 1. Install / run

`erm` needs **Python 3.11+** and **ffmpeg/ffprobe** on `PATH`.

1. Check ffmpeg: `ffmpeg -version`. If missing, suggest the OS install
   (`brew install ffmpeg`, `apt install ffmpeg`, `choco install ffmpeg`).
2. Resolve a launcher — **prefer uv** (broadest, no persistent install):
   - **Tier 1 — uvx (preferred).** If `uv --version` succeeds, run erm straight
     from PyPI with `uvx erm …` — no install step; uv fetches and caches the
     environment on first run, so later runs are fast. Pin a version with
     `uvx erm@<version> …` when needed. Verify: `uvx erm --help`.
   - **Tier 2 — venv fallback (no `uv` on PATH).** Create an isolated env and
     install from PyPI:
     ```sh
     python3 -m venv .venv
     source .venv/bin/activate
     pip install erm
     erm --help   # verify
     ```

**Launcher convention.** In the commands throughout this skill, `erm` means the
launcher you resolved above: prefix with `uvx ` under tier 1
(e.g. `uvx erm INPUT.wav --dry-run`), or use plain `erm` after activating the
venv under tier 2.

Transcription runs on CPU by default (no setup). GPU is optional and needs the
CUDA runtime libs; `--device auto` falls back to CPU. Add the CUDA wheels to the
same environment — `uvx --with nvidia-cublas-cu12 --with nvidia-cudnn-cu12 erm …`
under tier 1, or `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` into the venv
under tier 2. See the `transcription` docs page for details.

## 2. Ask before choosing a command

`erm`'s behavior forks on a couple of choices. Use AskUserQuestion to settle
these **only when they aren't already clear** from the request, then proceed:

- **What kind of audio?** podcast/interview · video (caption-timed or A/V sync) ·
  multitrack stem · already-clean studio. This selects the recipe.
- **Render mode?** `--mode remove` (default — excises fillers, timeline shrinks)
  vs `--mode silence` (mutes in place, **duration preserved** — required for
  video sync and multitrack stems).

If the user already implied the answers (e.g. "clean my podcast"), don't ask —
pick the sensible default and say what you chose.

Then read the **recipes** doc and use the matching copy-paste command.

## 3. Core workflow (the iterate loop)

1. **Inspect first:** `erm INPUT.wav --dry-run` — prints/writes the cut-list JSON
   (`*-cuts-*.json`); renders nothing. Review what it intends to cut.
2. **Render:** `erm INPUT.wav` — writes `INPUT-cleaned-<timestamp>.wav` next to the input.
3. **Validate:** `erm validate INPUT.wav OUTPUT.wav` — re-transcribes the output and
   asserts no fillers survive, plus container/duration sanity. Exit 0 = pass.

Useful flags (confirm with `erm --help`): `-o/--output`, `--json`, `--model`,
`--device`, `--fillers`. The full `usage` doc explains the workflow in depth.

**Adjusting the word list.** If the user wants to strip an extra word (e.g.
"also remove 'basically' / 'like'"), prefer `--add-fillers "basically,like"` —
it keeps the built-in defaults and unions the new words on top. Use
`--remove-fillers WORD` to drop a default that over-matches their voice. Reach
for `--fillers` only to replace the whole set, since it requires re-typing every
stem. Custom words match verbatim (no automatic elongation). See the
`recipes` doc → "Custom filler vocabulary".

## 4. When results aren't perfect

If fillers remain, real words get clipped, splices click/smear, the noise floor
pumps, or words run together — hand off to the **erm-tune** skill, which maps
each symptom to the right knob.
