# AGENTS.md — erm

`erm` is a local CLI that strips disfluencies (`um`, `uh`, `er`, `erm`, `ah`,
`hmm`, `mhm`, `mm`, `uh-huh`, and elongations) from English speech audio. This
file orients any AI coding agent (Codex, Copilot, OpenCode, Cursor, Gemini CLI,
pi.dev, Claude Code, …) working with the tool.

> Full docs: https://dougcalobrisi.github.io/erm/ — source in [`docs/`](docs/index.md).
> Always confirm exact flags with `erm --help`; never guess flag names or defaults.

## Install

Requires **Python 3.11+** and **ffmpeg/ffprobe** on `PATH`.

```sh
pipx install erm      # recommended (isolated)
# or: pip install erm
erm --help            # verify
```

ffmpeg: `brew install ffmpeg` / `apt install ffmpeg` / `choco install ffmpeg`.
Transcription runs on CPU by default; GPU needs CUDA runtime libs (`--device auto`
falls back to CPU).

## Use — the iterate loop

```sh
erm input.wav --dry-run                 # inspect the cut-list JSON, render nothing
erm input.wav                           # write input-cleaned-<timestamp>.wav
erm validate input.wav output.wav       # re-transcribe; assert no fillers survive
```

**Render mode** is the first decision:
- `--mode remove` (default) — excises fillers; timeline shrinks.
- `--mode silence` — mutes in place; **duration preserved** (use for video A/V
  sync and multitrack stems).

See [`docs/recipes.md`](docs/recipes.md) for copy-paste commands (podcast, video,
multitrack, noisy room, fastest, max coverage, custom fillers).

## Tune — five knob clusters

Tune by symptom; change one cluster at a time; re-check with `--dry-run` + `validate`.
See [`docs/troubleshooting.md`](docs/troubleshooting.md) (symptom → knob).

1. **Detection** (what gets cut) — `--model` (biggest lever), `--detect-gaps`,
   `--confirm-pitch`, `--gap-*`, `--intraword-min-ms`, `--fillers`. → [`docs/detection.md`](docs/detection.md)
2. **Refinement** — `--search-ms`, `--merge-gap-ms`. → [`docs/render-pipeline.md`](docs/render-pipeline.md)
3. **Splice spacing** (remove mode) — `--pad-pause-factor`, `--pad-min-ms`,
   `--pad-max-ms`, `--min-gap-ms`. → [`docs/render-pipeline.md`](docs/render-pipeline.md)
4. **Crossfade** — `--crossfade-factor`, `--min-crossfade-ms`, `--max-crossfade-ms`,
   `--crossfade-ms`. → [`docs/render-pipeline.md`](docs/render-pipeline.md)
5. **Denoise / room tone** — `--denoise none|pre|post|hybrid`, `--denoise-nr`,
   `--denoise-nf`, `--room-tone*`. → [`docs/denoise-and-room-tone.md`](docs/denoise-and-room-tone.md)

## Repo orientation (for contributors)

- CLI entry point + all flags: `src/erm/cli.py`
- Default filler list: `src/erm/fillers.py`
- Tests: `pytest` (some marked `slow` need the Whisper model download).
- Packaging: `pyproject.toml`; release via `.github/workflows/release.yml`.

## Bundled Claude Code / Cowork skills

`skills/erm/` and `skills/erm-tune/` are open-format Agent Skills usable beyond
Claude. Drop them into your agent's skills dir: `~/.claude/skills/` (Claude,
OpenCode) or `~/.agents/skills/` (Codex, OpenCode, pi.dev). Codex reads only
`.agents/skills/`.
