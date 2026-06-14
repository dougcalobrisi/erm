# Installation

`erm` is a local command-line tool. Nothing leaves your machine — no API keys,
no uploads.

## Requirements

- **Python 3.11+**
- **`ffmpeg` and `ffprobe` on your `PATH`** — `erm` shells out to them for every
  cut, mux, and probe. Install via your package manager (`brew install ffmpeg`,
  `apt install ffmpeg`, …) and confirm with `ffmpeg -version`.

## Run it with `uvx` (recommended)

If you have [`uv`](https://docs.astral.sh/uv/), you don't need to install
anything persistently — `uvx` fetches `erm` into a cached environment and runs
it:

```sh
uvx erm input.wav
```

The first run downloads the package; subsequent runs reuse the cache. This is
the recommended way to run `erm` and the path the bundled AI-agent skills use.

## Install into a virtualenv

Where `uv` isn't available, install the published package
([`erm` on PyPI](https://pypi.org/project/erm/)) into a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install erm
```

Then `erm input.wav` as usual.

## Editable install (development)

To hack on `erm` itself, clone the repo and install it editable with the dev
extras (test + lint tooling):

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Or, with `uv`: `uv sync --extra dev` (also: `make setup`).

## Transcription device (GPU vs CPU)

Transcription runs on **CPU by default** and needs no extra setup. If you have
an NVIDIA GPU, [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) can
use it — but only when the CUDA 12 runtime libraries (`libcublas`, `libcudnn`)
are installed. A machine with an NVIDIA GPU and driver but no CUDA runtime is
the common case that produces:

```
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

`erm` handles this automatically. With the default `--device auto`, if the GPU
can't be loaded it prints a warning and falls back to CPU, so transcription
still completes. Two ways to make it explicit:

- **Force CPU** (no warning, skips the GPU probe):

  ```sh
  erm input.wav --device cpu
  ```

- **Enable the GPU** by installing the CUDA wheels into the same environment:

  ```sh
  pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
  ```

  faster-whisper's CUDA backend needs CUDA 12 / cuDNN 9. See the
  [faster-whisper GPU notes](https://github.com/SYSTRAN/faster-whisper#gpu) for
  details.

## Use inside AI coding agents

`erm` ships agent guidance so an AI assistant can install, run, and tune it for
you. In Claude Code / Cowork:

```
/plugin marketplace add dougcalobrisi/erm
/plugin install erm@erm
```

This adds two skills — **`erm`** (install + clean a file) and **`erm-tune`**
(diagnose a bad result and map the symptom to the right knob). Other agents
(Codex, Copilot, Cursor, Gemini CLI, …) read the repo's
[`AGENTS.md`](https://github.com/dougcalobrisi/erm/blob/main/AGENTS.md) and the
open-format [Agent Skills](https://agentskills.io) in
[`skills/`](https://github.com/dougcalobrisi/erm/tree/main/skills).

## Next steps

- [Tuning & workflow](usage.md) — the `--dry-run` → read-the-cuts → render loop.
- [CLI reference](cli-reference.md) — every flag.
- [Recipes](recipes.md) — copy-paste command lines for common jobs.
