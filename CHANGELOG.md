# Changelog

All notable changes to `erm` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-12

Render modes and pause spacing. Every new behavior is off by default — a default
run renders byte-identical audio to 0.2.x, and the only cut-list change is two
additive fields.

### Added

- **`--mode {remove,silence}`** — choose how detected cuts are applied.
  - `remove` (default): excise each cut and splice the survivors with
    crossfades (the existing behavior; the timeline shrinks).
  - `silence`: mute each cut span in place with a single ffmpeg `volume` pass,
    preserving the input's exact duration. Use it to keep A/V sync, multi-track
    alignment, and caption/transcript timestamps intact. The room-tone overlay
    fills the muted holes with the natural floor.
- **`--pad-pause-factor`** (with **`--pad-min-ms`** / **`--pad-max-ms`**) —
  *remove mode.* Retain a fraction of the silence each cut snapped over so tight
  splices keep a little breathing room. Context-aware and never adds time;
  default `0.0` removes the whole cut.
- **`--min-gap-ms`** — *remove mode.* Guarantee at least N ms between the two
  words flanking a splice, injecting silence when the natural pause is shorter.
  Default `0.0` injects nothing; injected silence is filled by room tone.
- Cut-list JSON fields `mode` and `injected_gap_s`; `silence` mode also reports
  `muted_s`.
- Public API: `render_silenced`, `pad_cuts`, `inject_min_gaps` (exported from
  `erm`).
- `docs/modes-and-padding.md` design note and a README **Modes** section.
- This `CHANGELOG.md`.

### Changed

- `erm validate` now checks output duration per mode, reading `mode` /
  `injected_gap_s` from the cut list — `silence`: `output ≈ input`; `remove`:
  `output ≈ input − cuts + injected_gap_s`. Cut lists without these fields
  default to `remove` / `0.0` and validate exactly as before.
- `erm` warns when muted holes or injected gaps would be bare digital silence
  rather than a natural floor (`--mode silence` with `--no-room-tone --denoise
  none`, or `--min-gap-ms` with `--no-room-tone`).

## [0.2.0] - 2026-06-12

### Fixed

- Transcription gracefully falls back to CPU when the CUDA runtime libraries
  can't be loaded. `--device auto` (default) probes the GPU and falls back with
  a warning; `--device cpu` skips the probe entirely. (#3)

### Changed

- Expanded the test suite with CLI, integration, and pure-helper coverage. (#4)

## [0.1.1] - 2026-04-28

### Added

- Release tooling: `Makefile`, build/publish scripts, and GitHub Actions
  workflow. (#1)

## [0.1.0] - 2026-04-28

### Added

- Initial release. `erm` strips disfluencies (`um`, `uh`, `er`, `erm`, `ah`,
  `hmm`, `mhm`, `mm`, `uh-huh`, plus any-length elongations) from English
  speech using `faster-whisper` word timestamps, three audio-domain detectors
  for fillers Whisper hides, and ffmpeg for the cuts.
- Energy-minimum + zero-crossing boundary refinement, cut-size-scaled
  crossfades, optional denoising (`none` / `pre` / `post` / `hybrid`), and a
  looped room-tone undertone for a uniform noise floor.
- `erm validate` subcommand: container sanity, duration math, and a
  no-filler-survives invariant.

[Unreleased]: https://github.com/dougcalobrisi/erm/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/dougcalobrisi/erm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/dougcalobrisi/erm/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/dougcalobrisi/erm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/dougcalobrisi/erm/releases/tag/v0.1.0
