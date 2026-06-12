"""Tests for the CUDA -> CPU fallback in erm.asr.transcribe.

faster-whisper is mocked via a fake module injected into sys.modules, so these
run without downloading a model or installing the real (heavy) dependency.
"""

from __future__ import annotations

import sys
import types

import pytest

from erm.asr import _is_recoverable_cuda_error, transcribe


@pytest.mark.parametrize(
    "message",
    [
        "Library libcublas.so.12 is not found or cannot be loaded",
        "libcudnn_ops.so.9: cannot open shared object file",
        "CUDA driver version is insufficient",
    ],
)
def test_is_recoverable_cuda_error_matches(message):
    assert _is_recoverable_cuda_error(RuntimeError(message))


def test_is_recoverable_cuda_error_ignores_unrelated():
    assert not _is_recoverable_cuda_error(RuntimeError("ffmpeg returned non-zero exit"))


class _FakeSegment:
    def __init__(self):
        self.words = [types.SimpleNamespace(word=" hello ", start=0.0, end=0.5)]


class _FakeInfo:
    duration = 1.0


def _install_fake_faster_whisper(monkeypatch, *, fail_on):
    """Install a fake faster_whisper whose CUDA encode fails.

    `fail_on` is the device string ("cuda") that raises the libcublas error the
    first time its model is iterated, mimicking faster-whisper's lazy failure.
    """
    constructed_devices: list[str] = []

    class _FakeModel:
        def __init__(self, model_name, device, compute_type):
            constructed_devices.append(device)
            self._device = device

        def transcribe(self, *args, **kwargs):
            def _gen():
                if self._device == fail_on:
                    raise RuntimeError(
                        "Library libcublas.so.12 is not found or cannot be loaded"
                    )
                yield _FakeSegment()

            return _gen(), _FakeInfo()

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return constructed_devices


def test_auto_falls_back_to_cpu_on_cuda_failure(monkeypatch, capsys):
    # "auto" is what faster-whisper resolves to the GPU, so make that the device
    # whose lazy encode raises the libcublas error.
    devices = _install_fake_faster_whisper(monkeypatch, fail_on="auto")

    words, duration = transcribe("x.wav", device="auto")

    assert [w.text for w in words] == ["hello"]
    assert duration == 1.0
    # First tried auto (failed), then retried on cpu.
    assert devices == ["auto", "cpu"]
    assert "falling back to CPU" in capsys.readouterr().err


def test_explicit_cuda_does_not_fall_back(monkeypatch):
    _install_fake_faster_whisper(monkeypatch, fail_on="cuda")
    with pytest.raises(RuntimeError, match="libcublas"):
        transcribe("x.wav", device="cuda")


def test_cpu_path_runs_without_fallback(monkeypatch):
    devices = _install_fake_faster_whisper(monkeypatch, fail_on="cuda")
    words, _ = transcribe("x.wav", device="cpu")
    assert [w.text for w in words] == ["hello"]
    assert devices == ["cpu"]
