"""Denoising helpers for incoming audio streams."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .audio import AudioFrame


class Denoiser(Protocol):
    """Protocol describing denoising behaviour for streaming audio."""

    def reset(self) -> None:
        """Reset any internal state so a new stream can be processed."""

    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        """Return a denoised frame, potentially identical to the input."""


class NoOpDenoiser:
    """A pass-through denoiser that leaves audio untouched."""

    def reset(self) -> None:  # pragma: no cover - trivial
        return None

    def process_frame(self, frame: AudioFrame) -> AudioFrame:  # pragma: no cover - trivial
        return frame


class RNNoiseDenoiser:
    """RNNoise wrapper that performs lightweight denoising per frame."""

    def __init__(self, *, sample_rate: int = 48000, channels: int = 1) -> None:
        try:
            import rnnoise  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("rnnoise backend is not available") from exc

        self._rnnoise = rnnoise.RNNoise()
        self.sample_rate = sample_rate
        self.channels = channels

    def reset(self) -> None:  # pragma: no cover - stateless wrapper
        return None

    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        if frame.sample_rate != self.sample_rate or frame.channels != self.channels:
            return frame

        if len(frame.pcm) % 2 != 0:
            raise ValueError("RNNoise expects 16-bit PCM payloads")

        samples = np.frombuffer(frame.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        denoised = self._rnnoise.process_frame(samples)
        denoised_pcm = np.clip(denoised * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        return AudioFrame(
            pcm=denoised_pcm,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            timestamp=frame.timestamp,
        )

