"""Audio denoising interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .audio import AudioFrame


class Denoiser(Protocol):
    """Protocol describing a streaming audio denoiser."""

    def reset(self) -> None:  # pragma: no cover - interface only
        ...

    def process_frame(self, frame: AudioFrame) -> AudioFrame:  # pragma: no cover - interface only
        ...


@dataclass(slots=True)
class NoOpDenoiser(Denoiser):
    """Denoiser that returns frames untouched."""

    def reset(self) -> None:  # pragma: no cover - trivial
        return None

    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        return frame


class RNNoiseDenoiser(Denoiser):
    """RNNoise-based denoiser for mono 48 kHz PCM16 frames."""

    def __init__(self) -> None:
        try:
            from rnnoise import RNNoise
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "rnnoise is required for RNNoiseDenoiser; install it via pip"
            ) from exc
        self._rnnoise = RNNoise()

    def reset(self) -> None:  # pragma: no cover - trivial
        return None

    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        if frame.channels != 1 or frame.sample_rate != 48000:
            return frame
        filtered = self._rnnoise.filter(frame.pcm)
        return AudioFrame(
            pcm=filtered,
            sample_rate=frame.sample_rate,
            channels=frame.channels,
            timestamp=frame.timestamp,
        )
