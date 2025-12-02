"""Voice activity detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .audio import AudioBuffer, AudioFrame, AudioSegment


@dataclass
class VADDecision:
    segment: AudioSegment
    confidence: float
    start_time: float | None = None
    end_time: float | None = None


class VAD(Protocol):
    """Protocol describing the voice activity detector interface."""

    def reset(self) -> None:
        """Reset any internal state so a new stream can be processed."""

    def process_frame(self, frame: AudioFrame) -> List[VADDecision]:
        """Consume a single frame and emit decisions if utterances complete."""

    def flush(self) -> List[VADDecision]:
        """Emit any buffered segments when the stream ends."""


class _StreamingVAD:
    """Shared streaming segmentation logic for VAD implementations."""

    def __init__(self, *, min_speech_frames: int = 3, max_silence_frames: int = 4) -> None:
        self.min_speech_frames = min_speech_frames
        self.max_silence_frames = max_silence_frames
        self.reset()

    def reset(self) -> None:
        self._buffer = AudioBuffer()
        self._silence_streak = 0
        self._speech_started = False
        self._segment_start: float | None = None

    def detect(self, frames: Iterable[AudioFrame]) -> List[VADDecision]:
        self.reset()
        decisions: List[VADDecision] = []
        for frame in frames:
            decisions.extend(self.process_frame(frame))
        decisions.extend(self.flush())
        return decisions

    def process_frame(self, frame: AudioFrame) -> List[VADDecision]:
        decisions: List[VADDecision] = []
        is_speech = self._is_speech(frame)
        if is_speech:
            if not self._speech_started:
                self._segment_start = frame.timestamp
            self._buffer.append(frame)
            self._speech_started = True
            self._silence_streak = 0
            return decisions

        if not self._speech_started:
            return decisions

        self._silence_streak += 1
        if self._silence_streak >= self.max_silence_frames:
            decision = self._emit_segment()
            if decision:
                decisions.append(decision)
        return decisions

    def flush(self) -> List[VADDecision]:
        decisions: List[VADDecision] = []
        if self._speech_started and len(self._buffer) >= self.min_speech_frames:
            decision = self._emit_segment()
            if decision:
                decisions.append(decision)
        else:
            self._buffer.clear()
            self._speech_started = False
            self._silence_streak = 0
            self._segment_start = None
        return decisions

    def _emit_segment(self) -> VADDecision | None:
        if len(self._buffer) < self.min_speech_frames:
            self._buffer.clear()
            self._speech_started = False
            self._silence_streak = 0
            self._segment_start = None
            return None

        segment = self._buffer.pop_segment()
        if not segment:
            return None

        confidence = min(1.0, max(0.1, len(segment.frames) / 10.0))
        start_time = segment.start_time
        if start_time is None:
            start_time = self._segment_start
        end_time = segment.end_time

        decision = VADDecision(
            segment=segment,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
        )
        self._buffer.clear()
        self._speech_started = False
        self._silence_streak = 0
        self._segment_start = None
        return decision

    def _is_speech(self, frame: AudioFrame) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError


class EnergyVAD(_StreamingVAD):
    """Simple energy-based VAD for offline testing."""

    def __init__(
        self,
        threshold: float = 0.01,
        min_speech_frames: int = 3,
        max_silence_frames: int = 4,
    ) -> None:
        self.threshold = threshold
        super().__init__(
            min_speech_frames=min_speech_frames, max_silence_frames=max_silence_frames
        )

    def _is_speech(self, frame: AudioFrame) -> bool:
        energy = frame.rms()
        return energy >= self.threshold


class WebRTCVAD(_StreamingVAD):
    """WebRTC VAD backend using the upstream C implementation."""

    def __init__(
        self,
        *,
        aggressiveness: int = 2,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        min_speech_frames: int = 3,
        max_silence_frames: int = 4,
    ) -> None:
        try:
            import webrtcvad  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("webrtcvad backend is not available") from exc

        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be one of 10, 20, or 30")
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("Unsupported sample rate for webrtcvad")

        super().__init__(
            min_speech_frames=min_speech_frames, max_silence_frames=max_silence_frames
        )
        self._vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self._expected_bytes = int(sample_rate * (frame_duration_ms / 1000.0)) * 2

    def _is_speech(self, frame: AudioFrame) -> bool:
        if frame.sample_rate != self.sample_rate:
            raise ValueError(
                f"Frame sample rate {frame.sample_rate} does not match VAD setting {self.sample_rate}"
            )
        if frame.channels != 1:
            raise ValueError("WebRTC VAD only supports mono audio")
        if len(frame.pcm) != self._expected_bytes:
            raise ValueError(
                "Frame length must match the configured duration for WebRTC VAD"
            )
        return bool(self._vad.is_speech(frame.pcm, self.sample_rate))
