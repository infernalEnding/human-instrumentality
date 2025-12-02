"""Voice activity detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

import numpy as np

from .audio import AudioBuffer, AudioFrame, AudioSegment


@dataclass
class VADDecision:
    segment: AudioSegment
    confidence: float
    start_time: float | None = None
    end_time: float | None = None


class VoiceActivityDetector(Protocol):
    """Protocol describing the interface for VAD backends."""

    def reset(self) -> None:  # pragma: no cover - interface only
        ...

    def process_frame(self, frame: AudioFrame) -> List[VADDecision]:  # pragma: no cover - interface only
        ...

    def flush(self) -> List[VADDecision]:  # pragma: no cover - interface only
        ...


class EnergyVAD(VoiceActivityDetector):
    """Simple energy-based VAD for offline testing."""

    def __init__(
        self,
        threshold: float = 0.01,
        min_speech_frames: int = 3,
        max_silence_frames: int = 4,
    ) -> None:
        self.threshold = threshold
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
        energy = frame.rms()
        is_speech = energy >= self.threshold
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


class WebRTCVAD(VoiceActivityDetector):
    """Voice activity detector backed by ``webrtcvad``.

    The implementation performs light-weight buffering to emit segments
    compatible with the :class:`EnergyVAD` interface while delegating the
    speech decisions to the WebRTC algorithm when possible.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        *,
        frame_duration_ms: int = 20,
        min_speech_frames: int = 3,
        max_silence_frames: int = 4,
        energy_fallback_threshold: float = 0.01,
    ) -> None:
        if not 0 <= aggressiveness <= 3:
            raise ValueError("aggressiveness must be between 0 and 3")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be one of 10, 20, or 30")
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.min_speech_frames = min_speech_frames
        self.max_silence_frames = max_silence_frames
        self.energy_fallback_threshold = energy_fallback_threshold
        try:
            import webrtcvad
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "webrtcvad is required for WebRTCVAD; install it via pip"
            ) from exc
        self._vad = webrtcvad.Vad(self.aggressiveness)
        self._buffer = AudioBuffer()
        self._silence_streak = 0
        self._speech_started = False
        self._segment_start: float | None = None

    def reset(self) -> None:
        self._buffer = AudioBuffer()
        self._silence_streak = 0
        self._speech_started = False
        self._segment_start = None

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

    def _is_speech(self, frame: AudioFrame) -> bool:
        mono_pcm = self._prepare_mono_frame(frame)
        if mono_pcm is None:
            return frame.rms() >= self.energy_fallback_threshold
        try:
            return bool(self._vad.is_speech(mono_pcm, frame.sample_rate))
        except Exception:  # pragma: no cover - defensive fallback
            return frame.rms() >= self.energy_fallback_threshold

    def _prepare_mono_frame(self, frame: AudioFrame) -> bytes | None:
        """Return PCM ready for ``webrtcvad`` or ``None`` if unsupported."""

        if frame.sample_rate not in (8000, 16000, 32000, 48000):
            return None

        samples_per_channel = int(
            frame.sample_rate * self.frame_duration_ms / 1000
        )
        expected_length = samples_per_channel * frame.channels
        try:
            samples = np.frombuffer(frame.pcm, dtype=np.int16)
        except ValueError:
            return None
        if samples.size != expected_length:
            return None
        if frame.channels == 1:
            return frame.pcm
        reshaped = samples.reshape((-1, frame.channels))
        mono = reshaped.mean(axis=1).astype(np.int16)
        return mono.tobytes()
