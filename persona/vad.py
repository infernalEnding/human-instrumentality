"""Voice activity detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .audio import AudioBuffer, AudioFrame, AudioSegment


@dataclass
class VADDecision:
    segment: AudioSegment
    confidence: float
    start_time: float | None = None
    end_time: float | None = None


class EnergyVAD:
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
