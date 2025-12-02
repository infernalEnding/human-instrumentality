"""Audio-based emotion analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .audio import AudioSegment


@dataclass
class AudioEmotionResult:
    label: str
    score: float


class AudioEmotionAnalyzer(Protocol):
    def analyze(self, segment: AudioSegment) -> AudioEmotionResult:
        ...


class HuggingFaceAudioEmotionAnalyzer:
    """Wrapper around Hugging Face audio classification pipelines."""

    def __init__(
        self,
        model_id: str = "superb/hubert-large-superb-er",
        *,
        pipeline_factory=None,
    ) -> None:
        if pipeline_factory is None:
            from transformers import pipeline

            pipeline_factory = pipeline

        self._pipeline = pipeline_factory(
            "audio-classification",
            model=model_id,
        )

    def analyze(self, segment: AudioSegment) -> AudioEmotionResult:
        if not segment.frames:
            return AudioEmotionResult(label="neutral", score=0.0)

        audio_array = segment.to_numpy()
        if segment.channels > 1:
            audio_array = np.mean(audio_array, axis=1)

        result = self._pipeline(
            {"array": audio_array, "sampling_rate": segment.sample_rate}
        )
        if isinstance(result, list):
            result = result[0]
        label = str(result.get("label", "neutral"))
        score = float(result.get("score", 0.0))
        return AudioEmotionResult(label=label, score=score)
