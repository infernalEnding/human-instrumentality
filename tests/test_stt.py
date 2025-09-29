from __future__ import annotations

import numpy as np

from persona.audio import AudioFrame, AudioSegment
from persona.stt import HuggingFaceTranscriber


class StubASRPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, payload, **kwargs):
        self.calls += 1
        return {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "score": 0.9},
                {"text": "world", "score": 0.95},
            ],
        }


def test_huggingface_transcriber_averages_scores() -> None:
    pcm = (np.ones(320, dtype=np.int16)).tobytes()
    frame = AudioFrame(pcm=pcm, sample_rate=16000)
    segment = AudioSegment(frames=[frame])
    pipeline = StubASRPipeline()
    transcriber = HuggingFaceTranscriber(
        "fake/model",
        device="cpu",
        pipeline_factory=lambda *_, **__: pipeline,
    )

    result = transcriber.transcribe(segment)
    assert result.text == "hello world"
    assert abs(result.confidence - 0.925) < 1e-6
    assert pipeline.calls == 1
