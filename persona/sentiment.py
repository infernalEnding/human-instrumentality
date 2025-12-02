"""Lightweight sentiment analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SentimentResult:
    label: str
    score: float


class SentimentAnalyzer(Protocol):
    def analyze(self, text: str) -> SentimentResult:
        ...


class LightweightSentimentAnalyzer:
    """Run sentiment analysis with a compact text-classification model."""

    def __init__(
        self,
        model_id: str = "distilbert-base-uncased-finetuned-sst-2-english",
        *,
        pipeline_factory=None,
    ) -> None:
        if pipeline_factory is None:
            from transformers import pipeline

            pipeline_factory = pipeline

        self._pipeline = pipeline_factory(
            "sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
        )

    def analyze(self, text: str) -> SentimentResult:
        if not text.strip():
            return SentimentResult(label="neutral", score=0.0)
        try:
            raw = self._pipeline(text[:512])[0]
            label = str(raw.get("label", "neutral")).lower()
            score = float(raw.get("score", 0.0))
        except Exception:
            label = "neutral"
            score = 0.0
        return SentimentResult(label=label, score=score)
