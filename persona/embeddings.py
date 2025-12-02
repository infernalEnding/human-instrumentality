"""Embedding utilities for semantic memory recall."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class EmbeddingMetadata:
    key: str
    summary: str
    speaker_id: str | None
    importance: float
    source: str | None = None


@dataclass
class EmbeddingMatch:
    metadata: EmbeddingMetadata
    score: float


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:  # pragma: no cover - interface
        ...


class EmbeddingIndex(Protocol):
    def add(self, *, vector: Sequence[float], metadata: EmbeddingMetadata) -> None:  # pragma: no cover - interface
        ...

    def search(
        self,
        *,
        query: Sequence[float],
        limit: int = 3,
        speaker_id: str | None = None,
        min_importance: float = 0.0,
    ) -> list[EmbeddingMatch]:  # pragma: no cover - interface
        ...


class InMemoryEmbeddingIndex:
    def __init__(self, *, normalize: bool = True) -> None:
        self._normalize = normalize
        self._vectors: list[list[float]] = []
        self._metadata: list[EmbeddingMetadata] = []

    def add(self, *, vector: Sequence[float], metadata: EmbeddingMetadata) -> None:
        stored = self._normalize_vector(vector) if self._normalize else list(vector)
        self._vectors.append(stored)
        self._metadata.append(metadata)

    def search(
        self,
        *,
        query: Sequence[float],
        limit: int = 3,
        speaker_id: str | None = None,
        min_importance: float = 0.0,
    ) -> list[EmbeddingMatch]:
        if not self._vectors:
            return []
        query_vec = self._normalize_vector(query) if self._normalize else list(query)
        scores: list[EmbeddingMatch] = []
        for stored, metadata in zip(self._vectors, self._metadata):
            if metadata.importance < min_importance:
                continue
            similarity = sum(q * s for q, s in zip(query_vec, stored))
            score = similarity * (1.0 + (metadata.importance * 0.5))
            if speaker_id:
                if metadata.speaker_id == speaker_id:
                    score += 0.2
                elif metadata.speaker_id:
                    score -= 0.05
            scores.append(EmbeddingMatch(metadata=metadata, score=score))

        scores.sort(key=lambda match: match.score, reverse=True)
        return scores[:limit]

    def _normalize_vector(self, vector: Sequence[float]) -> list[float]:
        values = [float(v) for v in vector]
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:
            return [0.0 for _ in values]
        return [v / norm for v in values]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=False)

