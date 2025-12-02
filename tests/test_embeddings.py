from __future__ import annotations

from persona.embeddings import EmbeddingMetadata, InMemoryEmbeddingIndex
from persona.memory import MemoryLogger


class DummyEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            vectors.append(
                [
                    float(len(text)),
                    float(text.count("mission")),
                    float(text.count("plan")),
                ]
            )
        return vectors


def test_in_memory_index_boosts_importance_and_speaker() -> None:
    index = InMemoryEmbeddingIndex(normalize=False)
    base_vector = [1.0, 1.0, 0.0]
    index.add(
        vector=base_vector,
        metadata=EmbeddingMetadata(
            key="a",
            summary="General mission",
            speaker_id=None,
            importance=0.2,
        ),
    )
    index.add(
        vector=[value * 0.9 for value in base_vector],
        metadata=EmbeddingMetadata(
            key="b",
            summary="Speaker mission",
            speaker_id="42",
            importance=0.8,
        ),
    )

    matches = index.search(query=base_vector, limit=2, speaker_id="42")
    assert matches
    assert matches[0].metadata.key == "b"
    assert matches[0].score > matches[1].score


def test_memory_logger_recalls_embeddings(tmp_path) -> None:
    embedder = DummyEmbedder()
    index = InMemoryEmbeddingIndex(normalize=False)
    logger = MemoryLogger(tmp_path, embedder=embedder, embedding_index=index)

    logger.log(
        transcript="We agreed on the mission plan",
        response="Acknowledged",
        emotion="focused",
        importance=0.9,
        summary="Mission planning",
        speaker_id="7",
    )
    logger.log(
        transcript="Casual chat about music",
        response="Sounds good",
        emotion="happy",
        importance=0.4,
        summary="Music talk",
        speaker_id="5",
    )

    recalled = logger.recall_for_prompt(
        transcript="Tell me about the mission",
        limit=2,
        min_importance=0.3,
        speaker_id="7",
    )
    assert recalled
    assert any("Mission planning" in item for item in recalled)
    assert recalled[0].startswith("Mission planning")
