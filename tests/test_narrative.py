from datetime import datetime, timedelta
from pathlib import Path

from persona.llm import LLMNarrativeSuggestion
from persona.narrative import NarrativeEvent, NarrativeStore, format_narrative_context, make_event


def test_make_event_builds_narrative_event() -> None:
    suggestion = LLMNarrativeSuggestion(
        title="First Contact", summary="Met Alex at the cafe", tone="warm"
    )
    event = make_event(
        suggestion,
        tags=["coffee", "friendship", "coffee"],
        speaker_id="abc123",
        created_at=datetime(2024, 5, 1, 12, 0, 0),
    )
    assert event.title == suggestion.title
    assert event.summary == suggestion.summary
    assert event.tone == suggestion.tone
    assert event.tags == ("coffee", "friendship")
    assert event.speaker_id == "abc123"
    assert event.id.startswith("20240501")


def test_narrative_store_persists_and_links(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    store = NarrativeStore(path)
    first = store.add_event(
        make_event(
            LLMNarrativeSuggestion(summary="Discussed mission parameters"),
            tags=["mission", "apollo"],
            speaker_id="u1",
        )
    )
    second = store.add_event(
        make_event(
            LLMNarrativeSuggestion(title="Mission update", summary="Apollo timeline set"),
            tags=["apollo", "timeline"],
            speaker_id="u2",
            created_at=first.created_at + timedelta(minutes=5),
        )
    )

    assert second.related_ids == (first.id,)
    reloaded = NarrativeStore(path)
    assert reloaded.events[0].id == second.id
    assert second.id in reloaded.events[1].related_ids


def test_format_narrative_context_includes_descriptors(tmp_path: Path) -> None:
    store = NarrativeStore(tmp_path / "context.jsonl")
    store.add_event(
        make_event(
            LLMNarrativeSuggestion(summary="Visited the beach", tone="nostalgic"),
            tags=["travel", "beach"],
            speaker_id="u1",
        )
    )
    lines = store.format_context(speaker_id="u1", limit=2)
    assert any("Visited the beach" in line for line in lines)
    assert any("tags=travel, beach" in line for line in lines)
    assert any("tone=nostalgic" in line for line in lines)
