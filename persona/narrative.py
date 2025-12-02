"""Narrative event tracking and persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Sequence

from .llm import LLMNarrativeSuggestion


@dataclass
class NarrativeEvent:
    id: str
    created_at: datetime
    title: str
    summary: str
    tone: str | None = None
    tags: tuple[str, ...] = ()
    speaker_id: str | None = None
    related_ids: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() + "Z",
            "title": self.title,
            "summary": self.summary,
            "tone": self.tone,
            "tags": list(self.tags),
            "speaker_id": self.speaker_id,
            "related_ids": list(self.related_ids),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "NarrativeEvent":
        created_at_raw = payload.get("created_at") or payload.get("timestamp")
        created_at = (
            datetime.fromisoformat(created_at_raw.replace("Z", ""))
            if isinstance(created_at_raw, str)
            else datetime.utcnow()
        )
        return cls(
            id=str(payload.get("id")),
            created_at=created_at,
            title=str(payload.get("title", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            tone=(str(payload.get("tone")) if payload.get("tone") not in (None, "") else None),
            tags=tuple(str(tag).strip() for tag in payload.get("tags", []) if str(tag).strip()),
            speaker_id=(
                str(payload.get("speaker_id"))
                if payload.get("speaker_id") not in (None, "")
                else None
            ),
            related_ids=tuple(
                str(rel).strip() for rel in payload.get("related_ids", []) if str(rel).strip()
            ),
        )


def _normalize_tags(tags: Sequence[str] | None) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for tag in tags or ():
        value = str(tag).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def make_event(
    suggestion: LLMNarrativeSuggestion,
    *,
    tags: Sequence[str] | None = None,
    speaker_id: str | None = None,
    created_at: datetime | None = None,
) -> NarrativeEvent:
    if not (suggestion.title or suggestion.summary):
        raise ValueError("Narrative suggestions require a title or summary")
    timestamp = created_at or datetime.utcnow()
    event_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    title = (suggestion.title or suggestion.summary or "").strip() or "Narrative"
    summary = (suggestion.summary or suggestion.title or "").strip()
    tone = suggestion.tone.strip() if suggestion.tone else None
    return NarrativeEvent(
        id=event_id,
        created_at=timestamp,
        title=title,
        summary=summary,
        tone=tone,
        tags=_normalize_tags(tags),
        speaker_id=str(speaker_id) if speaker_id is not None else None,
    )


def format_narrative_context(events: Sequence[NarrativeEvent]) -> list[str]:
    context: list[str] = []
    for event in events:
        descriptors: list[str] = []
        if event.tone:
            descriptors.append(f"tone={event.tone}")
        if event.tags:
            descriptors.append("tags=" + ", ".join(event.tags))
        if event.related_ids:
            descriptors.append("linked_to=" + ", ".join(event.related_ids))
        descriptor_text = f" ({'; '.join(descriptors)})" if descriptors else ""
        context.append(f"Narrative — {event.title}: {event.summary}{descriptor_text}")
    return context


class NarrativeStore:
    """Stores narrative events in memory with optional JSONL persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._events: list[NarrativeEvent] = []
        if self.path and self.path.exists():
            self._load()

    @property
    def events(self) -> tuple[NarrativeEvent, ...]:
        return tuple(self._events)

    def add_event(self, event: NarrativeEvent) -> NarrativeEvent:
        linked_ids: set[str] = set()
        for existing in self._events:
            if not existing.tags or not event.tags:
                continue
            if set(existing.tags) & set(event.tags):
                linked_ids.add(existing.id)
                updated_related = set(existing.related_ids)
                updated_related.add(event.id)
                self._update_event_links(existing.id, updated_related)
        linked_event = replace(event, related_ids=tuple(sorted(linked_ids)))
        self._events.append(linked_event)
        self._events.sort(key=lambda item: item.created_at, reverse=True)
        self._persist()
        return linked_event

    def recent_events(
        self, *, speaker_id: str | None = None, limit: int = 3
    ) -> list[NarrativeEvent]:
        if limit <= 0:
            return []
        filtered: list[NarrativeEvent] = []
        for event in self._events:
            if speaker_id is None:
                filtered.append(event)
                continue
            if event.speaker_id is None or event.speaker_id == str(speaker_id):
                filtered.append(event)
        return filtered[:limit]

    def format_context(
        self, *, speaker_id: str | None = None, limit: int = 3
    ) -> list[str]:
        return format_narrative_context(self.recent_events(speaker_id=speaker_id, limit=limit))

    def _update_event_links(self, event_id: str, related: Iterable[str]) -> None:
        for idx, existing in enumerate(self._events):
            if existing.id != event_id:
                continue
            self._events[idx] = replace(
                existing, related_ids=tuple(sorted({rid for rid in related if rid != event_id}))
            )
            break

    def _load(self) -> None:
        if not self.path:
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                event = NarrativeEvent.from_json(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            self._events.append(event)
        self._events.sort(key=lambda item: item.created_at, reverse=True)

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(event.to_json()) for event in self._events]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
