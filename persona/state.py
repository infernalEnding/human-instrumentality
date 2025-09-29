"""Persona state management for medium-term memories and relationships."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


@dataclass
class PersonaRelationship:
    """Serializable record describing a known individual."""

    name: str
    relationship: str | None = None
    last_interaction: str | None = None
    notes: str | None = None
    discord_ids: tuple[str, ...] = ()
    sentiment: str | None = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "relationship": self.relationship,
            "last_interaction": self.last_interaction,
            "notes": self.notes,
            "discord_ids": list(self.discord_ids),
            "sentiment": self.sentiment,
        }


class PersonaStateManager:
    """Persisted persona profile that augments the live memory logger."""

    def __init__(
        self,
        state_path: Path | str,
        *,
        persona_name: str = "Astra",
        max_medium_term: int = 25,
    ) -> None:
        self.path = Path(state_path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.persona_name = persona_name
        self.max_medium_term = max(1, max_medium_term)
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    def _default_state(self) -> Dict[str, Any]:
        return {
            "persona_name": self.persona_name,
            "medium_term": [],
            "hobbies": [],
            "artistic_likes": [],
            "relationships": {},
        }

    def _load_state(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._default_state()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self._default_state()
        if not isinstance(data, MutableMapping):
            return self._default_state()
        # Ensure required keys exist
        for key, default in self._default_state().items():
            if key not in data:
                data[key] = default
        data["medium_term"] = [
            self._normalise_medium_term(entry)
            for entry in data.get("medium_term", [])
            if isinstance(entry, Mapping)
        ]
        data["hobbies"] = sorted(
            {str(item).strip() for item in data.get("hobbies", []) if str(item).strip()}
        )
        data["artistic_likes"] = sorted(
            {
                str(item).strip()
                for item in data.get("artistic_likes", [])
                if str(item).strip()
            }
        )
        relationships: Dict[str, Any] = {}
        for rel in data.get("relationships", {}).values():
            if not isinstance(rel, Mapping):
                continue
            record = self._normalise_relationship(rel)
            relationships[self._relationship_key(record.name)] = record.to_payload()
        data["relationships"] = relationships
        return data

    def _save_state(self) -> None:
        serialisable = {
            "persona_name": self.persona_name,
            "medium_term": self._state["medium_term"],
            "hobbies": self._state["hobbies"],
            "artistic_likes": self._state["artistic_likes"],
            "relationships": self._state["relationships"],
        }
        self.path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    def get_state(self) -> Dict[str, Any]:
        """Return a deep-ish copy suitable for inspection or serialization."""

        return json.loads(json.dumps(self._state))

    def prompt_context(
        self,
        *,
        max_medium_term: int = 5,
        max_relationships: int = 5,
    ) -> List[str]:
        lines: List[str] = []
        medium_entries = sorted(
            self._state["medium_term"],
            key=lambda item: (item.get("importance", 0.0), item.get("last_updated", "")),
            reverse=True,
        )[: max(0, max_medium_term)]
        if medium_entries:
            lines.append("Medium-term memories:")
            for entry in medium_entries:
                lines.append(
                    f"- {entry.get('summary')} (importance {entry.get('importance', 0.0):.2f})"
                )
        if self._state["hobbies"]:
            lines.append("Hobbies: " + ", ".join(self._state["hobbies"]))
        if self._state["artistic_likes"]:
            lines.append("Artistic likes: " + ", ".join(self._state["artistic_likes"]))
        relationship_values = list(self._state["relationships"].values())
        if relationship_values:
            relationship_values.sort(
                key=lambda rel: (
                    rel.get("last_interaction") or "",
                    rel.get("relationship") or "",
                    rel.get("name") or "",
                ),
                reverse=True,
            )
            lines.append("Key relationships:")
            for rel in relationship_values[: max(0, max_relationships)]:
                descriptor = rel.get("relationship") or "connection"
                note = rel.get("notes") or ""
                mention = f"- {rel.get('name')}: {descriptor}"
                if note:
                    mention += f" (notes: {note})"
                lines.append(mention)
        return lines

    def apply_updates(self, updates: Mapping[str, Any]) -> bool:
        """Merge LLM-provided updates into the persona state."""

        if not isinstance(updates, Mapping):
            return False
        changed = False
        if "medium_term" in updates:
            changed |= self._update_medium_term(updates.get("medium_term"))
        if "hobbies" in updates:
            changed |= self._update_string_set("hobbies", updates.get("hobbies"))
        if "artistic_likes" in updates:
            changed |= self._update_string_set("artistic_likes", updates.get("artistic_likes"))
        if "relationships" in updates:
            changed |= self._update_relationships(updates.get("relationships"))
        if changed:
            self._save_state()
        return changed

    # ------------------------------------------------------------------
    # Normalisation helpers
    def _normalise_medium_term(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        summary = str(entry.get("summary", "")).strip()
        importance = self._clamp_float(entry.get("importance", 0.5))
        timestamp = entry.get("last_updated") or entry.get("timestamp")
        if not timestamp:
            timestamp = datetime.utcnow().isoformat() + "Z"
        return {
            "summary": summary or "Untitled reflection",
            "importance": importance,
            "last_updated": str(timestamp),
        }

    def _normalise_relationship(self, entry: Mapping[str, Any]) -> PersonaRelationship:
        name = str(entry.get("name", "")).strip()
        discord_ids = entry.get("discord_ids") or entry.get("discord_id")
        if isinstance(discord_ids, str):
            discord_values = {discord_ids.strip()} if discord_ids.strip() else set()
        else:
            discord_values = {
                str(value).strip()
                for value in (discord_ids or [])
                if str(value).strip()
            }
        return PersonaRelationship(
            name=name or "Unknown contact",
            relationship=(str(entry.get("relationship")) if entry.get("relationship") else None),
            last_interaction=(
                str(entry.get("last_interaction"))
                if entry.get("last_interaction")
                else None
            ),
            notes=(str(entry.get("notes")) if entry.get("notes") else None),
            discord_ids=tuple(sorted(discord_values)),
            sentiment=(str(entry.get("sentiment")) if entry.get("sentiment") else None),
        )

    def _clamp_float(self, value: Any, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = minimum
        return max(minimum, min(maximum, parsed))

    def _relationship_key(self, name: str) -> str:
        return name.strip().lower()

    # ------------------------------------------------------------------
    # Update helpers
    def _update_medium_term(self, payload: Any) -> bool:
        if not isinstance(payload, Iterable):
            return False
        changed = False
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            summary = str(item.get("summary", "")).strip()
            if not summary:
                continue
            action = str(item.get("action", "upsert")).lower()
            key = summary.lower()
            entries = self._state["medium_term"]
            if action == "remove":
                remaining = [entry for entry in entries if entry.get("summary", "").lower() != key]
                if len(remaining) != len(entries):
                    self._state["medium_term"] = remaining
                    changed = True
                continue
            entry = self._normalise_medium_term(item)
            existing = next(
                (e for e in entries if e.get("summary", "").lower() == key),
                None,
            )
            if existing:
                if existing != entry:
                    existing.update(entry)
                    changed = True
            else:
                entries.append(entry)
                changed = True
        if changed:
            entries = self._state["medium_term"]
            entries.sort(
                key=lambda item: (item.get("importance", 0.0), item.get("last_updated", "")),
                reverse=True,
            )
            self._state["medium_term"] = entries[: self.max_medium_term]
        return changed

    def _update_string_set(self, key: str, payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            return False
        current = set(self._state[key])
        additions = payload.get("add", [])
        removals = payload.get("remove", [])
        before = set(current)
        for item in additions:
            text = str(item).strip()
            if text:
                current.add(text)
        for item in removals:
            text = str(item).strip()
            if text and text in current:
                current.remove(text)
        if current == before:
            return False
        self._state[key] = sorted(current)
        return True

    def _update_relationships(self, payload: Any) -> bool:
        if not isinstance(payload, Iterable):
            return False
        changed = False
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            record = self._normalise_relationship(item)
            key = self._relationship_key(record.name)
            if str(item.get("action", "upsert")).lower() == "remove":
                if key in self._state["relationships"]:
                    del self._state["relationships"][key]
                    changed = True
                continue
            existing = self._state["relationships"].get(key)
            payload_dict = record.to_payload()
            if existing:
                merged = existing.copy()
                for field, value in payload_dict.items():
                    if value in (None, "") and field in merged:
                        continue
                    merged[field] = value
                if merged != existing:
                    self._state["relationships"][key] = merged
                    changed = True
            else:
                self._state["relationships"][key] = payload_dict
                changed = True
        return changed
