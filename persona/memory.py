"""Markdown-backed memory logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

from .embeddings import EmbeddingIndex, EmbeddingMatch, EmbeddingMetadata, TextEmbedder


@dataclass
class MemoryEntry:
    path: Path
    timestamp: datetime
    summary: str
    emotion: str | None
    importance: float
    tags: tuple[str, ...] = ()


class MemoryLogger:
    def __init__(
        self,
        base_dir: Path | str,
        *,
        persona_name: str = "Astra",
        default_tags: Sequence[str] | None = None,
        importance_threshold: float = 0.55,
        cooldown_seconds: float = 45.0,
        embedder: TextEmbedder | None = None,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.persona_name = persona_name
        self.default_tags = tuple(default_tags or ("conversation",))
        self.importance_threshold = importance_threshold
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self._last_logged_at: datetime | None = None
        self.embedder = embedder
        self.embedding_index = embedding_index
        self._keyword_triggers = {
            "remember",
            "promise",
            "important",
            "goal",
            "feeling",
            "plan",
        }

    def log(
        self,
        *,
        transcript: str,
        response: str,
        emotion: str | None,
        importance: float,
        summary: str | None = None,
        tags: Sequence[str] | None = None,
        speaker_id: str | int | None = None,
    ) -> MemoryEntry:
        timestamp = datetime.utcnow()
        summary_text = summary or transcript[:120]
        safe_name = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        path = self.base_dir / f"{safe_name}.md"
        tag_values = list(tags or self._auto_tags(transcript, response, emotion))
        if speaker_id is not None:
            user_tag = f"user:{speaker_id}"
            if user_tag not in tag_values:
                tag_values.append(user_tag)
        tag_values = tuple(tag_values)
        content = self._render_markdown(
            timestamp=timestamp,
            transcript=transcript,
            response=response,
            emotion=emotion,
            importance=importance,
            summary=summary_text,
            tags=tag_values,
        )
        path.write_text(content, encoding="utf-8")
        self._last_logged_at = timestamp
        metadata = self._build_metadata(
            key=safe_name,
            summary=summary_text,
            importance=importance,
            speaker_id=speaker_id,
            source=str(path),
        )
        self._index_entry(
            metadata=metadata,
            transcript=transcript,
            response=response,
        )
        return MemoryEntry(
            path=path,
            timestamp=timestamp,
            summary=summary_text,
            emotion=emotion,
            importance=importance,
            tags=tag_values,
        )

    def list_entries(self, limit: int = 10) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for path in sorted(self.base_dir.glob("*.md"), reverse=True):
            if len(entries) >= limit:
                break
            entry = self._parse_entry(path)
            if entry:
                entries.append(entry)
        return entries

    def rank_entries(
        self,
        *,
        limit: int = 3,
        min_importance: float = 0.0,
        speaker_id: str | int | None = None,
    ) -> List[MemoryEntry]:
        entries: List[MemoryEntry] = []
        for path in self.base_dir.glob("*.md"):
            entry = self._parse_entry(path)
            if entry and entry.importance >= min_importance:
                entries.append(entry)
        sort_key = lambda item: (
            self._is_speaker_match(item.tags, speaker_id),
            item.importance,
            item.timestamp,
        )
        entries.sort(key=sort_key, reverse=True)
        return entries[:limit]

    def format_entries_for_prompt(
        self,
        *,
        limit: int = 3,
        min_importance: float = 0.4,
        speaker_id: str | int | None = None,
    ) -> List[str]:
        if limit <= 0:
            return []
        selected = self.rank_entries(
            limit=limit, min_importance=min_importance, speaker_id=speaker_id
        )
        if len(selected) < limit and min_importance > 0.0:
            fallback = self.rank_entries(
                limit=limit, min_importance=0.0, speaker_id=speaker_id
            )
            existing_paths = {entry.path for entry in selected}
            for entry in fallback:
                if entry.path not in existing_paths:
                    selected.append(entry)
                if len(selected) >= limit:
                    break
        selected.sort(
            key=lambda item: (
                self._is_speaker_match(item.tags, speaker_id),
                item.importance,
                item.timestamp,
            ),
            reverse=True,
        )
        formatted: List[str] = []
        for entry in selected[:limit]:
            metadata = [
                f"logged_at={entry.timestamp.isoformat()}Z",
                f"importance={entry.importance:.2f}",
            ]
            if entry.emotion:
                metadata.append(f"emotion={entry.emotion}")
            if entry.tags:
                metadata.append("tags=" + ", ".join(entry.tags))
            formatted.append(f"{entry.summary} ({'; '.join(metadata)})")
        return formatted

    def recall_for_prompt(
        self,
        *,
        transcript: str,
        limit: int = 3,
        min_importance: float = 0.4,
        speaker_id: str | int | None = None,
    ) -> List[str]:
        if not transcript or limit <= 0:
            return []
        speaker = str(speaker_id) if speaker_id is not None else None
        if self.embedder and self.embedding_index:
            return self._semantic_recall(
                transcript=transcript,
                limit=limit,
                min_importance=min_importance,
                speaker_id=speaker,
            )
        return self.format_entries_for_prompt(
            limit=limit, min_importance=min_importance, speaker_id=speaker
        )

    def retrieve(
        self,
        query_terms: Iterable[str],
        limit: int = 3,
        *,
        speaker_id: str | int | None = None,
    ) -> List[str]:
        terms = [term.lower() for term in query_terms if term]
        if not terms:
            return []
        scored: List[tuple[int, MemoryEntry]] = []
        for path in self.base_dir.glob("*.md"):
            entry = self._parse_entry(path)
            if not entry:
                continue
            text = path.read_text(encoding="utf-8").lower()
            score = sum(text.count(term) for term in terms)
            if score > 0:
                scored.append((score, entry))
        scored.sort(
            key=lambda item: (
                self._is_speaker_match(item[1].tags, speaker_id),
                item[0],
                item[1].timestamp,
            ),
            reverse=True,
        )
        snippets: List[str] = []
        for _, entry in scored[:limit]:
            lines = entry.path.read_text(encoding="utf-8").splitlines()
            summary_line = self._extract_field(lines, "Summary")
            snippets.append(summary_line or (lines[0] if lines else ""))
        return snippets

    def should_log(
        self,
        *,
        llm_flag: bool,
        importance: float,
        transcript: str,
        response: str,
    ) -> bool:
        score = importance
        transcript_lower = transcript.lower()
        if llm_flag:
            score += 0.2
        if any(keyword in transcript_lower for keyword in self._keyword_triggers):
            score += 0.25
        if "!" in transcript:
            score += 0.05
        if transcript.endswith("?"):
            score += 0.1
        now = datetime.utcnow()
        if self._last_logged_at and now - self._last_logged_at < self.cooldown:
            score -= 0.2
        return score >= self.importance_threshold

    def _build_metadata(
        self,
        *,
        key: str,
        summary: str,
        importance: float,
        speaker_id: str | int | None,
        source: str | None,
    ) -> EmbeddingMetadata:
        speaker = str(speaker_id) if speaker_id is not None else None
        return EmbeddingMetadata(
            key=key,
            summary=summary,
            speaker_id=speaker,
            importance=importance,
            source=source,
        )

    def _index_entry(
        self,
        *,
        metadata: EmbeddingMetadata,
        transcript: str,
        response: str,
    ) -> None:
        if not self.embedder or not self.embedding_index:
            return
        text = "\n".join(
            (
                f"Summary: {metadata.summary}",
                f"Transcript: {transcript}",
                f"Response: {response}",
            )
        )
        try:
            vector = self.embedder.embed([text])[0]
        except Exception:
            return
        self.embedding_index.add(vector=vector, metadata=metadata)

    def _semantic_recall(
        self,
        *,
        transcript: str,
        limit: int,
        min_importance: float,
        speaker_id: str | None,
    ) -> List[str]:
        if not self.embedder or not self.embedding_index:
            return []
        try:
            query_vec = self.embedder.embed([transcript])[0]
        except Exception:
            return []
        matches = self.embedding_index.search(
            query=query_vec,
            limit=limit,
            speaker_id=speaker_id,
            min_importance=min_importance,
        )
        return [self._format_match(match) for match in matches]

    def _format_match(self, match: EmbeddingMatch) -> str:
        metadata = match.metadata
        speaker_hint = (
            f"speaker={metadata.speaker_id}" if metadata.speaker_id is not None else ""
        )
        source_hint = f"source={metadata.source}" if metadata.source else ""
        hints = [hint for hint in (speaker_hint, source_hint) if hint]
        hints.append(f"importance={metadata.importance:.2f}")
        hints.append(f"score={match.score:.2f}")
        return f"{metadata.summary} ({'; '.join(hints)})"

    def _render_markdown(
        self,
        *,
        timestamp: datetime,
        transcript: str,
        response: str,
        emotion: str | None,
        importance: float,
        summary: str,
        tags: Sequence[str],
    ) -> str:
        metadata = [
            f"# Memory Log {timestamp.isoformat()}Z",
            f"Persona: {self.persona_name}",
            f"Emotion: {emotion or 'neutral'}",
            f"Importance: {importance:.2f}",
            f"Tags: {', '.join(tags) if tags else 'conversation'}",
            f"Summary: {summary}",
            "",
        ]
        body = [
            "## Transcript",
            transcript,
            "",
            "## Response",
            response,
        ]
        return "\n".join(metadata + body)

    def _parse_importance(self, lines: List[str]) -> float:
        for line in lines:
            if line.startswith("Importance:"):
                try:
                    return float(line.partition(":")[2].strip())
                except ValueError:
                    return 0.0
        return 0.0

    def _extract_field(self, lines: List[str], key: str) -> str:
        for line in lines:
            if line.startswith(f"{key}:"):
                return line.partition(":")[2].strip()
        return ""

    def _parse_entry(self, path: Path) -> MemoryEntry | None:
        try:
            timestamp = datetime.strptime(path.stem, "%Y%m%dT%H%M%S%fZ")
        except ValueError:
            return None

        lines = path.read_text(encoding="utf-8").splitlines()
        summary = self._extract_field(lines, "Summary") or self._derive_summary(lines)
        emotion = self._extract_field(lines, "Emotion") or None
        tags_line = self._extract_field(lines, "Tags")
        tags = (
            tuple(tag.strip() for tag in tags_line.split(",") if tag.strip())
            if tags_line
            else ()
        )
        summary = summary or "Untitled memory"
        return MemoryEntry(
            path=path,
            timestamp=timestamp,
            summary=summary,
            emotion=emotion,
            importance=self._parse_importance(lines),
            tags=tags,
        )

    def _derive_summary(self, lines: List[str]) -> str:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(prefix) for prefix in ("#", "Persona:", "Emotion:", "Importance:", "Tags:")):
                continue
            return stripped
        return ""

    def _auto_tags(
        self, transcript: str, response: str, emotion: str | None
    ) -> List[str]:
        tags = set(self.default_tags)
        if emotion:
            tags.add(f"emotion:{emotion.lower()}")
        combined = f"{transcript} {response}".lower()
        if "plan" in combined:
            tags.add("planning")
        if "remember" in combined:
            tags.add("memory")
        if "feeling" in combined or "emotion" in combined:
            tags.add("introspection")
        if "?" in transcript:
            tags.add("question")
        return sorted(tags)

    def _is_speaker_match(
        self, tags: Sequence[str], speaker_id: str | int | None
    ) -> bool:
        if speaker_id is None:
            return False
        user_tag = f"user:{speaker_id}"
        return user_tag in tags
