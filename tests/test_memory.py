from __future__ import annotations

import time

from persona.memory import MemoryLogger


def test_memory_logger_writes_metadata(tmp_path) -> None:
    logger = MemoryLogger(tmp_path, persona_name="Astra", default_tags=("origin",))
    entry = logger.log(
        transcript="We pledged to remember the mission",
        response="Absolutely, it's important",
        emotion="focused",
        importance=0.75,
        summary="Mission promise",
        tags=("mission",),
        speaker_id="42",
    )

    content = entry.path.read_text(encoding="utf-8")
    assert "Persona: Astra" in content
    assert "Tags: mission" in content
    assert "Summary: Mission promise" in content
    assert "user:42" in entry.tags


def test_memory_should_log_heuristics(tmp_path) -> None:
    logger = MemoryLogger(tmp_path)
    assert logger.should_log(
        llm_flag=False,
        importance=0.4,
        transcript="This is an important promise we'll remember",
        response="I'll keep it in mind",
    )
    logger.log(
        transcript="Logged", response="ok", emotion="calm", importance=0.8, summary="Logged"
    )
    assert not logger.should_log(
        llm_flag=False,
        importance=0.3,
        transcript="Just small talk",
        response="nothing",
    )


def test_memory_list_and_retrieve(tmp_path) -> None:
    logger = MemoryLogger(tmp_path)
    first = logger.log(
        transcript="We discussed breakfast",
        response="Sounds tasty",
        emotion="cheerful",
        importance=0.6,
        summary="Breakfast chat",
        speaker_id="carol",
    )
    time.sleep(0.001)
    second = logger.log(
        transcript="Mission details shared",
        response="Acknowledged the mission",
        emotion="focused",
        importance=0.8,
        summary="Mission brief",
        speaker_id="alice",
    )

    entries = logger.list_entries(limit=2)
    assert entries[0].path == second.path
    assert entries[1].path == first.path

    snippets = logger.retrieve(["mission"], limit=1)
    assert snippets and "Mission brief" in snippets[0]


def test_format_entries_for_prompt_prioritises_importance(tmp_path) -> None:
    logger = MemoryLogger(tmp_path)
    logger.log(
        transcript="Casual chit chat",
        response="Noted",
        emotion="calm",
        importance=0.2,
        summary="Small talk",
    )
    logger.log(
        transcript="Discussed the mission parameters",
        response="We agreed on the plan",
        emotion="focused",
        importance=0.85,
        summary="Mission parameters",
    )

    prompt_memories = logger.format_entries_for_prompt(limit=1, min_importance=0.5)
    assert len(prompt_memories) == 1
    top_memory = prompt_memories[0]
    assert "Mission parameters" in top_memory
    assert "importance=0.85" in top_memory
    assert "emotion=focused" in top_memory


def test_speaker_tag_prioritization_in_prompt(tmp_path) -> None:
    logger = MemoryLogger(tmp_path)
    logger.log(
        transcript="Shared mission prep",
        response="Ready to go",
        emotion="calm",
        importance=0.9,
        summary="Global mission",
    )
    logger.log(
        transcript="Personal reminder",
        response="I'll remember",
        emotion="focused",
        importance=0.65,
        summary="Speaker memory",
        speaker_id="alice",
    )

    prompt_memories = logger.format_entries_for_prompt(
        limit=2, min_importance=0.5, speaker_id="alice"
    )
    assert len(prompt_memories) == 2
    assert "Speaker memory" in prompt_memories[0]
    assert "Global mission" in prompt_memories[1]


def test_retrieve_filters_other_speakers(tmp_path) -> None:
    logger = MemoryLogger(tmp_path)
    logger.log(
        transcript="Alice discussed the mission",
        response="Confirmed",
        emotion="focused",
        importance=0.7,
        summary="Alice mission",
        speaker_id="alice",
    )
    logger.log(
        transcript="Bob discussed the mission",
        response="Confirmed",
        emotion="focused",
        importance=0.7,
        summary="Bob mission",
        speaker_id="bob",
    )
    logger.log(
        transcript="Mission overview shared",
        response="Logged",
        emotion="calm",
        importance=0.6,
        summary="Global mission",
    )

    snippets = logger.retrieve(["mission"], limit=2, speaker_id="alice")
    assert snippets and "Alice mission" in snippets[0]
    assert any("Global mission" in snippet for snippet in snippets)
    assert all("Bob mission" not in snippet for snippet in snippets)
