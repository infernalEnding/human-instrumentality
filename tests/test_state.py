from __future__ import annotations

from persona.state import PersonaStateManager


def test_persona_state_manager_updates_and_persists(tmp_path) -> None:
    state_file = tmp_path / "persona_state.json"
    manager = PersonaStateManager(state_file, persona_name="Astra", max_medium_term=3)

    updates = {
        "medium_term": [
            {"summary": "Helped Alex with code", "importance": 0.7},
            {"summary": "Joined a Discord art jam", "importance": 0.6},
        ],
        "hobbies": {"add": ["hiking", "painting"], "remove": ["skydiving"]},
        "artistic_likes": {"add": ["synthwave"], "remove": []},
        "relationships": [
            {
                "name": "Alex",
                "relationship": "friend",
                "notes": "Met in Discord server",
                "discord_ids": ["12345"],
            }
        ],
    }

    changed = manager.apply_updates(updates)
    assert changed is True

    state = manager.get_state()
    assert "Helped Alex" in state["medium_term"][0]["summary"]
    assert "painting" in state["hobbies"]
    assert state["artistic_likes"] == ["synthwave"]
    assert "alex" in state["relationships"]

    # Ensure persistence
    reloaded = PersonaStateManager(state_file, persona_name="Astra")
    assert reloaded.get_state()["hobbies"] == state["hobbies"]

    # Test removals
    manager.apply_updates(
        {
            "medium_term": [{"summary": "Joined a Discord art jam", "action": "remove"}],
            "hobbies": {"remove": ["hiking"]},
            "relationships": [{"name": "Alex", "action": "remove"}],
        }
    )
    state_after = manager.get_state()
    assert all("art jam" not in entry["summary"].lower() for entry in state_after["medium_term"])
    assert "hiking" not in state_after["hobbies"]
    assert "alex" not in state_after["relationships"]


def test_persona_state_prompt_context(tmp_path) -> None:
    manager = PersonaStateManager(tmp_path / "state.json")
    manager.apply_updates(
        {
            "medium_term": [
                {"summary": "Prepared for community stream", "importance": 0.9},
                {"summary": "Learned new guitar riff", "importance": 0.6},
            ],
            "hobbies": {"add": ["guitar"]},
            "relationships": [
                {
                    "name": "Jamie",
                    "relationship": "moderator",
                    "notes": "Keeps things positive",
                }
            ],
        }
    )

    lines = manager.prompt_context()
    assert any(line.startswith("Medium-term memories:") for line in lines)
    assert any("Hobbies" in line for line in lines)
    assert any("Jamie" in line for line in lines)
