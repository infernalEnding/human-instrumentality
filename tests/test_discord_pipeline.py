from __future__ import annotations

import struct
from pathlib import Path

from persona.llm import LLMResponse
from persona.pipeline import DiscordSpeakerPipeline
from persona.planner import ResponsePlanner
from persona.state import PersonaStateManager
from persona.stt import TranscriptionResult, Transcriber
from persona.tts import DebugSynthesizer
from persona.vad import EnergyVAD

from persona.audio import AudioFrame


class StaticTranscriber(Transcriber):
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, segment) -> TranscriptionResult:
        return TranscriptionResult(text=self.text, confidence=0.9, start_time=0.0, end_time=1.0)


class PromptCapturingLLM:
    def __init__(self) -> None:
        self.persona_state: list[str] | None = None

    def generate_reply(self, transcript: str, memories=None, persona_state=None) -> LLMResponse:
        self.persona_state = list(persona_state or [])
        return LLMResponse(
            text="Welcome!",
            should_log_memory=False,
            emotion="warm",
            importance=0.4,
            summary=None,
            state_updates={
                "relationships": [
                    {
                        "name": "Alex",
                        "notes": "Confirmed consent to be greeted by name.",
                    }
                ]
            },
        )


def frame_from_amplitude(amplitude: float, sample_rate: int = 16000, samples: int = 160) -> AudioFrame:
    value = int(max(-1.0, min(1.0, amplitude)) * 32767)
    pcm = struct.pack(f"<{samples}h", *([value] * samples))
    return AudioFrame(pcm=pcm, sample_rate=sample_rate)


def test_get_or_create_discord_relationship(tmp_path: Path) -> None:
    manager = PersonaStateManager(tmp_path / "persona_state.json")

    created = manager.get_or_create_discord_relationship("12345", display_name="CoolUser")
    assert created["needs_identification"] is True
    assert "12345" in created["discord_ids"]

    fetched = manager.get_or_create_discord_relationship("12345")
    assert fetched["name"] == created["name"]


def test_discord_pipeline_injects_identification_prompts(tmp_path: Path) -> None:
    state_manager = PersonaStateManager(tmp_path / "persona_state.json")
    llm = PromptCapturingLLM()
    pipeline = DiscordSpeakerPipeline(
        speaker_id="user-123",
        speaker_display_name="CoolUser",
        vad=EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1),
        transcriber=StaticTranscriber("Hello there"),
        llm=llm,
        planner=ResponsePlanner(),
        synthesizer=DebugSynthesizer(),
        persona_state_manager=state_manager,
    )

    frames = [frame_from_amplitude(0.5) for _ in range(4)]
    frames.extend(frame_from_amplitude(0.0) for _ in range(2))
    outputs = pipeline.process_frames(frames)

    assert outputs, "expected a response from the Discord pipeline"
    assert llm.persona_state is not None
    assert any("greet" in line.lower() for line in llm.persona_state)
    assert any("consent" in line.lower() for line in llm.persona_state)

    state = state_manager.get_state()
    assert "alex" in state["relationships"]
    alex = state["relationships"]["alex"]
    assert alex["needs_identification"] is False
    assert "user-123" in alex["discord_ids"]
    assert "consent" in (alex.get("notes") or "")
