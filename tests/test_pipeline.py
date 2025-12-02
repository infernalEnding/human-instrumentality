from __future__ import annotations

import struct
from pathlib import Path

from persona.audio import AudioFrame, AudioSegment
from persona.emotion import AudioEmotionResult
from persona.discord_integration import SpeakerContext
from persona.llm import LLMResponse, RuleBasedPersonaLLM
from persona.memory import MemoryLogger
from persona.pipeline import PersonaPipeline
from persona.planner import ResponsePlanner
from persona.stt import TranscriptionResult, Transcriber
from persona.tts import DebugSynthesizer
from persona.vad import EnergyVAD
from persona.state import PersonaStateManager


class ScriptedTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, segment) -> TranscriptionResult:
        return TranscriptionResult(text=self.text, confidence=0.95, start_time=0.0, end_time=1.0)


class CapturingLLM:
    def __init__(self) -> None:
        self.memories: list[str] | None = None
        self.persona_state: list[str] | None = None

    def generate_reply(
        self,
        transcript: str,
        memories: list[str] | None = None,
        persona_state: list[str] | None = None,
        sentiment=None,
        audio_emotion=None,
    ) -> LLMResponse:
        self.memories = list(memories or [])
        self.persona_state = list(persona_state or [])
        return LLMResponse(
            text="Acknowledged",
            should_log_memory=False,
            emotion="neutral",
            importance=0.3,
            summary=None,
        )


class WhitespaceTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, segment) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(text="   ", confidence=0.4, start_time=0.0, end_time=0.5)


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate_reply(
        self,
        transcript: str,
        memories: list[str] | None = None,
        persona_state: list[str] | None = None,
        sentiment=None,
        audio_emotion=None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="Hello",
            should_log_memory=False,
            emotion="neutral",
            importance=0.1,
            summary=None,
        )


class OverconfidentLLM:
    def generate_reply(
        self,
        transcript: str,
        memories: list[str] | None = None,
        persona_state: list[str] | None = None,
        sentiment=None,
        audio_emotion=None,
    ) -> LLMResponse:
        return LLMResponse(
            text="This matters a lot",
            should_log_memory=True,
            emotion="intense",
            importance=1.8,
            summary="Major development",
        )


class UpdatingLLM:
    def __init__(self) -> None:
        self.persona_state: list[str] | None = None

    def generate_reply(
        self,
        transcript: str,
        memories: list[str] | None = None,
        persona_state: list[str] | None = None,
        sentiment=None,
        audio_emotion=None,
    ) -> LLMResponse:
        self.persona_state = list(persona_state or [])
        return LLMResponse(
            text="Got it",
            should_log_memory=False,
            emotion="calm",
            importance=0.2,
            summary=None,
            state_updates={
                "hobbies": {"add": ["painting"], "remove": []},
                "medium_term": [
                    {
                        "summary": "Talked about creative pursuits",
                        "importance": 0.6,
                    }
                ],
                "relationships": [
                    {
                        "name": "Alex",
                        "relationship": "friend",
                        "notes": "Met via Discord", 
                        "discord_ids": ["12345"],
                    }
                ],
            },
        )


class StaticEmotionAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, segment) -> AudioEmotionResult:
        self.calls += 1
        return AudioEmotionResult(label="happy", score=0.87)


class EmotionAwareLLM:
    def __init__(self) -> None:
        self.received_audio_emotion: AudioEmotionResult | None = None

    def generate_reply(
        self,
        transcript: str,
        memories: list[str] | None = None,
        persona_state: list[str] | None = None,
        sentiment=None,
        audio_emotion: AudioEmotionResult | None = None,
    ) -> LLMResponse:
        self.received_audio_emotion = audio_emotion
        return LLMResponse(
            text="Hello with empathy",
            should_log_memory=False,
            emotion="engaged",
            importance=0.2,
            summary=None,
        )


def frame_from_amplitude(amplitude: float, sample_rate: int = 16000, samples: int = 160) -> AudioFrame:
    value = int(max(-1.0, min(1.0, amplitude)) * 32767)
    pcm = struct.pack(f"<{samples}h", *([value] * samples))
    return AudioFrame(pcm=pcm, sample_rate=sample_rate)


def test_pipeline_generates_response_and_logs_memory(tmp_path: Path) -> None:
    frames = [frame_from_amplitude(0.6) for _ in range(5)]
    frames.extend(frame_from_amplitude(0.0) for _ in range(3))

    pipeline = PersonaPipeline(
        vad=EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1),
        transcriber=ScriptedTranscriber("This is an important test"),
        llm=RuleBasedPersonaLLM(persona_name="Nova", mood="excited"),
        planner=ResponsePlanner(),
        synthesizer=DebugSynthesizer(),
        memory_logger=MemoryLogger(tmp_path),
    )

    outputs = pipeline.process_frames(frames)
    assert len(outputs) == 1
    output = outputs[0]
    assert "important" in output.transcription
    assert "Nova" in output.plan.response_text
    assert output.audio.payload

    memory_files = list(tmp_path.glob("*.md"))
    assert memory_files, "memory log file was not created"
    content = memory_files[0].read_text(encoding="utf-8")
    assert "Memory Log" in content
    assert "important test" in content.lower()
    assert "Persona:" in content
    assert "Tags:" in content


def test_memory_retrieval_returns_recent_entries(tmp_path: Path) -> None:
    logger = MemoryLogger(tmp_path)
    logger.log(
        transcript="We discussed the weather",
        response="Sunny reply",
        emotion="calm",
        sentiment=None,
        importance=0.3,
        summary="Weather chat",
    )
    logger.log(
        transcript="Critical mission briefing",
        response="Acknowledged",
        emotion="focused",
        sentiment=None,
        importance=0.9,
        summary="Mission briefing",
    )

    snippets = logger.retrieve(["mission"], limit=1)
    assert snippets, "expected to retrieve a mission memory"
    assert "Mission" in snippets[0]


def test_pipeline_formats_memories_for_llm(tmp_path: Path) -> None:
    logger = MemoryLogger(tmp_path)
    logger.log(
        transcript="Routine check-in",
        response="Cool",
        emotion="calm",
        sentiment=None,
        importance=0.3,
        summary="Routine chat",
    )
    logger.log(
        transcript="High stakes mission update",
        response="Acknowledged",
        emotion="focused",
        sentiment=None,
        importance=0.9,
        summary="High stakes mission",
    )

    capturing_llm = CapturingLLM()
    pipeline = PersonaPipeline(
        vad=EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1),
        transcriber=ScriptedTranscriber("Ready to brief"),
        llm=capturing_llm,
        planner=ResponsePlanner(),
        synthesizer=DebugSynthesizer(),
        memory_logger=logger,
        memory_window=1,
        memory_importance_threshold=0.6,
    )

    frames = [frame_from_amplitude(0.6) for _ in range(3)]
    frames.extend(frame_from_amplitude(0.0) for _ in range(2))

    pipeline.process_frames(frames)

    assert capturing_llm.memories is not None
    assert len(capturing_llm.memories) == 1
    memory_line = capturing_llm.memories[0]
    assert "High stakes mission" in memory_line
    assert "importance=0.90" in memory_line


def test_pipeline_ignores_whitespace_transcripts(tmp_path: Path) -> None:
    frames = [frame_from_amplitude(0.5) for _ in range(3)]
    frames.extend(frame_from_amplitude(0.0) for _ in range(2))

    transcriber = WhitespaceTranscriber()
    llm = CountingLLM()
    pipeline = PersonaPipeline(
        vad=EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1),
        transcriber=transcriber,
        llm=llm,
        planner=ResponsePlanner(),
        synthesizer=DebugSynthesizer(),
        memory_logger=MemoryLogger(tmp_path),
    )

    outputs = pipeline.process_frames(frames)
    assert outputs == []
    assert llm.calls == 0
    assert transcriber.calls == 1


def test_pipeline_passes_audio_emotion(tmp_path: Path) -> None:
    frames = [frame_from_amplitude(0.6) for _ in range(3)]
    frames.extend(frame_from_amplitude(0.0) for _ in range(2))

    analyzer = StaticEmotionAnalyzer()
    llm = EmotionAwareLLM()
    pipeline = PersonaPipeline(
        vad=EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1),
        transcriber=ScriptedTranscriber("Hearing emotions"),
        llm=llm,
        planner=ResponsePlanner(),
        synthesizer=DebugSynthesizer(),
        memory_logger=MemoryLogger(tmp_path),
        audio_emotion_analyzer=analyzer,
    )

    outputs = pipeline.process_frames(frames)

    assert analyzer.calls >= 1
    assert llm.received_audio_emotion is not None
    assert llm.received_audio_emotion.label == "happy"
    assert outputs and outputs[0].audio_emotion is not None
    assert outputs[0].audio_emotion.label == "happy"


def test_pipeline_clamps_importance_before_logging(tmp_path: Path) -> None:
    frames = [frame_from_amplitude(0.6) for _ in range(4)]
    frames.extend(frame_from_amplitude(0.0) for _ in range(2))

    pipeline = PersonaPipeline(
        vad=EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1),
        transcriber=ScriptedTranscriber("This is vital"),
        llm=OverconfidentLLM(),
        planner=ResponsePlanner(),
        synthesizer=DebugSynthesizer(),
        memory_logger=MemoryLogger(tmp_path),
    )

    outputs = pipeline.process_frames(frames)
    assert len(outputs) == 1
    assert outputs[0].plan.importance == 1.0

    memory_files = list(tmp_path.glob("*.md"))
    assert memory_files
    content = memory_files[0].read_text(encoding="utf-8")
    assert "Importance: 1.00" in content


def test_pipeline_logs_speaker_identifier(tmp_path: Path) -> None:
    segment = AudioSegment(
        frames=[AudioFrame(pcm=b"\x01\x00" * 80, sample_rate=16000, channels=1, timestamp=0.0)]
    )
    speaker_ctx = SpeakerContext(
        guild_id=None, channel_id=None, user_id=99, display_name="Tester"
    )
    pipeline = PersonaPipeline(
        vad=EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1),
        transcriber=ScriptedTranscriber("Logging speaker"),
        llm=OverconfidentLLM(),
        planner=ResponsePlanner(),
        synthesizer=DebugSynthesizer(),
        memory_logger=MemoryLogger(tmp_path),
    )

    pipeline.process_utterance(segment, speaker_ctx=speaker_ctx)

    entries = pipeline.memory_logger.list_entries()
    assert entries
    assert f"user:{speaker_ctx.user_id}" in entries[0].tags


def test_pipeline_updates_persona_state(tmp_path: Path) -> None:
    frames = [frame_from_amplitude(0.6) for _ in range(4)]
    frames.extend(frame_from_amplitude(0.0) for _ in range(2))

    state_file = tmp_path / "persona_state.json"
    state_manager = PersonaStateManager(state_file, persona_name="Astra")
    state_manager.apply_updates({"artistic_likes": {"add": ["synthwave"]}})

    llm = UpdatingLLM()
    pipeline = PersonaPipeline(
        vad=EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1),
        transcriber=ScriptedTranscriber("Let's talk about art"),
        llm=llm,
        planner=ResponsePlanner(),
        synthesizer=DebugSynthesizer(),
        memory_logger=None,
        persona_state_manager=state_manager,
    )

    outputs = pipeline.process_frames(frames)
    assert len(outputs) == 1
    assert outputs[0].plan.state_updates is not None
    assert any("Artistic likes" in line for line in (llm.persona_state or []))

    updated = state_manager.get_state()
    assert "painting" in updated["hobbies"]
    assert updated["medium_term"][0]["summary"].startswith("Talked about creative")
    assert "alex" in updated["relationships"]
