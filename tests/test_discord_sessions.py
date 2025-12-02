"""Tests for Discord speaker session handling."""

from __future__ import annotations

import struct
from types import SimpleNamespace

from persona.denoiser import NoOpDenoiser
from persona.discord_integration import (
    DiscordVoiceBridge,
    SpeakerContext,
    SpeakerSession,
)
from persona.pipeline import PipelineOutput
from persona.planner import ResponsePlan
from persona.vad import EnergyVAD


def _pcm_frame(amplitude: int, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    frame_samples = int(sample_rate * 0.02)
    values = [amplitude] * frame_samples * channels
    return struct.pack(f"<{len(values)}h", *values)


def _vad_factory() -> EnergyVAD:
    return EnergyVAD(threshold=0.001, min_speech_frames=2, max_silence_frames=1)


def test_speaker_session_enqueues_segments() -> None:
    context = SpeakerContext(
        guild_id=1, channel_id=2, user_id=3, display_name="User"
    )
    session = SpeakerSession(
        context=context,
        sample_rate=16000,
        channels=1,
        vad_factory=_vad_factory,
    )

    speech = _pcm_frame(1000)
    silence = _pcm_frame(0)

    segments: list = []
    segments.extend(session.process_pcm(speech))
    segments.extend(session.process_pcm(speech))
    segments.extend(session.process_pcm(silence))

    assert len(segments) == 1
    segment = segments[0]
    assert segment.start_time == 0.0
    assert abs(segment.end_time - 0.04) < 1e-3
    assert len(session.completed_segments) == 1
    assert abs(session._timestamp - 0.06) < 1e-3


class _DummyPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.denoiser_factory = NoOpDenoiser
        self.vad_factory = _vad_factory

    def process_utterance(self, segment, *, start_time=None, end_time=None, speaker_ctx=None):
        self.calls.append((segment, speaker_ctx))
        audio = SimpleNamespace(payload=b"audio", sample_rate=16000, encoding="pcm16")
        plan = ResponsePlan(
            response_text="resp",
            log_memory=False,
            memory_summary=None,
            emotion=None,
            importance=0.0,
        )
        return PipelineOutput(
            transcription="hello",
            plan=plan,
            audio=audio,
            start_time=start_time,
            end_time=end_time,
            speaker_ctx=speaker_ctx,
        )


def test_voice_bridge_routes_segments_to_pipeline() -> None:
    pipeline = _DummyPipeline()
    payloads: list[bytes] = []
    bridge = DiscordVoiceBridge(
        pipeline=pipeline,
        send_audio=payloads.append,
        vad_factory=_vad_factory,
    )

    bridge.handle_speaking(
        ssrc=1234,
        user_id=42,
        guild_id=99,
        channel_id=100,
        display_name="Tester",
    )

    speech = _pcm_frame(1000)
    silence = _pcm_frame(0)

    bridge.receive_discord_pcm(speech, sample_rate=16000, channels=1, ssrc=1234)
    bridge.receive_discord_pcm(speech, sample_rate=16000, channels=1, ssrc=1234)
    bridge.receive_discord_pcm(silence, sample_rate=16000, channels=1, ssrc=1234)

    assert len(pipeline.calls) == 1
    _, speaker_ctx = pipeline.calls[0]
    assert speaker_ctx.user_id == 42
    assert speaker_ctx.display_name == "Tester"
    assert payloads == [b"audio"]
    assert 42 in bridge.sessions
