"""Discord integration scaffolding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, Optional

from .audio import AudioBuffer, AudioFrame, AudioSegment
from .pipeline import PersonaPipeline
from .vad import EnergyVAD


@dataclass
class SpeakerContext:
    guild_id: int | None
    channel_id: int | None
    user_id: int
    display_name: str


class SpeakerSession:
    """Tracks voice activity and utterance boundaries for a single speaker."""

    def __init__(
        self,
        *,
        context: SpeakerContext,
        sample_rate: int,
        channels: int,
        vad_factory: Callable[[], EnergyVAD] | None = None,
    ) -> None:
        self.context = context
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad = vad_factory() if vad_factory else EnergyVAD()
        self.audio_buffer = AudioBuffer()
        self._timestamp = 0.0
        self.completed_segments: Deque[AudioSegment] = deque()

    def process_pcm(self, pcm: bytes) -> list[AudioSegment]:
        frame = AudioFrame(
            pcm=pcm,
            sample_rate=self.sample_rate,
            channels=self.channels,
            timestamp=self._timestamp,
        )
        self._timestamp += frame.duration_seconds
        self.audio_buffer.append(frame)
        decisions = self.vad.process_frame(frame)
        return self._handle_decisions(decisions)

    def flush(self) -> list[AudioSegment]:
        return self._handle_decisions(self.vad.flush())

    def _handle_decisions(self, decisions: Iterable) -> list[AudioSegment]:
        segments: list[AudioSegment] = []
        for decision in decisions:
            self.audio_buffer.clear()
            self.completed_segments.append(decision.segment)
            segments.append(decision.segment)
        return segments


@dataclass
class DiscordConfig:
    token: str
    guild_id: int | None = None
    channel_id: int | None = None
    persona_name: str = "Astra"
    input_mode: str = "bot"  # or "microphone"


class DiscordVoiceBridge:
    """High level coordinator for Discord audio bridging.

    The implementation keeps the networking responsibilities optional to avoid
    forcing discord.py as a hard dependency. Users are expected to inject the
    concrete audio adapters at runtime.
    """

    def __init__(
        self,
        pipeline: PersonaPipeline,
        send_audio: Callable[[bytes], None],
        on_transcript: Optional[Callable[..., None]] = None,
        frame_duration_s: float = 0.02,
        vad_factory: Callable[[], EnergyVAD] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.send_audio = send_audio
        self.on_transcript = on_transcript
        self.frame_duration_s = frame_duration_s
        self._timestamp = 0.0
        self.vad_factory = vad_factory
        self.sessions: Dict[int, SpeakerSession] = {}
        self.ssrc_to_user_id: Dict[int, int] = {}
        self._speaker_contexts: Dict[int, SpeakerContext] = {}

    def handle_speaking(
        self,
        *,
        ssrc: int,
        user_id: int,
        guild_id: int | None,
        channel_id: int | None,
        display_name: str,
    ) -> None:
        """Register SSRC-to-user mapping and context metadata."""

        self.ssrc_to_user_id[ssrc] = user_id
        context = SpeakerContext(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            display_name=display_name,
        )
        self._speaker_contexts[user_id] = context
        if user_id in self.sessions:
            self.sessions[user_id].context = context

    def handle_audio_frames(self, frames) -> None:
        outputs = self.pipeline.process_frames(frames)
        self._dispatch(outputs)

    def receive_discord_pcm(
        self,
        pcm: bytes,
        sample_rate: int,
        channels: int,
        *,
        ssrc: int | None = None,
    ) -> None:
        if ssrc is None:
            frame = AudioFrame(
                pcm=pcm,
                sample_rate=sample_rate,
                channels=channels,
                timestamp=self._timestamp,
            )
            self._timestamp += len(pcm) / (2 * channels * sample_rate)
            outputs = self.pipeline.process_stream_frame(frame)
            self._dispatch(outputs)
            return

        user_id = self.ssrc_to_user_id.get(ssrc)
        if user_id is None:
            return

        session = self.sessions.get(user_id)
        if session is None:
            context = self._speaker_contexts.get(
                user_id,
                SpeakerContext(
                    guild_id=None,
                    channel_id=None,
                    user_id=user_id,
                    display_name=str(user_id),
                ),
            )
            session = SpeakerSession(
                context=context,
                sample_rate=sample_rate,
                channels=channels,
                vad_factory=self.vad_factory,
            )
            self.sessions[user_id] = session

        segments = session.process_pcm(pcm)
        self._process_segments(segments, session.context)

    def flush(self) -> None:
        for session in self.sessions.values():
            segments = session.flush()
            self._process_segments(segments, session.context)
        outputs = self.pipeline.flush()
        self._dispatch(outputs)

    def _process_segments(
        self, segments: Iterable[AudioSegment], speaker_ctx: SpeakerContext | None
    ) -> None:
        outputs = []
        for segment in segments:
            output = self.pipeline.process_utterance(
                segment,
                start_time=segment.start_time,
                end_time=segment.end_time,
                speaker_ctx=speaker_ctx,
            )
            if output:
                outputs.append(output)
        if outputs:
            self._dispatch(outputs)

    def _dispatch(self, outputs: Iterable) -> None:
        for output in outputs:
            if self.on_transcript:
                try:
                    self.on_transcript(output.transcription, output.start_time, output.end_time)
                except TypeError:
                    self.on_transcript(output.transcription)
            self.send_audio(output.audio.payload)
