"""Discord integration scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .audio import AudioFrame
from .pipeline import PersonaPipeline


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
    ) -> None:
        self.pipeline = pipeline
        self.send_audio = send_audio
        self.on_transcript = on_transcript
        self.frame_duration_s = frame_duration_s
        self._timestamp = 0.0

    def handle_audio_frames(self, frames) -> None:
        outputs = self.pipeline.process_frames(frames)
        self._dispatch(outputs)

    def receive_discord_pcm(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        frame = AudioFrame(
            pcm=pcm,
            sample_rate=sample_rate,
            channels=channels,
            timestamp=self._timestamp,
        )
        self._timestamp += len(pcm) / (2 * channels * sample_rate)
        outputs = self.pipeline.process_stream_frame(frame)
        self._dispatch(outputs)

    def flush(self) -> None:
        outputs = self.pipeline.flush()
        self._dispatch(outputs)

    def _dispatch(self, outputs: Iterable) -> None:
        for output in outputs:
            if self.on_transcript:
                try:
                    self.on_transcript(output.transcription, output.start_time, output.end_time)
                except TypeError:
                    self.on_transcript(output.transcription)
            self.send_audio(output.audio.payload)
