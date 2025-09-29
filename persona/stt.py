"""Transcription interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .audio import AudioSegment


@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    language: str = "en"
    start_time: float | None = None
    end_time: float | None = None


class Transcriber(Protocol):
    """Interface for speech-to-text backends."""

    def transcribe(self, segment: AudioSegment) -> TranscriptionResult:
        ...


class EchoTranscriber:
    """Debug transcriber that treats PCM payload as UTF-8 text."""

    def __init__(self, default_confidence: float = 0.9) -> None:
        self.default_confidence = default_confidence

    def transcribe(self, segment: AudioSegment) -> TranscriptionResult:
        try:
            text = segment.pcm.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            text = ""
        else:
            text = text.strip("\x00").rstrip()
        return TranscriptionResult(
            text=text,
            confidence=self.default_confidence,
            start_time=segment.start_time,
            end_time=segment.end_time,
        )


class HuggingFaceTranscriber:
    """Speech-to-text wrapper around Hugging Face ASR pipelines."""

    def __init__(
        self,
        model_id: str,
        *,
        device: int | str | None = None,
        chunk_length_s: float | None = None,
        decoder: str | None = None,
        pipeline_factory=None,
    ) -> None:
        """Create a transcriber backed by a Hugging Face pipeline.

        Parameters
        ----------
        model_id:
            Identifier of an Automatic Speech Recognition model on the Hugging
            Face Hub (e.g. ``openai/whisper-large-v3``).
        device:
            Torch device index or string. Defaults to CUDA when available.
        chunk_length_s:
            Optional streaming chunk size for long-form transcription.
        decoder:
            Optional decoder type supported by the ASR pipeline (e.g.
            ``"flash``" for Whisper models).
        pipeline_factory:
            Dependency injection hook used in tests; defaults to
            :func:`transformers.pipeline`.
        """

        if pipeline_factory is None:
            from transformers import pipeline

            pipeline_factory = pipeline

        if device is None:
            import torch

            device = 0 if torch.cuda.is_available() else "cpu"

        self._pipeline = pipeline_factory(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            chunk_length_s=chunk_length_s,
            decoder=decoder,
        )
        self._return_timestamps = True

    def transcribe(self, segment: AudioSegment) -> TranscriptionResult:
        if not segment.frames:
            return TranscriptionResult(text="", confidence=0.0)

        audio_array = segment.to_numpy()
        if segment.channels > 1:
            audio_array = np.mean(audio_array, axis=1)

        kwargs = {}
        if self._return_timestamps:
            kwargs["return_timestamps"] = True

        try:
            result = self._pipeline(
                {"array": audio_array, "sampling_rate": segment.sample_rate},
                **kwargs,
            )
        except TypeError:
            result = self._pipeline(
                {"array": audio_array, "sampling_rate": segment.sample_rate}
            )
        if isinstance(result, list):
            result = result[0]
        chunks = result.get("chunks") or []
        if chunks:
            text = " ".join((chunk.get("text") or "").strip() for chunk in chunks).strip()
        else:
            text = (result.get("text") or "").strip()

        confidence = 0.0
        if chunks:
            scores = [chunk.get("score") for chunk in chunks if chunk.get("score") is not None]
            if scores:
                confidence = float(np.mean(scores))
        if not confidence:
            confidence = float(result.get("score") or 0.0)

        base_start = segment.start_time or 0.0
        start_time = segment.start_time
        end_time = segment.end_time
        if chunks:
            timestamps = [chunk.get("timestamp") for chunk in chunks if chunk.get("timestamp")]
            if timestamps:
                starts = [ts[0] for ts in timestamps if ts and ts[0] is not None]
                ends = [ts[1] for ts in timestamps if ts and ts[1] is not None]
                if starts:
                    start_time = base_start + float(min(starts))
                if ends:
                    end_time = base_start + float(max(ends))

        return TranscriptionResult(
            text=text,
            confidence=confidence or 0.0,
            start_time=start_time,
            end_time=end_time,
        )
