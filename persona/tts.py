"""Text-to-speech abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    return pcm.tobytes()


@dataclass
class SynthesizedAudio:
    payload: bytes
    sample_rate: int
    encoding: str = "pcm16"


class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str) -> SynthesizedAudio:
        ...


class DebugSynthesizer:
    """Synthesizer that returns UTF-8 payload for testing."""

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> SynthesizedAudio:
        return SynthesizedAudio(payload=text.encode("utf-8"), sample_rate=self.sample_rate, encoding="text/debug")


class HuggingFaceSynthesizer:
    """Text-to-speech synthesizer powered by SpeechT5 on Hugging Face."""

    def __init__(
        self,
        model_id: str = "microsoft/speecht5_tts",
        *,
        vocoder_id: str = "microsoft/speecht5_hifigan",
        speaker_embeddings: np.ndarray | None = None,
        speaker_dataset: str = "Matthijs/cmu-arctic-xvectors",
        speaker_sample: int = 7306,
        device: str = "cuda",
    ) -> None:
        import torch
        from datasets import load_dataset
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"

        self.processor = SpeechT5Processor.from_pretrained(model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_id).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id).to(self.device)

        if speaker_embeddings is None:
            dataset = load_dataset(speaker_dataset, split="validation")
            speaker_embeddings = np.array(dataset[speaker_sample]["xvector"])  # type: ignore[index]
        self.speaker_embeddings = (
            torch.tensor(speaker_embeddings).unsqueeze(0).to(self.device)
        )

    def synthesize(self, text: str) -> SynthesizedAudio:
        import torch

        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=self.speaker_embeddings,
                vocoder=self.vocoder,
            )
        audio = speech.cpu().numpy()
        payload = _float_to_pcm16(audio)
        sample_rate = int(self.processor.feature_extractor.sampling_rate)
        return SynthesizedAudio(payload=payload, sample_rate=sample_rate, encoding="pcm16")
