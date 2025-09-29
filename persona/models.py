"""Model profile utilities for configuring Hugging Face components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ModelProfile:
    """Pre-configured collection of Hugging Face model identifiers."""

    name: str
    description: str
    asr_model: str
    llm_model: str
    tts_model: str
    tts_vocoder: str | None = None
    tts_speaker_dataset: str | None = None
    tts_speaker_sample: int | None = None


def _default_profiles() -> Dict[str, ModelProfile]:
    """Return the built-in model profile registry."""

    return {
        "balanced": ModelProfile(
            name="balanced",
            description=(
                "High quality Whisper, Mixtral, and SpeechT5 models tuned for an RTX 5090."
            ),
            asr_model="openai/whisper-large-v3",
            llm_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            tts_model="microsoft/speecht5_tts",
            tts_vocoder="microsoft/speecht5_hifigan",
            tts_speaker_dataset="Matthijs/cmu-arctic-xvectors",
            tts_speaker_sample=7306,
        ),
        "light": ModelProfile(
            name="light",
            description=(
                "Lower VRAM footprint using Whisper base, Phi-3-mini, and Bark-small."
            ),
            asr_model="openai/whisper-small",
            llm_model="microsoft/Phi-3-mini-4k-instruct",
            tts_model="suno/bark-small",
            tts_vocoder=None,
            tts_speaker_dataset=None,
            tts_speaker_sample=None,
        ),
        "edge": ModelProfile(
            name="edge",
            description=(
                "Latency-focused stack pairing Distil-Whisper, Llama-3.1-8B, and Edge-TTS."
            ),
            asr_model="distil-whisper/distil-large-v3",
            llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tts_model="microsoft/speecht5_tts",
            tts_vocoder="microsoft/speecht5_hifigan",
            tts_speaker_dataset="Matthijs/cmu-arctic-xvectors",
            tts_speaker_sample=1188,
        ),
    }


_PROFILES: Dict[str, ModelProfile] = _default_profiles()

DEFAULT_PROFILE_NAME = "balanced"


def list_model_profiles() -> List[str]:
    """Return the sorted list of available profile names."""

    return sorted(_PROFILES.keys())


def get_model_profile(name: str) -> ModelProfile:
    """Fetch a model profile by name."""

    try:
        return _PROFILES[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown model profile: {name}") from exc


def describe_model_profiles() -> List[str]:
    """Return human-readable lines describing each profile."""

    lines: List[str] = []
    for name in list_model_profiles():
        profile = _PROFILES[name]
        lines.append(f"{name}: {profile.description}")
        lines.append(
            "  ASR: {asr}\n  LLM: {llm}\n  TTS: {tts}".format(
                asr=profile.asr_model,
                llm=profile.llm_model,
                tts=profile.tts_model,
            )
        )
        if profile.tts_vocoder:
            lines.append(f"  Vocoder: {profile.tts_vocoder}")
        if profile.tts_speaker_dataset:
            lines.append(
                f"  Speaker dataset: {profile.tts_speaker_dataset}"
            )
        if profile.tts_speaker_sample is not None:
            lines.append(f"  Speaker sample: {profile.tts_speaker_sample}")
    return lines


def apply_profile_defaults(namespace, profile: ModelProfile, fields: Iterable[str]) -> None:
    """Populate missing attributes on a namespace from the provided profile."""

    for field in fields:
        if hasattr(namespace, field):
            continue
        value = getattr(profile, field)
        setattr(namespace, field, value)

