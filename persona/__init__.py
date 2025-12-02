"""Core modules for the Human Instrumentality persona pipeline."""

from .audio import AudioFrame, AudioSegment, write_wav
from .vad import EnergyVAD
from .stt import EchoTranscriber, HuggingFaceTranscriber, TranscriptionResult, Transcriber
from .llm import (
    HuggingFacePersonaLLM,
    LLMNarrativeSuggestion,
    LLMResponse,
    PersonaLLM,
    RuleBasedPersonaLLM,
)
from .memory import MemoryLogger
from .planner import ResponsePlan, ResponsePlanner
from .models import (
    ModelProfile,
    apply_profile_defaults,
    describe_model_profiles,
    get_model_profile,
    list_model_profiles,
)
from .sentiment import LightweightSentimentAnalyzer, SentimentAnalyzer, SentimentResult
from .state import PersonaStateManager
from .tts import DebugSynthesizer, HuggingFaceSynthesizer, SynthesizedAudio, SpeechSynthesizer
from .pipeline import DiscordSpeakerPipeline, PersonaPipeline, PipelineOutput

__all__ = [
    "AudioFrame",
    "AudioSegment",
    "write_wav",
    "EnergyVAD",
    "TranscriptionResult",
    "Transcriber",
    "EchoTranscriber",
    "HuggingFaceTranscriber",
    "LLMNarrativeSuggestion",
    "LLMResponse",
    "PersonaLLM",
    "RuleBasedPersonaLLM",
    "HuggingFacePersonaLLM",
    "MemoryLogger",
    "SentimentAnalyzer",
    "SentimentResult",
    "LightweightSentimentAnalyzer",
    "PersonaStateManager",
    "ResponsePlan",
    "ResponsePlanner",
    "SynthesizedAudio",
    "SpeechSynthesizer",
    "DebugSynthesizer",
    "HuggingFaceSynthesizer",
    "PersonaPipeline",
    "DiscordSpeakerPipeline",
    "PipelineOutput",
    "ModelProfile",
    "list_model_profiles",
    "get_model_profile",
    "describe_model_profiles",
    "apply_profile_defaults",
]
