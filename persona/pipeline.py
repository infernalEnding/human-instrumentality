"""High-level orchestration for the persona pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

from .audio import AudioFrame, AudioSegment
from .emotion import AudioEmotionAnalyzer, AudioEmotionResult
from .denoiser import Denoiser, NoOpDenoiser
from .llm import PersonaLLM
from .memory import MemoryLogger
from .narrative import NarrativeStore, make_event
from .planner import ResponsePlan, ResponsePlanner
from .sentiment import SentimentAnalyzer
from .state import PersonaStateManager
from .stt import Transcriber
from .tts import SpeechSynthesizer, SynthesizedAudio
from .vad import EnergyVAD, VADDecision, VoiceActivityDetector


@dataclass
class PipelineOutput:
    transcription: str
    plan: ResponsePlan
    audio: SynthesizedAudio
    audio_emotion: AudioEmotionResult | None = None
    start_time: float | None = None
    end_time: float | None = None
    speaker_ctx: object | None = None


class PersonaPipeline:
    def __init__(
        self,
        *,
        vad: VoiceActivityDetector | None = None,
        vad_factory: Callable[[], VoiceActivityDetector] | None = None,
        denoiser_factory: Callable[[], Denoiser] | None = None,
        transcriber: Transcriber,
        llm: PersonaLLM,
        planner: ResponsePlanner,
        synthesizer: SpeechSynthesizer,
        memory_logger: MemoryLogger | None = None,
        persona_state_manager: PersonaStateManager | None = None,
        narrative_store: NarrativeStore | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        audio_emotion_analyzer: AudioEmotionAnalyzer | None = None,
        memory_window: int = 3,
        memory_importance_threshold: float = 0.5,
        narrative_importance_threshold: float | None = None,
        narrative_window: int | None = None,
    ) -> None:
        if vad_factory is None:
            if vad is None:
                vad_factory = lambda: EnergyVAD()
            else:
                vad_factory = lambda: vad
        self.vad_factory = vad_factory
        self.vad = self.vad_factory()
        self.denoiser_factory = denoiser_factory or (lambda: NoOpDenoiser())
        self.denoiser = self.denoiser_factory()
        self.transcriber = transcriber
        self.llm = llm
        self.planner = planner
        self.synthesizer = synthesizer
        self.memory_logger = memory_logger
        self.persona_state_manager = persona_state_manager
        self.narrative_store = narrative_store
        self.sentiment_analyzer = sentiment_analyzer
        self.audio_emotion_analyzer = audio_emotion_analyzer
        self.memory_window = max(0, memory_window)
        self.memory_importance_threshold = max(0.0, memory_importance_threshold)
        self.narrative_importance_threshold = (
            narrative_importance_threshold
            if narrative_importance_threshold is not None
            else self.memory_importance_threshold
        )
        self.narrative_window = (
            max(0, narrative_window)
            if narrative_window is not None
            else max(0, self.memory_window)
        )

    def process_frames(
        self, frames: Iterable[AudioFrame], *, speaker_ctx: object | None = None
    ) -> List[PipelineOutput]:
        self.vad.reset()
        self.denoiser.reset()
        outputs: List[PipelineOutput] = []
        for frame in frames:
            outputs.extend(self.process_stream_frame(frame, speaker_ctx=speaker_ctx))
        outputs.extend(self.flush(speaker_ctx=speaker_ctx))
        return outputs

    def process_stream_frame(
        self, frame: AudioFrame, *, speaker_ctx: object | None = None
    ) -> List[PipelineOutput]:
        outputs: List[PipelineOutput] = []
        frame = self.denoiser.process_frame(frame)
        decisions = self.vad.process_frame(frame)
        outputs.extend(self._consume_decisions(decisions, speaker_ctx=speaker_ctx))
        return outputs

    def flush(self, *, speaker_ctx: object | None = None) -> List[PipelineOutput]:
        decisions = self.vad.flush()
        return self._consume_decisions(decisions, speaker_ctx=speaker_ctx)

    def process_utterance(
        self,
        segment: AudioSegment,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        speaker_ctx: object | None = None,
        already_denoised: bool = False,
    ) -> PipelineOutput | None:
        if not already_denoised:
            segment = self._denoise_segment(segment)
        decision = VADDecision(
            segment=segment,
            confidence=1.0,
            start_time=start_time or segment.start_time,
            end_time=end_time or segment.end_time,
        )
        return self._process_segment(decision, speaker_ctx=speaker_ctx)

    def _consume_decisions(
        self, decisions: Iterable[VADDecision], *, speaker_ctx: object | None = None
    ) -> List[PipelineOutput]:
        outputs: List[PipelineOutput] = []
        for decision in decisions:
            output = self._process_segment(decision, speaker_ctx=speaker_ctx)
            if output:
                outputs.append(output)
        return outputs

    def _denoise_segment(self, segment: AudioSegment) -> AudioSegment:
        frames = [self.denoiser.process_frame(frame) for frame in segment.frames]
        return AudioSegment(frames=frames)

    def _process_segment(
        self, decision: VADDecision, *, speaker_ctx: object | None = None
    ) -> PipelineOutput | None:
        transcription = self.transcriber.transcribe(decision.segment)
        transcript_text = transcription.text.strip() if transcription.text else ""
        if not transcript_text:
            return None
        sentiment_note: str | None = None
        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.analyze(transcript_text)
            sentiment_note = f"Sentiment: {sentiment.label} ({sentiment.score:.2f})"
        audio_emotion: AudioEmotionResult | None = None
        if self.audio_emotion_analyzer:
            audio_emotion = self.audio_emotion_analyzer.analyze(decision.segment)
        speaker_id: str | int | None = None
        if speaker_ctx is not None:
            speaker_id = getattr(speaker_ctx, "user_id", None)
        if speaker_id is None and isinstance(self, DiscordSpeakerPipeline):
            speaker_id = self.speaker_id
        elif speaker_id is not None:
            speaker_id = str(speaker_id)
        if self.memory_logger:
            memory_text = self.memory_logger.recall_for_prompt(
                transcript=transcript_text,
                limit=self.memory_window,
                min_importance=self.memory_importance_threshold,
                speaker_id=speaker_id,
            )
        else:
            memory_text = []
        persona_state = self._prepare_prompt_context(
            speaker_id=speaker_id if speaker_id is not None else None
        )
        llm_response = self.llm.generate_reply(
            transcript_text,
            memory_text,
            persona_state=persona_state,
            sentiment=sentiment_note,
            audio_emotion=audio_emotion,
        )
        plan = self.planner.create_plan(llm_response)
        plan.importance = max(0.0, min(1.0, plan.importance))
        should_log = False
        if self.memory_logger:
            should_log = self.memory_logger.should_log(
                llm_flag=plan.log_memory,
                importance=plan.importance,
                transcript=transcript_text,
                response=plan.response_text,
            )
        if should_log and self.memory_logger:
            log_importance = max(plan.importance, transcription.confidence)
            self.memory_logger.log(
                transcript=transcript_text,
                response=plan.response_text,
                emotion=plan.emotion,
                sentiment=sentiment_note,
                importance=log_importance,
                summary=plan.memory_summary,
                tags=plan.memory_tags,
                speaker_id=speaker_id,
            )
        self._persist_state_updates(plan)
        self._persist_narrative_event(plan, speaker_id=speaker_id)
        audio = self.synthesizer.synthesize(plan.response_text)
        return PipelineOutput(
            transcription=transcript_text,
            plan=plan,
            audio=audio,
            audio_emotion=audio_emotion,
            start_time=transcription.start_time,
            end_time=transcription.end_time,
            speaker_ctx=speaker_ctx,
        )

    def _prepare_prompt_context(
        self, *, speaker_id: str | None = None
    ) -> List[str] | None:
        context: list[str] = []
        if self.persona_state_manager:
            context.extend(self.persona_state_manager.prompt_context())
        if self.narrative_store and self.narrative_window > 0:
            context.extend(
                self.narrative_store.format_context(
                    speaker_id=speaker_id, limit=self.narrative_window
                )
            )
        return context or None

    def _persist_state_updates(self, plan: ResponsePlan) -> None:
        if self.persona_state_manager and plan.state_updates:
            self.persona_state_manager.apply_updates(plan.state_updates)

    def _persist_narrative_event(
        self, plan: ResponsePlan, *, speaker_id: str | None
    ) -> None:
        if not self.narrative_store or not plan.narrative:
            return
        if plan.importance < self.narrative_importance_threshold:
            return
        event = make_event(
            plan.narrative,
            tags=plan.memory_tags,
            speaker_id=speaker_id,
        )
        self.narrative_store.add_event(event)


class DiscordSpeakerPipeline(PersonaPipeline):
    def __init__(
        self,
        *,
        speaker_id: str,
        speaker_display_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.speaker_id = str(speaker_id)
        self.speaker_display_name = speaker_display_name
        self._relationship_record: dict | None = None
        self._session_initialised = False

    def _ensure_relationship(self) -> None:
        if self._session_initialised or not self.persona_state_manager:
            return
        self._relationship_record = self.persona_state_manager.get_or_create_discord_relationship(
            self.speaker_id, display_name=self.speaker_display_name
        )
        self._session_initialised = True

    def _prepare_prompt_context(
        self, *, speaker_id: str | None = None
    ) -> List[str] | None:
        self._ensure_relationship()
        base_context = super()._prepare_prompt_context(speaker_id=speaker_id) or []
        if not self._relationship_record:
            return base_context
        hints = [
            "Discord safety: greet the speaker and confirm their preferred name and consent to chat.",
        ]
        if self.speaker_display_name:
            hints.append(f"Discord handle: {self.speaker_display_name}")
        if self._relationship_record.get("needs_identification"):
            hints.append(
                "This contact still needs identification—politely ask for their name and whether it's okay to continue."
            )
        return base_context + hints

    def _persist_state_updates(self, plan: ResponsePlan) -> None:
        if not self.persona_state_manager or not plan.state_updates:
            return
        relationships = plan.state_updates.get("relationships")
        removals: list[dict] = []
        if isinstance(relationships, list):
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                ids = rel.get("discord_ids") or []
                is_match = self.speaker_id in ids or not ids
                if self._relationship_record and not is_match:
                    recorded_name = self._relationship_record.get("name", "").strip().lower()
                    rel_name = str(rel.get("name", "")).strip().lower()
                    is_match = recorded_name and recorded_name == rel_name
                if not is_match:
                    continue
                if self._relationship_record:
                    existing_name = self._relationship_record.get("name")
                    updated_name = rel.get("name")
                    if existing_name and updated_name and existing_name != updated_name:
                        removals.append({"name": existing_name, "action": "remove"})
                if self.speaker_id not in ids:
                    rel.setdefault("discord_ids", [])
                    if self.speaker_id not in rel["discord_ids"]:
                        rel["discord_ids"].append(self.speaker_id)
                rel["needs_identification"] = False
                if self.speaker_display_name and not rel.get("notes"):
                    rel["notes"] = f"Discord handle: {self.speaker_display_name}"
            relationships.extend(removals)
        self.persona_state_manager.apply_updates(plan.state_updates)
