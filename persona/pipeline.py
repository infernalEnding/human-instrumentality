"""High-level orchestration for the persona pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .audio import AudioFrame, AudioSegment
from .llm import PersonaLLM
from .memory import MemoryLogger
from .planner import ResponsePlan, ResponsePlanner
from .state import PersonaStateManager
from .stt import Transcriber
from .tts import SpeechSynthesizer, SynthesizedAudio
from .vad import EnergyVAD, VADDecision


@dataclass
class PipelineOutput:
    transcription: str
    plan: ResponsePlan
    audio: SynthesizedAudio
    start_time: float | None = None
    end_time: float | None = None
    speaker_ctx: object | None = None


class PersonaPipeline:
    def __init__(
        self,
        *,
        vad: EnergyVAD,
        transcriber: Transcriber,
        llm: PersonaLLM,
        planner: ResponsePlanner,
        synthesizer: SpeechSynthesizer,
        memory_logger: MemoryLogger | None = None,
        persona_state_manager: PersonaStateManager | None = None,
        memory_window: int = 3,
        memory_importance_threshold: float = 0.5,
    ) -> None:
        self.vad = vad
        self.transcriber = transcriber
        self.llm = llm
        self.planner = planner
        self.synthesizer = synthesizer
        self.memory_logger = memory_logger
        self.persona_state_manager = persona_state_manager
        self.memory_window = max(0, memory_window)
        self.memory_importance_threshold = max(0.0, memory_importance_threshold)

    def process_frames(self, frames: Iterable[AudioFrame]) -> List[PipelineOutput]:
        self.vad.reset()
        outputs: List[PipelineOutput] = []
        for frame in frames:
            outputs.extend(self.process_stream_frame(frame))
        outputs.extend(self.flush())
        return outputs

    def process_stream_frame(self, frame: AudioFrame) -> List[PipelineOutput]:
        outputs: List[PipelineOutput] = []
        decisions = self.vad.process_frame(frame)
        outputs.extend(self._consume_decisions(decisions))
        return outputs

    def flush(self) -> List[PipelineOutput]:
        decisions = self.vad.flush()
        return self._consume_decisions(decisions)

    def process_utterance(
        self,
        segment: AudioSegment,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
        speaker_ctx: object | None = None,
    ) -> PipelineOutput | None:
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

    def _process_segment(
        self, decision: VADDecision, *, speaker_ctx: object | None = None
    ) -> PipelineOutput | None:
        transcription = self.transcriber.transcribe(decision.segment)
        transcript_text = transcription.text.strip() if transcription.text else ""
        if not transcript_text:
            return None
        speaker_id: str | int | None = None
        if speaker_ctx is not None:
            speaker_id = getattr(speaker_ctx, "user_id", None)
        if speaker_id is None and isinstance(self, DiscordSpeakerPipeline):
            speaker_id = self.speaker_id
        elif speaker_id is not None:
            speaker_id = str(speaker_id)
        if self.memory_logger:
            memory_text = self.memory_logger.format_entries_for_prompt(
                limit=self.memory_window,
                min_importance=self.memory_importance_threshold,
                speaker_id=speaker_id,
            )
        else:
            memory_text = []
        persona_state = self._prepare_prompt_context()
        llm_response = self.llm.generate_reply(
            transcript_text,
            memory_text,
            persona_state=persona_state,
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
                importance=log_importance,
                summary=plan.memory_summary,
                speaker_id=speaker_id,
            )
        self._persist_state_updates(plan)
        audio = self.synthesizer.synthesize(plan.response_text)
        return PipelineOutput(
            transcription=transcript_text,
            plan=plan,
            audio=audio,
            start_time=transcription.start_time,
            end_time=transcription.end_time,
            speaker_ctx=speaker_ctx,
        )

    def _prepare_prompt_context(self) -> List[str] | None:
        if self.persona_state_manager:
            return self.persona_state_manager.prompt_context()
        return None

    def _persist_state_updates(self, plan: ResponsePlan) -> None:
        if self.persona_state_manager and plan.state_updates:
            self.persona_state_manager.apply_updates(plan.state_updates)


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

    def _prepare_prompt_context(self) -> List[str] | None:
        self._ensure_relationship()
        base_context = super()._prepare_prompt_context() or []
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
