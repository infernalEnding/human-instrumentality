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

    def _consume_decisions(self, decisions: Iterable[VADDecision]) -> List[PipelineOutput]:
        outputs: List[PipelineOutput] = []
        for decision in decisions:
            output = self._process_segment(decision)
            if output:
                outputs.append(output)
        return outputs

    def _process_segment(self, decision: VADDecision) -> PipelineOutput | None:
        transcription = self.transcriber.transcribe(decision.segment)
        transcript_text = transcription.text.strip() if transcription.text else ""
        if not transcript_text:
            return None
        if self.memory_logger:
            memory_text = self.memory_logger.format_entries_for_prompt(
                limit=self.memory_window,
                min_importance=self.memory_importance_threshold,
            )
        else:
            memory_text = []
        if self.persona_state_manager:
            persona_state = self.persona_state_manager.prompt_context()
        else:
            persona_state = None
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
            )
        if self.persona_state_manager and plan.state_updates:
            self.persona_state_manager.apply_updates(plan.state_updates)
        audio = self.synthesizer.synthesize(plan.response_text)
        return PipelineOutput(
            transcription=transcript_text,
            plan=plan,
            audio=audio,
            start_time=transcription.start_time,
            end_time=transcription.end_time,
        )
