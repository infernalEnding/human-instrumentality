"""Response planning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .llm import LLMResponse


@dataclass
class ResponsePlan:
    response_text: str
    log_memory: bool
    memory_summary: Optional[str]
    emotion: Optional[str]
    importance: float
    state_updates: Mapping[str, object] | None = None


class ResponsePlanner:
    """Decides post-processing actions for LLM output."""

    def create_plan(self, llm_response: LLMResponse) -> ResponsePlan:
        return ResponsePlan(
            response_text=llm_response.text,
            log_memory=llm_response.should_log_memory,
            memory_summary=llm_response.summary,
            emotion=llm_response.emotion,
            importance=llm_response.importance,
            state_updates=llm_response.state_updates,
        )
