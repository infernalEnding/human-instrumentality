"""Persona LLM abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Protocol

import json
import time


@dataclass
class LLMResponse:
    text: str
    should_log_memory: bool
    emotion: str | None = None
    importance: float = 0.0
    summary: str | None = None
    state_updates: Mapping[str, Any] | None = None


class PersonaLLM(Protocol):
    def generate_reply(
        self,
        transcript: str,
        memories: List[str] | None = None,
        persona_state: List[str] | None = None,
    ) -> LLMResponse:
        ...


class RuleBasedPersonaLLM:
    """Fallback LLM that fabricates persona-like responses."""

    def __init__(self, persona_name: str = "Astra", mood: str = "curious") -> None:
        self.persona_name = persona_name
        self.mood = mood

    def generate_reply(
        self,
        transcript: str,
        memories: List[str] | None = None,
        persona_state: List[str] | None = None,
    ) -> LLMResponse:
        memories = memories or []
        persona_state = persona_state or []
        memory_hint = f" I remember {memories[0]}" if memories else ""
        state_hint = f" I'm also keeping in mind: {persona_state[0]}" if persona_state else ""
        response = (
            f"{self.persona_name}: I heard you say '{transcript}'."
            f" I'm feeling {self.mood} today.{memory_hint}{state_hint}"
        )
        should_log = "important" in transcript.lower()
        summary = None
        if should_log:
            summary = f"Discussed an important topic: {transcript[:120]}"
        return LLMResponse(
            text=response,
            should_log_memory=should_log,
            emotion=self.mood,
            importance=0.8 if should_log else 0.2,
            summary=summary,
            state_updates=None,
        )


class HuggingFacePersonaLLM:
    """Persona-aware response generator backed by a Hugging Face model."""

    def __init__(
        self,
        model_id: str,
        *,
        persona_name: str = "Astra",
        persona_backstory: str | None = None,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        device_map: str | dict | None = "auto",
        pipeline_factory=None,
        max_retries: int = 2,
        retry_delay: float = 0.25,
        max_latency: float | None = 30.0,
        fallback_response: str | None = None,
    ) -> None:
        if pipeline_factory is None:
            from transformers import pipeline

            pipeline_factory = pipeline

        generation_kwargs = {
            "model": model_id,
            "torch_dtype": None,
            "device_map": device_map,
        }

        self._pipeline = pipeline_factory(
            "text-generation",
            **generation_kwargs,
        )
        self.persona_name = persona_name
        self.persona_backstory = persona_backstory or (
            f"You are {persona_name}, an empathetic AI companion who listens closely and "
            "offers thoughtful, emotionally aware replies."
        )
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max(1, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.max_latency = max_latency
        self.fallback_response = fallback_response or (
            "I'm having trouble speaking right now, but I'm still here with you."
        )

    def _build_prompt(
        self,
        transcript: str,
        memories: List[str] | None,
        persona_state: List[str] | None,
    ) -> List[dict[str, str]]:
        memory_lines = "\n".join(f"- {memory}" for memory in (memories or []))
        memory_block = (
            "Relevant memories you recall from past conversations:\n"
            f"{memory_lines}\n\n"
            if memory_lines
            else ""
        )
        state_lines = "\n".join(f"- {line}" for line in (persona_state or []))
        state_block = (
            "Current persona state and preferences:\n"
            f"{state_lines}\n\n"
            if state_lines
            else ""
        )
        return [
            {
                "role": "system",
                "content": (
                    f"{self.persona_backstory}\n"
                    "Always answer in JSON with the following schema:\n"
                    "{\n"
                    '  "reply": string,\n'
                    '  "log_memory": boolean,\n'
                    '  "emotion": string,\n'
                    '  "importance": number between 0 and 1,\n'
                    '  "summary": string or null,\n'
                    '  "state_updates": object with optional medium_term, hobbies, artistic_likes, relationships\n'
                    "}\n"
                    "The reply should be natural conversational text."
                ),
            },
            {
                "role": "user",
                "content": f"{memory_block}{state_block}Latest user transcript:\n{transcript}",
            },
        ]

    def _parse_response(self, raw: str) -> LLMResponse:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {
                "reply": raw.strip(),
                "log_memory": False,
                "emotion": "neutral",
                "importance": 0.3,
                "summary": None,
                "state_updates": None,
            }

        raw_importance = payload.get("importance", 0.0)
        try:
            importance = float(raw_importance)
        except (TypeError, ValueError):
            importance = 0.0
        updates = payload.get("state_updates")
        if not isinstance(updates, Mapping):
            updates = None
        return LLMResponse(
            text=str(payload.get("reply", "")).strip(),
            should_log_memory=bool(payload.get("log_memory", False)),
            emotion=(payload.get("emotion") or "neutral"),
            importance=max(0.0, min(1.0, importance)),
            summary=payload.get("summary") if payload.get("summary") not in (None, "") else None,
            state_updates=updates,
        )

    def generate_reply(
        self,
        transcript: str,
        memories: List[str] | None = None,
        persona_state: List[str] | None = None,
    ) -> LLMResponse:
        messages = self._build_prompt(transcript, memories, persona_state)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()
                raw_text = self._invoke_pipeline(messages)
                duration = time.perf_counter() - start
                if self.max_latency and duration > self.max_latency:
                    raise TimeoutError(
                        f"LLM response exceeded max latency ({duration:.2f}s)"
                    )
                response = self._parse_response(raw_text)
                if not response.text:
                    raise ValueError("LLM returned empty reply")
                return response
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        fallback = LLMResponse(
            text=f"{self.fallback_response} (debug: {last_error})" if last_error else self.fallback_response,
            should_log_memory=False,
            emotion="neutral",
            importance=0.1,
            summary=None,
        )
        return fallback

    def _invoke_pipeline(self, messages: List[dict[str, str]]) -> str:
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
                return_full_text=False,
            )
            raw_text = outputs[0]["generated_text"]
        else:
            prompt = "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
            )
            raw_text = outputs[0]["generated_text"]

        return raw_text
