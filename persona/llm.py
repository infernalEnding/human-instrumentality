"""Persona LLM abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Protocol, TYPE_CHECKING

import json
import os
import time

import requests

if TYPE_CHECKING:
    from .emotion import AudioEmotionResult


@dataclass
class LLMNarrativeSuggestion:
    title: str | None = None
    summary: str | None = None
    tone: str | None = None


@dataclass
class LLMResponse:
    text: str
    should_log_memory: bool
    emotion: str | None = None
    importance: float = 0.0
    summary: str | None = None
    state_updates: Mapping[str, Any] | None = None
    memory_tags: list[str] = field(default_factory=list)
    narrative: LLMNarrativeSuggestion | None = None


def _parse_structured_response(raw: str) -> LLMResponse:
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
            "memory_tags": [],
            "narrative": None,
        }

    raw_importance = payload.get("importance", 0.0)
    try:
        importance = float(raw_importance)
    except (TypeError, ValueError):
        importance = 0.0
    updates = payload.get("state_updates")
    if not isinstance(updates, Mapping):
        updates = None

    raw_tags = payload.get("memory_tags")
    if isinstance(raw_tags, list):
        memory_tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    else:
        memory_tags = []

    narrative: LLMNarrativeSuggestion | None = None
    narrative_payload = payload.get("narrative")
    if isinstance(narrative_payload, Mapping):
        title = str(narrative_payload.get("title", "")).strip()
        summary = str(narrative_payload.get("summary", "")).strip()
        tone = str(narrative_payload.get("tone", "")).strip()
        if title or summary or tone:
            narrative = LLMNarrativeSuggestion(
                title=title or None, summary=summary or None, tone=tone or None
            )
    return LLMResponse(
        text=str(payload.get("reply", "")).strip(),
        should_log_memory=bool(payload.get("log_memory", False)),
        emotion=(payload.get("emotion") or "neutral"),
        importance=max(0.0, min(1.0, importance)),
        summary=payload.get("summary") if payload.get("summary") not in (None, "") else None,
        state_updates=updates,
        memory_tags=memory_tags,
        narrative=narrative,
    )


class PersonaLLM(Protocol):
    def generate_reply(
        self,
        transcript: str,
        memories: List[str] | None = None,
        persona_state: List[str] | None = None,
        sentiment: str | None = None,
        audio_emotion: "AudioEmotionResult" | None = None,
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
        sentiment: str | None = None,
        audio_emotion: "AudioEmotionResult" | None = None,
    ) -> LLMResponse:
        memories = memories or []
        persona_state = persona_state or []
        sentiment_hint = f" The speaker seems {sentiment}." if sentiment else ""
        emotion_hint = (
            f" I sense a {audio_emotion.label} tone."
            if audio_emotion is not None
            else ""
        )
        memory_hint = f" I remember {memories[0]}" if memories else ""
        state_hint = f" I'm also keeping in mind: {persona_state[0]}" if persona_state else ""
        response = (
            f"{self.persona_name}: I heard you say '{transcript}'."
            f" I'm feeling {self.mood} today.{memory_hint}{state_hint}{sentiment_hint}{emotion_hint}"
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
        sentiment: str | None,
        audio_emotion: "AudioEmotionResult" | None,
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
        sentiment_block = f"Observed sentiment: {sentiment}\n\n" if sentiment else ""
        emotion_block = (
            f"Detected vocal emotion: {audio_emotion.label} ({audio_emotion.score:.2f})\n\n"
            if audio_emotion
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
                    '  "memory_tags": array of short keywords for logging,\n'
                    '  "state_updates": object with optional medium_term, hobbies, artistic_likes, relationships,\n'
                    '  "narrative": object or null with title, summary, and tone fields\n'
                    "}\n"
                    "The reply should be natural conversational text."
                    " Use the provided sentiment and vocal emotion cues to guide tone,"
                    " and include concise narrative context that captures the evolving story arc."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{memory_block}{state_block}{sentiment_block}{emotion_block}"
                    f"Latest user transcript:\n{transcript}"
                ),
            },
        ]

    def _parse_response(self, raw: str) -> LLMResponse:
        return _parse_structured_response(raw)

    def generate_reply(
        self,
        transcript: str,
        memories: List[str] | None = None,
        persona_state: List[str] | None = None,
        sentiment: str | None = None,
        audio_emotion: "AudioEmotionResult" | None = None,
    ) -> LLMResponse:
        messages = self._build_prompt(
            transcript, memories, persona_state, sentiment, audio_emotion
        )
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


class OpenRouterPersonaLLM:
    """Persona-aware response generator backed by the OpenRouter API."""

    def __init__(
        self,
        model: str,
        *,
        persona_name: str = "Astra",
        persona_backstory: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        base_url: str = "https://openrouter.ai/api/v1",
        request_timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.persona_backstory = persona_backstory or (
            f"You are {persona_name}, an empathetic AI companion who listens closely and "
            "offers thoughtful, emotionally aware replies."
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY or pass api_key."
            )

    def _build_prompt(
        self,
        transcript: str,
        memories: List[str] | None,
        persona_state: List[str] | None,
        sentiment: str | None,
        audio_emotion: "AudioEmotionResult" | None,
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
        sentiment_block = f"Observed sentiment: {sentiment}\n\n" if sentiment else ""
        emotion_block = (
            f"Detected vocal emotion: {audio_emotion.label} ({audio_emotion.score:.2f})\n\n"
            if audio_emotion
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
                    '  "memory_tags": array of short keywords for logging,\n'
                    '  "state_updates": object with optional medium_term, hobbies, artistic_likes, relationships,\n'
                    '  "narrative": object or null with title, summary, and tone fields\n'
                    "}\n"
                    "The reply should be natural conversational text."
                    " Use the provided sentiment and vocal emotion cues to guide tone,"\
                    " and include concise narrative context that captures the evolving story arc."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{memory_block}{state_block}{sentiment_block}{emotion_block}"
                    f"Latest user transcript:\n{transcript}"
                ),
            },
        ]

    def _invoke_api(self, messages: List[dict[str, str]]) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices")
        if not choices:
            raise ValueError("OpenRouter returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise ValueError("OpenRouter returned empty content")
        return str(content)

    def generate_reply(
        self,
        transcript: str,
        memories: List[str] | None = None,
        persona_state: List[str] | None = None,
        sentiment: str | None = None,
        audio_emotion: "AudioEmotionResult" | None = None,
    ) -> LLMResponse:
        messages = self._build_prompt(
            transcript, memories, persona_state, sentiment, audio_emotion
        )
        raw_text = self._invoke_api(messages)
        response = _parse_structured_response(raw_text)
        if not response.text:
            raise ValueError("LLM returned empty reply")
        return response
