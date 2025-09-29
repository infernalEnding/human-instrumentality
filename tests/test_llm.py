from __future__ import annotations

from typing import Any

from persona.llm import HuggingFacePersonaLLM


class StubPipeline:
    def __init__(self, response: str) -> None:
        self.response = response
        self.tokenizer = None

    def __call__(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        return [{"generated_text": self.response}]


class StubTokenizer:
    def __init__(self) -> None:
        self.history: list[Any] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.history.append(messages)
        return "prompt"


class StubChatPipeline(StubPipeline):
    def __init__(self, response: str) -> None:
        super().__init__(response)
        self.tokenizer = StubTokenizer()


def test_huggingface_llm_parses_json() -> None:
    llm = HuggingFacePersonaLLM(
        "fake/model",
        pipeline_factory=lambda *_, **__: StubPipeline(
            "{\"reply\": \"Hello\", \"log_memory\": true, \"emotion\": \"calm\", \"importance\": 0.6, \"summary\": \"Greeting\"}"
        ),
    )

    response = llm.generate_reply("Hi there", memories=["Met yesterday"])
    assert response.text == "Hello"
    assert response.should_log_memory is True
    assert response.emotion == "calm"
    assert response.importance == 0.6
    assert response.summary == "Greeting"
    assert response.state_updates is None


def test_huggingface_llm_handles_non_json() -> None:
    llm = HuggingFacePersonaLLM(
        "fake/model",
        pipeline_factory=lambda *_, **__: StubPipeline("Plain text response"),
    )

    response = llm.generate_reply("Tell me something")
    assert response.text == "Plain text response"
    assert response.should_log_memory is False
    assert response.state_updates is None


def test_huggingface_llm_uses_chat_template_when_available() -> None:
    pipeline = StubChatPipeline("{\"reply\": \"Howdy\", \"log_memory\": false, \"emotion\": \"cheerful\", \"importance\": 0.4, \"summary\": null}")
    llm = HuggingFacePersonaLLM(
        "fake/model",
        pipeline_factory=lambda *_, **__: pipeline,
    )

    response = llm.generate_reply("Howdy?", memories=["We love greetings"])
    assert response.text == "Howdy"
    assert pipeline.tokenizer.history, "chat template should be applied"


class FailingPipeline:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any):
        self.calls += 1
        raise RuntimeError("decoder crashed")


def test_huggingface_llm_retries_and_falls_back() -> None:
    failing = FailingPipeline()
    llm = HuggingFacePersonaLLM(
        "fake/model",
        pipeline_factory=lambda *_, **__: failing,
        max_retries=2,
        retry_delay=0.0,
        fallback_response="Sorry, can't respond",
    )

    response = llm.generate_reply("Hello there")
    assert "Sorry" in response.text
    assert response.should_log_memory is False
    assert failing.calls == 2


def test_huggingface_llm_handles_invalid_importance() -> None:
    llm = HuggingFacePersonaLLM(
        "fake/model",
        pipeline_factory=lambda *_, **__: StubPipeline(
            '{"reply": "Hi", "log_memory": false, "emotion": "neutral", "importance": "not-a-number", "summary": null}'
        ),
    )

    response = llm.generate_reply("Test transcript")
    assert response.importance == 0.0


def test_huggingface_llm_parses_state_updates() -> None:
    llm = HuggingFacePersonaLLM(
        "fake/model",
        pipeline_factory=lambda *_, **__: StubPipeline(
            '{"reply": "Sure", "log_memory": false, "emotion": "neutral", "importance": 0.2, "summary": null, "state_updates": {"hobbies": {"add": ["guitar"]}}}'
        ),
    )

    response = llm.generate_reply(
        "Remember our jam session",
        persona_state=["Hobbies: piano"],
    )
    assert response.state_updates == {"hobbies": {"add": ["guitar"]}}
