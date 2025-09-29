# Work Plan: Persona Voice Pipeline

This tracker keeps the implementation steps aligned with roadmap items 2 through 5. Each task captures status, next actions, and relevant notes so we can update progress as we deliver features.

## 2. Prototype the streaming audio loop
| Task | Status | Notes / Next Actions |
| --- | --- | --- |
| Implement microphone capture with configurable backends (PyAudio/SoundDevice fallback). | Complete | `persona.audio` exposes `MicrophoneStream` with SoundDevice/PyAudio backends and a synthetic frame source for tests. |
| Add voice-activity detection gating to produce utterance chunks with timestamps. | Complete | `EnergyVAD` now runs incrementally, tracks timestamps, and surfaces `VADDecision` boundaries for downstream modules. |
| Integrate a low-latency Whisper inference call path for incremental transcription. | Complete | The Hugging Face transcriber requests timestamped chunks, averages confidences, and tolerates pipelines without timestamp support. |
| Validate the loop via CLI demo that logs detected utterances. | Complete | `persona.cli --mic` streams live frames through the pipeline, prints transcript summaries, and supports streaming WAV export. |

## 3. Implement and tune the memory service
| Task | Status | Notes / Next Actions |
| --- | --- | --- |
| Finalize Markdown logging format (metadata headers, tagging schema, emotion markers). | Complete | Memory entries include persona metadata, auto-tag heuristics, and emotion markers stored with microsecond timestamps. |
| Build importance scoring heuristics to decide when to write memories. | Complete | MemoryLogger blends LLM directives with keyword boosts and cooldown penalties before persisting reflections. |
| Expose retrieval interface for persona prompts with recency/semantic filters. | Complete | `MemoryLogger.retrieve` returns scored summaries; follow up with semantic search once embeddings are available. |
| Add tests covering serialization, rotation, and retrieval ordering. | Complete | Added `tests/test_memory.py` with metadata parsing, heuristics, and ordering checks. |

## 4. Harden the dialogue planner and LLM orchestration
| Task | Status | Notes / Next Actions |
| --- | --- | --- |
| Standardize JSON schema for persona reasoning, response planning, and memory directives. | Complete | `HuggingFacePersonaLLM` enforces JSON schema with reply/log_memory/emotion/importance/summary fields; continue validating outputs. |
| Add guardrails (validation, retries, safe defaults) around local LLM calls. | Complete | `HuggingFacePersonaLLM` now retries transient failures, enforces latency caps, sanitizes schema outputs, and falls back to a safe reply. |
| Incorporate configurable system/persona prompts and runtime parameter loading. | Complete | CLI accepts persona prompt files and environment overrides before constructing the Hugging Face persona wrapper. |
| Create regression tests for planner outputs and LLM orchestration paths. | Complete | `tests/test_llm.py` exercises rule-based + HF adapters; add coverage for error handling once guardrails land. |

## 5. Add Discord transport integration
| Task | Status | Notes / Next Actions |
| --- | --- | --- |
| Evaluate approach (mic replacement vs. bot account) and capture requirements. | Complete | Runbook documents the trade-offs and recommends starting with mic proxying before moving to a standalone bot. |
| Implement PCM streaming bridge into `PersonaPipeline.process_frames`. | Complete | `DiscordVoiceBridge.receive_discord_pcm` wraps PCM packets into timestamped frames and reuses the streaming pipeline. |
| Support synthesized audio playback into Discord voice channels. | Complete | The voice bridge dispatches synthesized PCM payloads through the injected `send_audio` callback for Opus encoding/playback. |
| Document setup instructions and operational considerations. | Complete | The runbook covers live microphone streaming, Discord transport wiring, and safety levers for deployments. |

> _Update the tables as work completes to keep this plan current._
