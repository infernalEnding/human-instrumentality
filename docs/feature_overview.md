# Persona Voice Assistant Feature Overview

This document outlines the planned feature set for a locally hosted AI persona capable of real-time voice interaction, Discord participation, and long-term memory logging.

## High-Level Goals
- Run all language, reasoning, and speech models locally on an RTX 5090 GPU.
- Provide a natural voice conversation experience: microphone input, speech recognition, persona-driven reasoning, and synthesized voice responses.
- Integrate with Discord, either by proxying the user's microphone or operating as an independent bot account.
- Maintain a reflective memory system by persisting the persona's notable thoughts and emotions in Markdown journals that can be reloaded into context.
- Capture medium-term persona traits—hobbies, artistic tastes, Discord relationships—in a profile the LLM can refresh dynamically.

## System Architecture
```
Audio Input --> Activity Detection --> Speech-to-Text --> Dialogue Orchestrator -->
    ├─> Persona Memory Logger (optional writes)
    └─> LLM Persona Core --> Response Planner --> Text-to-Speech --> Audio Output
```

### Core Components
1. **Audio Input & Activity Detection**
   - Capture microphone audio with low-latency streaming (e.g., PyAudio, PortAudio, or WebRTC).
   - Perform voice activity detection (VAD) to decide when to trigger a transcription cycle.

2. **Speech-to-Text (STT)**
   - Deploy a local STT model (e.g., Whisper large-v3 via Hugging Face pipelines) optimized for RTX 5090.
   - Support streaming transcription to keep latency low.

3. **Dialogue Orchestrator**
   - Manage conversation state, turn taking, and persona grounding.
   - Provide hooks for logging memories and retrieving context.

4. **Persona Memory System**
   - Store Markdown journal entries in a versioned directory tree.
   - Tag entries by topic, emotion, or interaction partner.
   - Periodically load relevant memories into the LLM context using retrieval (e.g., embeddings) or scheduled prompts.
5. **Persona State Profile**
   - Persist medium-term reflections, hobbies, and artistic preferences outside of the high-churn Markdown memory stream.
   - Maintain a roster of recognised Discord contacts, tracking relationship sentiment and notable history.
   - Surface condensed snapshots of this profile to the LLM before each reply and merge structured updates the model emits.

6. **LLM Persona Core**
   - Run a local instruction-tuned LLM (e.g., Mixtral, Llama 3) via Hugging Face with GPU acceleration.
   - Maintain persona traits, backstory, and interaction style while emitting structured JSON for downstream planning.

7. **Response Planner**
   - Convert raw LLM text into structured responses (e.g., handle actions, emotions, or Discord-specific commands).
   - Decide whether to append new Markdown memories based on message salience.

8. **Text-to-Speech (TTS)**
   - Use a local neural voice model (e.g., SpeechT5 with HiFi-GAN vocoder) to synthesize persona voice via Hugging Face APIs.
   - Provide configurable voice timbre and speaking speed.

9. **Audio Output**
   - Play synthesized audio locally or stream to Discord voice channels.
   - Support push-to-talk or continuous speech modes.

## Discord Integration Options
1. **Microphone Replacement**
   - Capture Discord audio via virtual audio cable.
   - Feed transcriptions into the persona pipeline and return TTS audio as microphone output.

2. **Dedicated Bot User**
   - Authenticate a Discord bot that joins voice channels.
   - Stream audio input using Discord's voice gateway and produce persona responses as TTS.

## Memory Logging Strategy
- **Triggering Conditions**: Log after emotionally significant exchanges, major decisions, or explicit user instructions.
- **Markdown Structure**: Each entry contains timestamp, context summary, emotional state, and any commitments.
- **Retention & Retrieval**: Use embeddings or keyword search to find relevant entries. Load a curated subset into the LLM prompt when generating responses.

## Operational Considerations
- **Performance**: Optimize GPU usage by pinning STT, LLM, and TTS workloads to separate CUDA streams when possible.
- **Safety & Privacy**: Store logs locally with encryption options. Provide user controls for memory deletion.
- **Extensibility**: Design modular interfaces for swapping models or integrating additional platforms (e.g., VRChat, Zoom).

## Next Steps
1. Evaluate candidate STT, LLM, and TTS models compatible with RTX 5090.
2. Prototype the audio pipeline with streaming transcription and playback.
3. Implement the persona memory logging service and retrieval strategy.
4. Develop Discord integration module and define deployment workflow.
5. Follow the operational guidance in the [runbook](runbook.md) to harden your local
   environment and deployment practices.

## Implementation Snapshot
- `persona.audio`, `persona.vad`: PCM helpers and energy-based voice-activity detection usable for early simulations.
- `persona.stt`, `persona.llm`: Hugging Face integrations for Whisper-class ASR and JSON-structured persona replies, plus a rule-based fallback persona.
- `persona.memory`: Markdown logging with retrieval utilities to surface prior reflections.
- `persona.pipeline`, `persona.tts`: orchestration and SpeechT5-powered synthesis for end-to-end runs with optional debug shims.
- `persona.discord_integration`: lightweight bridge for wiring the pipeline into Discord transport adapters.
