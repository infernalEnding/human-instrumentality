# Human Instrumentality

This repository implements a production-ready, locally hosted AI persona pipeline featuring Hugging Face-powered speech-to-text, persona reasoning, and neural text-to-speech. It still ships light-weight debug shims for unit testing, but the default configuration now targets RTX-class GPUs capable of running the full speech stack offline. See [docs/feature_overview.md](docs/feature_overview.md) for architectural details and the new [runbook](docs/runbook.md) for end-to-end setup guidance.

## Components
- **Audio & VAD** – `persona.audio` now includes microphone capture backends (SoundDevice and PyAudio) alongside PCM helpers, while `persona.vad` exposes a streaming detector that emits timestamped utterances for Whisper to consume incrementally.
- **Speech-to-Text** – `persona.stt` exposes a `HuggingFaceTranscriber` that wraps Whisper- and NeMo-style ASR pipelines for GPU inference, alongside a debug echo transcriber for tests.
- **Persona Reasoning** – `persona.llm` adds `HuggingFacePersonaLLM`, which prompts chat-optimized instruction models and parses structured JSON replies to decide on memory logging and emotional tone.
- **Memory System** – `persona.memory` persists Markdown journals with persona metadata, tag heuristics, and cooldown-aware importance scoring so the persona only saves high-signal reflections.
- **Medium-Term Persona State** – `persona.state.PersonaStateManager` maintains hobbies, artistic preferences, cross-session reflections, and Discord relationship notes in a JSON profile that the LLM both reads and updates each turn.
- **Response Planning & TTS** – `persona.planner` keeps reply orchestration deterministic while `persona.tts` integrates SpeechT5 for high quality, local neural synthesis. A debug synthesizer remains available for dry runs.
- **Pipeline Orchestration** – `persona.pipeline` handles streaming frames, VAD state, and guardrails for both CLI and Discord entry points. `persona.discord_integration` ships a voice bridge that ingests PCM frames from the gateway and streams synthesized audio back.
- **CLI Runner** – `persona.cli` ingests WAV files or live microphone audio (`--mic`), streams them through Hugging Face models, emits synthesized replies plus Markdown memories, and optionally exports speech as numbered WAV files.

## Installation
Install the package in editable mode. The base dependencies include `torch`, `transformers`, and `datasets`; install the CUDA build of PyTorch that matches your driver if you plan to run on an RTX 5090.

```bash
pip install -e .[dev]
# Optional: reinstall torch with your preferred CUDA wheel
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## Running the Persona Locally
Feed either raw text (debug mode) or a 16-bit PCM WAV recording. By default the CLI spins up a "balanced" profile (Whisper large-v3, Mixtral instruct, SpeechT5) so you can swap stacks with a single flag instead of overriding every model manually.

```bash
# Fast debug flow without downloading models
python -m persona.cli --debug "This is an important systems test"

# Full GPU pipeline on local Hugging Face models with exported audio
python -m persona.cli \
  --input-wav path/to/input.wav \
  --memory-dir ./memory_logs \
  --output-wav ./exports/reply.wav \
  --persona-name Astra \
  --profile balanced

# Live microphone capture
python -m persona.cli \
  --mic \
  --mic-backend sounddevice \
  --mic-sample-rate 16000 \
  --mic-frame-ms 30
```

### Swapping Hugging Face model stacks

The CLI exposes curated profiles so you can pivot between production-quality, lightweight, and latency-optimised bundles in a single flag. Inspect the available options with:

```bash
python -m persona.cli --list-model-profiles
```

Then launch the persona with your preferred stack. For example, the `light` profile trades quality for lower VRAM requirements while still exercising the full pipeline:

```bash
python -m persona.cli --profile light --input-wav path/to/input.wav
```

You can continue to override individual model identifiers (`--asr-model`, `--llm-model`, `--tts-model`, etc.) after selecting a profile when you want to mix and match specific components.

Pass `--persona-backstory`, `--tts-speaker-sample`, or model identifiers to fine-tune the persona. All generated memories land under `./memory_logs` by default. Passing a directory to `--output-wav` (for example `--output-wav ./exports`) stores numbered replies when the pipeline produces multiple turns from a single audio clip.

Memory recall can be tuned with `--memory-window` (how many past entries to load) and `--memory-min-importance` (the minimum score a memory needs before it reaches the prompt). This keeps the persona focused on high-signal reflections while still allowing fallback to more recent context when few important memories exist.

Longer-lived persona preferences and relationships live in `persona_state.json` (override with `--persona-state-file`). The CLI feeds condensed summaries of these hobbies, artistic tastes, medium-term reflections, and Discord contacts into the LLM prompt so it can keep track of friends and revisit themes across sessions. Any state updates the model emits are merged back into the JSON profile automatically.

You can also supply a richer system prompt via `--persona-backstory-file path/to/prompt.md` or the `PERSONA_BACKSTORY` environment variable to keep persona tuning outside of shell history.

## Discord Integration
`persona.discord_integration.DiscordVoiceBridge` now accepts decoded Discord PCM via `receive_discord_pcm`, forwards timestamped frames into the pipeline, and streams synthesized PCM back through your `send_audio` callback. Use the `on_transcript` hook to capture transcripts (and their timestamps) for moderation dashboards whether you proxy your microphone or run a bot account.

## Development
Run the unit test suite after making changes:

```bash
pytest
```
