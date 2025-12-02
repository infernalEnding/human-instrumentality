# Persona Runbook

This runbook captures the practical steps for preparing the environment, launching the
persona locally, and extending it into Discord once you are ready for real-time
conversations. It assumes an RTX 5090-class workstation that can host the speech,
reasoning, and synthesis models entirely offline.

## 1. System Preparation

1. **GPU Drivers** – Install the latest NVIDIA drivers that match CUDA 12.4 or newer.
   Verify the driver stack with `nvidia-smi` before continuing.
2. **Python Environment** – Use Python 3.11 or newer in a virtual environment
   (`python -m venv .venv && source .venv/bin/activate`).
3. **Base Dependencies** – Install the project with development extras, then pin a CUDA
   build of PyTorch that matches your driver:

   ```bash
   pip install -e .[dev]
   pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
   ```

4. **Model Cache** – Log in to the Hugging Face CLI (`huggingface-cli login`) so the
   required checkpoints can be downloaded once and reused offline. Set
   `HF_HOME=/mnt/fastssd/hf-cache` (or similar) to control cache placement.

## 2. Running the Pipeline Locally

### Quick Functional Test

Use the debug pathway to confirm the CLI wiring without pulling heavyweight models:

```bash
python -m persona.cli --debug "System check"
```

You should see a fabricated transcription and response without attempting GPU work.

### Full GPU Pipeline

To execute the full speech loop against local Hugging Face models, provide either a WAV
recording or stream live audio from your microphone:

```bash
python -m persona.cli \
  --input-wav samples/greeting.wav \
  --memory-dir ./memory_logs \
  --output-wav ./exports/greeting.wav \
  --persona-name Astra \
  --persona-backstory "You are Astra, a grounded synth-pop vocalist." \
  --llm-model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --asr-model openai/whisper-large-v3 \
  --tts-model microsoft/speecht5_tts
```

The CLI prints the transcription, persona reply, detected emotion, and—when
`--output-wav` is supplied—the saved WAV file path. Passing a directory instead of a
file name (for example `--output-wav ./exports`) saves numbered files when multiple
responses are produced in sequence.

Switching to a live microphone is as simple as:

```bash
python -m persona.cli \
  --mic \
  --mic-backend sounddevice \
  --mic-sample-rate 16000 \
  --mic-frame-ms 30
```

The CLI streams frames through the VAD gate in real time, printing each detected
utterance with confidence, start/end timestamps, and optional memory summaries. Use
`--mic-backend pyaudio` if you prefer that capture stack or need WASAPI-specific
behaviour on Windows.

### Remote LLMs via OpenRouter

You can bypass local LLM weights entirely by targeting the OpenRouter
OpenAI-compatible endpoint:

```bash
python -m persona.cli \
  --llm-backend openrouter \
  --llm-model openai/gpt-4o-mini \
  --input-wav path/to/input.wav
```

Supply `OPENROUTER_API_KEY` (or `--openrouter-api-key`) and note the privacy trade-
off: transcripts, persona state summaries, and backstory text leave your machine and
are subject to the retention policies of OpenRouter and the upstream model vendor.
Avoid sending confidential data through this path and review the provider's terms
before enabling it in production.

### Memory Introspection

Markdown reflections accumulate in `--memory-dir` with microsecond timestamps. Each entry
now includes persona metadata, auto-tagged topics (for example `memory`, `planning`, or
`emotion:calm`), and the planner's importance score. The logger blends the LLM directive
with keyword heuristics and a cooldown timer to avoid flooding the journal with small
talk, keeping retrieval focused on genuinely memorable events.

Alongside the Markdown journal, the CLI keeps a `persona_state.json` profile (override
with `--persona-state-file`) that tracks medium-term reflections, hobbies, artistic
preferences, and Discord relationships. The prompt builder summarises this profile before
every reply, and any structured updates the LLM emits are merged back into the JSON file
so the persona remembers friends and tastes across sessions.

## 3. Discord Deployment Blueprint

1. **Choose the Integration Mode** – Decide between proxying your microphone (via a
   virtual audio cable such as VB-Audio) or running a standalone Discord bot account.
   The runbook recommends starting with mic replacement for quick testing, then
   promoting to a bot once you are ready for unattended sessions.
2. **Wire the Transport** – `DiscordVoiceBridge` now exposes `receive_discord_pcm` for
   feeding decoded 20 ms PCM frames straight from Discord. The helper converts them into
   timestamped `AudioFrame` objects so the VAD and Whisper streaming logic mirror the CLI
   behaviour. Use the same class to flush synthesized audio back through your
   voice-client by passing the returned payloads into your library's Opus encoder.
3. **Session Control** – Reuse the CLI configuration for model identifiers. Build a
   thin harness that instantiates the pipeline once and calls
   `receive_discord_pcm(...)` on the cadence Discord provides frames. Register an
   `on_transcript` callback to capture transcripts (with timestamps) for moderation and
   logging dashboards.
4. **Safety Levers** – Surface toggles for muting synthesis, purging memory logs, and
   forcing debug components when you need to operate in constrained environments. The
   bridge keeps VAD state internally, so you can pause ingress without losing partially
   buffered speech.

## 4. Troubleshooting Checklist

| Symptom | Suggested Fix |
| --- | --- |
| GPU OOM during generation | Lower `--llm-max-new-tokens`, switch to a smaller LLM, or enable CPU offload with `--llm-device-map balanced_low_0`. |
| Whisper struggles with accents | Try an alternative ASR model such as `openai/whisper-large-v3-turbo` or decrease `--vad-threshold` so pauses are detected earlier. |
| TTS audio distorts | Adjust the HiFi-GAN vocoder (`--tts-vocoder`) or pick a different speaker embedding sample via `--tts-speaker-sample`. |
| Memories never log | Confirm the LLM replies flag `log_memory: true`; increase the planner's importance heuristics or switch to debug mode to verify the pipeline wiring. |

With the environment validated, iteratively tighten latency by enabling mixed precision,
streaming ASR chunks, and Discord voice timers so the persona feels responsive in live
conversations.
