"""Command line entry point for running the persona pipeline locally."""

from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
from typing import List, Sequence
import wave

from .audio import AudioFrame, MicrophoneConfig, MicrophoneStream, write_wav
from .llm import HuggingFacePersonaLLM, RuleBasedPersonaLLM
from .models import (
    DEFAULT_PROFILE_NAME,
    apply_profile_defaults,
    describe_model_profiles,
    get_model_profile,
    list_model_profiles,
)
from .memory import MemoryLogger
from .pipeline import PersonaPipeline
from .planner import ResponsePlanner
from .state import PersonaStateManager
from .stt import EchoTranscriber, HuggingFaceTranscriber
from .tts import DebugSynthesizer, HuggingFaceSynthesizer
from .vad import EnergyVAD


def build_pipeline(args: argparse.Namespace) -> PersonaPipeline:
    logger = MemoryLogger(args.memory_dir)
    persona_state = PersonaStateManager(
        args.persona_state_file,
        persona_name=args.persona_name,
    )

    if args.debug:
        transcriber = EchoTranscriber()
        llm = RuleBasedPersonaLLM(persona_name=args.persona_name, mood="curious")
        synthesizer = DebugSynthesizer()
    else:
        transcriber = HuggingFaceTranscriber(
            args.asr_model,
            device=args.asr_device,
            chunk_length_s=args.asr_chunk_length,
            decoder=args.asr_decoder,
        )
        llm = HuggingFacePersonaLLM(
            args.llm_model,
            persona_name=args.persona_name,
            persona_backstory=args.persona_backstory,
            temperature=args.llm_temperature,
            max_new_tokens=args.llm_max_new_tokens,
            device_map=args.llm_device_map,
        )
        synthesizer = HuggingFaceSynthesizer(
            args.tts_model,
            vocoder_id=args.tts_vocoder,
            speaker_dataset=args.tts_speaker_dataset,
            speaker_sample=args.tts_speaker_sample,
            device=args.tts_device,
        )

    pipeline = PersonaPipeline(
        vad=EnergyVAD(
            threshold=args.vad_threshold, min_speech_frames=2, max_silence_frames=2
        ),
        transcriber=transcriber,
        llm=llm,
        planner=ResponsePlanner(),
        synthesizer=synthesizer,
        memory_logger=logger,
        persona_state_manager=persona_state,
        memory_window=args.memory_window,
        memory_importance_threshold=args.memory_min_importance,
    )
    return pipeline


def _resolve_output_paths(base: Path, count: int) -> List[Path]:
    """Compute file paths for exporting synthesized audio."""

    if count <= 0:
        return []

    if base.suffix.lower() == ".wav":
        if count == 1:
            return [base]
        directory = base.parent if base.parent != Path("") else Path(".")
        stem = base.stem or "response"
        return [directory / f"{stem}_{idx:02d}.wav" for idx in range(1, count + 1)]

    directory = base
    return [directory / f"response_{idx:02d}.wav" for idx in range(1, count + 1)]


def text_to_frames(
    text: str,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    trailing_silence_frames: int = 3,
) -> list[AudioFrame]:
    """Generate synthetic PCM frames so text input can exercise the pipeline."""

    if not text:
        return []

    samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
    frame_byte_length = samples_per_frame * 2
    speech_bytes = text.encode("utf-8")
    if len(speech_bytes) % 2 != 0:
        speech_bytes += b" "

    speech_frames = max(2, math.ceil(len(speech_bytes) / frame_byte_length))
    frames: list[AudioFrame] = []
    timestamp = 0.0
    seconds_per_frame = frame_ms / 1000.0

    for index in range(speech_frames):
        start = index * frame_byte_length
        end = start + frame_byte_length
        chunk = speech_bytes[start:end]
        if len(chunk) < frame_byte_length:
            chunk = chunk.ljust(frame_byte_length, b" ")
        frames.append(
            AudioFrame(
                pcm=chunk,
                sample_rate=sample_rate,
                timestamp=timestamp,
            )
        )
        timestamp += seconds_per_frame

    for _ in range(trailing_silence_frames):
        frames.append(
            AudioFrame(
                pcm=b"\x00" * frame_byte_length,
                sample_rate=sample_rate,
                timestamp=timestamp,
            )
        )
        timestamp += seconds_per_frame

    return frames


def wav_to_frames(path: Path, frame_ms: int = 30) -> list[AudioFrame]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported")
        samples_per_frame = int(sample_rate * frame_ms / 1000)
        frames: list[AudioFrame] = []
        timestamp = 0.0
        while True:
            pcm = wav_file.readframes(samples_per_frame)
            if not pcm:
                break
            frame = AudioFrame(
                pcm=pcm,
                sample_rate=sample_rate,
                channels=channels,
                timestamp=timestamp,
            )
            frames.append(frame)
            timestamp += frame.duration_seconds
    return frames


class _StreamingExporter:
    def __init__(self, base: Path | None) -> None:
        self.base = base
        self.count = 0

    def next_path(self) -> Path | None:
        if self.base is None:
            return None
        self.count += 1
        if self.base.suffix.lower() == ".wav" and self.count == 1:
            return self.base
        if self.base.suffix.lower() == ".wav":
            directory = self.base.parent if self.base.parent != Path("") else Path(".")
            stem = self.base.stem or "response"
        else:
            directory = self.base
            stem = "response"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{stem}_{self.count:02d}.wav"


def _emit_outputs(
    outputs: Sequence,
    *,
    export_paths: Sequence[Path] | None = None,
    streaming_exporter: _StreamingExporter | None = None,
    start_index: int = 1,
) -> None:
    for idx, output in enumerate(outputs, start=start_index):
        print("TRANSCRIPTION:", output.transcription)
        if output.plan and output.plan.memory_summary:
            print("MEMORY_SUMMARY:", output.plan.memory_summary)
        if output.plan:
            print("RESPONSE:", output.plan.response_text)
            print("EMOTION:", output.plan.emotion)
        else:
            print("RESPONSE:", "")
        target_path: Path | None = None
        if export_paths is not None:
            target_path = export_paths[idx - start_index]
        elif streaming_exporter is not None:
            target_path = streaming_exporter.next_path()
        if target_path:
            try:
                saved_path = write_wav(target_path, output.audio)
            except ValueError as exc:
                print(f"SKIPPED AUDIO EXPORT ({target_path}): {exc}")
            else:
                print(f"AUDIO_SAVED: {saved_path}")
        print("AUDIO SAMPLE RATE:", output.audio.sample_rate)
        print("AUDIO ENCODING:", output.audio.encoding)


def _run_microphone_loop(pipeline: PersonaPipeline, args: argparse.Namespace) -> None:
    config = MicrophoneConfig(
        sample_rate=args.mic_sample_rate,
        channels=args.mic_channels,
        frame_ms=args.mic_frame_ms,
    )
    exporter = _StreamingExporter(args.output_wav)
    print("Starting microphone capture. Press Ctrl+C to stop.")
    try:
        with MicrophoneStream(config=config, backend=args.mic_backend) as stream:
            for frame in stream.frames():
                outputs = pipeline.process_stream_frame(frame)
                if outputs:
                    _emit_outputs(outputs, streaming_exporter=exporter)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        raise SystemExit(f"Microphone capture failed: {exc}") from exc
    finally:
        outputs = pipeline.flush()
        if outputs:
            _emit_outputs(outputs, streaming_exporter=exporter)


def main() -> None:  # pragma: no cover - exercised via CLI usage
    parser = argparse.ArgumentParser(description="Persona pipeline CLI")
    parser.add_argument("text", nargs="?", help="Text to feed into the pipeline")
    parser.add_argument("--input-wav", type=Path, help="Path to a WAV file to process")
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Capture audio from the default microphone instead of using an input file",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path("./memory_logs"),
        help="Directory where Markdown memories are stored",
    )
    parser.add_argument(
        "--persona-state-file",
        type=Path,
        default=Path("./persona_state.json"),
        help="Path to a JSON file storing medium-term memories and relationships",
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=3,
        help="Maximum number of past memories to provide to the persona prompt",
    )
    parser.add_argument(
        "--memory-min-importance",
        type=float,
        default=0.5,
        help="Minimum importance score a memory must have to be surfaced to the LLM",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=None,
        help=(
            "Optional path where synthesized replies are exported. Provide a"
            " .wav file name for a single reply or a directory for multiple"
            " responses."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=list_model_profiles(),
        default=DEFAULT_PROFILE_NAME,
        help="Preconfigured Hugging Face model bundle to use",
    )
    parser.add_argument(
        "--list-model-profiles",
        action="store_true",
        help="Print the available model profiles and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use lightweight debug components instead of Hugging Face models",
    )
    parser.add_argument("--persona-name", default="Astra", help="Name of the persona")
    parser.add_argument(
        "--persona-backstory",
        default=None,
        help="Optional custom backstory for the persona prompt",
    )
    parser.add_argument(
        "--persona-backstory-file",
        type=Path,
        default=None,
        help="Path to a markdown or text file containing the persona system prompt",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.05,
        help="Energy threshold for the VAD detector",
    )
    parser.add_argument(
        "--mic-backend",
        default="sounddevice",
        help="Microphone backend to use (sounddevice or pyaudio)",
    )
    parser.add_argument(
        "--mic-sample-rate",
        dest="mic_sample_rate",
        type=int,
        default=16000,
        help="Sample rate to capture microphone audio",
    )
    parser.add_argument(
        "--mic-channels",
        dest="mic_channels",
        type=int,
        default=1,
        help="Number of channels to record from the microphone",
    )
    parser.add_argument(
        "--mic-frame-ms",
        dest="mic_frame_ms",
        type=int,
        default=30,
        help="Frame size in milliseconds when chunking microphone audio",
    )
    parser.add_argument(
        "--asr-model",
        default=argparse.SUPPRESS,
        help="Hugging Face ASR model identifier",
    )
    parser.add_argument(
        "--asr-device",
        default=None,
        help="Device for ASR inference (e.g. cuda:0)",
    )
    parser.add_argument(
        "--asr-chunk-length",
        dest="asr_chunk_length",
        type=float,
        default=30.0,
        help="Chunk length in seconds for streaming ASR",
    )
    parser.add_argument(
        "--asr-decoder",
        default=None,
        help="Optional decoder identifier supported by the ASR pipeline",
    )
    parser.add_argument(
        "--llm-model",
        default=argparse.SUPPRESS,
        help="Hugging Face LLM model identifier",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for persona responses",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per reply",
    )
    parser.add_argument(
        "--llm-device-map",
        default="auto",
        help="Device map for accelerating large language model inference",
    )
    parser.add_argument(
        "--tts-model",
        default=argparse.SUPPRESS,
        help="Text-to-speech model identifier",
    )
    parser.add_argument(
        "--tts-vocoder",
        default=argparse.SUPPRESS,
        help="HiFi-GAN vocoder identifier",
    )
    parser.add_argument(
        "--tts-speaker-dataset",
        default=argparse.SUPPRESS,
        help="Dataset providing speaker embeddings",
    )
    parser.add_argument(
        "--tts-speaker-sample",
        type=int,
        default=argparse.SUPPRESS,
        help="Sample index from the speaker embedding dataset",
    )
    parser.add_argument(
        "--tts-device",
        default="cuda",
        help="Device to run text-to-speech inference on",
    )
    args = parser.parse_args()

    if args.persona_backstory_file:
        args.persona_backstory = args.persona_backstory_file.read_text(encoding="utf-8")
    if not args.persona_backstory:
        args.persona_backstory = os.environ.get("PERSONA_BACKSTORY")

    if args.list_model_profiles:
        for line in describe_model_profiles():
            print(line)
        return

    profile = get_model_profile(args.profile)
    profile_fields = (
        "asr_model",
        "llm_model",
        "tts_model",
        "tts_vocoder",
        "tts_speaker_dataset",
        "tts_speaker_sample",
    )
    apply_profile_defaults(args, profile, profile_fields)

    pipeline = build_pipeline(args)

    if args.mic:
        _run_microphone_loop(pipeline, args)
        return

    if args.input_wav:
        frames = wav_to_frames(args.input_wav)
    elif args.text:
        frames = text_to_frames(args.text)
    else:
        parser.error("Provide --mic, --input-wav, or text input")

    outputs = pipeline.process_frames(frames)
    if not outputs:
        print("No response generated")
        return
    export_paths: list[Path] | None = None
    if args.output_wav:
        export_paths = _resolve_output_paths(args.output_wav, len(outputs))

    _emit_outputs(outputs, export_paths=export_paths)


if __name__ == "__main__":  # pragma: no cover
    main()
