"""Audio primitives for the persona pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, TYPE_CHECKING
import math
import struct

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tts import SynthesizedAudio


@dataclass(frozen=True)
class AudioFrame:
    """Represents a short chunk of 16-bit PCM audio.

    Parameters
    ----------
    pcm:
        Raw 16-bit PCM bytes.
    sample_rate:
        Sampling rate of the frame in Hz.
    channels:
        Number of audio channels contained in ``pcm``.
    timestamp:
        Optional timestamp in seconds representing when the frame started.
        This is used by streaming voice-activity detection to surface
        utterance boundaries with absolute timing information.
    """

    pcm: bytes
    sample_rate: int
    channels: int = 1
    timestamp: float | None = None

    @property
    def frame_count(self) -> int:
        """Number of samples per channel in this frame."""

        if self.channels <= 0:
            raise ValueError("channels must be positive")
        width = 2  # 16-bit PCM
        return len(self.pcm) // (width * self.channels)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / float(self.sample_rate)

    def rms(self) -> float:
        """Compute the root-mean-square amplitude for the frame."""

        if not self.pcm:
            return 0.0
        width = 2
        if len(self.pcm) % width != 0:
            raise ValueError("PCM payload must be aligned to 16-bit samples")
        count = len(self.pcm) // width
        if count == 0:
            return 0.0
        unpack_fmt = f"<{count}h"
        samples = struct.unpack(unpack_fmt, self.pcm)
        square_sum = sum(s * s for s in samples)
        return math.sqrt(square_sum / count) / 32768.0


@dataclass
class AudioSegment:
    """Collection of contiguous frames that belong to a single utterance."""

    frames: List[AudioFrame]

    @property
    def duration_seconds(self) -> float:
        return sum(frame.duration_seconds for frame in self.frames)

    @property
    def pcm(self) -> bytes:
        return b"".join(frame.pcm for frame in self.frames)

    @property
    def sample_rate(self) -> int:
        if not self.frames:
            raise ValueError("audio segment has no frames")
        rates = {frame.sample_rate for frame in self.frames}
        if len(rates) != 1:
            raise ValueError("frames in a segment must share a sample rate")
        return self.frames[0].sample_rate

    @property
    def channels(self) -> int:
        if not self.frames:
            raise ValueError("audio segment has no frames")
        channels = {frame.channels for frame in self.frames}
        if len(channels) != 1:
            raise ValueError("frames in a segment must share the same channel count")
        return self.frames[0].channels

    @property
    def start_time(self) -> float | None:
        if not self.frames:
            return None
        for frame in self.frames:
            if frame.timestamp is not None:
                return frame.timestamp
        return None

    @property
    def end_time(self) -> float | None:
        if not self.frames:
            return None
        last_with_timestamp = None
        for frame in self.frames:
            if frame.timestamp is not None:
                last_with_timestamp = frame.timestamp + frame.duration_seconds
        return last_with_timestamp

    def to_numpy(self) -> np.ndarray:
        """Convert PCM payload into a float32 NumPy array in [-1, 1]."""

        pcm = self.pcm
        if not pcm:
            return np.zeros(0, dtype=np.float32)
        if len(pcm) % 2 != 0:
            raise ValueError("PCM payload must be aligned to 16-bit samples")
        array = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        array /= 32768.0
        if self.channels > 1:
            array = array.reshape(-1, self.channels)
        return array


class AudioBuffer:
    """Utility helper to accumulate frames into segments."""

    def __init__(self) -> None:
        self._frames: List[AudioFrame] = []

    def append(self, frame: AudioFrame) -> None:
        self._frames.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    def pop_segment(self) -> AudioSegment | None:
        if not self._frames:
            return None
        segment = AudioSegment(frames=list(self._frames))
        self.clear()
        return segment

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._frames)

    def __iter__(self) -> Iterable[AudioFrame]:  # pragma: no cover - trivial
        return iter(self._frames)


def write_wav(path: Path, audio: "SynthesizedAudio", *, channels: int = 1) -> Path:
    """Persist synthesized PCM16 audio to a WAV file.

    Parameters
    ----------
    path:
        Destination file path. Parent directories are created automatically.
    audio:
        Synthesized audio payload to save.
    channels:
        Number of channels to encode in the WAV header. Defaults to mono.

    Returns
    -------
    Path
        The path where the WAV file was written.
    """

    if audio.encoding != "pcm16":
        raise ValueError(
            "Only pcm16 encoded audio can be exported to WAV; "
            f"got {audio.encoding!r}"
        )

    path = path.with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)

    import wave

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(audio.payload)

    return path


class MicrophoneError(RuntimeError):
    """Raised when a microphone backend cannot be initialised."""


@dataclass
class MicrophoneConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 20

    @property
    def samples_per_frame(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000.0))


class MicrophoneStream:
    """Best-effort microphone capture with optional third-party backends.

    The implementation tries to import :mod:`sounddevice` first, followed by
    :mod:`pyaudio`. Tests can inject synthetic frames via the ``frame_source``
    argument to avoid real hardware access.
    """

    def __init__(
        self,
        *,
        config: MicrophoneConfig | None = None,
        backend: str | None = None,
        frame_source=None,
    ) -> None:
        self.config = config or MicrophoneConfig()
        self.backend = backend
        self._stream = None
        self._frame_source = frame_source
        self._timestamp = 0.0
        self._pyaudio = None

    def __enter__(self) -> "MicrophoneStream":
        if self._frame_source is not None:
            return self

        backend = (self.backend or "sounddevice").lower()
        if backend == "sounddevice":
            self._stream = self._open_sounddevice()
        elif backend == "pyaudio":
            self._stream = self._open_pyaudio()
        else:
            raise MicrophoneError(f"Unsupported microphone backend: {backend}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - exercised indirectly
        if self._stream is None:
            return
        stop = (
            getattr(self._stream, "stop", None)
            or getattr(self._stream, "stop_stream", None)
        )
        close = (
            getattr(self._stream, "close", None)
            or getattr(self._stream, "close_stream", None)
        )
        if stop:
            stop()
        if close:
            close()
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None

    def frames(self) -> Iterable[AudioFrame]:
        if self._frame_source is not None:
            yield from self._generate_frames(self._frame_source)
            return

        if self._stream is None:
            raise MicrophoneError("Microphone stream has not been started")

        generator = getattr(self._stream, "generator", None)
        if generator is not None:
            yield from self._generate_frames(generator)
            return

        read = getattr(self._stream, "read", None)
        if read is None:
            raise MicrophoneError("Microphone backend does not expose a read method")

        samples_per_frame = self.config.samples_per_frame
        while True:  # pragma: no cover - requires real device
            pcm, overflow = read(samples_per_frame)
            if overflow:
                continue
            if not pcm:
                break
            yield self._build_frame(bytes(pcm))

    def _generate_frames(self, source) -> Iterable[AudioFrame]:
        for chunk in source:
            if not chunk:
                continue
            yield self._build_frame(bytes(chunk))

    def _build_frame(self, pcm: bytes) -> AudioFrame:
        frame = AudioFrame(
            pcm=pcm,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            timestamp=self._timestamp,
        )
        self._timestamp += frame.duration_seconds
        return frame

    def _open_sounddevice(self):
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise MicrophoneError("sounddevice backend is not available") from exc

        config = self.config
        stream = sd.InputStream(
            samplerate=config.sample_rate,
            channels=config.channels,
            dtype="int16",
            blocksize=config.samples_per_frame,
        )
        stream.start()

        def _generator():  # pragma: no cover - requires real device
            while True:
                pcm, overflowed = stream.read(config.samples_per_frame)
                if overflowed:
                    continue
                yield pcm.tobytes()

        stream.generator = _generator()
        return stream

    def _open_pyaudio(self):
        try:
            import pyaudio  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise MicrophoneError("PyAudio backend is not available") from exc

        config = self.config
        self._pyaudio = pyaudio.PyAudio()
        stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=config.channels,
            rate=config.sample_rate,
            input=True,
            frames_per_buffer=config.samples_per_frame,
        )

        def _reader():  # pragma: no cover - requires real device
            while True:
                yield stream.read(config.samples_per_frame, exception_on_overflow=False)

        stream.generator = _reader()
        return stream
