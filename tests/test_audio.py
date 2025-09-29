from __future__ import annotations

import numpy as np

from persona.audio import (
    AudioFrame,
    AudioSegment,
    MicrophoneConfig,
    MicrophoneStream,
    write_wav,
)
from persona.tts import SynthesizedAudio


def _pcm_bytes() -> bytes:
    samples = np.linspace(-1.0, 1.0, num=160, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.tobytes()


def test_audio_segment_numpy_conversion() -> None:
    sample_rate = 16000
    pcm = (np.arange(0, 160, dtype=np.int16) - 80).tobytes()
    frame = AudioFrame(pcm=pcm, sample_rate=sample_rate, timestamp=0.1)
    segment = AudioSegment(frames=[frame])

    array = segment.to_numpy()
    assert array.shape == (160,)
    assert np.isclose(array.mean(), 0.0, atol=1e-2)
    assert segment.sample_rate == sample_rate
    assert segment.start_time == 0.1
    assert np.isclose(segment.end_time, 0.1 + frame.duration_seconds)


def test_write_wav_roundtrip(tmp_path) -> None:
    payload = _pcm_bytes()
    audio = SynthesizedAudio(payload=payload, sample_rate=22050)

    path = write_wav(tmp_path / "out.wav", audio)

    import wave

    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getframerate() == 22050
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        data = wav_file.readframes(wav_file.getnframes())
        assert data == payload


def test_write_wav_rejects_non_pcm16(tmp_path) -> None:
    audio = SynthesizedAudio(payload=b"debug", sample_rate=16000, encoding="text/debug")

    import pytest

    with pytest.raises(ValueError):
        write_wav(tmp_path / "fail.wav", audio)


def test_microphone_stream_with_synthetic_source() -> None:
    config = MicrophoneConfig(sample_rate=16000, channels=1, frame_ms=20)
    pcm_frame = (np.ones(config.samples_per_frame, dtype=np.int16) * 1000).tobytes()
    source = [pcm_frame, pcm_frame]

    with MicrophoneStream(config=config, frame_source=source) as stream:
        frames = list(stream.frames())

    assert len(frames) == 2
    assert frames[0].timestamp == 0.0
    assert np.isclose(frames[1].timestamp, frames[0].duration_seconds)
