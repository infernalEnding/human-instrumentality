from __future__ import annotations

from pathlib import Path

from persona.cli import _resolve_output_paths, text_to_frames


def test_resolve_output_paths_single_file() -> None:
    base = Path("reply.wav")
    paths = _resolve_output_paths(base, 1)
    assert paths == [Path("reply.wav")]


def test_resolve_output_paths_multiple_files_from_file() -> None:
    base = Path("reply.wav")
    paths = _resolve_output_paths(base, 3)
    assert paths == [
        Path("reply_01.wav"),
        Path("reply_02.wav"),
        Path("reply_03.wav"),
    ]


def test_resolve_output_paths_directory(tmp_path) -> None:
    base = tmp_path / "exports"
    paths = _resolve_output_paths(base, 2)
    assert paths == [
        base / "response_01.wav",
        base / "response_02.wav",
    ]


def test_text_to_frames_generates_pcm_payloads() -> None:
    frames = text_to_frames("Important mission update", sample_rate=16000)
    assert len(frames) >= 5
    energies = [frame.rms() for frame in frames]
    assert max(energies) > 0.05
    assert energies[-1] == 0.0
    assert all(len(frame.pcm) % 2 == 0 for frame in frames)
