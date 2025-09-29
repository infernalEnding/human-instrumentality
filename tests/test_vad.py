import struct

from persona.audio import AudioFrame
from persona.vad import EnergyVAD


def test_energy_vad_streaming_with_timestamps() -> None:
    vad = EnergyVAD(threshold=0.05, min_speech_frames=2, max_silence_frames=1)
    frames = []
    timestamp = 0.0
    for amplitude in [0.4, 0.4, 0.0, 0.0]:
        value = int(amplitude * 32767)
        pcm_payload = struct.pack("<160h", *([value] * 160))
        frame = AudioFrame(
            pcm=pcm_payload,
            sample_rate=16000,
            channels=1,
            timestamp=timestamp,
        )
        frames.append(frame)
        timestamp += frame.duration_seconds

    outputs = []
    for frame in frames:
        outputs.extend(vad.process_frame(frame))
    outputs.extend(vad.flush())

    assert outputs, "Expected at least one VAD decision"
    decision = outputs[0]
    assert decision.start_time is not None
    if decision.end_time is not None:
        assert decision.end_time >= decision.start_time
