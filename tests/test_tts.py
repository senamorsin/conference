from io import BytesIO
import wave

import pytest

from src.tts.piper_engine import PiperSpeechSynthesizer


class StubVoice:
    def synthesize_wav(self, text, wav_file, syn_config=None, set_wav_format=True, include_alignments=False):
        assert text == "HELLO"
        if set_wav_format:
            wav_file.setframerate(16000)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
        wav_file.writeframes(b"\x00\x00" * 160)


def test_piper_speech_synthesizer_rejects_empty_text(tmp_path) -> None:
    synthesizer = PiperSpeechSynthesizer(
        model_path=tmp_path / "voice.onnx",
        config_path=tmp_path / "voice.onnx.json",
    )

    with pytest.raises(ValueError):
        synthesizer.synthesize_wav_bytes("   ")


def test_piper_speech_synthesizer_builds_valid_wav_with_stub_voice(tmp_path) -> None:
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    model_path.write_bytes(b"model")
    config_path.write_text("{}", encoding="utf-8")

    synthesizer = PiperSpeechSynthesizer(
        model_path=model_path,
        config_path=config_path,
    )
    synthesizer._voice = StubVoice()

    wav_bytes = synthesizer.synthesize_wav_bytes("HELLO")

    assert wav_bytes.startswith(b"RIFF")
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 160
