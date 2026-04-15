from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
import threading
import wave

from piper.config import SynthesisConfig
from piper.voice import PiperVoice


def _default_voice_path() -> Path:
    return Path("models/tts/en_US-lessac-low.onnx")


@dataclass(slots=True)
class PiperSpeechSynthesizer:
    model_path: Path
    config_path: Path
    use_cuda: bool = False
    length_scale: float = 0.95
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    volume: float = 1.0
    _voice: PiperVoice | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def is_available(self) -> bool:
        return self.model_path.is_file() and self.config_path.is_file()

    @property
    def unavailable_reason(self) -> str | None:
        if self.is_available:
            return None
        if not self.model_path.is_file():
            return f"Missing Piper voice model: {self.model_path}"
        return f"Missing Piper voice config: {self.config_path}"

    def synthesize_wav_bytes(self, text: str) -> bytes:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            raise ValueError("Text for speech synthesis must not be empty")
        if not self.is_available:
            raise FileNotFoundError(self.unavailable_reason or "Piper voice files are not available")

        voice = self._get_or_load_voice()
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            voice.synthesize_wav(
                normalized,
                wav_file,
                syn_config=SynthesisConfig(
                    length_scale=self.length_scale,
                    noise_scale=self.noise_scale,
                    noise_w_scale=self.noise_w_scale,
                    volume=self.volume,
                ),
            )
        return buffer.getvalue()

    def preload(self) -> None:
        if not self.is_available:
            raise FileNotFoundError(self.unavailable_reason or "Piper voice files are not available")
        self._get_or_load_voice()

    def _get_or_load_voice(self) -> PiperVoice:
        if self._voice is not None:
            return self._voice

        with self._lock:
            if self._voice is None:
                self._voice = PiperVoice.load(
                    model_path=self.model_path,
                    config_path=self.config_path,
                    use_cuda=self.use_cuda,
                )
        return self._voice


def create_default_speech_synthesizer() -> PiperSpeechSynthesizer:
    model_path = _default_voice_path()
    config_path = Path(f"{model_path}.json")
    return PiperSpeechSynthesizer(
        model_path=model_path,
        config_path=config_path,
        use_cuda=False,
    )
