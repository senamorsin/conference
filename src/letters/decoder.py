from __future__ import annotations

from collections import deque

from src.letters.classifier import LetterPrediction


class LetterDecoder:
    def __init__(self, stable_steps: int, accept_threshold: float, blank_steps_to_reset: int) -> None:
        self.stable_steps = stable_steps
        self.accept_threshold = accept_threshold
        self.blank_steps_to_reset = blank_steps_to_reset
        self._history: deque[str] = deque(maxlen=stable_steps)
        self._accepted: list[str] = []
        self._hold_letter: str | None = None
        self._blank_steps = 0
        self.current_letter: str | None = None
        self.current_confidence: float = 0.0

    @property
    def accepted_letters(self) -> str:
        return "".join(self._accepted)

    @property
    def blank_steps(self) -> int:
        return self._blank_steps

    def step(self, prediction: LetterPrediction | None) -> None:
        if prediction is None:
            self.current_letter = None
            self.current_confidence = 0.0
            self._history.clear()
            self._blank_steps += 1
            if self._blank_steps >= self.blank_steps_to_reset:
                self._hold_letter = None
            return

        self._blank_steps = 0
        self.current_letter = prediction.label
        self.current_confidence = prediction.confidence

        if prediction.confidence < self.accept_threshold:
            self._history.clear()
            return

        self._history.append(prediction.label)
        if len(self._history) < self.stable_steps:
            return

        stable_letter = self._history[0]
        if any(letter != stable_letter for letter in self._history):
            return

        if stable_letter == self._hold_letter:
            return

        self._accepted.append(stable_letter)
        self._hold_letter = stable_letter
        self._history.clear()

    def reset(self) -> None:
        self._history.clear()
        self._accepted.clear()
        self._hold_letter = None
        self._blank_steps = 0
        self.current_letter = None
        self.current_confidence = 0.0
