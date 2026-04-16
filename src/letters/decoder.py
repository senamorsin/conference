from __future__ import annotations

from collections import deque

from src.letters.classifier import LetterPrediction


class LetterDecoder:
    def __init__(
        self,
        stable_steps: int,
        accept_threshold: float,
        blank_steps_to_reset: int,
        repeat_hold_steps: int = 4,
    ) -> None:
        self.stable_steps = stable_steps
        self.accept_threshold = accept_threshold
        self.blank_steps_to_reset = blank_steps_to_reset
        self.repeat_hold_steps = repeat_hold_steps
        self._history: deque[str] = deque(maxlen=stable_steps)
        self._accepted: list[str] = []
        self._hold_letter: str | None = None
        self._hold_repeat_steps = 0
        self._blank_steps = 0
        self.current_letter: str | None = None
        self.current_confidence: float = 0.0

    @property
    def accepted_letters(self) -> str:
        return "".join(self._accepted)

    @property
    def blank_steps(self) -> int:
        return self._blank_steps

    def _can_accept_letter(self, letter: str) -> bool:
        if len(self._accepted) < 2:
            return True
        return not (self._accepted[-1] == letter and self._accepted[-2] == letter)

    def step(self, prediction: LetterPrediction | None) -> None:
        if prediction is None:
            self.current_letter = None
            self.current_confidence = 0.0
            self._history.clear()
            self._blank_steps += 1
            self._hold_repeat_steps = 0
            if self._blank_steps >= self.blank_steps_to_reset:
                self._hold_letter = None
            return

        self._blank_steps = 0
        self.current_letter = prediction.label
        self.current_confidence = prediction.confidence

        if prediction.confidence < self.accept_threshold:
            self._history.clear()
            self._hold_repeat_steps = 0
            return

        self._history.append(prediction.label)
        if len(self._history) < self.stable_steps:
            return

        stable_letter = self._history[0]
        if any(letter != stable_letter for letter in self._history):
            self._hold_repeat_steps = 0
            return

        if stable_letter == self._hold_letter:
            self._hold_repeat_steps += 1
            if self._hold_repeat_steps >= self.repeat_hold_steps:
                if self._can_accept_letter(stable_letter):
                    self._accepted.append(stable_letter)
                self._hold_repeat_steps = 0
                self._history.clear()
            return

        if self._can_accept_letter(stable_letter):
            self._accepted.append(stable_letter)
        self._hold_letter = stable_letter
        self._hold_repeat_steps = 0
        self._history.clear()

    def delete_last(self) -> None:
        if self._accepted:
            self._accepted.pop()
        self._history.clear()
        self._hold_letter = None
        self._hold_repeat_steps = 0
        self._blank_steps = 0

    def reset(self) -> None:
        self._history.clear()
        self._accepted.clear()
        self._hold_letter = None
        self._hold_repeat_steps = 0
        self._blank_steps = 0
        self.current_letter = None
        self.current_confidence = 0.0
