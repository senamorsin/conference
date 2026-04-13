from __future__ import annotations

from dataclasses import dataclass

from src.postprocess.dictionary import DictionaryCorrector


@dataclass(slots=True)
class FinalizedWord:
    raw_word: str
    final_word: str
    status: str
    score: float


class LetterWordBuilder:
    def __init__(
        self,
        corrector: DictionaryCorrector,
        finalize_blank_steps: int = 7,
        min_letters: int = 1,
    ) -> None:
        self.corrector = corrector
        self.finalize_blank_steps = finalize_blank_steps
        self.min_letters = min_letters
        self.latest: FinalizedWord | None = None
        self._history: list[str] = []

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    def maybe_finalize(self, accepted_letters: str, blank_steps: int) -> FinalizedWord | None:
        normalized = accepted_letters.strip().upper()
        if not normalized:
            return None
        if len(normalized) < self.min_letters:
            return None
        if blank_steps < self.finalize_blank_steps:
            return None

        suggestion = self.corrector.correct(normalized)
        built = FinalizedWord(
            raw_word=suggestion.raw_word,
            final_word=suggestion.normalized_word,
            status=suggestion.status,
            score=suggestion.score,
        )
        self.latest = built
        self._history.append(built.final_word)
        return built

    def reset(self) -> None:
        self.latest = None
        self._history.clear()
