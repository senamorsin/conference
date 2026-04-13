from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


DEFAULT_DICTIONARY_PATH = Path(__file__).resolve().parents[2] / "configs" / "letters_dictionary.txt"


@dataclass(slots=True)
class WordSuggestion:
    raw_word: str
    normalized_word: str
    status: str
    score: float


class DictionaryCorrector:
    def __init__(self, vocabulary: Iterable[str], correction_threshold: float = 0.78) -> None:
        normalized = sorted({word.strip().upper() for word in vocabulary if word.strip()})
        self.vocabulary = tuple(normalized)
        self.correction_threshold = correction_threshold

    @classmethod
    def from_path(
        cls,
        path: str | Path = DEFAULT_DICTIONARY_PATH,
        correction_threshold: float = 0.78,
    ) -> "DictionaryCorrector":
        dictionary_path = Path(path).expanduser().resolve()
        if not dictionary_path.exists():
            return cls(vocabulary=(), correction_threshold=correction_threshold)

        vocabulary = dictionary_path.read_text(encoding="utf-8").splitlines()
        return cls(vocabulary=vocabulary, correction_threshold=correction_threshold)

    def correct(self, raw_word: str) -> WordSuggestion:
        normalized = raw_word.strip().upper()
        if not normalized:
            return WordSuggestion(raw_word="", normalized_word="", status="empty", score=0.0)

        if normalized in self.vocabulary:
            return WordSuggestion(
                raw_word=normalized,
                normalized_word=normalized,
                status="exact",
                score=1.0,
            )

        best_word = normalized
        best_score = 0.0

        for candidate in self.vocabulary:
            if abs(len(candidate) - len(normalized)) > 2:
                continue

            score = SequenceMatcher(a=normalized, b=candidate).ratio()
            if score > best_score:
                best_word = candidate
                best_score = score

        if best_score >= self.correction_threshold:
            return WordSuggestion(
                raw_word=normalized,
                normalized_word=best_word,
                status="corrected",
                score=best_score,
            )

        return WordSuggestion(
            raw_word=normalized,
            normalized_word=normalized,
            status="unknown",
            score=best_score,
        )
