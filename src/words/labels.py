from __future__ import annotations

from typing import Final

from src.letters.labels import FEATURE_COLUMNS as LETTER_FEATURE_COLUMNS


WORD_SEQUENCE_LENGTH: Final[int] = 12
WORD_LABELS: Final[tuple[str, ...]] = (
    "AGAIN",
    "BABY",
    "BATHROOM",
    "BOOK",
    "BOY",
    "BROTHER",
    "DOCTOR",
    "DRINK",
    "EAT",
    "FAMILY",
    "FATHER",
    "FINISH",
    "FRIEND",
    "GIRL",
    "GO",
    "GOOD",
    "HAPPY",
    "HELLO",
    "HELP",
    "HOME",
    "KNOW",
    "LEARN",
    "LIKE",
    "LOVE",
    "MONEY",
    "MORE",
    "MOTHER",
    "NAME",
    "NEED",
    "NO",
    "PHONE",
    "PLEASE",
    "SAD",
    "SCHOOL",
    "SICK",
    "SISTER",
    "SORRY",
    "THANK_YOU",
    "TIRED",
    "TODAY",
    "TOMORROW",
    "UNDERSTAND",
    "WANT",
    "WATER",
    "WHAT",
    "WHERE",
    "WHO",
    "WORK",
    "YES",
    "YESTERDAY",
)
MSASL_LABEL_ALIASES: Final[dict[str, str]] = {
    "again": "AGAIN",
    "baby": "BABY",
    "bathroom": "BATHROOM",
    "toilet": "BATHROOM",
    "book": "BOOK",
    "boy": "BOY",
    "brother": "BROTHER",
    "doctor": "DOCTOR",
    "drink": "DRINK",
    "eat": "EAT",
    "family": "FAMILY",
    "father": "FATHER",
    "dad": "FATHER",
    "finish": "FINISH",
    "done": "FINISH",
    "friend": "FRIEND",
    "girl": "GIRL",
    "go": "GO",
    "good": "GOOD",
    "happy": "HAPPY",
    "hello": "HELLO",
    "help": "HELP",
    "home": "HOME",
    "know": "KNOW",
    "learn": "LEARN",
    "like": "LIKE",
    "love": "LOVE",
    "money": "MONEY",
    "more": "MORE",
    "mother": "MOTHER",
    "mom": "MOTHER",
    "name": "NAME",
    "need": "NEED",
    "no": "NO",
    "phone": "PHONE",
    "please": "PLEASE",
    "sad": "SAD",
    "school": "SCHOOL",
    "sick": "SICK",
    "sister": "SISTER",
    "sorry": "SORRY",
    "thanks": "THANK_YOU",
    "thank you": "THANK_YOU",
    "tired": "TIRED",
    "today": "TODAY",
    "tomorrow": "TOMORROW",
    "understand": "UNDERSTAND",
    "want": "WANT",
    "water": "WATER",
    "what": "WHAT",
    "where": "WHERE",
    "who": "WHO",
    "work": "WORK",
    "yesterday": "YESTERDAY",
    "yes": "YES",
}
ASL_CITIZEN_LABEL_ALIASES: Final[dict[str, str]] = {
    "again": "AGAIN",
    "baby": "BABY",
    "bathroom": "BATHROOM",
    "toilet": "BATHROOM",
    "book": "BOOK",
    "boy": "BOY",
    "brother": "BROTHER",
    "doctor": "DOCTOR",
    "family": "FAMILY",
    "father": "FATHER",
    "dad": "FATHER",
    "finish": "FINISH",
    "done": "FINISH",
    "friend": "FRIEND",
    "girl": "GIRL",
    "go": "GO",
    "good": "GOOD",
    "happy": "HAPPY",
    "hello": "HELLO",
    "help": "HELP",
    "home": "HOME",
    "know": "KNOW",
    "learn": "LEARN",
    "like": "LIKE",
    "love": "LOVE",
    "money": "MONEY",
    "more": "MORE",
    "mother": "MOTHER",
    "mom": "MOTHER",
    "name": "NAME",
    "need": "NEED",
    "no": "NO",
    "phone": "PHONE",
    "please": "PLEASE",
    "sad": "SAD",
    "school": "SCHOOL",
    "sick": "SICK",
    "sister": "SISTER",
    "sorry": "SORRY",
    "thankyou": "THANK_YOU",
    "tired": "TIRED",
    "today": "TODAY",
    "tomorrow": "TOMORROW",
    "understand": "UNDERSTAND",
    "water": "WATER",
    "where": "WHERE",
    "who": "WHO",
    "work": "WORK",
    "yesterday": "YESTERDAY",
    "yes": "YES",
}
ASL_CITIZEN_WORD_LABELS: Final[tuple[str, ...]] = tuple(sorted(set(ASL_CITIZEN_LABEL_ALIASES.values())))
WORD_MOTION_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "wrist_x",
    "wrist_y",
    "thumb_tip_x",
    "thumb_tip_y",
    "index_tip_x",
    "index_tip_y",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_width",
    "bbox_height",
)
WORD_FRAME_FEATURE_NAMES: Final[tuple[str, ...]] = LETTER_FEATURE_COLUMNS + WORD_MOTION_FEATURE_NAMES
WORD_DELTA_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f"d{time_step}_{feature_name}"
    for time_step in range(WORD_SEQUENCE_LENGTH - 1)
    for feature_name in WORD_MOTION_FEATURE_NAMES
)
WORD_FRAME_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"t{time_step}_{feature_name}"
    for time_step in range(WORD_SEQUENCE_LENGTH)
    for feature_name in WORD_FRAME_FEATURE_NAMES
)
WORD_FEATURE_COLUMNS: Final[tuple[str, ...]] = WORD_FRAME_COLUMNS + WORD_DELTA_FEATURE_NAMES


def normalize_msasl_word_label(raw_label: str) -> str | None:
    key = raw_label.strip().lower()
    if not key:
        return None
    return MSASL_LABEL_ALIASES.get(key)


def normalize_asl_citizen_word_label(raw_label: str) -> str | None:
    key = raw_label.strip().lower().replace(" ", "")
    if not key:
        return None
    return ASL_CITIZEN_LABEL_ALIASES.get(key)


def display_word_label(label: str | None) -> str | None:
    if label is None:
        return None
    return label.replace("_", " ")
