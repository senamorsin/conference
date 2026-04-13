# ASL Real-Time MVP

Первый вертикальный срез под буквенный режим ASL:

- загрузка кадра через веб-интерфейс,
- извлечение hand landmarks через MediaPipe,
- заглушка-классификатор буквы,
- антидребезг и буфер принятых букв,
- API для предсказания и сброса состояния.

## Что уже есть

- `src/landmarks/mediapipe_extractor.py` — извлечение и нормализация landmarks руки.
- `src/letters/classifier.py` — временный placeholder-классификатор вместо обученной модели.
- `src/letters/decoder.py` — логика подтверждения буквы и защиты от повторов.
- `scripts/prepare_asl_alphabet.py` — подготовка `grassknoted/asl-alphabet` в landmarks CSV.
- `scripts/train_letters.py` — обучение baseline-модели буквенного режима.
- `src/app/` — FastAPI-приложение, HTML-страница и API маршруты.
- `models/mediapipe/hand_landmarker.task` — model asset для `mediapipe.tasks`.
- `tests/` — базовые тесты для decoder, нормализации и health endpoint.

## Как запустить

```bash
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run python main.py
```

Если порт `8000` занят, можно запустить на другом:

```bash
PORT=8010 uv run python main.py
```

## Как перевести проект на Python 3.11

```bash
uv python install 3.11
uv venv --python 3.11
uv sync
```

## Датасет для буквенного baseline

Для текущего шага выбран `grassknoted/asl-alphabet`:

- большой image dataset с папками по классам,
- хорошо подходит под наш pipeline `image -> MediaPipe landmarks -> classifier`,
- для честного baseline мы по умолчанию используем только статические буквы `A-I, K-Y` и `NOTHING`,
- `J` и `Z` исключены из baseline, потому что это motion-буквы.

Скачать датасет можно через Kaggle CLI, например так:

```bash
kaggle datasets download -d grassknoted/asl-alphabet -p data/raw
mkdir -p data/raw/asl-alphabet
unzip data/raw/asl-alphabet.zip -d data/raw/asl-alphabet
```

Ожидаемая структура по умолчанию:

```text
data/raw/asl-alphabet/asl_alphabet_train/<LABEL>/*.jpg
```

## Как подготовить признаки и обучить baseline

Извлечение landmarks из датасета:

```bash
uv run python scripts/prepare_asl_alphabet.py --config configs/model_letters.yaml
```

Обучение baseline-модели:

```bash
uv run python scripts/train_letters.py --config configs/model_letters.yaml
```

После этого приложение автоматически подхватит модель из:

```text
models/letters/asl_letters_random_forest.joblib
```

## GPU-режим

Оба скрипта теперь умеют использовать GPU, но по-разному:

- `scripts/prepare_asl_alphabet.py` использует `MediaPipe Tasks` delegate и может работать с `GPU` или `CPU`.
- `scripts/train_letters.py` использует `XGBoost` на `cuda`, если установлен пакет `xgboost` и доступна CUDA; иначе автоматически падает обратно на CPU `RandomForest`.

Текущий конфиг уже стоит в `auto`-режиме:

- `features.delegate: auto`
- `training.backend: auto`
- `training.device: auto`

Если хочешь принудительно GPU-тренировку для второго скрипта, нужен `xgboost`:

```bash
uv add xgboost
uv sync
```

На машине без CUDA или без `xgboost` второй скрипт честно останется на CPU.

После запуска открой:

```text
http://127.0.0.1:8000
```

## Что проверить

1. Открывается стартовая страница.
2. `GET /health` возвращает статус `ok`.
3. При загрузке изображения руки endpoint `/api/letters/predict` возвращает:
   - `current_letter`,
   - `confidence`,
   - `accepted_letters`,
   - `landmarks_detected`.
4. Кнопка reset очищает буфер букв.

## Ограничения текущей версии

- Это пока не real-time поток с камеры, а upload одного кадра.
- До обучения на датасете приложение использует placeholder-классификатор.
- Нет correction, word boundary detection и TTS.

## Следующий шаг

Подобрать датасет ASL для буквенного режима и заменить placeholder-классификатор на baseline-модель по landmarks.
