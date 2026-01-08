Credit Default Prediction (MLOps Project)
Описание проекта

Проект реализует сквозной (end-to-end) MLOps-пайплайн для модели прогнозирования вероятности дефолта клиента в кредитном скоринге.

Проект охватывает полный жизненный цикл ML-модели:

подготовка и валидация данных,

feature engineering,

обучение и подбор гиперпараметров,

логирование экспериментов,

контроль версий данных и моделей,

CI-тестирование,

контейнеризация и REST API,

мониторинг data drift (PSI).

Датасет: Default of Credit Card Clients Dataset (UCI ML Repository).

Структура проекта
.
├── data/
│   ├── raw/                     # исходные данные
│   │   └── UCI_Credit_Card.csv
│   └── processed/               # данные после подготовки
│       ├── train.csv
│       └── test.csv
│
├── src/
│   ├── data/
│   │   └── make_dataset.py      # загрузка, очистка, split
│   ├── features/
│   │   └── build_features.py    # feature engineering
│   ├── models/
│   │   ├── pipeline.py          # sklearn Pipeline
│   │   └── train.py             # обучение + CV + MLflow
│   ├── api/
│   │   └── app.py               # FastAPI
│   ├── monitoring/
│   │   └── simulate_stream_and_psi.py  # PSI / drift monitoring
│   └── experiments_mlflow.py    # эксперименты с MLflow
│
├── tests/
│   ├── test_metrics.py
│   └── test_data_validation.py
│
├── models/                      # сохранённые модели
│   └── credit_default_model.joblib
│
├── artifacts/                   # метрики и графики
│   ├── metrics.json
│   └── roc_curve.png
│
├── great_expectations/
│   ├── great_expectations.yml
│   ├── expectations/
│   │   └── credit_data_suite.json
│   └── checkpoints/
│       └── processed_checkpoint.yml
│
├── .github/workflows/ci.yml     # GitHub Actions CI
├── dvc.yaml                     # DVC pipeline
├── .dvc/config                  # DVC init (manual)
├── Dockerfile
├── scripts/docker_run.sh
├── requirements.txt
└── README.md

Установка и окружение
1. Клонирование репозитория
git clone <repo_url>
cd <repo_name>

2. Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
#.venv\Scripts\activate    # Windows

3. Установка зависимостей
pip install -r requirements.txt

DVC пайплайн (подготовка данных и обучение)

Проект использует DVC pipeline с двумя стадиями:

prepare — подготовка данных

train — обучение модели

Запуск полного пайплайна
dvc repro


После выполнения будут созданы:

data/processed/train.csv

data/processed/test.csv

models/credit_default_model.joblib

artifacts/metrics.json

artifacts/roc_curve.png

DVC инициализирован вручную через .dvc/config (ограничения среды).

Обучение модели и MLflow

Обучение происходит в:

src/models/train.py


Используется:

sklearn Pipeline

RandomizedSearchCV

метрики: ROC-AUC, Precision, Recall, F1

логирование в MLflow

MLflow UI
mlflow ui


Открой:

http://localhost:5000

Тестирование и CI
Локальный запуск тестов
pytest -q

Проверка стиля кода
black --check .
flake8 .

Валидация данных (Great Expectations)
great_expectations checkpoint run processed_checkpoint


В CI (GitHub Actions) автоматически выполняются:

black

flake8

pytest

Great Expectations validation

FastAPI REST API
Запуск API локально
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Проверка
curl http://localhost:8000/

Endpoint /predict

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 50000,
    "BILL_AMT2": 48000,
    "BILL_AMT3": 47000,
    "BILL_AMT4": 46000,
    "BILL_AMT5": 45000,
    "BILL_AMT6": 44000,
    "PAY_AMT1": 2000,
    "PAY_AMT2": 2000,
    "PAY_AMT3": 2000,
    "PAY_AMT4": 2000,
    "PAY_AMT5": 2000,
    "PAY_AMT6": 2000
  }'


Целевая переменная:

{
  "default_prediction": 0,
  "default_probability": 0.12
}

Docker
Сборка и запуск
bash scripts/docker_run.sh


Скрипт:

выполняет dvc repro

собирает Docker-образ

запускает контейнер с API

Мониторинг и Data Drift (PSI)

Скрипт:

src/monitoring/simulate_stream_and_psi.py


Функциональность:

берёт train.csv как baseline,

имитирует поток данных из test.csv,

отправляет данные в API,

считает PSI по вероятностям и признакам.

Запуск
python -m src.monitoring.simulate_stream_and_psi \
  --api http://localhost:8000


Интерпретация PSI:

< 0.1 — OK

0.1 – 0.25 — WARNING

> 0.25 — ALERT
