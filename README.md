# Банковская классификация NLP

REST API для автоматической классификации обращений клиентов банка по 8 категориям.

## Описание задачи

Система анализирует тексты обращений клиентов и автоматически определяет их категорию для маршрутизации в соответствующие отделы банка.

### Классы классификации

| Код | Описание | Пример |
|-----|----------|--------|
| APP_LOGIN | Проблемы с входом в приложение | Face ID крутится и возвращает на экран логина |
| APP_TECH | Технические проблемы приложения | Приложение вылетает при открытии истории операций |
| CARD_ISSUE | Проблемы с банковской картой | Моя карта заблокирована, как ее разблокировать |
| PAYMENT_OUT_FAIL | Ошибки исходящих платежей | Не удалось отправить перевод, постоянная ошибка |
| INCOMING_DELAY | Задержки входящих платежей | Кэшбэк задерживается, когда он будет начислен |
| FRAUD_SUSPECTED | Подозрение на мошенничество | Мне пришло странное смс о списании, это мошенники |
| ACCOUNT_SEIZED | Арест счетов | Мои счета арестовали судебные приставы |
| KYC_VERIFICATION | Проблемы верификации | Не проходит верификация по паспорту в приложении |

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск API сервера

```bash
cd api
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Сервер запустится на http://localhost:8000

### 3. Проверка работы

- **Swagger UI**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **Метрики Prometheus**: http://localhost:8000/metrics

## Тестирование

### Запуск всех тестов

```bash
cd api
pytest tests/ -v
```

### Запуск с покрытием кода

```bash
pytest tests/ --cov=app --cov-report=html
```

## API Endpoints

### GET `/`

Корневой эндпоинт.

**Ответ:**
```json
```

### GET `/health`

Проверка здоровья сервиса и статуса загрузки модели.

**Ответ:**
```json
{"status": "ok", "model_loaded": true, "model_name": "TF-IDF + LogisticRegression"}
```

### POST `/classify`

Классификация текста обращения клиента.

**Request Body:**
```json
{"text": "Не могу войти в приложение, Face ID не работает"}
```

**Response:**
```json
{"label": "APP_LOGIN", "score": 0.9234}
```

**Коды ответов:**
- `200` — успешная классификация
- `422` — ошибка валидации входных данных
- `500` — внутренняя ошибка сервера
- `503` — модель не загружена

### GET `/metrics`

Метрики Prometheus для мониторинга.

**Возвращает:**
- `classify_requests_total{label, status}` — количество запросов по категориям
- `classify_latency_seconds` — время обработки запросов
- `classify_errors_total` — количество ошибок
- `model_loaded` — статус загрузки модели

## Структура проекта

```
integ_ai_in_dev_nlp/
├── api/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI приложение
│   │   ├── ml_service.py        # Сервис ML модели
│   │   └── schemas.py           # Схемы запросов/ответов + LabelEnum
│   ├── tests/
│   │   ├── conftest.py          # Фикстуры для тестов
│   │   ├── test_api.py          # Тесты API
│   │   └── test_data.py         # Тесты целостности данных
│   ├── model.pkl                # Обученная модель
│   ├── Dockerfile               # Docker образ
│   ├── pytest.ini               # Конфигурация pytest
│   └── requirements.txt         # Зависимости
── data/
│   ├── processed/
│   │   ├── train.csv            # Обучающая выборка
│   │   └── test.csv             # Тестовая выборка
│   └── raw/                     # Исходные данные
├── notebooks/
│   └── 01_eda_preprocessing_baseline.ipynb
└── README.md
```

## ML Модель

### Текущая модель

- **Архитектура**: TF-IDF + LogisticRegression
- **Библиотеки**: scikit-learn, pandas
- **Accuracy на train**: ~0.97
- **Размер**: ~187 KB

### Обучение модели

```bash
python train_model.py
```

Модель сохраняется в `model.pkl`

## Docker

### Сборка образа

```bash
docker build -t bank-nlp-api .
```

### Запуск контейнера

```bash
docker run -p 8000:8000 bank-nlp-api
```

API будет доступно на http://localhost:8000

## Примеры использования

### curl

```bash
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" -d '{"text": "Заблокировали карту"}'
```

### Python

```python
import httpx

response = httpx.post(
    "http://localhost:8000/classify",
    json={"text": "Не проходит верификация по паспорту"}
)

print(response.json())
```

## Roadmap

- [ ] Замена TF-IDF на Transformer (rubert-tiny2)
- [ ] Добавление логирования запросов
- [ ] Rate limiting
- [ ] CI/CD через GitHub Actions
- [ ] Деплой на Kubernetes
