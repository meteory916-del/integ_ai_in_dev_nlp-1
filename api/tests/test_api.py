import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas import LabelEnum


pytestmark = pytest.mark.asyncio


REALISTIC_EXAMPLES = [
    ("Face ID крутится и возвращает на экран логина", LabelEnum.APP_LOGIN),
    ("Приложение вылетает при открытии истории операций", LabelEnum.APP_TECH),
    ("Моя карта заблокирована, как ее разблокировать?", LabelEnum.CARD_ISSUE),
    ("Ошибка при переводе средств, операция не выполнена", LabelEnum.PAYMENT_OUT_FAIL),
    ("Кэшбэк задерживается, когда он будет начислен?", LabelEnum.INCOMING_DELAY),
    ("Мне пришло странное смс о списании, это мошенники?", LabelEnum.FRAUD_SUSPECTED),
    ("Мои счета арестовали судебные приставы", LabelEnum.ACCOUNT_SEIZED),
    ("Не проходит верификация по паспорту в приложении", LabelEnum.KYC_VERIFICATION),
]


async def test_health_check():
    """Проверка доступности сервиса и загрузки модели."""

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "LinearSVC" in data["model_name"]


@pytest.mark.parametrize("text,expected_label", REALISTIC_EXAMPLES)
async def test_classify_all_classes(text, expected_label):
    """Проверка правильной классификации обращений всех 8 классов."""

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/classify",
            json={"text": text}
        )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == expected_label.value
    assert isinstance(data["decision_margin"], float)
    assert data["decision_margin"] >= 0.0


async def test_classify_empty_text():
    """Пустой текст должен вызывать ошибку валидации."""

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/classify",
            json={"text": ""}
        )

    assert response.status_code == 422


async def test_classify_too_long_text():
    """Текст длиннее 5000 символов должен отклоняться."""

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/classify",
            json={"text": "A" * 5001}
        )

    assert response.status_code == 422


async def test_classify_missing_text():
    """Запрос без обязательного поля text должен отклоняться."""

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/classify",
            json={}
        )

    assert response.status_code == 422


async def test_metrics_endpoint():
    """Проверка Prometheus-метрик."""

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:

        await client.post(
            "/v1/classify",
            json={"text": "Не могу войти в приложение"}
        )

        response = await client.get("/metrics")

    assert response.status_code == 200

    assert "text/plain" in response.headers["content-type"]
    assert "classify_requests_total" in response.text
    assert "classify_latency_seconds" in response.text
    assert "model_loaded" in response.text