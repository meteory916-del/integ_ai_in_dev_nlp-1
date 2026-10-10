import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.schemas import LabelEnum

# Включаем асинхронный режим для всех тестов в файле
pytestmark = pytest.mark.asyncio


# Список реальных примеров для тестирования
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
    """Тест эндпоинта /health: модель должна быть загружена."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health")
    
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == expected_label.value
    data = r.json()
    assert data["label"] == expected_label.value
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_name" in data


@pytest.mark.parametrize("text,expected_label", REALISTIC_EXAMPLES)
async def test_classify_all_classes(text, expected_label):
    """
    Тест классификации на реальных примерах для всех 8 классов.
    Проверяет валидность ответа и диапазон score.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/v1/classify", json={"text": text})
    
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == expected_label.value
    data = r.json()
    assert data["label"] == expected_label.value
    data = r.json()
    
    # 1. Проверяем, что вернулся один из 8 допустимых классов
    assert data["label"] in [e.value for e in LabelEnum]
    
    # 2. Проверяем, что score строго в диапазоне [0.0, 1.0]
    assert isinstance(data["decision_margin"], float)


async def test_classify_empty_text():
    """Тест валидации: пустой текст должен возвращать 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/v1/classify", json={"text": ""})
    
    assert r.status_code == 422


async def test_classify_too_long_text():
    """Тест валидации: текст длиннее 5000 символов должен возвращать 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/v1/classify", json={"text": "A" * 5001})
    
    assert r.status_code == 422


async def test_classify_missing_text():
    """Тест валидации: отсутствие поля text должно возвращать 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/v1/classify", json={})
    
    assert r.status_code == 422


async def test_metrics_endpoint():
    """Тест эндпоинта /metrics."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Сначала сделаем запрос, чтобы накрутить счетчик
        await ac.post("/v1/classify", json={"text": "Тест для метрик"})
        r = await ac.get("/metrics")
    
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == expected_label.value
    data = r.json()
    assert data["label"] == expected_label.value
    assert "classify_requests_total" in r.text
    assert "classify_latency_seconds" in r.text
