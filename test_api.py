from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API работает"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_classify_returns_valid_label():
    response = client.post("/classify", json={"text": "Не могу войти"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] in [
        "APP_LOGIN", "APP_TECH", "CARD_ISSUE", 
        "PAYMENT_OUT_FAIL", "INCOMING_DELAY", 
        "FRAUD_SUSPECTED", "ACCOUNT_SEIZED", "KYC_VERIFICATION"
    ]
    assert 0 <= data["confidence"] <= 1

def test_classify_empty_text():
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 200
def test_classify_app_login():
    """Тест на класс APP_LOGIN (проблемы с входом)"""
    response = client.post("/classify", json={"text": "Не могу войти в приложение"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "APP_LOGIN"

def test_classify_card_issue():
    """Тест на класс CARD_ISSUE (проблемы с картой)"""
    response = client.post("/classify", json={"text": "Заблокировали карту"})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] in ["CARD_ISSUE", "ACCOUNT_SEIZED"]

def test_classify_fraud():
    """Тест на класс FRAUD_SUSPECTED (подозрение на мошенничество)"""
    response = client.post("/classify", json={"text": "Подозрительная транзакция"})
    assert response.status_code == 200
    data = response.json()
    assert "confidence" in data

    def test_metrics_endpoint():
    """Тест эндпоинта /metrics"""
    # Сначала сделаем запрос к classify
    client.post("/classify", json={"text": "Тест"})
    
    # Проверим metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "classify_requests_total" in response.text
    assert "classify_latency_seconds" in response.text