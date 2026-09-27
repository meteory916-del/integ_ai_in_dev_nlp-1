import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/v1/classify" in response.json()["paths"]

def test_validate_empty_text():
    response = client.post("/v1/classify", json={"text": ""})
    assert response.status_code == 422

def test_validate_missing_field():
    response = client.post("/v1/classify", json={})
    assert response.status_code == 422

def test_validate_too_long():
    response = client.post("/v1/classify", json={"text": "A" * 5001})
    assert response.status_code == 422

def test_classify_endpoint_exists():
    """Тест что эндпоинт существует (может вернуть 200 или 500)"""
    response = client.post("/v1/classify", json={"text": "test"})
    assert response.status_code in [200, 500]
