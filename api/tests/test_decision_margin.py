import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.model import ml_service


@pytest.mark.asyncio
async def test_decision_margin_matches_model_scores():
    """API должен возвращать разницу двух максимальных score."""

    text = "Не могу войти в мобильное приложение"

    assert ml_service.is_loaded()

    scores = ml_service.model.decision_function([text])[0]
    top_two = sorted(scores)[-2:]

    expected_margin = float(top_two[1] - top_two[0])
    expected_label = str(ml_service.model.predict([text])[0])

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/classify",
            json={"text": text}
        )

    assert response.status_code == 200

    result = response.json()

    assert result["label"] == expected_label
    assert result["decision_margin"] == pytest.approx(
        expected_margin,
        abs=1e-9
    )