from starlette.testclient import TestClient

from llm_sdk_view.app import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "project": "llm-sdk-view"}


def test_preview():
    response = TestClient(app).post(
        "/api/preview",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "response_inclusion": "excluded",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["execution"] == "disabled-in-initial-scaffold"
    assert data["kwargs"]["tools"][0]["type"] == "web_search_20260318"
