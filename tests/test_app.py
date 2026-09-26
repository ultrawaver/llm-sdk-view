from starlette.testclient import TestClient

from llm_sdk_view.app import app


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "project": "llm-sdk-view"}


def test_there_is_no_second_request_template():
    """The removed /api/preview rendered a hand-maintained request shape.

    Its whole purpose was a second copy of the request, which could drift from
    the one llm-anthropic builds, so it must not come back.
    """
    response = TestClient(app).post("/api/preview", json={"messages": []})

    assert response.status_code == 404
