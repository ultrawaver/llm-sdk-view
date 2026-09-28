from starlette.testclient import TestClient

from llm_sdk_view.app import app
from llm_sdk_view.chat import ChatOptions, ChatSession


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "project": "llm-sdk-view"}


def test_the_page_can_ask_where_the_rates_came_from():
    """Prices are fetched at runtime, so the footer needs to be able to say so."""
    body = TestClient(app).get("/api/rates").json()

    assert body["rates_state"] in {"live", "cached", "unavailable"}
    assert body["rates_url"].startswith("https://")
    assert isinstance(body["models"], int)


def test_the_preview_is_not_a_second_request_template():
    """/api/preview must be the real request, not a second copy of it.

    The version that was deleted rendered a hand-maintained request shape,
    which could drift from what llm-anthropic builds. What came back instead
    runs the same prepare() path, so the rule to prove is that a preview and a
    real turn produce byte-identical code - not that the endpoint is gone.
    """
    client = TestClient(app)
    payload = {"session_id": "preview-vs-send", "text": "hi", "model": "claude-sonnet-5"}
    preview = client.post("/api/preview", json=payload).json()

    assert preview["sent"] is False
    assert "client.messages.stream(" in preview["code"]
    assert preview["code"] == ChatSession(ChatOptions(model="claude-sonnet-5")).prepare(
        "hi"
    ).code
