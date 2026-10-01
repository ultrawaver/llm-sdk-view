from starlette.testclient import TestClient

from native_api_chat.app import app
from native_api_chat.chat import ChatOptions, ChatSession


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "project": "native-api-chat"}


def test_the_page_can_ask_where_the_rates_came_from():
    """Prices are fetched at runtime, so the footer needs to be able to say so.

    One answer per provider, because the two do not read the same document:
    Anthropic's pricing page, OpenRouter's own catalogue. The page asks about
    the provider it is showing.
    """
    body = TestClient(app).get("/api/rates").json()

    assert set(body["sources"]) == {"anthropic", "openrouter"}
    for name, source in body["sources"].items():
        assert source["rates_state"] in {"live", "cached", "unavailable"}, name
        assert source["rates_url"].startswith("https://"), name
        assert isinstance(source["models"], int), name


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
