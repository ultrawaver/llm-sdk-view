import os
import time

from starlette.testclient import TestClient

from native_api_chat.app import app, server_state
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


def test_the_build_question_answers_for_the_server_too():
    """A stale tab and a stale server are different failures with one symptom.

    The asset fingerprint cannot tell them apart: it is read off the disk on
    every request, so a process that imported yesterday's code still answers
    with today's file. `/api/version` carries the fact that separates them.
    """
    body = TestClient(app).get("/api/version").json()

    assert body["version"]
    assert body["server"]["stale"] is False
    assert body["server"]["loaded_at"]
    assert body["server"]["newest_source"]


def test_a_server_older_than_its_own_files_says_so(tmp_path):
    """Compared against this process's import, not against "now"."""
    (tmp_path / "a.py").write_text("x = 1")
    loaded_at = time.time_ns()

    assert server_state(loaded_at, tmp_path)["stale"] is False

    later = tmp_path / "b.py"
    later.write_text("y = 2")
    os.utime(later, ns=(loaded_at + 1_000_000_000,) * 2)

    assert server_state(loaded_at, tmp_path)["stale"] is True


def test_the_pages_own_files_are_not_part_of_that_question(tmp_path):
    """static/ is read per request, so it can never be the stale half.

    Only the imported modules freeze at their import. Counting an asset here
    would report a stale server every time the js was edited - and that is the
    one change a reload really does fix.
    """
    (tmp_path / "a.py").write_text("x = 1")
    loaded_at = time.time_ns()
    asset = tmp_path / "app.js"
    asset.write_text("//")
    os.utime(asset, ns=(loaded_at + 1_000_000_000,) * 2)

    assert server_state(loaded_at, tmp_path)["stale"] is False


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
