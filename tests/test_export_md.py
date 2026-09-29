"""The conversation export: one Markdown document, and the delete beside it.

The unit tests build records by hand so every section (thinking, tools,
sources, system change, cost) is exercised exactly once; the endpoint tests
go through the app with the fake provider so what is downloaded is what a
real turn left in llm's own database.
"""

from itertools import cycle
from unittest.mock import PropertyMock, patch

import pytest
from starlette.testclient import TestClient

from llm_sdk_view import store
from llm_sdk_view.app import SESSIONS, app
from llm_sdk_view.export_md import (
    content_disposition,
    conversation_markdown,
    export_filename,
)
from llm_sdk_view.records import ResponseView, TurnRecord

pytestmark = pytest.mark.usefixtures("key")

SONNET = "claude-sonnet-5"


def make_turn(
    user="问个问题",
    text="这是回答",
    thinking=None,
    citations=None,
    tools=None,
    system="",
    usage=None,
    timestamp="2026-09-29T05:15:18+00:00",
):
    response = ResponseView.from_dict(
        {
            "model": SONNET,
            "text": text,
            "thinking": thinking,
            "citations": citations or [],
            "server_tool_blocks": tools or [],
            "usage": usage
            or {"input_tokens": 2391, "output_tokens": 270,
                "cache_creation_input_tokens": 6589},
            "duration_ms": 24300,
        }
    )
    return TurnRecord(
        conversation_id="conv1",
        turn_id=f"turn-{user}",
        user_input=user,
        options={"model": SONNET, "system": system},
        request_kwargs={},
        rendered_code="",
        response=response,
        context={},
        timestamp=timestamp,
    )


def loaded(turns, name="我的对话"):
    return {"id": "conv1", "name": name, "model": SONNET, "turns": turns}


# --- the document ---------------------------------------------------------


def test_the_header_carries_what_a_reader_needs():
    document = conversation_markdown(loaded([make_turn(system="你是助手")]))

    assert document.startswith("# 我的对话\n")
    assert f"- **Model:** {SONNET}" in document
    assert "- **Turns:** 1" in document
    assert "- **System:** 你是助手" in document


def test_a_turn_is_user_then_answer_then_figures():
    document = conversation_markdown(loaded([make_turn()]))

    assert "**User**\n\n问个问题" in document
    assert "**Assistant**\n\n这是回答" in document
    # One metadata line: latency, tokens with the cache write, no invention.
    assert "> 24.3s · in 2,391 + 6,589 cache write · out 270" in document


def test_thinking_is_wrapped_the_way_claude_wraps_it():
    document = conversation_markdown(loaded([make_turn(thinking="先想想")]))

    assert "<thinking>\n先想想\n</thinking>" in document


def test_sections_that_never_happened_leave_no_trace():
    document = conversation_markdown(loaded([make_turn()]))

    assert "<thinking>" not in document
    assert "Sources:" not in document


def test_tools_and_sources_are_lines_not_payloads():
    turn = make_turn(
        tools=[{"type": "server_tool_use", "name": "web_search",
                "input": {"query": "EDA IP market"}},
               {"type": "web_search_tool_result", "content": []}],
        citations=[{"url": "https://example.com/a", "title": "报告"}],
    )
    document = conversation_markdown(loaded([turn]))

    assert '> web_search: "EDA IP market"' in document
    # A result block is not a call; only the query line appears.
    assert document.count("web_search") == 1
    assert "> Sources: [报告](https://example.com/a)" in document


def test_a_system_change_mid_conversation_is_noted():
    turns = [
        make_turn(user="one", system="v1"),
        make_turn(user="two", system="v2", timestamp="2026-09-29T05:16:00+00:00"),
    ]
    document = conversation_markdown(loaded(turns))

    assert "- **System:** v1" in document
    assert "> System prompt changed this turn: v2" in document


def test_the_total_cost_sums_priced_turns():
    turns = [make_turn(), make_turn(user="two")]
    with patch.object(ResponseView, "cost", new_callable=PropertyMock) as cost:
        cost.return_value = {"total": 0.02, "lines": []}
        document = conversation_markdown(loaded(turns))

    assert "- **Total cost:** $0.0400" in document
    assert document.count("$0.0200") == 2


def test_unpriced_turns_say_so_instead_of_summing_silently():
    turns = [make_turn(), make_turn(user="two")]
    # Priced and unpriced alternate across the two turns, however many
    # times the document asks (the header and each turn's meta both read).
    priced = cycle([{"total": 0.02, "lines": []}, None])
    with patch.object(ResponseView, "cost", new_callable=PropertyMock) as cost:
        cost.side_effect = lambda: next(priced)
        document = conversation_markdown(loaded(turns))

    assert "(partial: not every turn is priced)" in document


# --- the download name ------------------------------------------------------


def test_the_filename_keeps_cjk_and_drops_path_bytes():
    name = export_filename('我的/对话: "EDA"?', [make_turn()])

    assert "/" not in name and '"' not in name and "?" not in name
    assert name.startswith("我的对话")
    assert name.endswith("-2026-09-29.md")


def test_an_untitled_conversation_still_has_a_name():
    assert export_filename("  ", []).startswith("conversation")


def test_the_disposition_carries_utf8_per_rfc5987():
    header = content_disposition("我的对话-2026-09-29.md")

    assert header.startswith("attachment; ")
    # RFC 5987's filename* is the real name; the ASCII fallback a client that
    # ignores it would save must still read as a name, not "-2026-09-29.md".
    assert 'filename="conversation-2026-09-29.md"' in header
    assert "filename*=UTF-8''%E6%88%91%E7%9A%84%E5%AF%B9%E8%AF%9D" in header


def test_the_ascii_fallback_keeps_a_title_it_can_spell():
    header = content_disposition("My Notes-2026-09-28.md")

    assert 'filename="My-Notes-2026-09-28.md"' in header
    assert "filename*=UTF-8''My%20Notes-2026-09-28.md" in header


# --- through the app --------------------------------------------------------


@pytest.fixture
def client():
    SESSIONS.clear()
    yield TestClient(app)
    SESSIONS.clear()


def send(client, text, **kwargs):
    payload = {"session_id": "default", "text": text, "model": SONNET}
    payload.update(kwargs)
    return client.post("/api/chat", json=payload).json()


def test_the_export_is_the_conversation_that_was_stored(
    client, fake_provider, transports
):
    saved = send(client, "导出我")
    conversation_id = saved["record"]["conversation_id"]

    response = client.get(f"/api/conversations/{conversation_id}/export.md")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "attachment;" in response.headers["content-disposition"]
    assert "# 导出我" in response.text
    assert "**User**\n\n导出我" in response.text
    assert saved["text"] in response.text


def test_an_unknown_conversation_exports_nothing(client, fake_provider):
    response = client.get("/api/conversations/does-not-exist/export.md")

    assert response.status_code == 404


def test_delete_removes_every_trace_of_the_conversation(
    client, fake_provider, transports, isolated_history
):
    saved = send(client, "删了我")
    conversation_id = saved["record"]["conversation_id"]

    response = client.request("DELETE", f"/api/conversations/{conversation_id}")

    assert response.status_code == 200
    assert response.json() == {"deleted": conversation_id}
    assert client.get("/api/conversations").json()["items"] == []
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
    database = store.connect()
    assert database["threads"].count == 0
    assert database["turns"].count == 0
    assert database[store.SIDECAR_TABLE].count == 0


def test_delete_leaves_other_conversations_alone(client, fake_provider, transports):
    first = send(client, "留下的一")
    second = send(client, "删掉的二", new_conversation=True)

    client.request("DELETE", f"/api/conversations/{second['record']['conversation_id']}")

    items = client.get("/api/conversations").json()["items"]
    assert [item["id"] for item in items] == [first["record"]["conversation_id"]]


def test_deleting_nothing_is_a_404_not_a_success(client, fake_provider):
    response = client.request("DELETE", "/api/conversations/does-not-exist")

    assert response.status_code == 404
