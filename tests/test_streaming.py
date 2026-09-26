"""The Streaming control: what it changes, and what it cannot.

``stream`` is a UI presentation choice, not a provider parameter. These tests
pin down both halves of that sentence offline:

- ON yields each chunk as the provider produces it; OFF yields nothing until
  the turn is finished, then delivers the final accumulated Message.
- The SDK code always shows the transport the installed plugin really uses, so
  a buffered turn must never be rendered as ``messages.create()`` when the
  plugin opens a stream.
"""

import pytest
from starlette.testclient import TestClient

from llm_sdk_view.app import app
from llm_sdk_view.capabilities import capabilities_for, plugin_transport
from llm_sdk_view.chat import (
    STREAMING_TRANSPORT_NOTE,
    ChatOptions,
    final_message_text,
)

FOUR_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)


# --- what the installed plugin really does -----------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_every_model_reports_the_same_transport(model_id):
    """Streaming support is not a model capability, so nothing varies."""
    import llm

    model = llm.get_model(capabilities_for(model_id).llm_id)

    assert plugin_transport(model) == "stream"


@pytest.mark.parametrize("stream", (True, False))
def test_the_plugin_opens_a_stream_either_way(make_session, transports, stream):
    """A buffered turn is still sent through the streaming transport."""
    chat = make_session(stream=stream)
    chat.run_turn("Hello")

    assert transports == ["stream"]


def test_no_create_call_is_ever_made(make_session, transports, fake_provider):
    make_session(stream=False).run_turn("Hello")

    assert "create" not in transports
    assert fake_provider, "the turn still went out, just buffered"


# --- what the control changes ------------------------------------------------


def test_stream_on_yields_each_chunk(make_session):
    events = list(make_session(stream=True).stream_turn("Hello"))

    assert [e["text"] for e in events if e["type"] == "text"] == ["Hello", " world"]
    assert events[-1]["type"] == "done"


def test_stream_off_yields_no_chunks(make_session):
    events = list(make_session(stream=False).stream_turn("Hello"))

    assert [e for e in events if e["type"] == "text"] == []
    assert events[-1] == {"type": "done", "text": "Hello world"}


def test_both_modes_end_with_the_same_reply(make_session):
    streamed = make_session(stream=True).run_turn("Hello")
    buffered = make_session(stream=False).run_turn("Hello")

    assert streamed["text"] == buffered["text"] == "Hello world"


def test_both_modes_send_the_same_provider_parameters(make_session, fake_provider):
    make_session(stream=True).run_turn("Hello")
    streamed = dict(fake_provider[-1])
    make_session(stream=False).run_turn("Hello")
    buffered = dict(fake_provider[-1])

    assert streamed == buffered, "streaming must not change the request"


def test_the_final_message_is_what_both_modes_show(make_session):
    """Display and persistence come from the finished Message, not fragments."""
    chat = make_session(stream=True)
    result = chat.run_turn("Hello")

    assert result["chunks"] == ["Hello", " world"]
    assert result["text"] == "Hello world"
    assert final_message_text(chat.conversation.responses[-1]) == "Hello world"


def test_the_turn_is_recorded_on_the_conversation_in_both_modes(make_session):
    for stream in (True, False):
        chat = make_session(stream=stream)
        chat.run_turn("First")
        chat.run_turn("Second")

        roles = [m["role"] for m in chat.prepare("Third").kwargs["messages"]]
        assert roles == ["user", "assistant", "user", "assistant", "user"]


# --- what the code pane must say ---------------------------------------------


def test_stream_on_renders_the_streaming_transport(make_session):
    prepared = make_session(stream=True).prepare("Hello")

    assert prepared.transport == "stream"
    assert "with client.messages.stream(" in prepared.code
    assert "stream.get_final_message()" in prepared.code
    assert "client.messages.create(" not in prepared.code


def test_stream_off_still_renders_the_streaming_transport(make_session):
    """No fake create(): the plugin does not call it."""
    prepared = make_session(stream=False).prepare("Hello")

    assert prepared.transport == "stream"
    assert "with client.messages.stream(" in prepared.code
    assert "client.messages.create(" not in prepared.code
    # ...and the reason is stated rather than silently substituted.
    assert "buffered" in prepared.code
    assert "ten minutes" in prepared.code


def test_the_note_explains_that_streaming_is_not_a_provider_parameter():
    assert "not which SDK method is used" in STREAMING_TRANSPORT_NOTE


def test_stream_is_not_a_provider_parameter(make_session, fake_provider):
    """Nothing about `stream` may appear in the request."""
    make_session(stream=False).run_turn("Hello")

    assert "stream" not in fake_provider[-1]
    assert ChatOptions(stream=False).max_tokens == 16384


# --- the form surface ---------------------------------------------------------


def test_the_form_reports_the_default_and_the_transport():
    data = TestClient(app).get("/api/form").json()

    assert data["defaults"]["stream"] is True
    assert data["transport"]["sdk_method"] == "stream"
    assert "ten minutes" in data["transport"]["streaming_note"]


def test_a_buffered_turn_over_http(monkeypatch, fake_provider, transports):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    from llm_sdk_view.app import SESSIONS

    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/chat", json={"text": "Hello", "stream": False}
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Hello world"
    assert response.json()["transport"] == "stream"
    assert transports == ["stream"]


def test_the_ui_shows_the_transport_rather_than_assuming_one():
    from pathlib import Path

    import llm_sdk_view

    html = (Path(llm_sdk_view.__file__).parent / "static" / "index.html").read_text("utf-8")

    assert "client.messages.create(" not in html
    assert "transport" in html
