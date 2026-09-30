"""The streaming transport: what it is, and why the form cannot choose it.

The Anthropic API supports streaming and non-streaming calls, but the installed
``llm-anthropic`` opens ``messages.stream()`` for every turn: the API rejects
non-streaming requests whose ``max_tokens`` could run past ten minutes. So the
form shows ``stream: ON`` as a read-only fact, offers no OFF, and the right pane
renders the call that really happens. These tests pin all three down offline.
"""

import ast

import pytest
from starlette.testclient import TestClient

from native_api_chat.app import app
from native_api_chat.capabilities import capabilities_for, plugin_transport
from native_api_chat.chat import (
    RUNTIME_FIXED,
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
    """Streaming is not a model capability, so nothing varies."""
    import llm

    model = llm.get_model(capabilities_for(model_id).llm_id)

    assert plugin_transport(model) == "stream"


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_the_plugin_opens_a_stream_and_never_create(model_id, make_session, transports):
    chat = make_session(model=model_id)
    chat.run_turn("Hello")

    assert transports == ["stream"]
    assert "create" not in transports


def test_there_is_no_stream_option_to_send(make_session, fake_provider):
    """`stream` is not a provider parameter, so it cannot appear in a request."""
    make_session().run_turn("Hello")

    assert "stream" not in fake_provider[-1]


# --- what the turn produces ---------------------------------------------------


def test_the_turn_yields_each_chunk(make_session):
    events = list(make_session().stream_turn("Hello"))

    assert [e["text"] for e in events if e["type"] == "text"] == ["Hello", " world"]
    assert events[-1]["type"] == "done"
    assert events[-1]["text"] == "Hello world"


def test_reasoning_streams_as_its_own_event(make_session, response_scenario):
    """Thinking deltas reach the page as reasoning, never mixed into text."""
    response_scenario(
        {
            "id": "msg_think",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [
                {"type": "thinking", "thinking": "let me think about this"},
                {"type": "text", "text": "Hello world"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
    )
    events = list(make_session().stream_turn("Hello"))

    reasoning = [e["text"] for e in events if e["type"] == "reasoning"]
    assert reasoning == ["let me think about this"]
    # Reasoning is its own channel: text events stay pure answer text.
    assert [e["text"] for e in events if e["type"] == "text"] == ["Hello", " world"]
    # The closing event carries the finished Message's thinking, so the
    # strip can be re-rendered from the record rather than from fragments.
    done = events[-1]
    assert done["type"] == "done"
    assert done["thinking"] == "let me think about this"


def test_a_turn_without_thinking_sends_no_reasoning(make_session):
    events = list(make_session().stream_turn("Hello"))

    assert [e for e in events if e["type"] == "reasoning"] == []
    assert events[-1]["thinking"] is None


def test_the_final_message_is_what_the_UI_shows_and_keeps(make_session):
    """Display and persistence come from the finished Message, not fragments."""
    chat = make_session()
    result = chat.run_turn("Hello")

    assert result["chunks"] == ["Hello", " world"]
    assert result["text"] == "Hello world"
    assert final_message_text(chat.conversation.responses[-1]) == "Hello world"


def test_the_turn_is_recorded_on_the_conversation(make_session):
    chat = make_session()
    chat.run_turn("First")
    chat.run_turn("Second")

    roles = [m["role"] for m in chat.prepare("Third").kwargs["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


# --- what the code pane must say ---------------------------------------------


@pytest.mark.parametrize("model_id", FOUR_MODELS)
def test_the_code_renders_the_streaming_transport(model_id, make_session):
    prepared = make_session(model=model_id).prepare("Hello")

    assert prepared.transport == "stream"
    assert "with client.messages.stream(" in prepared.code
    assert "client.messages.create(" not in prepared.code
    # The reply is read off the stream, as the official Playground shows it -
    # a lone get_final_message() would read like a one-shot fetch.
    assert "for text in stream.text_stream:" in prepared.code


def test_the_code_keeps_the_order_the_plugin_built_the_request_in(make_session):
    """The key order is llm-anthropic's assembly order, not a sorting."""
    prepared = make_session().prepare("Hello")
    call = ast.parse(prepared.code).body[-1].items[0].context_expr

    assert [keyword.arg for keyword in call.keywords] == list(prepared.kwargs)


def test_the_code_pane_carries_no_commentary(make_session):
    """The code pane is the request. A request carries no explanation."""
    code = make_session().prepare("Hello").code

    assert "#" not in code
    assert "ten minutes" not in code
    assert "llm-anthropic" not in code


def test_the_note_explains_that_streaming_is_fixed_by_the_runtime():
    assert "fixed by the runtime" in STREAMING_TRANSPORT_NOTE
    assert "ten minutes" in STREAMING_TRANSPORT_NOTE


# --- the form surface ---------------------------------------------------------


def test_the_form_offers_no_stream_choice():
    data = TestClient(app).get("/api/form").json()
    control = data["controls"]["stream"]

    assert control["status"] == RUNTIME_FIXED
    assert control["value"] == "ON"
    assert control["editable"] is False
    assert control["sdk_method"] == "client.messages.stream(...)"
    options = {option["value"]: option for option in control["options"]}
    assert options["ON"]["disabled"] is False
    assert options["OFF"]["disabled"] is True
    assert "not used by llm-anthropic" in options["OFF"]["note"]


def test_the_form_reports_the_transport():
    data = TestClient(app).get("/api/form").json()

    assert data["transport"]["sdk_method"] == "stream"
    assert "ten minutes" in data["transport"]["streaming_note"]


def test_a_turn_over_http_uses_the_streaming_transport(
    monkeypatch, fake_provider, transports
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    from native_api_chat.app import SESSIONS

    SESSIONS.clear()
    response = TestClient(app).post("/api/chat", json={"text": "Hello"})

    assert response.status_code == 200
    assert response.json()["text"] == "Hello world"
    assert response.json()["transport"] == "stream"
    assert transports == ["stream"]


def test_the_ui_shows_the_transport_rather_than_assuming_one(static_page):
    assert "client.messages.create(" not in static_page
    assert "controls.stream" in static_page
    assert "API supported" in static_page


def test_no_stream_field_survives_on_the_options():
    assert "stream" not in set(ChatOptions.__dataclass_fields__)
