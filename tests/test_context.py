"""The context meter: what it counts, what it guesses, and what it refuses.

There is no tokenizer bundled with this project and counting tokens would be a
paid API call, so the figure shown before a turn is an estimate and says so.
Once the API answers, its own usage numbers replace the estimate. No part of
this compacts, summarises or silently trims the history: when the request and
its reserved output cannot fit, sending is refused instead.
"""

import pytest
from starlette.testclient import TestClient

from llm_sdk_view import chat as chat_module
from llm_sdk_view.app import app
from llm_sdk_view.chat import (
    ContextState,
    context_state,
    estimate_request_tokens,
)

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"


# --- the estimate -------------------------------------------------------------


def test_a_plain_request_can_be_estimated(make_session):
    prepared = make_session(model=HAIKU).prepare("Hello")

    assert prepared.context.tokens is not None
    assert prepared.context.tokens > 0
    assert prepared.context.source == "estimated"
    assert prepared.context.limit == 200_000
    assert prepared.context.percent is not None


def test_the_system_prompt_and_tools_are_counted(make_session):
    short = make_session(model=HAIKU).prepare("Hi")
    longer = make_session(model=HAIKU, system="x" * 4000).prepare("Hi")

    assert longer.context.tokens > short.context.tokens


def test_an_unmeasurable_block_means_unknown_rather_than_a_guess():
    kwargs = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image", "source": {"type": "base64", "data": "..."}},
                ],
            }
        ]
    }

    assert estimate_request_tokens(kwargs) is None


def test_unknown_tokens_leave_the_percentage_blank(make_session, capabilities):
    state = context_state(
        {"messages": [{"role": "user", "content": [{"type": "image"}]}]},
        capabilities(HAIKU),
    )

    assert state.tokens is None
    assert state.source == "unknown"
    assert state.percent is None
    assert state.fits is None


# --- the limit ----------------------------------------------------------------


def test_the_limit_is_labelled_with_its_source(make_session, capabilities):
    state = context_state({}, capabilities(HAIKU))

    assert state.limit == 200_000
    assert "fallback profile" in state.limit_source
    assert state.limit_data_source == "Fallback capability data"


# --- real usage wins ----------------------------------------------------------


def test_usage_replaces_the_estimate_once_the_api_answers(make_session):
    chat = make_session(model=HAIKU)
    result = chat.run_turn("Hello")

    assert result["context"]["source"] == "API usage"
    # The fake provider reports 10 input and 3 output tokens.
    assert result["context"]["tokens"] == 13


def test_the_estimate_is_still_shown_before_the_first_turn(make_session):
    chat = make_session(model=HAIKU)
    prepared = chat.prepare("Hello")

    assert prepared.context.source == "estimated"
    assert chat.context().source == "estimated"


# --- refusing rather than trimming --------------------------------------------


def test_a_request_that_cannot_fit_is_refused(make_session, monkeypatch, fake_provider):
    """No silent trimming: the turn is refused and says what to do instead."""
    chat = make_session(model=HAIKU)
    monkeypatch.setattr(
        chat_module, "estimate_request_tokens", lambda kwargs: 199_000
    )

    with pytest.raises(ValueError, match="does not fit"):
        chat.prepare("A very long conversation")

    assert fake_provider == []


def test_the_refusal_names_the_two_ways_out(make_session, monkeypatch):
    chat = make_session(model=HAIKU)
    monkeypatch.setattr(
        chat_module, "estimate_request_tokens", lambda kwargs: 199_000
    )

    with pytest.raises(ValueError) as caught:
        chat.prepare("A very long conversation")

    message = str(caught.value)
    assert "max_tokens" in message
    assert "new conversation" in message


def test_lowering_max_tokens_can_make_it_fit(make_session, monkeypatch):
    chat = make_session(model=HAIKU, max_tokens=1000)
    monkeypatch.setattr(
        chat_module, "estimate_request_tokens", lambda kwargs: 199_000
    )

    assert chat.prepare("A very long conversation").context.fits is True


def test_the_history_is_never_rewritten_to_make_room(make_session, fake_provider):
    """Even a long conversation is sent whole, or not at all."""
    chat = make_session(model=HAIKU)
    chat.run_turn("First")
    chat.run_turn("Second")
    prepared = chat.prepare("Third")

    texts = [
        block.get("text")
        for message in prepared.kwargs["messages"]
        for block in message["content"]
        if block.get("type") == "text"
    ]
    assert texts == ["First", "Hello world", "Second", "Hello world", "Third"]


# --- the surface ---------------------------------------------------------------


def test_the_form_reports_the_context_figure():
    data = TestClient(app).get("/api/form").json()

    assert data["context"]["limit"] == 200_000
    assert data["context"]["limit_source"]
    assert set(data["context"]) >= {"tokens", "limit", "source", "percent", "fits"}


def test_the_turn_events_carry_the_context(make_session):
    events = list(make_session(model=HAIKU).stream_turn("Hello"))

    assert events[0]["context"]["source"] == "estimated"
    assert events[-1]["context"]["source"] == "API usage"


def test_the_state_serialises_for_the_page():
    state = ContextState(
        tokens=10,
        limit=100,
        source="estimated",
        limit_source="fallback profile",
        limit_data_source="Fallback capability data",
        percent=10.0,
        reserved=5,
        fits=True,
    )

    assert state.as_dict()["percent"] == 10.0
    assert state.as_dict()["fits"] is True


def test_the_ui_shows_the_source_and_the_limit():
    from pathlib import Path

    import llm_sdk_view

    html = (Path(llm_sdk_view.__file__).parent / "static" / "index.html").read_text("utf-8")

    assert "contextUsage" in html
    assert "source:" in html
    assert "does not fit" in html


def test_adaptive_models_report_their_own_window(make_session):
    prepared = make_session(model=SONNET).prepare("Hello")

    assert prepared.context.limit == 1_000_000
