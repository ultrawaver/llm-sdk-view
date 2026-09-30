"""The context meter: what it measures, what it guesses, and what it refuses.

The figure comes from the provider whenever the provider can answer - its free
``count_tokens`` endpoint for a request that has not been sent, its own ``usage``
for a conversation that has. A character estimate is the last resort and says
so, because the two numbers it cannot see are large: Anthropic expands this
app's 70-character ``web_search`` stub into roughly 2,200 tokens, and CJK text
costs about one token per character rather than one per four. No part of this
compacts, summarises or silently trims the history: when the request and its
reserved output cannot fit, sending is refused instead.
"""

import pytest
from starlette.testclient import TestClient

from native_api_chat import chat as chat_module
from native_api_chat.app import app
from native_api_chat.chat import (
    ChatOptions,
    ChatSession,
    ContextState,
    _estimate_text,
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


def test_the_estimate_does_not_read_chinese_as_english():
    """One token per CJK character, one per four Latin ones - measured.

    The rule this replaces was four characters per token for everything. That
    is right about English and out by a factor of four on Chinese, which is the
    language most of this app's own conversations are written in.
    """
    assert _estimate_text("中国的文字就是这样的") == 10
    assert _estimate_text("abcdefghij") == 3
    # Mixed text counts each part its own way.
    assert _estimate_text("你好世界 hello") == 4 + 2


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


def test_cache_tokens_count_toward_the_context_figure(make_session, capabilities):
    """The cached prefix occupies context exactly like fresh tokens do."""
    state = context_state(
        {},
        capabilities(HAIKU),
        usage={"input": 50, "output": 10, "cache_creation": 0, "cache_read": 8000},
    )

    # 8060, not the 60 the uncached-only figure used to report.
    assert state.tokens == 8060
    assert state.source == "API usage"


# --- where the figure comes from ----------------------------------------------


def test_the_estimate_is_still_shown_before_the_first_turn(make_session):
    chat = make_session(model=HAIKU)
    prepared = chat.prepare("Hello")

    assert prepared.context.source == "estimated"
    assert chat.context().source == "estimated"


def test_the_apis_own_count_beats_everything(capabilities):
    """What the counter says is the request's size; nothing here outranks it.

    The draft is deliberately not added: the count already covers the request
    including the message being typed.
    """
    state = context_state(
        {},
        capabilities(HAIKU),
        usage={"input": 100, "output": 10, "cache_creation": 0, "cache_read": 0},
        counted=2225,
        draft="Hello",
    )

    assert state.tokens == 2225
    assert state.source == "API count"


def test_a_conversation_the_provider_has_measured_is_not_estimated(make_session):
    """The bug this replaces: a two-turn conversation whose meter said 49.

    The provider had already reported 2,265 input and 70 output tokens for it,
    and the context after that turn is both. The meter counted characters
    instead and showed a forty-seventh of the truth.
    """
    chat = ChatSession(
        ChatOptions(model=HAIKU),
        baseline_usage={
            "input": 2265,
            "output": 70,
            "cache_creation": 0,
            "cache_read": 0,
        },
    )

    prepared = chat.prepare("Hello")

    # The message being typed is added to the measured baseline, and the sum
    # says which part of it is the estimate.
    assert prepared.context.tokens == 2335 + 2
    assert prepared.context.source == "API usage + estimated draft"
    # With nothing beyond the stored conversation, there is nothing to qualify.
    assert chat.measure(prepared.kwargs, "").tokens == 2335
    assert chat.measure(prepared.kwargs, "").source == "API usage"


def test_a_draft_on_a_measured_conversation_is_labelled_where_it_is_ours(
    capabilities,
):
    state = context_state(
        {},
        capabilities(HAIKU),
        usage={"input": 2225, "output": 27},
        draft="你好世界",
    )

    assert state.tokens == 2252 + 4
    assert state.source == "API usage + estimated draft"


def test_a_measured_baseline_needs_no_qualification(capabilities):
    state = context_state({}, capabilities(HAIKU), usage={"input": 2225, "output": 27})

    assert state.tokens == 2252
    assert state.source == "API usage"


def test_a_count_in_flight_is_not_the_default(capabilities):
    state = context_state({}, capabilities(HAIKU), usage={"input": 10})

    assert state.pending is False
    assert state.as_dict()["pending"] is False


def test_the_fit_check_uses_the_measured_figure_too(make_session, monkeypatch):
    """The refusal and the meter must not disagree about the same request.

    A conversation near the window must be refused by the same number the
    meter shows, not by a character count that cannot see the tools.
    """
    chat = ChatSession(
        ChatOptions(model=HAIKU),
        baseline_usage={
            "input": 199_000,
            "output": 0,
            "cache_creation": 0,
            "cache_read": 0,
        },
    )

    with pytest.raises(ValueError, match="API usage"):
        chat.prepare("A very long conversation")


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


def test_the_ui_shows_the_source_and_the_limit(static_page):
    assert "contextUsage" in static_page
    assert "source:" in static_page
    assert "does not fit" in static_page


def test_opening_a_conversation_repaints_the_context_meter(static_page):
    """The meter counts the whole conversation, not the empty draft.

    Opening a stored conversation once left the meter at the empty-page
    figure - one token - because only a typed draft triggered a preview.
    Now opening a conversation fires the baseline preview, and clearing the
    draft falls back to it instead of leaving a stale number.
    """
    assert "async function refreshContext()" in static_page
    assert "refreshContext();" in static_page
    assert "state.formContext = data.context;" in static_page


def test_the_percentage_escalates_like_the_bar(static_page):
    """Green while there is room, amber past 70%, red past 90% or when the
    request no longer fits - the escalation Claude's own UI uses."""
    assert "pct >= 70" in static_page
    assert "pct >= 90" in static_page
    assert ".pct.warn" in static_page
    assert ".pct.danger" in static_page


def test_the_meter_only_qualifies_the_figures_that_are_partly_ours(static_page):
    """A count and a usage report are the provider's own numbers and are shown
    as they are; the figures this app partly composed say which part is ours.

    The page has to be able to tell them apart, because only one of the three
    can see what Anthropic adds for the server tool this app always sends.
    """
    assert "const CONTEXT_LABEL = {" in static_page
    assert "'API usage + estimated draft': ' · draft estimated'" in static_page
    assert "'unknown': ' · cannot be measured'" in static_page
    assert "CONTEXT_LABEL[context.source] || ''" in static_page


def test_the_page_looks_again_for_a_count_that_is_still_running(static_page):
    """A fallback figure is replaced by the API's own number when it lands.

    A bounded number of times, so an unreachable API leaves the meter on a
    labelled estimate instead of turning the page into a polling loop.
    """
    assert "function lookAgainForTheCount(context)" in static_page
    assert "lookAgainForTheCount(context);" in static_page
    assert "contextLooksLeft <= 0" in static_page


def test_typing_a_message_clears_the_selection(static_page):
    """Typing drops the selected turn, and that has to reach the panes.

    The cost bar used to be part of this: it rendered the *selected* turn, so
    an old total stayed under a request that had not been built yet. The footer
    is the conversation's own total now (test_cost_scopes.py), which the
    selection cannot make stale - so typing no longer repaints it, and this
    pins both halves.
    """
    flat = " ".join(static_page.split())

    assert "byId('prompt').addEventListener('input', () => {" in flat
    # markSelected() is the same event reaching the bubbles: a turn that is
    # no longer selected must stop looking like the one in the pane.
    assert "state.selected = null; markSelected(); renderPane(); }" in flat
    assert "renderPane(); refreshCost(); }" not in flat


def test_adaptive_models_report_their_own_window(make_session):
    prepared = make_session(model=SONNET).prepare("Hello")

    assert prepared.context.limit == 1_000_000
