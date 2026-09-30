"""The right pane has to be readable before the call is paid for.

These tests prove the preview is the real request and not a second model of
it: it runs the same ``prepare()`` path, renders with the same renderer, and
is the only thing in the app that is allowed to build a request without
sending it.
"""

import pytest
from starlette.testclient import TestClient

from native_api_chat import token_count
from native_api_chat.app import SESSIONS, app
from native_api_chat.app import _prior_usage as prior_usage

SONNET = "claude-sonnet-5"
HAIKU = "claude-haiku-4-5-20251001"

FOUR_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)


@pytest.fixture
def client():
    SESSIONS.clear()
    yield TestClient(app)
    SESSIONS.clear()


@pytest.fixture
def key(monkeypatch):
    """A key for the send path only. The preview needs none."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")


def payload(**kwargs):
    base = {"session_id": "preview-tests", "text": "Hello", "model": SONNET}
    base.update(kwargs)
    return base


def test_a_preview_is_built_without_sending(client, fake_provider, transports):
    data = client.post("/api/preview", json=payload()).json()

    assert data["sent"] is False
    assert "client.messages.stream(" in data["code"]
    # Building a request is not sending one.
    assert transports == []


def test_the_preview_is_the_request_send_would_have_built(
    client, fake_provider, transports, key
):
    preview = client.post("/api/preview", json=payload()).json()
    sent = client.post("/api/chat", json=payload()).json()

    assert preview["code"] == sent["code"]
    assert preview["kwargs"] == sent["kwargs"]
    assert transports == ["stream"]


def test_a_preview_never_records_a_turn(client, fake_provider, key):
    client.post("/api/preview", json=payload(text="one"))
    client.post("/api/preview", json=payload(text="two"))

    # Looking at a request must not start a conversation of its own.
    assert "preview-tests" not in SESSIONS


def test_a_preview_of_a_saved_conversation_carries_its_history(
    client, fake_provider, transports, key
):
    """The pane must show the request Send will build, history included.

    A page that has just opened a stored conversation holds no live session
    yet. Building the preview on an empty one showed only the message being
    typed while Send appended it to the conversation: the right pane was
    quietly a second model of the request, and the context figure counted a
    single message worth of tokens.
    """
    client.post("/api/chat", json=payload(text="first"))
    conversation_id = client.get("/api/conversations").json()["items"][0]["id"]
    # The page has the conversation open but has not sent anything yet.
    SESSIONS.clear()

    preview = client.post(
        "/api/preview", json=payload(text="second", conversation_id=conversation_id)
    ).json()
    sent = client.post(
        "/api/chat", json=payload(text="second", conversation_id=conversation_id)
    ).json()

    assert [message["role"] for message in preview["kwargs"]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert preview["kwargs"] == sent["kwargs"]
    assert preview["code"] == sent["code"]


def test_looking_at_another_model_does_not_discard_the_history(
    client, fake_provider, transports, key
):
    """A model switch in the form must not silently start a new conversation.

    The preview builds on a throwaway session, so the stored conversation keeps
    its history until the user actually sends something.
    """
    client.post("/api/chat", json=payload(text="first"))
    client.post("/api/preview", json=payload(text="just looking", model=HAIKU))
    second = client.post("/api/chat", json=payload(text="second")).json()

    # user, assistant, user: the first turn is still in the request.
    assert [message["role"] for message in second["kwargs"]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]


@pytest.mark.parametrize("model", FOUR_MODELS)
def test_every_model_can_be_previewed(client, fake_provider, model):
    data = client.post("/api/preview", json=payload(model=model)).json()

    assert "client.messages.stream(" in data["code"]
    assert data["context"]["limit"] > 0


def test_an_illegal_combination_is_refused_before_anything_is_sent(
    client, fake_provider, transports
):
    """Refusing early is the point: the mistake has to be free to fix."""
    response = client.post(
        "/api/preview", json=payload(model=HAIKU, thinking="on", max_tokens=512)
    )

    assert response.status_code == 400
    assert transports == []


def test_an_empty_draft_reports_the_conversation_baseline(
    client, fake_provider, transports, key
):
    """The meter counts the whole conversation, from the provider's own numbers.

    Opening a stored conversation used to leave the meter at the empty-page
    figure - one token - because only a typed draft triggered a preview and
    an empty one was refused. The baseline preview builds the request Send
    would start from, history included, so the figure is the conversation's
    real weight before a word is typed.

    It is the provider's own usage rather than an estimate: the stored turn
    reported input 10 and output 3, and both are part of the next request.
    """
    client.post("/api/chat", json=payload(text="first message in a stored conversation"))
    conversation_id = client.get("/api/conversations").json()["items"][0]["id"]
    # The page has the conversation open but has not sent anything yet.
    SESSIONS.clear()
    sent_so_far = list(transports)

    response = client.post(
        "/api/preview", json=payload(text="", conversation_id=conversation_id)
    )

    assert response.status_code == 200
    data = response.json()
    assert data["baseline"] is True
    assert data["sent"] is False
    # The stored turn outweighs the empty-page figure by far.
    assert data["context"]["tokens"] == 13
    assert data["context"]["source"] == "API usage"
    assert data["context"]["pending"] is False
    # A baseline builds no request to show, and sends nothing either.
    assert "code" not in data
    assert transports == sent_so_far


def test_an_empty_draft_without_a_conversation_is_refused(client, fake_provider):
    """No history, no baseline.

    With nothing stored, no block exists for the cache marker to ride on,
    so the request cannot be built: the fresh page keeps the form's own
    figure and never asks for this one.
    """
    response = client.post("/api/preview", json=payload(text=""))

    assert response.status_code == 400


# --- the context figure the preview reports -----------------------------------


def test_the_preview_reports_the_apis_own_count(client, fake_provider, monkeypatch):
    """The counter's answer is what the meter shows, and it is not an estimate."""
    monkeypatch.setattr(token_count, "lookup", lambda kwargs: 2225)

    data = client.post("/api/preview", json=payload()).json()

    assert data["context"]["tokens"] == 2225
    assert data["context"]["source"] == "API count"
    assert data["context"]["pending"] is False


def test_a_count_that_is_still_running_is_announced(client, fake_provider, monkeypatch):
    """A fallback figure says a count is on its way, rather than passing itself
    off as the API's number or waiting for the network."""
    monkeypatch.setattr(token_count, "lookup", lambda kwargs: None)
    monkeypatch.setattr(token_count, "kick", lambda kwargs: True)
    monkeypatch.setattr(token_count, "api_key", lambda: "fake-key-for-tests")

    data = client.post("/api/preview", json=payload()).json()

    assert data["context"]["pending"] is True
    assert data["context"]["source"] != "API count"


def test_a_measured_conversation_is_not_counted_again(
    client, fake_provider, transports, key, monkeypatch
):
    """The provider already answered for a conversation it has run.

    Asking the counter to re-measure a request the conversation's own usage
    already describes would spend a round trip to be told the same number.
    """
    client.post("/api/chat", json=payload(text="a stored turn"))
    conversation_id = client.get("/api/conversations").json()["items"][0]["id"]
    SESSIONS.clear()

    def never(kwargs):
        raise AssertionError("a measured conversation must not be counted again")

    monkeypatch.setattr(token_count, "lookup", never)
    data = client.post(
        "/api/preview", json=payload(text="", conversation_id=conversation_id)
    ).json()

    assert data["context"]["source"] == "API usage"
    assert data["context"]["pending"] is False


def test_the_baseline_comes_from_the_turns_own_usage(client, fake_provider, key):
    """The stored usage is normalised into the four counters context means.

    llm's database holds the provider's own key names; the meter sums uncached
    input, cache writes, cache reads and the reply it produced.
    """
    client.post("/api/chat", json=payload(text="a stored turn"))
    conversation_id = client.get("/api/conversations").json()["items"][0]["id"]

    usage = prior_usage(conversation_id)

    # The fake provider reports 10 input and 3 output tokens.
    assert usage["input"] == 10
    assert usage["output"] == 3
    assert prior_usage("no-such-conversation") is None


def test_an_omitted_thinking_value_still_means_the_model_default(
    client, fake_provider, transports, key
):
    """"Not set" is the model's default, never "thinking off".

    A form change applied to a live session once left it as None, and the
    request builder read None as off and sent {"type": "disabled"} on a model
    that is always thinking.
    """
    client.post("/api/chat", json=payload(text="first"))
    second = client.post("/api/chat", json=payload(text="again")).json()

    assert second["kwargs"]["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_an_empty_effort_is_the_default_not_a_level(client, fake_provider):
    """A greyed-out select sends '' after a model switch.

    The form always carries the effort key, so '' means "nothing selected" -
    the model default - and must not be validated as a level Haiku does not
    have. This once blocked every Haiku send from a page that had visited
    Sonnet first.
    """
    data = client.post("/api/preview", json=payload(model=HAIKU, effort="")).json()

    assert "error" not in data
    assert "output_config" not in data["kwargs"]


def test_the_rebuild_is_handed_the_level_it_must_end_up_with(static_page):
    """The level and the rebuild have to arrive together.

    Replacing a select's options clears its selection, so a value written
    before the rebuild is wiped by it. The page used to rely on that clearing
    as the reset, and a reopened conversation was written first and rebuilt
    after - which left it on "default" and had the composer report the
    difference as a change the user had made. The stored level now goes to
    the rebuild, which keeps it when this model still offers it.

    This pins the wiring. The behaviour - a level the model cannot send is
    never left selected, a level it can send is never dropped - is measured
    in tests/test_conversation_state_in_a_browser.py, because a string in a
    page cannot be asked what a select ends up holding.
    """
    flat = " ".join(static_page.split())
    assert "function applyEffortState(chosen)" in flat
    assert "applyEffortState(options.effort);" in flat
    # Back to writing the select and rebuilding it afterwards would be the
    # old shape again, whichever order the two lines were in.
    assert "restore('effort'" not in flat


# --- the page can only ever send what the form displays ------------------------


def _page_payload(form: dict) -> dict:
    """Exactly what the page sends: the values its controls display.

    Every bug found in live smoke was of one class: a value the form shows
    being refused, or a value the form never showed being required. This
    payload is derived from the form itself, so the invariant below dies the
    moment those two views of the world drift apart.
    """
    controls = form["controls"]
    defaults = form["defaults"]
    return {
        "session_id": "preview-tests",
        "text": "Hello",
        "model": form["model"]["id"],
        "max_tokens": defaults["max_tokens"],
        "system": defaults["system"],
        "thinking": controls["thinking"]["value"],
        "effort": controls["effort"]["value"],
        "web_search": defaults["web_search"],
        "web_search_type": controls["web_search_type"]["value"],
        "allowed_callers": controls["allowed_callers"]["value"],
        # The select keeps its HTML default when the control is unsupported.
        "response_inclusion": defaults["response_inclusion"],
        "max_uses": defaults["max_uses"],
        "cache_control": defaults["cache_control"],
    }


@pytest.mark.parametrize("model", FOUR_MODELS)
def test_every_displayed_combination_previews_cleanly(
    client, fake_provider, transports, model
):
    """What the form shows must always be a request the runtime can build.

    Haiku once failed here three separate ways: a misread Models API entry,
    a stale effort level, and a caller check written against one model's
    default. One invariant now covers the whole class.
    """
    form = client.get(f"/api/form?model={model}").json()
    data = client.post("/api/preview", json=_page_payload(form)).json()

    assert "error" not in data, data.get("error")
    assert data["kwargs"]["model"] == form["model"]["sends"]
    assert data["sent"] is False


@pytest.mark.parametrize("model", FOUR_MODELS)
def test_the_displayed_caller_is_the_caller_the_request_really_uses(
    client, fake_provider, model
):
    """The form's caller fact and the request must never disagree."""
    form = client.get(f"/api/form?model={model}").json()
    displayed = form["controls"]["allowed_callers"]["value"]
    data = client.post("/api/preview", json=_page_payload(form)).json()

    assert "error" not in data, data.get("error")
    assert data["allowed_callers"] == displayed


def test_an_explicit_new_default_caller_on_the_basic_tool_is_resolved(
    make_session,
):
    """code_execution is meaningless to web_search_20250305.

    The field is never sent, so the request is the bare tool either way; the
    session resolves the value to the caller that will really apply instead
    of refusing one that has no effect on the wire.
    """
    prepared = make_session(
        model=HAIKU, allowed_callers="code_execution_20260120"
    ).prepare("Hi")

    assert prepared.allowed_callers == "direct"
