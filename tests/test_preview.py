"""The right pane has to be readable before the call is paid for.

These tests prove the preview is the real request and not a second model of
it: it runs the same ``prepare()`` path, renders with the same renderer, and
is the only thing in the app that is allowed to build a request without
sending it.
"""

import pytest
from starlette.testclient import TestClient

from llm_sdk_view.app import SESSIONS, app

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


def test_an_empty_prompt_has_nothing_to_preview(client, fake_provider):
    response = client.post("/api/preview", json=payload(text=""))

    assert response.status_code == 400


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


def test_a_greyed_effort_select_never_keeps_a_stale_level():
    """The page resets the select when its options are replaced.

    Browsers clear a select to '' when the selected option is removed, which
    used to leak the previous model's level into the next request.
    """
    from pathlib import Path

    import llm_sdk_view

    html = (Path(llm_sdk_view.__file__).parent / "static" / "index.html").read_text(
        "utf-8"
    )

    assert "if (!byId('effort').value || byId('effort').selectedOptions[0].disabled)" in (
        html
    )


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
