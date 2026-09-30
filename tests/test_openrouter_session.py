"""One whole OpenRouter turn, through the same session the page uses.

The other OpenRouter files check the request in isolation. This one runs the
loop end to end - settle options, build, verify, send, stream, record - because
that is the path a real key exercises, and the pieces can each be right while
the joins between them are not. A session built on an Anthropic model used to
be the only one that could reach ``stream_turn`` at all.

The OpenAI client is replaced, so nothing reaches the network and no test needs
a key. What is *not* replaced is any part of this project or of
``llm-openrouter``: the request sent and the counters reported are the real
ones, produced by the real code.
"""

import json

import llm
import pytest
from starlette.testclient import TestClient

from native_api_chat.app import SESSIONS, app
from native_api_chat.chat import ChatSession
from native_api_chat.providers.openrouter import OpenRouterOptions, OpenRouterProvider
from native_api_chat.turn import MissingKeyError

pytest.importorskip("llm_openrouter")

PROVIDER = OpenRouterProvider()


@pytest.fixture
def client():
    SESSIONS.clear()
    yield TestClient(app)
    SESSIONS.clear()


@pytest.fixture
def session(openrouter_registry, fake_openrouter):
    """A session on an OpenRouter model, with the reply scripted."""

    def factory(text="Hi there.", *, usage=None, incomplete=False, **fields):
        (model_id,) = openrouter_registry("openai/gpt-5.4")
        fake_openrouter.answers(text, usage=usage, incomplete=incomplete)
        return ChatSession(OpenRouterOptions(model=model_id, max_tokens=64, **fields))

    return factory


# --- the loop ---------------------------------------------------------------


def test_a_whole_turn_runs_on_an_openrouter_model(session, fake_openrouter):
    """The closed loop: the answer comes back and the call was really made.

    This is the test that would have caught the session being Anthropic-shaped,
    which it was until the provider seam existed - it failed reaching for a
    ``claude_model_id`` on an OpenRouter model.
    """
    turn = session("Hi there.").run_turn("hello")

    assert turn["text"] == "Hi there."
    assert turn["chunks"] == ["Hi there."]
    assert [method for method, _ in fake_openrouter] == ["responses.create"]


def test_the_transport_reported_is_the_method_called(session, fake_openrouter):
    """A name the page shows and a call nobody made is the defect this repo
    already refuses for ``messages.create``. It has to hold per provider."""
    turn = session().run_turn("hello")

    (method, _) = fake_openrouter[-1]
    assert turn["transport"] == method == "responses.create"
    assert "client.responses.create(" in turn["code"]
    assert "chat.completions.create(" not in turn["code"]


def test_the_other_transport_calls_the_other_method(session, fake_openrouter):
    """``chat_completions`` is a per-request choice, so the loop has to close
    on both paths, not only on the default one."""
    turn = session(chat_completions=True).run_turn("hello")

    (method, _) = fake_openrouter[-1]
    assert turn["transport"] == method == "chat.completions.create"
    assert "client.chat.completions.create(" in turn["code"]


def test_the_request_sent_is_the_request_shown(session, fake_openrouter):
    """The pane is only worth reading if it is the sent request, and the two
    are built by different code: ours renders, the plugin sends."""
    turn = session(reasoning_effort="low").run_turn("hello")

    (_, sent) = fake_openrouter[-1]
    for field, value in turn["kwargs"].items():
        assert sent.get(field) == value, field


def test_the_record_holds_the_turn(session):
    """One record, or the transcript and the pane can disagree about a turn."""
    record = session("Hi there.").run_turn("hello")["record"]

    assert record["user_input"] == "hello"
    assert record["response"]["text"] == "Hi there."
    assert record["rendered_code"].startswith("import os")


# --- what the turn cost -----------------------------------------------------


def test_the_cache_read_is_taken_from_where_openrouter_puts_it(session):
    """Nested under ``input_tokens_details``, not at the top level.

    Anthropic reports its cache counters as siblings of the token counts;
    OpenRouter nests them. Reading the Anthropic spelling here finds nothing
    and reports every turn as a cache miss, which is a number, so nothing
    downstream can tell that it is a wrong one.
    """
    chat = session(
        usage={
            "input_tokens": 1200,
            "output_tokens": 8,
            "input_tokens_details": {"cached_tokens": 1024},
        }
    )
    chat.run_turn("hello")

    assert chat.baseline_usage == {
        "input": 1200,
        "output": 8,
        "cache_creation": None,
        "cache_read": 1024,
    }


def test_no_cache_hits_and_no_cache_report_are_different_answers(session):
    """``llm`` strips zeros, so the parent's presence is the whole signal.

    A model that cached nothing and a provider that said nothing both arrive
    as a missing ``cached_tokens``. Only the first may be shown as 0%.
    """
    cached_nothing = session(
        usage={
            "input_tokens": 12,
            "output_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0},
        }
    )
    cached_nothing.run_turn("hello")
    assert cached_nothing.baseline_usage["cache_read"] == 0

    said_nothing = session(usage={"input_tokens": 12, "output_tokens": 8})
    said_nothing.run_turn("hello")
    assert said_nothing.baseline_usage["cache_read"] is None


def test_a_cache_write_is_unknown_rather_than_inferred(session):
    """OpenRouter prices a cache write for some models and reports none here.
    Deriving one from the read would invent the more expensive half."""
    chat = session(
        usage={
            "input_tokens": 1200,
            "output_tokens": 8,
            "input_tokens_details": {"cached_tokens": 1024},
        }
    )
    chat.run_turn("hello")

    assert chat.baseline_usage["cache_creation"] is None


def test_a_truncated_reply_reports_no_figure_rather_than_zero(session):
    """The defect a real free-model turn exposed, and the worst kind.

    A reply cut off at the ceiling ends the Responses stream with
    ``response.incomplete``, which llm does not read, so every count arrives
    None. The meter added four Nones to nothing and reported 0 tokens as
    ``API usage`` - a measured figure, invented. Being measured is the whole
    reason that label is trusted over the estimate beside it.
    """
    chat = session("Hi", incomplete=True)
    chat.run_turn("hello")

    assert chat.baseline_usage is None
    assert chat.context().as_dict()["source"] == "estimated"


def test_a_truncated_responses_turn_has_no_stop_reason_to_report(session):
    """Not a choice this project gets to make: llm discards the event.

    ``response.incomplete`` carries the reason, and llm's Responses handler
    reads only ``response.completed``, so nothing about the truncation reaches
    this process. The page says "unreported", which is the truth - the turn is
    not recorded as having stopped normally.

    This test states the gap rather than papering over it. Inferring "must
    have hit the ceiling" from a missing Message would be a guess presented as
    the provider's own word, and the fix belongs in llm.
    """
    chat = session("Hi", incomplete=True)
    record = chat.run_turn("hello")["record"]

    assert record["response"]["stop_reason"] is None
    assert record["response"]["response_json"] is None
    # The reply itself still survives, off the stream.
    assert record["response"]["text"] == "Hi"


def test_a_finished_reply_says_so_too(session):
    """The other half: a status is reported when nothing went wrong either."""
    record = session("Hi there.").run_turn("hello")["record"]

    assert record["response"]["stop_reason"] == "completed"


def test_the_other_transport_reports_its_own_word_for_it(session):
    """Chat Completions calls it ``finish_reason`` and llm does read it, so
    this path keeps the counts the Responses path loses."""
    chat = session("Hi", incomplete=True, chat_completions=True)
    turn = chat.run_turn("hello")

    assert turn["record"]["response"]["stop_reason"] == "length"


def test_the_context_figure_follows_the_reported_usage(session):
    """Measured, then reported, then estimated. Once the provider has said how
    many tokens the request was, our own guess stops being the answer."""
    chat = session(usage={"input_tokens": 1200, "output_tokens": 8})
    before = chat.context().as_dict()
    chat.run_turn("hello")
    after = chat.context().as_dict()

    assert before["source"] == "estimated"
    assert after["source"].startswith("API usage")
    assert after["tokens"] >= 1208
    assert after["limit"] == 1_000_000


# --- what must not be asked -------------------------------------------------


def test_anthropics_counter_is_never_asked_about_an_openrouter_turn(
    client, openrouter_registry, fake_openrouter, monkeypatch
):
    """The counter is one provider's, and it is handed the request just built.

    The Chat Completions request carries ``messages``, which is exactly the
    field ``count_tokens`` wants, so the call would be well-formed enough to
    succeed - posting an OpenRouter conversation to Anthropic under the user's
    Anthropic key. The Responses request carries no messages and would fail,
    but only after the same round trip, and it would leave the real counter in
    its failure cooldown.
    """
    from native_api_chat import token_count

    asked: list[dict] = []
    monkeypatch.setattr(token_count, "lookup", lambda kwargs: asked.append(kwargs))
    monkeypatch.setattr(token_count, "kick", lambda kwargs: asked.append(kwargs))
    (model_id,) = openrouter_registry("openai/gpt-5.4")

    for transport in (False, True):
        response = client.post(
            "/api/preview",
            json={"text": "hello", "model": model_id, "chat_completions": transport},
        )
        assert response.status_code == 200

    assert asked == []


def test_the_context_figure_says_it_is_an_estimate_when_nobody_counted(
    client, openrouter_registry, fake_openrouter
):
    """No counter and no turn yet leaves the estimate, and it has to say so
    rather than present a guess as the provider's own number."""
    (model_id,) = openrouter_registry("openai/gpt-5.4")

    body = client.post(
        "/api/preview", json={"text": "hello", "model": model_id}
    ).json()

    assert body["context"]["source"] == "estimated"
    assert body["context"]["pending"] is False


# --- the route --------------------------------------------------------------


def test_the_chat_route_answers_for_an_openrouter_model(
    client, openrouter_registry, fake_openrouter
):
    """The dispatch: a model id is the only thing that picks the provider, and
    the SDK named is the one whose code was rendered."""
    (model_id,) = openrouter_registry("openai/gpt-5.4")
    fake_openrouter.answers("Hi there.")

    response = client.post("/api/chat", json={"text": "hello", "model": model_id})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Hi there."
    assert body["sdk"] == "openai-python"
    assert "openrouter.ai/api/v1" in body["code"]


def test_the_route_reads_the_openrouter_form(client, openrouter_registry, fake_openrouter):
    """The payload is the provider's own, so a field only OpenRouter has must
    survive the trip and land on the wire."""
    (model_id,) = openrouter_registry("openai/gpt-5.4")
    fake_openrouter.answers()

    response = client.post(
        "/api/chat",
        json={"text": "hello", "model": model_id, "reasoning_effort": "high"},
    )

    assert response.status_code == 200
    (_, sent) = fake_openrouter[-1]
    assert sent["reasoning"]["effort"] == "high"


def test_no_key_is_ever_rendered_or_returned(client, openrouter_registry, fake_openrouter):
    """The key reaches the client and nothing else. One test on the whole
    loop's output, because that is where a leak would actually surface."""
    (model_id,) = openrouter_registry("openai/gpt-5.4", key="sk-or-secret-value")
    fake_openrouter.answers("Hi there.")

    body = client.post(
        "/api/chat", json={"text": "hello", "model": model_id}
    ).json()

    assert "sk-or-secret-value" not in json.dumps(body)
    assert 'os.environ["OPENROUTER_KEY"]' in body["code"]


@pytest.mark.parametrize("route", ["/api/preview", "/api/chat", "/api/chat/stream"])
def test_every_route_answers_a_missing_key_rather_than_failing(
    client, openrouter_registry, monkeypatch, route
):
    """A missing key now surfaces while the session is being built, which is
    before the guard each route used to have. Two of the three answered 500."""
    (model_id,) = openrouter_registry("openai/gpt-5.4")
    llm.get_model(model_id)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)

    response = client.post(route, json={"text": "hello", "model": model_id})

    assert response.status_code == 400
    assert response.json()["error"] == "missing_api_key"
    assert "OPENROUTER_KEY" in response.json()["detail"]


def test_a_keyless_machine_is_told_about_the_key_not_the_model(
    openrouter_registry, monkeypatch
):
    """The refusal has to name the thing the user can fix.

    ``llm-openrouter`` registers nothing without a key, so llm's own answer is
    "Unknown model: openrouter/openai/gpt-5.4" - which sends the user hunting
    for a typo in an id that is perfectly correct.
    """
    (model_id,) = openrouter_registry("openai/gpt-5.4")
    # Register with the key present - the plugin offers no models without one -
    # then take it away, which is the machine a first run is actually on.
    llm.get_model(model_id)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)

    with pytest.raises(MissingKeyError, match="OPENROUTER_KEY"):
        ChatSession(OpenRouterOptions(model=model_id, max_tokens=64))
