"""The API's own token counter: what it counts, what it caches, what it refuses.

The counter is free, so these tests could technically call it. They do not:
what has to hold is the plumbing around it - which fields make up a countable
request, that one exact request is counted once, that a failure leaves the
caller with a labelled estimate instead of an exception, and that a dead
network cannot cost a call per keystroke.
"""

import json
import time
import types
from datetime import timedelta

import pytest

from llm_sdk_view import token_count


def wait_until(predicate, timeout: float = 3.0) -> bool:
    """Wait for a background count. Generous, but it normally returns first try."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class _CountingClient:
    """A client whose answers come from a table, and which remembers the asks."""

    def __init__(self, counts: dict, asked: list):
        self.counts = counts
        self.asked = asked
        self.messages = self

    def count_tokens(self, **payload):
        # The app passes the request fields and a timeout; only the fields are
        # the request, so only they are recorded.
        payload.pop("timeout", None)
        self.asked.append(payload)
        if payload.get("model") not in self.counts:
            raise RuntimeError("no count for this request")
        return types.SimpleNamespace(input_tokens=self.counts[payload["model"]])


@pytest.fixture
def counting(monkeypatch):
    """A counter that answers from a table instead of the network."""

    def install(counts: dict, key: str | None = "fake-key-for-tests"):
        asked: list[dict] = []
        monkeypatch.setattr(token_count, "api_key", lambda: key)
        monkeypatch.setattr(
            token_count, "client", lambda api_key: _CountingClient(counts, asked)
        )
        token_count.reset()
        return asked

    return install


# --- what gets counted --------------------------------------------------------


def test_only_the_fields_that_change_the_count_are_sent():
    """max_tokens, temperature and stream are not input tokens.

    Sending them would be counting something the counter was not asked about;
    the endpoint rejects some of them outright.
    """
    payload = token_count.counted_payload(
        {
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
            "temperature": 1.0,
            "stream": True,
            "extra_body": {"temperature": 1.0},
        }
    )

    assert set(payload) == {"model", "messages"}


def test_empty_fields_are_left_out_rather_than_sent_as_empty():
    """An empty system prompt is not a system prompt."""
    payload = token_count.counted_payload({"model": "m", "messages": [], "system": ""})

    assert payload == {"model": "m"}


def test_the_signature_names_one_exact_request():
    one = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    same = {"messages": [{"role": "user", "content": "hi"}], "model": "m"}
    other = {"model": "m", "messages": [{"role": "user", "content": "hi!"}]}

    assert token_count.signature(one) == token_count.signature(same)
    assert token_count.signature(one) != token_count.signature(other)


def test_the_signature_covers_the_tools_that_cost_two_thousand_tokens():
    """The 70-character web search stub is the request's largest single cost.

    A signature that ignored tools would hand back the count for a request
    without them - and, measured against the live counter, that is a difference
    of about 2,200 tokens.
    """
    plain = {"model": "m", "messages": []}
    searching = {
        "model": "m",
        "messages": [],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
    }

    assert token_count.signature(plain) != token_count.signature(searching)


# --- asking the counter -------------------------------------------------------


def test_no_key_means_the_caller_is_told_rather_than_guessed_at(monkeypatch):
    monkeypatch.setattr(token_count, "api_key", lambda: None)

    with pytest.raises(token_count.TokenCountUnavailable, match="no Anthropic key"):
        token_count.count_now({"model": "m", "messages": []})


def test_the_counter_is_asked_about_exactly_the_request(monkeypatch, counting):
    asked = counting({"m": 2225})

    assert token_count.count_now({"model": "m", "messages": []}, timeout=2.5) == 2225
    assert asked == [{"model": "m", "messages": []}]


def test_a_provider_error_is_refused_not_swallowed(monkeypatch):
    def boom(api_key):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(token_count, "api_key", lambda: "k")
    monkeypatch.setattr(token_count, "client", boom)

    with pytest.raises(token_count.TokenCountUnavailable, match="connection reset"):
        token_count.count_now({"model": "m", "messages": []})


def test_a_counter_that_reports_nothing_is_unavailable(monkeypatch):
    class Empty:
        def __init__(self, **kwargs):
            self.messages = self

        def count_tokens(self, **kwargs):
            return object()

    monkeypatch.setattr(token_count, "api_key", lambda: "k")
    monkeypatch.setattr(token_count, "client", lambda api_key: Empty())

    with pytest.raises(token_count.TokenCountUnavailable):
        token_count.count_now({"model": "m", "messages": []})


# --- caching and the background worker ----------------------------------------


def test_an_unknown_request_has_no_count_to_report():
    assert token_count.lookup({"model": "m", "messages": []}) is None


def test_a_counted_request_is_remembered(counting):
    counting({"m": 42})

    assert token_count.kick({"model": "m", "messages": []}) is True
    assert wait_until(lambda: token_count.lookup({"model": "m", "messages": []}) == 42)


def test_the_same_request_is_counted_once(counting):
    asked = counting({"m": 42})
    request = {"model": "m", "messages": []}

    token_count.kick(request)
    token_count.kick(request)
    assert wait_until(lambda: token_count.lookup(request) == 42)
    # A third ask has nothing left to fetch.
    assert token_count.kick(request) is False
    assert len(asked) == 1


def test_a_counted_request_is_not_asked_for_again(counting):
    asked = counting({"m": 42})
    request = {"model": "m", "messages": []}
    token_count.kick(request)
    assert wait_until(lambda: token_count.lookup(request) == 42)

    assert token_count.kick(request) is False
    assert len(asked) == 1


def test_no_key_means_nothing_is_queued(monkeypatch, no_token_counting):
    monkeypatch.setattr(token_count, "api_key", lambda: None)

    assert token_count.kick({"model": "m", "messages": []}) is False
    assert token_count.lookup({"model": "m", "messages": []}) is None


def test_an_empty_request_is_not_worth_counting(counting):
    counting({"m": 42})

    assert token_count.kick({}) is False
    assert token_count.kick({"max_tokens": 1024}) is False


# --- failing without taking the page down -------------------------------------


def test_a_failed_count_is_reported_rather_than_raised(monkeypatch, no_token_counting):
    monkeypatch.setattr(token_count, "api_key", lambda: "k")

    def dead(payload, timeout=None):
        raise token_count.TokenCountUnavailable("the network is down")

    monkeypatch.setattr(token_count, "count_now", dead)
    request = {"model": "m", "messages": []}

    assert token_count.kick(request) is True
    assert wait_until(lambda: token_count.last_failure() == "the network is down")
    assert token_count.lookup(request) is None


def test_a_dead_network_does_not_cost_a_call_per_keystroke(
    monkeypatch, no_token_counting
):
    """Typing must not queue one doomed call per character."""
    monkeypatch.setattr(token_count, "api_key", lambda: "k")
    asked = []

    def dead(payload, timeout=None):
        asked.append(payload)
        raise token_count.TokenCountUnavailable("the network is down")

    monkeypatch.setattr(token_count, "count_now", dead)
    assert token_count.kick({"model": "m", "messages": [{"content": "a"}]}) is True
    assert wait_until(lambda: len(asked) == 1)

    for character in "bcdefghijk":
        assert token_count.kick({"model": "m", "messages": [{"content": character}]}) is False
    assert len(asked) == 1


def test_a_failure_reason_does_not_outlive_a_later_success(
    monkeypatch, no_token_counting
):
    monkeypatch.setattr(token_count, "api_key", lambda: "k")
    outcomes = [token_count.TokenCountUnavailable("down"), 42]

    def flaky(payload, timeout=None):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(token_count, "count_now", flaky)
    token_count.kick({"model": "m", "messages": [{"content": "one"}]})
    assert wait_until(lambda: token_count.last_failure() is not None)

    # Past the cooldown the counter is asked again, and the reason clears.
    monkeypatch.setattr(token_count, "FAILURE_COOLDOWN", timedelta(0))
    request = {"model": "m", "messages": [{"content": "two"}]}
    assert token_count.kick(request) is True
    assert wait_until(lambda: token_count.lookup(request) == 42)
    assert token_count.last_failure() is None


# --- the request the app actually builds --------------------------------------


def test_a_prepared_request_can_be_counted_as_it_stands(make_session):
    """The counted payload is the built request, not a model of it."""
    prepared = make_session(model="claude-sonnet-5").prepare("Hello")

    payload = token_count.counted_payload(prepared.kwargs)

    assert payload["model"] == prepared.kwargs["model"]
    assert payload["messages"] == prepared.kwargs["messages"]
    assert payload["tools"] == prepared.kwargs["tools"]
    # The tools really are the cost this module exists for.
    assert json.dumps(payload["tools"]) != "[]"
