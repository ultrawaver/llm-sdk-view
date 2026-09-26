"""Offline proof that the conversation path works end to end.

Every test here drives the real LLM Python API and the real llm-anthropic
execute() loop. Only the Anthropic transport is faked, so no request leaves
the machine and no key is needed.
"""

import inspect

import llm
import llm_anthropic
import pytest

from llm_sdk_view.chat import ChatOptions, ChatSession, MissingKeyError


class _Block:
    def __init__(self, type="text", **extra):
        self.type = type
        self.__dict__.update(extra)


class _Delta:
    def __init__(self, type, text=None):
        self.type = type
        self.text = text


class _Chunk:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FinalMessage:
    def model_dump(self):
        return {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "Hello world"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }


class _FakeStream:
    def __init__(self, kwargs, recorder):
        self.kwargs = kwargs
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield _Chunk(type="content_block_start", index=0, content_block=_Block())
        yield _Chunk(
            type="content_block_delta", index=0, delta=_Delta("text_delta", "Hello")
        )
        yield _Chunk(
            type="content_block_delta", index=0, delta=_Delta("text_delta", " world")
        )
        yield _Chunk(type="content_block_stop", index=0)

    def get_final_message(self):
        return _FinalMessage()


class _FakeMessages:
    def __init__(self, recorder):
        self.recorder = recorder

    def stream(self, **kwargs):
        self.recorder.append(kwargs)
        return _FakeStream(kwargs, self.recorder)


class _FakeClient:
    def __init__(self, recorder, **kwargs):
        self.recorder = recorder
        self.messages = _FakeMessages(recorder)


@pytest.fixture
def fake_provider(monkeypatch):
    """Replace the Anthropic transport, keep everything else real."""
    sent = []

    def factory(**kwargs):
        return _FakeClient(sent, **kwargs)

    monkeypatch.setattr(llm_anthropic, "Anthropic", factory)
    return sent


@pytest.fixture
def session(monkeypatch, fake_provider):
    chat = ChatSession(ChatOptions())
    # A working key, but a fake transport: no request can reach Anthropic.
    monkeypatch.setattr(chat.model, "get_key", lambda key=None: "fake-key-for-tests")
    return chat


def _response_inclusion_supported() -> bool:
    from llm_anthropic import WebSearch

    return "response_inclusion" in inspect.signature(WebSearch.__init__).parameters


def test_first_turn_reaches_the_left_pane(session):
    events = list(session.stream_turn("What changed this week?"))

    assert events[0]["type"] == "prepared"
    assert [e["text"] for e in events if e["type"] == "text"] == ["Hello", " world"]
    assert events[-1] == {"type": "done", "text": "Hello world"}
    assert "client.messages.create(" in events[0]["code"]


def test_second_turn_carries_the_full_history(session, fake_provider):
    session.run_turn("First question")
    prepared = session.prepare("Second question")

    roles = [message["role"] for message in prepared.kwargs["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert prepared.kwargs["messages"][-1]["content"][-1]["text"] == "Second question"
    assert prepared.kwargs["messages"][1]["content"][-1]["text"] == "Hello world"
    assert len(fake_provider) == 1, "prepare() must not send anything"


def test_right_pane_matches_the_request_that_was_sent(session, fake_provider):
    result = session.run_turn("Does the right pane match?")

    assert fake_provider, "the turn should have gone through the provider"
    assert fake_provider[0] == result["kwargs"], (
        "the SDK code must be rendered from the parameters that were sent"
    )
    # The rendered code carries the same values, not a paraphrase of them.
    for value in ('"web_search_20260318"', 'model="claude-sonnet-5"', "messages="):
        assert value in result["code"]


def test_prepared_request_keeps_the_four_anthropic_behaviours(session):
    prepared = session.prepare("Check the four behaviours")

    tool = next(t for t in prepared.kwargs["tools"] if t["name"] == "web_search")
    assert tool["type"] == "web_search_20260318"
    # Dynamic filtering stays on: nothing pins the tool to direct calls.
    assert "allowed_callers" not in tool
    assert prepared.kwargs["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    if _response_inclusion_supported():
        assert tool["response_inclusion"] == "excluded"
        assert '"response_inclusion": "excluded"' in prepared.code
    else:
        pytest.skip(
            "installed llm-anthropic has no response_inclusion; "
            "see docs/upstream-contributions.md"
        )


def test_missing_key_sends_nothing(monkeypatch, fake_provider):
    chat = ChatSession(ChatOptions())
    monkeypatch.setattr(
        chat.model, "get_key", lambda key=None: (_ for _ in ()).throw(
            llm.NeedsKeyException("No key found")
        )
    )
    with pytest.raises(MissingKeyError):
        chat.run_turn("Anything")

    assert fake_provider == [], "no request may be sent without a key"


def test_no_request_is_made_by_preparing(monkeypatch, fake_provider):
    chat = ChatSession(ChatOptions())
    monkeypatch.setattr(chat.model, "get_key", lambda key=None: "fake-key-for-tests")
    chat.prepare("Preview only")
    assert fake_provider == []
