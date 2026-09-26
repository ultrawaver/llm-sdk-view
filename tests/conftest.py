"""Shared offline fixtures for the conversation tests.

Only the Anthropic transport is faked. Everything above it - the real
``llm.Conversation``, the real ``llm-anthropic`` ``execute()`` loop and the
real ``model.build_kwargs()`` - runs untouched, so a prepared turn is the
request the provider would have received.
"""

import llm_anthropic
import pytest

from llm_sdk_view.chat import ChatOptions, ChatSession


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
    """The two SDK entry points a provider might use.

    Which one the plugin picks is the thing under test, so both are recorded.
    """

    def __init__(self, recorder, transports):
        self.recorder = recorder
        self.transports = transports

    def stream(self, **kwargs):
        self.transports.append("stream")
        self.recorder.append(kwargs)
        return _FakeStream(kwargs, self.recorder)

    def create(self, **kwargs):
        self.transports.append("create")
        self.recorder.append(kwargs)
        return _FinalMessage()


class _FakeClient:
    def __init__(self, recorder, transports, **kwargs):
        self.recorder = recorder
        self.messages = _FakeMessages(recorder, transports)


@pytest.fixture
def transports() -> list:
    """The SDK methods the provider actually opened, in order."""
    return []


@pytest.fixture
def fake_provider(monkeypatch, transports):
    """Replace the Anthropic transport, keep everything else real."""
    sent = []

    def factory(**kwargs):
        return _FakeClient(sent, transports, **kwargs)

    monkeypatch.setattr(llm_anthropic, "Anthropic", factory)
    return sent


@pytest.fixture
def make_session(monkeypatch, fake_provider):
    """Build a ChatSession that can run without a key and without network.

    The key is stubbed but the transport is faked, so no request can leave the
    machine either way.
    """

    def factory(**kwargs):
        chat = ChatSession(ChatOptions(**kwargs))
        monkeypatch.setattr(chat.model, "get_key", lambda key=None: "fake-key-for-tests")
        return chat

    return factory


@pytest.fixture
def session(make_session):
    return make_session()


@pytest.fixture
def capabilities():
    """The capability matrix entry for a model, read off the installed plugin."""

    def read(model_id: str):
        from llm_sdk_view.capabilities import capabilities_for

        return capabilities_for(model_id)

    return read
