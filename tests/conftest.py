"""Shared offline fixtures for the conversation tests.

Only the Anthropic transport is faked. Everything above it - the real
``llm.Conversation``, the real ``llm-anthropic`` ``execute()`` loop and the
real ``model.build_kwargs()`` - runs untouched, so a prepared turn is the
request the provider would have received.
"""

import copy
import os
import shutil
import tempfile

import llm_anthropic
import pytest

from llm_sdk_view import capabilities as capabilities_module
from llm_sdk_view import model_api
from llm_sdk_view.chat import ChatOptions, ChatSession


@pytest.fixture(autouse=True)
def isolated_machine(monkeypatch):
    """Every test starts from a machine with no key and no Models API cache.

    The developer's own key and a warm cache would otherwise decide what these
    tests see: the same suite has to pass on a laptop that has never called the
    API and on one that called it five minutes ago. Tests that want a key or a
    cache install their own afterwards.
    """
    directory = tempfile.mkdtemp(prefix="llm-sdk-view-cache-")
    monkeypatch.setenv("LLM_SDK_VIEW_CACHE_DIR", directory)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # The Models API layer reads the key store, which on this machine may hold
    # a real key. Tests must see exactly the key they install themselves, so
    # the environment is the only source here.
    monkeypatch.setattr(model_api, "api_key", lambda: os.environ.get("ANTHROPIC_API_KEY"))
    capabilities_module.reset_model_data()
    try:
        yield
    finally:
        capabilities_module.reset_model_data()
        shutil.rmtree(directory, ignore_errors=True)


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
    """The Message the API finished with.

    The default is a plain text answer. Tests that need proof a field is read
    off the Message rather than invented set ``SCENARIO`` to a richer one -
    citations, thinking, server tool results and cache counters only exist
    there, which is exactly why a view rebuilt from chat text would lose them.
    """

    SCENARIO: dict | None = None

    def model_dump(self):
        if _FinalMessage.SCENARIO is not None:
            return _FinalMessage.SCENARIO
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


# Turn storage writes to llm's own SQLite. Point every test at its own
# database so a run can neither read nor damage the history the user built up
# with `llm` itself (and so CI starts from nothing every time).
@pytest.fixture(autouse=True)
def isolated_history(monkeypatch, tmp_path_factory):
    root = tempfile.mkdtemp(prefix="llm-sdk-view-logs-")
    monkeypatch.setenv("LLM_SDK_VIEW_LOGS_DB", os.path.join(root, "logs.db"))
    _FinalMessage.SCENARIO = None
    try:
        yield root
    finally:
        _FinalMessage.SCENARIO = None
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def response_scenario():
    """Replace the finished Message the fake transport hands back.

    Used to prove the Response pane reads citations, thinking, server tool
    results and cache counts off the Message instead of deducing them from
    the text that streamed past.
    """

    def install(message: dict):
        # The plugin moves usage off the Message as it consumes the stream, so
        # each install gets its own copy to mutate.
        _FinalMessage.SCENARIO = copy.deepcopy(message)

    return install


@pytest.fixture
def key(monkeypatch):
    """A key for the send path only. Nothing reaches the provider in tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")


@pytest.fixture
def fake_provider(monkeypatch, transports):
    """Replace the Anthropic transport, keep everything else real."""
    sent = []

    def factory(**kwargs):
        return _FakeClient(sent, transports, **kwargs)

    monkeypatch.setattr(llm_anthropic, "Anthropic", factory)
    return sent


@pytest.fixture
def isolated_api_entry():
    """Install one Models API entry as the whole catalog and re-read it.

    The entry is written in the exact shape the Models API returns, because
    that shape is what the capability reader has to survive.
    """

    def install(model_id, max_input=200_000, max_output=64_000, effort=None, thinking=None):
        model_api.save_cache(
            [
                {
                    "id": model_id,
                    "display_name": model_id,
                    "created_at": "2026-09-01T00:00:00Z",
                    "type": "model",
                    "max_input_tokens": max_input,
                    "max_tokens": max_output,
                    "capabilities": {"effort": effort, "thinking": thinking},
                }
            ]
        )
        capabilities_module.reset_model_data()
        return model_id

    return install


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
    """A conversation on the adaptive-thinking model.

    The form default is Claude Haiku 4.5; the conversation tests exercise the
    richer path so that thinking, effort and response_inclusion are all present
    in the request they inspect.
    """
    return make_session(model="claude-sonnet-5")


@pytest.fixture
def capabilities():
    """The capability matrix entry for a model, read off the installed plugin."""

    def read(model_id: str):
        from llm_sdk_view.capabilities import capabilities_for

        return capabilities_for(model_id)

    return read
