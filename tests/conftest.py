"""Shared offline fixtures for the conversation tests.

Only the provider transports are faked. Everything above them - the real
``llm.Conversation``, the real plugin ``execute()`` loops and the real request
builders - runs untouched, so a prepared turn is the request the provider
would have received. That is what makes "the pane shows the call that really
happens" something the suite can check rather than something it assumes.

No fixture here reads a key, and none can reach the network.
"""

import copy
import os
import shutil
import socket
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import llm
import llm_anthropic
import pytest
import uvicorn

import native_api_chat
from native_api_chat import capabilities as capabilities_module
from native_api_chat import model_api, rates_page
from native_api_chat.app import create_app
from native_api_chat.chat import ChatOptions, ChatSession

# The rates the pricing page currently publishes, for the models the suite
# prices. Tests read these instead of the page, because a cost assertion that
# only held while Anthropic's site was reachable would fail for the wrong
# reason.
KNOWN_RATES = {
    "claude-fable-5-1": {
        "input": 10.0,
        "output": 50.0,
        "cache_write_5m": 12.50,
        "cache_write_1h": 20.0,
        "cache_read": 0.25,
    },
    "claude-opus-5-5": {
        "input": 4.0,
        "output": 20.0,
        "cache_write_5m": 5.0,
        "cache_write_1h": 8.0,
        "cache_read": 0.20,
    },
    "claude-sonnet-5": {
        "input": 2.0,
        "output": 10.0,
        "cache_write_5m": 2.50,
        "cache_write_1h": 4.0,
        "cache_read": 0.20,
    },
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_write_5m": 1.25,
        "cache_write_1h": 2.0,
        "cache_read": 0.10,
    },
}


@pytest.fixture(autouse=True)
def known_rates(monkeypatch):
    """Unit prices without a network.

    Prices come from the pricing page at runtime, so without this every cost
    assertion would depend on a live fetch. The page-shaped cache below is
    installed as if it had just been fetched; tests that exercise the fetching
    itself replace it afterwards.
    """

    def cached():
        return {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "rates": copy.deepcopy(KNOWN_RATES),
        }

    real = {
        "load_cache": rates_page.load_cache,
        "kick_refresh": rates_page.kick_refresh,
    }

    def use_real_cache():
        """Give a test the real disk cache back, inside the isolated dir."""
        monkeypatch.setattr(rates_page, "load_cache", real["load_cache"])
        monkeypatch.setattr(rates_page, "kick_refresh", real["kick_refresh"])

    monkeypatch.setattr(rates_page, "load_cache", cached)
    monkeypatch.setattr(rates_page, "kick_refresh", lambda *args, **kwargs: False)
    real["use_real_cache"] = use_real_cache
    return real


@pytest.fixture
def static_page() -> str:
    """Everything the page is built from, concatenated.

    Assertions about what the page shows or does must survive the assets
    being split across index.html / app.css / app.js - the behaviour they
    protect lives in the page, not in one file. Vendored third-party code
    (static/vendor/) is excluded on purpose.
    """
    root = Path(native_api_chat.__file__).parent / "static"
    return "\n".join(
        path.read_text("utf-8") for path in sorted(root.iterdir()) if path.is_file()
    )


@pytest.fixture(autouse=True)
def no_token_counting(monkeypatch):
    """The API's token counter is free, but the suite still never reaches it.

    The counter itself is replaced, not the code that calls it: ``kick``,
    ``lookup`` and the cache are the things under test, and they behave exactly
    as they do in the app once the client underneath them is a refusal. A test
    that wants counts replaces the client with a table instead.
    """
    from native_api_chat import token_count

    def refuse(api_key):
        raise RuntimeError("tests never call the API")

    token_count.reset()
    monkeypatch.setattr(token_count, "client", refuse)
    try:
        yield token_count
    finally:
        token_count.reset()


@pytest.fixture(autouse=True)
def isolated_machine(monkeypatch):
    """Every test starts from a machine with no key and no Models API cache.

    The developer's own key and a warm cache would otherwise decide what these
    tests see: the same suite has to pass on a laptop that has never called the
    API and on one that called it five minutes ago. Tests that want a key or a
    cache install their own afterwards.
    """
    directory = tempfile.mkdtemp(prefix="native-api-chat-cache-")
    monkeypatch.setenv("NATIVE_API_CHAT_CACHE_DIR", directory)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # The Models API layer and the send path both read the key store, which on
    # this machine may hold a real key. A test that silently found one would
    # pass here and fail on a CI machine that has none, so the environment is
    # the only source either of them sees.
    monkeypatch.setattr(model_api, "api_key", lambda: os.environ.get("ANTHROPIC_API_KEY"))
    def get_key(explicit_key=None, key_alias=None, env_var=None, **kwargs):
        """The environment and the caller: never the developer's key store.

        ``llm-anthropic`` asks for its key by alias with no environment
        fallback, so a test that found the key stored on this machine would
        pass here and fail on CI. Whatever the key fixture installs is what
        every send path sees.

        An explicit key still wins, because upstream's own ``get_key`` prefers
        it and a stand-in that quietly dropped it would make this suite
        disagree with the runtime it exists to check. It comes from the test,
        not from the machine, so it cannot be the developer's.
        """
        return explicit_key or os.environ.get(env_var or "ANTHROPIC_API_KEY")

    monkeypatch.setattr(llm, "get_key", get_key)
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
    def __init__(self, type, text=None, thinking=None):
        self.type = type
        self.text = text
        self.thinking = thinking


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
    """The streamed half of the fake transport.

    The default turn is two text chunks. A SCENARIO whose content holds a
    thinking block first streams that block as thinking deltas, exactly the
    order the API uses (reasoning before the answer), so tests can prove the
    app forwards reasoning as its own event type.
    """

    def __init__(self, kwargs, recorder):
        self.kwargs = kwargs
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        index = 0
        scenario = _FinalMessage.SCENARIO or {}
        for block in scenario.get("content", []):
            if block.get("type") != "thinking":
                continue
            yield _Chunk(
                type="content_block_start", index=index,
                content_block=_Block(type="thinking"),
            )
            yield _Chunk(
                type="content_block_delta", index=index,
                delta=_Delta("thinking_delta", thinking=block.get("thinking", "")),
            )
            yield _Chunk(type="content_block_stop", index=index)
            index += 1
        yield _Chunk(type="content_block_start", index=index, content_block=_Block())
        yield _Chunk(
            type="content_block_delta", index=index, delta=_Delta("text_delta", "Hello")
        )
        yield _Chunk(
            type="content_block_delta", index=index, delta=_Delta("text_delta", " world")
        )
        yield _Chunk(type="content_block_stop", index=index)

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
    root = tempfile.mkdtemp(prefix="native-api-chat-logs-")
    monkeypatch.setenv("NATIVE_API_CHAT_LOGS_DB", os.path.join(root, "logs.db"))
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


# --- OpenRouter -------------------------------------------------------------
#
# llm-openrouter registers no models without a key, so a test builds the model
# by hand. The kwargs below are the ones `register_models()` really passes,
# including the two headers that go on the wire - a fake that dropped them
# would let the rendered client drift from the real one unnoticed.
OPENROUTER_MODEL_KWARGS = dict(
    model_id="openrouter/anthropic/claude-sonnet-5",
    model_name="anthropic/claude-sonnet-5",
    vision=True,
    reasoning=True,
    verbosity=False,
    supports_schema=True,
    supports_tools=True,
    api_base="https://openrouter.ai/api/v1",
    headers={
        "HTTP-Referer": "https://llm.datasette.io/",
        "X-OpenRouter-Title": "LLM",
    },
)


class _FakeOpenAIStream:
    """An empty stream: these tests are about the request, not the reply."""

    def __iter__(self):
        return iter(())


class _FakeOpenAICalls:
    def __init__(self, name, sent):
        self.name = name
        self.sent = sent

    def create(self, **kwargs):
        self.sent.append((self.name, kwargs))
        return _FakeOpenAIStream()


class _FakeOpenAIClient:
    """Records which of the two OpenRouter calls was made, and with what."""

    def __init__(self, sent, **init):
        self.init = init
        self.responses = _FakeOpenAICalls("responses.create", sent)
        self.chat = type(
            "_Chat", (), {"completions": _FakeOpenAICalls("chat.completions.create", sent)}
        )()


@pytest.fixture
def openrouter_model():
    """An OpenRouter model built the way the plugin builds one."""
    llm_openrouter = pytest.importorskip("llm_openrouter")
    return llm_openrouter.OpenRouterResponses(**OPENROUTER_MODEL_KWARGS)


@pytest.fixture
def fake_openrouter(monkeypatch):
    """Replace the OpenAI client, keep llm-openrouter's own execute() real.

    Returns the list of ``(method, kwargs)`` the plugin actually called, which
    is what lets a test compare the rendered request against the sent one
    without either side being derived from the other.
    """
    from llm.default_plugins import openai_models

    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        openai_models.openai,
        "OpenAI",
        lambda **init: _FakeOpenAIClient(sent, **init),
    )
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
        from native_api_chat.capabilities import capabilities_for

        return capabilities_for(model_id)

    return read


# --- the browser harness -------------------------------------------------------
# Shared by every file that measures the real page (test_topbar_in_a_browser,
# test_chat_reading_in_a_browser). One definition, because a session-scoped
# browser must be one browser: the same fixture re-defined per module is a
# second instantiation, and the second one lands inside an asyncio loop.


def _no_browser(reason: str):
    """Skip, unless the caller said a skip is not an acceptable answer.

    CI sets ``NATIVE_API_CHAT_REQUIRE_BROWSER``: a check that quietly skips is
    a check that is not running, and a suite that stays green while these
    never execute is precisely how the defect they exist for shipped.
    """
    if os.environ.get("NATIVE_API_CHAT_REQUIRE_BROWSER"):
        pytest.fail(f"NATIVE_API_CHAT_REQUIRE_BROWSER is set but {reason}")
    pytest.skip(reason)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _launch(playwright):
    """Whatever chromium this machine has: the bundled one, or Chrome."""
    problems = []
    for options in ({}, {"channel": "chrome"}):
        try:
            return playwright.chromium.launch(headless=True, **options)
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not handled
            name = options.get("channel", "bundled chromium")
            problems.append(f"{name}: {str(exc).splitlines()[0]}")
    _no_browser(
        "there is no browser to drive (" + "; ".join(problems) + "). Install "
        "one with `playwright install chromium`, or install Google Chrome."
    )


@pytest.fixture(scope="session")
def browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _no_browser(
            "playwright is not installed. The browser checks need the "
            "optional extra: pip install -e '.[test,browser]'"
        )
    with sync_playwright() as playwright:
        launched = _launch(playwright)
        try:
            yield launched
        finally:
            launched.close()


@pytest.fixture
def live_app():
    """The real application on a real port.

    In-process and function-scoped, so the suite's isolation fixtures still
    apply: no API key, a throwaway cache directory, a throwaway database and
    a token counter that refuses. The page falls back to the versioned model
    profile, which is what a machine with no key sees anyway.
    """
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the application did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)


@pytest.fixture
def page(browser, live_app):
    context = browser.new_context(viewport={"width": 1500, "height": 820})
    opened = context.new_page()
    opened.goto(live_app)
    # The pills render once the form and the capabilities have arrived.
    opened.wait_for_selector('#settingsPills [data-pill="model"]')
    try:
        yield opened
    finally:
        context.close()
