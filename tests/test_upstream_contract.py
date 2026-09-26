"""Contract tests against the *installed* llm and llm-anthropic.

These tests never make a network call. They exist because the four required
Anthropic behaviours are properties of `llm-anthropic`'s source, not of this
repository, and a scaffold claim about them is worthless until it is checked
against the version that is actually installed.
"""

import ast
import importlib.metadata
import inspect

import pytest

llm = pytest.importorskip("llm")
llm_anthropic = pytest.importorskip("llm_anthropic")

from llm_anthropic import WebSearch  # noqa: E402

from llm_sdk_view.codegen import render_kwargs  # noqa: E402

TARGET_MODEL = "claude-sonnet-5"


def _version_tuple(name: str) -> tuple[int, ...]:
    raw = importlib.metadata.version(name).split(".")
    return tuple(int(part) for part in raw if part.isdigit())


def test_installed_versions_satisfy_pyproject():
    """pyproject requires llm>=0.36 and llm-anthropic>=0.29."""
    assert _version_tuple("llm") >= (0, 36)
    assert _version_tuple("llm-anthropic") >= (0, 29)


def test_target_model_is_registered_and_adaptive():
    model = llm.get_model(TARGET_MODEL)
    assert model.supports_web_search
    # `supports_adaptive_thinking` is the flag llm-anthropic keys the modern
    # web search tool version off, so it is load-bearing for this project.
    assert model.supports_adaptive_thinking


def test_web_search_uses_20260318_without_forcing_direct_callers():
    model = llm.get_model(TARGET_MODEL)
    spec = WebSearch(max_uses=5).tool_spec(model)

    assert spec["type"] == "web_search_20260318"
    # Dynamic content filtering is the default for web_search_20260318. Forcing
    # allowed_callers=["direct"] would disable it, so it must never be sent.
    assert "allowed_callers" not in spec


def test_web_search_response_inclusion_state():
    """What the installed llm-anthropic can actually do with
    `response_inclusion`.

    The released plugin cannot express it at all; the contribution branch can.
    Either way this project must never show a value the installed plugin has
    no way to send, so the assertion adapts to what is installed rather than
    asserting a fixed answer.
    """
    if "response_inclusion" not in inspect.signature(WebSearch.__init__).parameters:
        pytest.skip(
            "installed llm-anthropic has no response_inclusion - the gap "
            "described in docs/upstream-contributions.md"
        )
    model = llm.get_model(TARGET_MODEL)
    spec = WebSearch(max_uses=5, response_inclusion="excluded").tool_spec(model)
    assert spec["type"] == "web_search_20260318"
    assert spec["response_inclusion"] == "excluded"
    # Only web_search_20260318 accepts it, so an older model must fail loudly
    # instead of silently dropping the value.
    older = llm.get_model("claude-opus-4.1")
    with pytest.raises(ValueError):
        WebSearch(response_inclusion="excluded").tool_spec(older)


def test_upstream_prompt_caching_shape():
    model = llm.get_model(TARGET_MODEL)
    prompt = llm.Prompt(
        "Hello",
        model=model,
        options=model.Options(cache=True),
        tools=[WebSearch(max_uses=5)],
    )
    messages = model.build_kwargs(prompt, None)["messages"]

    assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_renderer_reproduces_the_upstream_request():
    """The right pane renders build_kwargs() output, so it cannot drift.

    Every top-level key the plugin produced has to appear in the rendered code
    with the same value; there is no second request model to compare against.
    """
    model = llm.get_model(TARGET_MODEL)
    upstream = model.build_kwargs(
        llm.Prompt(
            "Hello",
            model=model,
            options=model.Options(cache=True),
            tools=[WebSearch(max_uses=5)],
        ),
        None,
    )
    code = render_kwargs(upstream)
    call = ast.parse(code).body[-1].value
    rendered = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}

    # The rendered call has to evaluate back to exactly what was built.
    assert rendered == upstream
    assert '"type": "web_search_20260318"' in code
