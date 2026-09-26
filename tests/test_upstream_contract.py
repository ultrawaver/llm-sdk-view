"""Contract tests against the *installed* llm and llm-anthropic.

These tests never make a network call. They exist because the four required
Anthropic behaviours are properties of `llm-anthropic`'s source, not of this
repository, and a scaffold claim about them is worthless until it is checked
against the version that is actually installed.
"""

import importlib.metadata
import inspect

import pytest

llm = pytest.importorskip("llm")
llm_anthropic = pytest.importorskip("llm_anthropic")

from llm_anthropic import WebSearch  # noqa: E402

from llm_sdk_view.codegen import anthropic_kwargs  # noqa: E402
from llm_sdk_view.models import AnthropicTurn, Message  # noqa: E402

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


def test_web_search_does_not_expose_response_inclusion_yet():
    """The gap that requires an upstream PR.

    When llm-anthropic gains `response_inclusion`, this test fails and the
    upstream patch is no longer needed - delete it rather than relaxing it.
    """
    assert "response_inclusion" not in inspect.signature(WebSearch.__init__).parameters


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


def test_codegen_matches_upstream_request_shape():
    """The generated request must agree with what llm-anthropic really sends."""
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
    ours = anthropic_kwargs(
        AnthropicTurn(messages=(Message(role="user", content="Hello"),), max_searches=5)
    )

    assert ours["messages"] == upstream["messages"]
    # Everything except the field we still need upstream to accept.
    assert {k: v for k, v in ours["tools"][0].items() if k != "response_inclusion"} == (
        upstream["tools"][0]
    )
