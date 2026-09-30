"""The provider seam: who sends a model, and what the call really is.

These tests exist because the seam's whole job is to stop one provider's
assumptions leaking into another's. Anthropic was the only provider when this
project's rules were written, and every one of those rules - render the call
that happens, refuse rather than substitute, never branch on a model id - is
easy to keep with one provider and easy to lose with two.
"""

import ast

import llm
import pytest

from native_api_chat.capabilities import capabilities_for
from native_api_chat.codegen import render_kwargs
from native_api_chat.providers import PROVIDERS, provider_for
from native_api_chat.providers.anthropic import CREATE, STREAM
from native_api_chat.providers.base import Provider, Transport

llm_anthropic = pytest.importorskip("llm_anthropic")


def _model(model_id: str = "claude-sonnet-5"):
    return llm.get_model(capabilities_for(model_id).llm_id)


# --- who owns what ----------------------------------------------------------


def test_every_registered_provider_satisfies_the_protocol():
    for provider in PROVIDERS:
        assert isinstance(provider, Provider)
        assert provider.id and provider.label


def test_a_claude_model_is_claimed_by_anthropic():
    assert provider_for("claude-sonnet-5").id == "anthropic"


def test_an_unclaimed_model_is_refused_rather_than_guessed():
    """No provider is a catch-all.

    Handing an unrecognised id to whichever provider happens to be last would
    render a request to a service it was never going to reach, and the failure
    would surface somewhere far from the mistake.
    """
    with pytest.raises(ValueError, match="no provider claims"):
        provider_for("gpt-5.6-luna")


# --- the call that really happens -------------------------------------------


def test_the_transport_is_read_off_the_plugin_not_assumed():
    """llm-anthropic always opens a stream, and the seam has to find that out
    from the plugin rather than carry it as a constant of its own."""
    provider = provider_for("claude-sonnet-5")

    assert provider.transport(_model()) is STREAM
    assert STREAM.name == "messages.stream"
    assert STREAM.context is True


def test_the_renderer_has_no_default_transport():
    """A default here is a guess about someone else's runtime, and a guess is
    how a pane ends up printing a request nobody sends."""
    with pytest.raises(TypeError):
        render_kwargs({"model": "x"})


@pytest.mark.parametrize("transport", [STREAM, CREATE])
def test_both_call_forms_render_as_runnable_python(transport: Transport):
    code = render_kwargs({"model": "claude-sonnet-5", "max_tokens": 1}, transport)
    tree = ast.parse(code)

    assert transport.header.rstrip() in code
    assert f"{transport.call}(" in code
    # The last statement is the call, in whichever form the transport takes.
    assert isinstance(tree.body[-1], ast.With if transport.context else ast.Assign)


# --- assembly ---------------------------------------------------------------


def test_assemble_returns_the_whole_request_not_just_the_options():
    """The pane prints what assemble() returns, so anything the call site adds
    separately - model, messages - has to already be in it."""
    model = _model()
    prompt = llm.Prompt("Hello", model=model, options=model.Options(max_tokens=16))
    kwargs = provider_for("claude-sonnet-5").assemble(model, prompt, None, None)

    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["messages"][-1]["content"][-1]["text"] == "Hello"
    assert kwargs["max_tokens"] == 16


def test_assemble_adds_nothing_of_its_own():
    """A provider that improved the plugin's request would be the second copy
    of it that this whole design exists to prevent."""
    model = _model()
    prompt = llm.Prompt("Hello", model=model, options=model.Options(max_tokens=16))

    assert provider_for("claude-sonnet-5").assemble(
        model, prompt, None, None
    ) == model.build_kwargs(prompt, None)
