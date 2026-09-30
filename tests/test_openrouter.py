"""OpenRouter, checked against what llm-openrouter really sends.

Every test here is offline and keyless. The plugin's own ``execute()`` runs
untouched; only the OpenAI client underneath it is replaced, so the request
the provider assembles and the request that reached the SDK are two
independently produced objects that the tests then compare. Deriving one from
the other would prove nothing.
"""

import ast

import llm
import pytest

from native_api_chat.codegen import render_kwargs
from native_api_chat.providers import provider_for
from native_api_chat.providers.openrouter import (
    API_BASE,
    CHAT_COMPLETIONS,
    HEADERS,
    RESPONSES,
)

llm_openrouter = pytest.importorskip("llm_openrouter")

MODEL_ID = "openrouter/anthropic/claude-sonnet-5"
PROVIDER = provider_for(MODEL_ID)


def _turn(model, options=None, tools=()):
    """One prepared turn, with nothing sent."""
    conversation = llm.Conversation(model=model)
    response = conversation.prompt(
        "hello",
        system="be brief",
        options=options or {"max_tokens": 1024},
        tools=list(tools),
        stream=True,
    )
    return conversation, response


# --- ownership --------------------------------------------------------------


def test_openrouter_claims_its_own_models_and_nothing_else():
    assert PROVIDER.id == "openrouter"
    assert PROVIDER.owns(MODEL_ID)
    assert not PROVIDER.owns("claude-sonnet-5")


def test_the_same_claude_through_openrouter_is_a_different_provider():
    """It is a different entry with a different call, not a cheaper Claude.

    The day these two resolve to one provider, the pane starts describing an
    Anthropic request for a turn that went to OpenRouter.
    """
    assert provider_for("claude-sonnet-5").id == "anthropic"
    assert provider_for(MODEL_ID).id == "openrouter"


# --- the request is the one that gets sent ----------------------------------


def test_the_responses_request_is_the_one_the_plugin_sends(
    openrouter_model, fake_openrouter
):
    conversation, response = _turn(
        openrouter_model,
        {"max_tokens": 1024, "reasoning_effort": "high"},
        [llm_openrouter.WebSearch(max_uses=2)],
    )
    ours = PROVIDER.assemble(
        openrouter_model, response.prompt, conversation, response.prompt.options
    )
    list(openrouter_model.execute(response.prompt, True, response, conversation, key="k"))
    method, sent = fake_openrouter[-1]

    assert method == "responses.create"
    assert ours == sent


def test_the_chat_completions_request_is_the_one_the_plugin_sends(
    openrouter_model, fake_openrouter
):
    """The delegate builds it, so the provider has to ask the delegate too."""
    conversation, response = _turn(
        openrouter_model, {"max_tokens": 1024, "chat_completions": True}
    )
    ours = PROVIDER.assemble(
        openrouter_model, response.prompt, conversation, response.prompt.options
    )
    list(openrouter_model.execute(response.prompt, True, response, conversation, key="k"))
    method, sent = fake_openrouter[-1]

    assert method == "chat.completions.create"
    assert ours == sent


def test_the_named_transport_is_the_method_that_was_called(
    openrouter_model, fake_openrouter
):
    for options, expected in (
        ({"max_tokens": 16}, "responses.create"),
        ({"max_tokens": 16, "chat_completions": True}, "chat.completions.create"),
    ):
        conversation, response = _turn(openrouter_model, options)
        named = PROVIDER.transport(openrouter_model, response.prompt.options).name
        list(
            openrouter_model.execute(
                response.prompt, True, response, conversation, key="k"
            )
        )
        method, _ = fake_openrouter[-1]

        assert named == method == expected


# --- one model, two calls ---------------------------------------------------


def test_the_transport_is_decided_per_request_not_per_model(openrouter_model):
    """llm-openrouter registers Responses models only and delegates inside
    execute(), so the same id can produce either call."""
    _, plain = _turn(openrouter_model, {"max_tokens": 16})
    _, delegated = _turn(openrouter_model, {"max_tokens": 16, "chat_completions": True})

    assert PROVIDER.transport(openrouter_model, plain.prompt.options) is RESPONSES
    assert (
        PROVIDER.transport(openrouter_model, delegated.prompt.options)
        is CHAT_COMPLETIONS
    )


def test_the_two_transports_carry_different_field_names(openrouter_model):
    """Not cosmetic: the same turn is a different request on each path, which
    is why one rendered shape could never serve both."""
    conversation, responses_turn = _turn(openrouter_model, {"max_tokens": 16})
    on_responses = PROVIDER.assemble(
        openrouter_model, responses_turn.prompt, conversation, responses_turn.prompt.options
    )
    conversation, chat_turn = _turn(
        openrouter_model, {"max_tokens": 16, "chat_completions": True}
    )
    on_chat = PROVIDER.assemble(
        openrouter_model, chat_turn.prompt, conversation, chat_turn.prompt.options
    )

    assert on_responses["max_output_tokens"] == 16
    assert on_responses["instructions"] == "be brief"
    assert "input" in on_responses and "messages" not in on_responses

    assert on_chat["max_tokens"] == 16
    assert on_chat["messages"][0] == {"role": "system", "content": "be brief"}
    assert "messages" in on_chat and "input" not in on_chat


def test_server_tools_are_refused_on_the_chat_completions_path(
    openrouter_model, fake_openrouter
):
    """A runtime limit, reported rather than worked around: switching
    transport takes web search away, and the plugin says so itself."""
    conversation, response = _turn(
        openrouter_model,
        {"max_tokens": 16, "chat_completions": True},
        [llm_openrouter.WebSearch(max_uses=1)],
    )

    with pytest.raises(ValueError, match="Server-side tools cannot be used"):
        list(
            openrouter_model.execute(
                response.prompt, True, response, conversation, key="k"
            )
        )


# --- what the pane prints ---------------------------------------------------


@pytest.mark.parametrize("transport", [RESPONSES, CHAT_COMPLETIONS])
def test_the_rendered_call_is_runnable_python(openrouter_model, transport):
    conversation, response = _turn(openrouter_model, {"max_tokens": 16})
    kwargs = PROVIDER.assemble(
        openrouter_model, response.prompt, conversation, response.prompt.options
    )

    ast.parse(render_kwargs(kwargs, transport))


def test_the_rendered_client_is_the_one_the_plugin_builds(
    openrouter_model, fake_openrouter
):
    """A pane printing a bare openai.OpenAI() would describe a request to
    OpenAI, so the base URL and both headers have to be in the code."""
    conversation, response = _turn(openrouter_model, {"max_tokens": 16})
    list(openrouter_model.execute(response.prompt, True, response, conversation, key="k"))
    kwargs = PROVIDER.assemble(
        openrouter_model, response.prompt, conversation, response.prompt.options
    )
    code = render_kwargs(kwargs, RESPONSES)

    assert f'base_url="{API_BASE}"' in code
    for name, value in HEADERS.items():
        assert f'"{name}": "{value}"' in code
    assert openrouter_model.headers == HEADERS
    assert openrouter_model.api_base == API_BASE


def test_the_rendered_code_carries_no_secret(openrouter_model):
    conversation, response = _turn(openrouter_model, {"max_tokens": 16})
    kwargs = PROVIDER.assemble(
        openrouter_model, response.prompt, conversation, response.prompt.options
    )
    code = render_kwargs(kwargs, RESPONSES)

    assert 'os.environ["OPENROUTER_KEY"]' in code
    assert "sk-or-" not in code


def test_openrouter_native_options_reach_the_request(openrouter_model):
    """Provider routing and the reasoning controls are OpenRouter's own API,
    and are what makes this a provider rather than a compatibility layer."""
    conversation, response = _turn(
        openrouter_model,
        {
            "max_tokens": 16,
            "reasoning_effort": "high",
            "provider": {"quantizations": ["fp8"]},
        },
    )
    kwargs = PROVIDER.assemble(
        openrouter_model, response.prompt, conversation, response.prompt.options
    )

    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["extra_body"]["provider"] == {"quantizations": ["fp8"]}
