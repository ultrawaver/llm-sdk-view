import ast

import pytest

from llm_sdk_view.codegen import anthropic_kwargs, render_anthropic_python
from llm_sdk_view.models import AnthropicTurn, Message


def test_anthropic_request_has_four_required_capabilities():
    turn = AnthropicTurn(messages=(Message(role="user", content="Hello"),))
    kwargs = anthropic_kwargs(turn)

    # Prompt caching lives on a content block, never at the top level of
    # messages.create(), which is not a valid Anthropic parameter.
    assert "cache_control" not in kwargs
    assert kwargs["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tools"] == [
        {
            "type": "web_search_20260318",
            "name": "web_search",
            "max_uses": 5,
            "response_inclusion": "excluded",
        }
    ]
    assert "allowed_callers" not in kwargs["tools"][0]


def test_cache_control_lands_on_the_final_message_only():
    turn = AnthropicTurn(
        messages=(
            Message(role="user", content="First"),
            Message(role="assistant", content="Second"),
            Message(role="user", content="Third"),
        )
    )
    messages = anthropic_kwargs(turn)["messages"]

    assert "cache_control" not in messages[0]["content"][-1]
    assert "cache_control" not in messages[1]["content"][-1]
    assert messages[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_features_can_be_disabled():
    turn = AnthropicTurn(
        messages=(Message(role="user", content="Hello"),),
        web_search=False,
        prompt_cache=False,
    )
    kwargs = anthropic_kwargs(turn)
    assert "tools" not in kwargs
    assert all("cache_control" not in block for block in kwargs["messages"][0]["content"])


def test_generated_code_is_sdk_code_and_contains_no_secret():
    turn = AnthropicTurn(
        messages=(Message(role="user", content="Compare A and B"),),
        response_inclusion="excluded",
    )
    code = render_anthropic_python(turn)

    assert "client.messages.create(" in code
    assert '"web_search_20260318"' in code
    assert '"response_inclusion": "excluded"' in code
    assert "api_key=" not in code
    assert "ANTHROPIC_API_KEY" not in code
    # Must be valid Python, not merely plausible-looking text.
    ast.parse(code)


def test_rejects_unknown_response_inclusion():
    with pytest.raises(ValueError):
        AnthropicTurn(response_inclusion="sometimes")
