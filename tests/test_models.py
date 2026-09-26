import pytest

from llm_sdk_view.models import AnthropicTurn


def test_rejects_invalid_max_tokens():
    with pytest.raises(ValueError):
        AnthropicTurn(max_tokens=0)


def test_rejects_invalid_max_searches():
    with pytest.raises(ValueError):
        AnthropicTurn(max_searches=0)
