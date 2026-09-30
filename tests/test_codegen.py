"""The renderer, driven by what llm-anthropic really builds.

There is no hand-maintained request shape to compare against any more: the only
input to the renderer is a ``model.build_kwargs()`` result, so the tests build
one for real (offline, no transport involved) and check the rendering.
"""

import ast

import llm
import pytest

from native_api_chat import capabilities
from native_api_chat.codegen import render_kwargs
from native_api_chat.providers.anthropic import CREATE, STREAM

llm_anthropic = pytest.importorskip("llm_anthropic")
from llm_anthropic import WebSearch  # noqa: E402


def _built_kwargs(model_id: str = "claude-sonnet-5", cache: bool = True) -> dict:
    caps = capabilities.capabilities_for(model_id)
    model = llm.get_model(caps.llm_id)
    prompt = llm.Prompt(
        "Hello",
        model=model,
        options=model.Options(max_tokens=1024, cache=cache),
        tools=[WebSearch(max_uses=1)],
    )
    return model.build_kwargs(prompt, None)


def test_rendered_code_is_valid_python():
    ast.parse(render_kwargs(_built_kwargs(), CREATE))


def test_rendered_code_carries_the_request_values():
    code = render_kwargs(_built_kwargs(), CREATE)

    assert 'model="claude-sonnet-5"' in code
    assert "max_tokens=1024" in code
    assert '"name": "web_search"' in code
    assert '"cache_control"' in code


def test_cache_control_is_rendered_where_the_plugin_puts_it():
    """Prompt caching lives on a content block, never at the top level."""
    kwargs = _built_kwargs()
    code = render_kwargs(kwargs, CREATE)

    assert "cache_control" not in code.split("messages=")[0]
    assert kwargs["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_every_top_level_key_is_rendered():
    kwargs = _built_kwargs()
    code = render_kwargs(kwargs, CREATE)

    for key in kwargs:
        assert f"{key}=" in code


def test_rendered_code_contains_no_secret():
    code = render_kwargs(_built_kwargs(), CREATE)

    assert "api_key=" not in code
    assert "ANTHROPIC_API_KEY" not in code


def test_the_layout_matches_the_official_playground():
    """Byte-for-byte the shape the official Playground prints.

    Pasted from a real Playground request: short values stay on one line, long
    ones explode, and the reply is read off the stream.
    """
    kwargs = {
        "model": "claude-fable-5-1",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "你是Claude的啥模型 一句话答"},
            {"role": "assistant", "content": "我是Claude 3.5 Sonnet模型。"},
        ],
        "tools": [
            {
                "name": "web_search",
                "type": "web_search_20260318",
                "allowed_callers": ["direct"],
                "max_uses": 1,
            },
        ],
        "thinking": {"type": "disabled"},
    }

    assert render_kwargs(kwargs, STREAM) == (
        "import anthropic\n"
        "\n"
        "client = anthropic.Anthropic()\n"
        "\n"
        "with client.messages.stream(\n"
        '    model="claude-fable-5-1",\n'
        "    max_tokens=1024,\n"
        "    messages=[\n"
        '        {"role": "user", "content": "你是Claude的啥模型 一句话答"},\n'
        '        {"role": "assistant", "content": "我是Claude 3.5 Sonnet模型。"},\n'
        "    ],\n"
        "    tools=[\n"
        "        {\n"
        '            "name": "web_search",\n'
        '            "type": "web_search_20260318",\n'
        '            "allowed_callers": ["direct"],\n'
        '            "max_uses": 1,\n'
        "        },\n"
        "    ],\n"
        '    thinking={"type": "disabled"},\n'
        ") as stream:\n"
        "    for text in stream.text_stream:\n"
        '        print(text, end="", flush=True)\n'
    )


def test_a_value_that_fits_is_never_exploded():
    """The layout is chosen by width, not by type: dicts can stay inline."""
    code = render_kwargs({"a": {"x": 1}, "b": [1, 2, 3]}, STREAM)

    assert 'a={"x": 1},' in code
    assert "b=[1, 2, 3]," in code


def test_booleans_and_numbers_render_as_python_not_json():
    code = render_kwargs({"a": True, "b": False, "c": None, "d": 1.5, "e": 3}, CREATE)

    assert "a=True" in code
    assert "b=False" in code
    assert "c=None" in code
    assert "d=1.5" in code
    assert "e=3" in code
    ast.parse(code)
