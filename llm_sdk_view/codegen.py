import json

from .models import AnthropicTurn


def _python(value, indent: int = 1) -> str:
    """Render a JSON-shaped value as Python source at a given indent level."""
    pad = "    " * indent
    inner = "    " * (indent + 1)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        items = ",\n".join(f"{inner}{_python(item, indent + 1)}" for item in value)
        return f"[\n{items},\n{pad}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ",\n".join(
            f"{inner}{json.dumps(key)}: {_python(item, indent + 1)}"
            for key, item in value.items()
        )
        return f"{{\n{items},\n{pad}}}"
    raise TypeError(f"Unsupported value: {type(value)!r}")


def anthropic_messages(turn: AnthropicTurn) -> list[dict]:
    """Build Anthropic `messages` in the shape llm-anthropic sends.

    Content must be a list of blocks rather than a bare string: prompt caching
    is expressed with `cache_control` on a block, so a string content field
    could not carry it. `cache_control` goes on the last block of the final
    message, which is where `llm_anthropic._Shared.build_messages` puts it when
    `options.cache` is set.
    """
    messages = [
        {"role": message.role, "content": [{"type": "text", "text": message.content}]}
        for message in turn.messages
    ]
    if turn.prompt_cache and messages:
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return messages


def anthropic_kwargs(turn: AnthropicTurn) -> dict:
    kwargs = {
        "model": turn.model,
        "max_tokens": turn.max_tokens,
        "messages": anthropic_messages(turn),
    }
    if turn.web_search:
        # `web_search_20260318` is the version llm-anthropic emits for models
        # with supports_adaptive_thinking. Dynamic content filtering is the
        # default for this version and is preserved by never sending
        # `allowed_callers`, which would force direct-only calling.
        kwargs["tools"] = [
            {
                "type": "web_search_20260318",
                "name": "web_search",
                "max_uses": turn.max_searches,
                "response_inclusion": turn.response_inclusion,
            }
        ]
    return kwargs


def render_anthropic_python(turn: AnthropicTurn) -> str:
    kwargs = anthropic_kwargs(turn)
    arguments = ",\n".join(f"    {key}={_python(value)}" for key, value in kwargs.items())
    return (
        "import anthropic\n\n"
        "client = anthropic.Anthropic()\n\n"
        "message = client.messages.create(\n"
        f"{arguments},\n"
        ")\n"
    )
