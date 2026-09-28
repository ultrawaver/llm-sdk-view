"""Render the provider request as Python source.

There is exactly one input: the dictionary the execution path is about to hand
to the Anthropic SDK. Nothing in this module is allowed to construct, default
or "improve" that dictionary - a renderer that invents values would let the
right pane drift from the request, which is the one thing it exists to prevent.
"""

import json

# The official Playground inlines a value when it fits on one line and explodes
# it when it does not. Its sample inlines an item 58 columns wide and explodes
# one 104 columns wide, so any threshold between the two reproduces it; 88 is
# the usual Python line length and sits inside that window.
LINE_WIDTH = 88

# How the official Playground ends the call: the reply is read off the stream,
# not handed back as one finished object.
STREAM_BODY = (
    "    for text in stream.text_stream:\n"
    '        print(text, end="", flush=True)\n'
)


def _inline(value) -> str:
    """Render a JSON-shaped value on a single line, however long that is."""
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
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[" + ", ".join(_inline(item) for item in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return (
            "{"
            + ", ".join(
                f"{json.dumps(key)}: {_inline(item)}" for key, item in value.items()
            )
            + "}"
        )
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _fits(indent: int, text: str) -> bool:
    return len("    " * indent) + len(text) <= LINE_WIDTH


def _render(value, indent: int) -> str:
    """Render a value at an indent, exploding it only when it does not fit."""
    pad = "    " * indent
    one_line = _inline(value)
    if _fits(indent, one_line):
        return one_line
    inner = "    " * (indent + 1)
    if isinstance(value, list) and value:
        items = ",\n".join(f"{inner}{_render(item, indent + 1)}" for item in value)
        return f"[\n{items},\n{pad}]"
    if isinstance(value, dict) and value:
        items = ",\n".join(
            f"{inner}{_pair(key, item, indent)}" for key, item in value.items()
        )
        return f"{{\n{items},\n{pad}}}"
    # A scalar too long to fit: there is nothing to explode.
    return one_line


def _pair(key: str, value, indent: int) -> str:
    """Render ``"key": value``, exploding the value only when it does not fit."""
    prefix = f"{json.dumps(key)}: "
    one_line = prefix + _inline(value)
    if _fits(indent + 1, one_line):
        return one_line
    if isinstance(value, (list, dict)) and value:
        return prefix + _render(value, indent + 1)
    return one_line


def _argument(key: str, value) -> str:
    """Render one top-level ``key=value`` of the call."""
    one_line = f"{key}={_inline(value)}"
    if _fits(1, one_line):
        return one_line
    if isinstance(value, (list, dict)) and value:
        return f"{key}={_render(value, 1)}"
    return one_line


def render_kwargs(kwargs: dict, transport: str = "create") -> str:
    """Render SDK code from provider parameters.

    The input must be the dictionary the execution path actually sends, not a
    re-derivation of it, so that the right pane cannot drift from the request.
    Its key order is the order ``llm-anthropic`` assembled the request in and is
    reproduced as-is; grouping or sorting keys would hide that.

    ``transport`` is the SDK method the plugin really calls - passing "create"
    for a request that goes out as a stream would be a lie about the call.

    Whitespace is the only thing this renderer chooses: values and their order
    are the provider's, the layout is the official Playground's. Nothing is
    emitted as commentary - a request carries no explanation, and why streaming
    is fixed belongs to the form control that is greyed out.
    """
    arguments = ",\n".join(f"    {_argument(key, value)}" for key, value in kwargs.items())
    header = "import anthropic\n\nclient = anthropic.Anthropic()\n"
    if transport == "stream":
        return (
            f"{header}\n"
            "with client.messages.stream(\n"
            f"{arguments},\n"
            ") as stream:\n"
            f"{STREAM_BODY}"
        )
    return (
        f"{header}\n"
        "message = client.messages.create(\n"
        f"{arguments},\n"
        ")\n"
    )
