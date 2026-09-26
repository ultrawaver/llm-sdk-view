"""Render the provider request as Python source.

There is exactly one input: the dictionary the execution path is about to hand
to the Anthropic SDK. Nothing in this module is allowed to construct, default
or "improve" that dictionary - a renderer that invents values would let the
right pane drift from the request, which is the one thing it exists to prevent.
"""

import json


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
    if isinstance(value, float):
        return repr(value)
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


def render_kwargs(kwargs: dict, transport: str = "create", note: str | None = None) -> str:
    """Render SDK code from provider parameters.

    The input must be the dictionary the execution path actually sends, not a
    re-derivation of it, so that the right pane cannot drift from the request.
    ``transport`` is the SDK method the plugin really calls - passing "create"
    for a request that goes out as a stream would be a lie about the call.
    """
    arguments = ",\n".join(f"    {key}={_python(value)}" for key, value in kwargs.items())
    header = "import anthropic\n\nclient = anthropic.Anthropic()\n"
    if note:
        header += "\n" + "".join(f"# {line}\n" for line in note.splitlines())
    if transport == "stream":
        return (
            f"{header}\n"
            "with client.messages.stream(\n"
            f"{arguments},\n"
            ") as stream:\n"
            "    message = stream.get_final_message()\n"
        )
    return (
        f"{header}\n"
        "message = client.messages.create(\n"
        f"{arguments},\n"
        ")\n"
    )
