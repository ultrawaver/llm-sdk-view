"""Anthropic through ``llm-anthropic``, as the first provider behind the seam.

Nothing here is new behaviour. It is the assembly this project has always
done, written down in one place now that it is no longer the only one:
``llm-anthropic`` builds the whole request in ``build_kwargs()``, and it always
opens ``client.messages.stream()`` because the API rejects non-streaming
requests whose ``max_tokens`` could run past ten minutes.
"""

from __future__ import annotations

import inspect
from typing import Any

from .base import Transport

# The client the generated code builds. No key is ever written into it: the
# SDK reads ANTHROPIC_API_KEY from the environment, and this project does not
# put a secret in a pane the user is about to copy.
HEADER = "import anthropic\n\nclient = anthropic.Anthropic()\n"

# How the official Playground ends the call: the reply is read off the stream,
# not handed back as one finished object.
STREAM_BODY = (
    "    for text in stream.text_stream:\n"
    '        print(text, end="", flush=True)\n'
)

STREAM = Transport(
    name="messages.stream",
    header=HEADER,
    call="client.messages.stream",
    binding="stream",
    context=True,
    body=STREAM_BODY,
)

# Never reached with the installed plugin, and kept for exactly that reason:
# if llm-anthropic ever stops forcing the stream, the pane must follow it
# rather than keep printing a call that no longer happens.
CREATE = Transport(
    name="messages.create",
    header=HEADER,
    call="client.messages.create",
    binding="message",
    context=False,
)

UNKNOWN = Transport(
    name="unknown",
    header=HEADER,
    call="client.messages.stream",
    binding="stream",
    context=True,
    body=STREAM_BODY,
)


class AnthropicProvider:
    """Claude through ``llm-anthropic`` and Anthropic's official Python SDK."""

    id = "anthropic"
    label = "Anthropic"

    def owns(self, model_id: str) -> bool:
        """Claimed by prefix, and deliberately not as a catch-all.

        A provider that accepted every unrecognised id would answer for a
        model it cannot send, and the failure would surface somewhere far
        from the mistake. Every id this project offers comes from the
        Anthropic Models API, so they all begin the same way.
        """
        return model_id.startswith("claude")

    def transport(self, model: Any, options: Any = None) -> Transport:
        """Which SDK method the installed plugin actually calls.

        Read off the plugin's own ``execute()`` rather than assumed, so a
        plugin that changes its mind moves the pane with it. The options are
        accepted and ignored: Anthropic's transport is a property of the
        runtime, and there is no request that can change it.
        """
        source = inspect.getsource(type(model).execute)
        if ".create(" in source:
            return CREATE
        if ".stream(" in source:
            return STREAM
        return UNKNOWN

    def assemble(
        self, model: Any, prompt: Any, conversation: Any, options: Any = None
    ) -> dict:
        """``build_kwargs()`` is already the whole request, so there is
        nothing for this project to add to it - and adding anything would be
        the second copy of the request this design exists to prevent."""
        return model.build_kwargs(prompt, conversation)
