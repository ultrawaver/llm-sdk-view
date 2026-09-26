"""The real conversation path.

    Browser -> llm-sdk-view -> LLM Python API -> llm-anthropic -> Anthropic SDK

This module deliberately owns no request-shaping logic of its own. The
parameters shown in the right pane are the dictionary ``llm_anthropic`` hands
to ``client.messages.create()``: they come from ``model.build_kwargs()``, which
is the last thing the plugin calls before sending. There is no second,
hand-maintained copy of the request anywhere in this project.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import llm

from .codegen import render_kwargs


class MissingKeyError(RuntimeError):
    """No Anthropic key is available, so no request is sent."""


@dataclass(frozen=True)
class ChatOptions:
    model: str = "claude-sonnet-5"
    max_tokens: int = 4096
    web_search: bool = True
    max_searches: int = 5
    response_inclusion: str = "excluded"
    prompt_cache: bool = True


def _web_search_tool(model, options: ChatOptions):
    """Instantiate the plugin's own WebSearch server-side tool.

    Taking the class from ``model.supported_server_side_tools`` keeps the
    project dependent on the installed plugin rather than on a copy of it:
    when the plugin gains ``response_inclusion`` this code starts sending it
    with no change here.
    """
    for tool_class in getattr(model, "supported_server_side_tools", ()):
        if getattr(tool_class, "name", None) != "web_search":
            continue
        kwargs: dict[str, Any] = {"max_uses": options.max_searches}
        parameters = inspect.signature(tool_class.__init__).parameters
        if "response_inclusion" in parameters:
            kwargs["response_inclusion"] = options.response_inclusion
        return tool_class(**kwargs)
    raise ValueError(f"{model.model_id} does not support the web search tool")


@dataclass
class PreparedTurn:
    response: Any
    kwargs: dict
    code: str


class ChatSession:
    """One browser conversation backed by an ``llm.Conversation``."""

    def __init__(self, options: ChatOptions | None = None):
        self.options = options or ChatOptions()
        self.model = llm.get_model(self.options.model)
        self.conversation = llm.Conversation(model=self.model)

    def _tools(self) -> list:
        if not self.options.web_search:
            return []
        return [_web_search_tool(self.model, self.options)]

    def _option_dict(self) -> dict:
        return {
            "max_tokens": self.options.max_tokens,
            "cache": self.options.prompt_cache,
        }

    def prepare(self, text: str) -> PreparedTurn:
        """Build this turn without sending it.

        ``conversation.prompt()`` is lazy: it bakes the full history from
        earlier turns into ``response.prompt.messages`` and returns a Response
        that has not contacted the provider. That lets the right pane show the
        real request before anything is sent.
        """
        response = self.conversation.prompt(
            text,
            options=self._option_dict(),
            tools=self._tools(),
            stream=True,
        )
        kwargs = self.model.build_kwargs(response.prompt, self.conversation)
        return PreparedTurn(
            response=response,
            kwargs=kwargs,
            code=render_kwargs(kwargs),
        )

    def _require_key(self) -> None:
        """Fail locally rather than send a request nobody authorised."""
        try:
            self.model.get_key()
        except llm.NeedsKeyException as ex:
            raise MissingKeyError(str(ex)) from ex

    def stream_turn(self, text: str) -> Iterator[dict]:
        """Yield events: the prepared request, then streamed text, then done."""
        prepared = self.prepare(text)
        self._require_key()
        yield {"type": "prepared", "code": prepared.code, "kwargs": prepared.kwargs}
        for event in prepared.response.stream_events():
            if event.type == "text":
                yield {"type": "text", "text": event.chunk}
        yield {"type": "done", "text": prepared.response.text_or_raise()}

    def run_turn(self, text: str) -> dict:
        events = list(self.stream_turn(text))
        return {
            "text": events[-1]["text"],
            "code": events[0]["code"],
            "kwargs": events[0]["kwargs"],
        }
