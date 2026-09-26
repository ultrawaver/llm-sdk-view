"""The real conversation path.

    Browser -> llm-sdk-view -> LLM Python API -> llm-anthropic -> Anthropic SDK

This module deliberately owns no request-shaping logic of its own. The
parameters shown in the right pane are the dictionary ``llm_anthropic`` hands
to ``client.messages.create()``: they come from ``model.build_kwargs()``, which
is the last thing the plugin calls before sending. There is no second,
hand-maintained copy of the request anywhere in this project.

The form values below are *requests*, not facts: every one of them is checked
against the built request before a turn is allowed to stream. When the
installed plugin cannot express a value the form asks for, the turn is refused
instead of quietly sending something else.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import llm

from .codegen import render_kwargs

# Explicit, versioned tool types only. "latest" aliases are never sent.
WEB_SEARCH_TYPES = ("web_search_20260318", "web_search_20250305")
RESPONSE_INCLUSIONS = ("excluded", "full")
ALLOWED_CALLERS = ("code_execution_20260120", "direct")

# The Anthropic API default for web_search_20260318: the tool may also be
# called from code execution, which is what enables dynamic content filtering.
# allowed_callers=["direct"] is the value that turns filtering off.
DYNAMIC_FILTERING_CALLER = "code_execution_20260120"

# 0 means "no limit" in the form. The provider expresses that by omitting the
# field and rejects max_uses=0, so 0 must never be forwarded.
UNLIMITED_MAX_USES = 0


class MissingKeyError(RuntimeError):
    """No Anthropic key is available, so no request is sent."""


class UnsupportedOptionError(ValueError):
    """The form asked for something the installed llm-anthropic cannot send.

    A ``ValueError`` so that the routes report it as a bad request: silently
    sending the request without the value would contradict the form.
    """


@dataclass(frozen=True)
class ChatOptions:
    """Every value the chat form can set.

    These defaults are the defaults the form shows. There is intentionally no
    second copy of them in the template: the UI reads them from
    :func:`form_schema`.
    """

    model: str = "claude-sonnet-5"
    max_tokens: int = 16384
    system: str = ""
    web_search: bool = True
    web_search_type: str = "web_search_20260318"
    allowed_callers: str = DYNAMIC_FILTERING_CALLER
    response_inclusion: str = "excluded"
    max_uses: int = 1
    cache_control: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if self.max_uses < 0:
            raise ValueError("max_uses must be 0 (unlimited) or a positive integer")
        if self.web_search_type not in WEB_SEARCH_TYPES:
            raise ValueError(f"web_search_type must be one of {WEB_SEARCH_TYPES}")
        if self.allowed_callers not in ALLOWED_CALLERS:
            raise ValueError(f"allowed_callers must be one of {ALLOWED_CALLERS}")
        if self.response_inclusion not in RESPONSE_INCLUSIONS:
            raise ValueError(f"response_inclusion must be one of {RESPONSE_INCLUSIONS}")


def _web_search_class(model):
    """The plugin's own WebSearch class, never a copy of it."""
    for tool_class in getattr(model, "supported_server_side_tools", ()):
        if getattr(tool_class, "name", None) == "web_search":
            return tool_class
    raise ValueError(f"{model.model_id} does not support the web search tool")


def installed_capabilities(model) -> dict[str, bool]:
    """What the installed llm-anthropic can actually put on the request.

    Read off the installed source rather than assumed, because whether a
    control is real depends entirely on the plugin version.
    """
    parameters = inspect.signature(_web_search_class(model).__init__).parameters
    option_fields = getattr(model.Options, "model_fields", {})
    return {
        "response_inclusion": "response_inclusion" in parameters,
        "allowed_callers": "allowed_callers" in parameters,
        # A prompt cache TTL would be a prompt option, not a tool field.
        "cache_ttl": any("ttl" in name.lower() for name in option_fields),
    }


def _build_web_search_tool(model, options: ChatOptions, capabilities: dict):
    tool_class = _web_search_class(model)
    parameters = inspect.signature(tool_class.__init__).parameters
    kwargs: dict[str, Any] = {}
    # Unlimited is expressed by omission; max_uses=0 is not a legal value.
    if options.max_uses != UNLIMITED_MAX_USES:
        kwargs["max_uses"] = options.max_uses
    # response_inclusion is only accepted by web_search_20260318, and only when
    # the installed plugin can express it at all. When it cannot, the value is
    # dropped rather than faked, and the form reports the control as dead.
    if options.web_search_type == "web_search_20260318" and capabilities["response_inclusion"]:
        kwargs["response_inclusion"] = options.response_inclusion
    # Omitting allowed_callers is what selects the API default
    # (code_execution_20260120), so it is only sent when the form asks for
    # something else and the plugin can express it.
    if options.allowed_callers != DYNAMIC_FILTERING_CALLER:
        if "allowed_callers" not in parameters:
            raise UnsupportedOptionError(
                "the installed llm-anthropic cannot send allowed_callers, so "
                "dynamic filtering cannot be turned off from this UI"
            )
        kwargs["allowed_callers"] = [options.allowed_callers]
    return tool_class(**kwargs)


def _find_web_search_tool(kwargs: dict) -> dict | None:
    for tool in kwargs.get("tools", ()):
        if tool.get("name") == "web_search":
            return tool
    return None


def dynamic_filtering_state(kwargs: dict) -> str:
    """Read dynamic filtering off the request, never off the form."""
    tool = _find_web_search_tool(kwargs)
    if tool is None:
        return "off"
    callers = tool.get("allowed_callers")
    if callers is None:
        # Omitted means the API default, code_execution_20260120, applies.
        return "active"
    return "active" if DYNAMIC_FILTERING_CALLER in callers else "disabled"


def _verify(options: ChatOptions, kwargs: dict, capabilities: dict) -> None:
    """Refuse to stream a request that contradicts the form."""
    if not options.system.strip() and "system" in kwargs:
        raise ValueError("system is empty but the request carries a system prompt")
    if options.system.strip() and kwargs.get("system") != options.system:
        raise ValueError("the request system prompt does not match the form")

    tool = _find_web_search_tool(kwargs)
    if not options.web_search:
        if tool is not None:
            raise ValueError("web_search is off but a web search tool was built")
        return
    if tool is None:
        raise ValueError("web_search is on but no web search tool was built")
    if tool["type"] != options.web_search_type:
        raise ValueError(
            f"{options.model} sends {tool['type']}, not {options.web_search_type}: "
            "llm-anthropic derives the web search tool version from the model"
        )
    if options.max_uses == UNLIMITED_MAX_USES:
        if "max_uses" in tool:
            raise ValueError("max_uses=0 means unlimited and must be omitted")
    elif tool.get("max_uses") != options.max_uses:
        raise ValueError("the request max_uses does not match the form")
    if tool["type"] == "web_search_20260318" and capabilities["response_inclusion"]:
        if tool.get("response_inclusion") != options.response_inclusion:
            raise ValueError("the request response_inclusion does not match the form")
    expected = "active" if options.allowed_callers == DYNAMIC_FILTERING_CALLER else "disabled"
    state = dynamic_filtering_state(kwargs)
    if state != expected:
        raise ValueError(f"dynamic filtering is {state} but the form asked for {expected}")


def form_schema(model_id: str) -> dict:
    """Everything the chat form needs, read off the installed plugin."""
    model = llm.get_model(model_id)
    capabilities = installed_capabilities(model)
    defaults = ChatOptions(model=model_id)
    return {
        "defaults": {
            "model": model_id,
            "max_tokens": defaults.max_tokens,
            "system": defaults.system,
            "web_search": defaults.web_search,
            "web_search_type": defaults.web_search_type,
            "allowed_callers": defaults.allowed_callers,
            "response_inclusion": defaults.response_inclusion,
            "max_uses": defaults.max_uses,
            "cache_control": defaults.cache_control,
        },
        "model": {
            "id": model_id,
            # The id the plugin actually puts on the request.
            "sends": model.claude_model_id,
            "max_tokens": getattr(model, "default_max_tokens", None),
            # Asked of the plugin rather than re-derived here.
            "web_search_type": _web_search_class(model)().tool_spec(model)["type"],
        },
        "web_search_types": list(WEB_SEARCH_TYPES),
        "response_inclusions": list(RESPONSE_INCLUSIONS),
        "allowed_callers": {
            "options": list(ALLOWED_CALLERS),
            "api_default": DYNAMIC_FILTERING_CALLER,
            "sent": capabilities["allowed_callers"],
        },
        "capabilities": capabilities,
        "cache_ttl": {
            "supported": capabilities["cache_ttl"],
            "note": (
                "llm-anthropic hard-codes cache_control={'type': 'ephemeral'} and "
                "exposes no TTL option, so the form offers no TTL control."
            ),
        },
    }


@dataclass
class PreparedTurn:
    response: Any
    kwargs: dict
    code: str
    dynamic_filtering: str


class ChatSession:
    """One browser conversation backed by an ``llm.Conversation``."""

    def __init__(self, options: ChatOptions | None = None):
        self.options = options or ChatOptions()
        self.model = llm.get_model(self.options.model)
        self.capabilities = installed_capabilities(self.model)
        self._check_max_tokens()
        self.conversation = llm.Conversation(model=self.model)

    def _check_max_tokens(self) -> None:
        ceiling = getattr(self.model, "default_max_tokens", None)
        if ceiling and self.options.max_tokens > ceiling:
            raise ValueError(
                f"max_tokens must be at most {ceiling} for {self.model.model_id}"
            )

    def _tools(self) -> list:
        if not self.options.web_search:
            return []
        return [_build_web_search_tool(self.model, self.options, self.capabilities)]

    def _option_dict(self) -> dict:
        return {
            "max_tokens": self.options.max_tokens,
            "cache": self.options.cache_control,
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
            system=self.options.system or None,
            options=self._option_dict(),
            tools=self._tools(),
            stream=True,
        )
        kwargs = self.model.build_kwargs(response.prompt, self.conversation)
        _verify(self.options, kwargs, self.capabilities)
        return PreparedTurn(
            response=response,
            kwargs=kwargs,
            code=render_kwargs(kwargs),
            dynamic_filtering=dynamic_filtering_state(kwargs),
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
        yield {
            "type": "prepared",
            "code": prepared.code,
            "kwargs": prepared.kwargs,
            "dynamic_filtering": prepared.dynamic_filtering,
        }
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
            "dynamic_filtering": events[0]["dynamic_filtering"],
        }
