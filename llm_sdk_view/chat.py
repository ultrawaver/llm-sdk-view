"""The real conversation path.

    Browser -> llm-sdk-view -> LLM Python API -> llm-anthropic -> Anthropic SDK

This module deliberately owns no request-shaping logic of its own. The
parameters shown in the right pane are the dictionary ``llm_anthropic`` hands
to ``client.messages.create()``: they come from ``model.build_kwargs()``, which
is the last thing the plugin calls before sending. There is no second,
hand-maintained copy of the request anywhere in this project.

The form values below are *requests*, not facts: every one of them is checked
against the built request, and against :mod:`llm_sdk_view.capabilities`, before
a turn is allowed to stream. When the installed plugin cannot express a value
the form asks for, the turn is refused instead of quietly sending something
else.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import llm

from .capabilities import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    DISABLED_EFFORT,
    ModelCapabilities,
    capabilities_for,
    fallback_models,
    model_ids,
    plugin_transport,
    profile,
    provenance,
    resolve_model_id,
)
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

# Why llm-anthropic never calls messages.create(), quoted from its execute():
# "The Anthropic SDK rejects non-streaming requests with large max_tokens
# values because they may take longer than ten minutes."
STREAMING_TRANSPORT_NOTE = (
    "llm-anthropic always opens messages.stream(), even for a buffered turn: "
    "the API rejects non-streaming requests whose max_tokens could run past "
    "ten minutes. The Streaming control decides whether this UI shows text as "
    "it arrives, not which SDK method is used."
)


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

    model: str = DEFAULT_MODEL
    max_tokens: int = 16384
    system: str = ""
    effort: str = DEFAULT_EFFORT
    web_search: bool = True
    # None means "whatever tool version llm-anthropic gives this model".
    web_search_type: str | None = None
    allowed_callers: str = DYNAMIC_FILTERING_CALLER
    response_inclusion: str = "excluded"
    max_uses: int = 1
    cache_control: bool = True
    # Whether the UI shows text as it arrives. It is not a provider
    # parameter: see _transport_note() for what it can and cannot change.
    stream: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if self.max_uses < 0:
            raise ValueError("max_uses must be 0 (unlimited) or a positive integer")
        if self.web_search_type is not None and self.web_search_type not in WEB_SEARCH_TYPES:
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


def _build_web_search_tool(model, options: ChatOptions, capabilities: ModelCapabilities):
    tool_class = _web_search_class(model)
    kwargs: dict[str, Any] = {}
    # Unlimited is expressed by omission; max_uses=0 is not a legal value.
    if options.max_uses != UNLIMITED_MAX_USES:
        kwargs["max_uses"] = options.max_uses
    # response_inclusion is only accepted by web_search_20260318, and only when
    # the installed plugin can express it at all. When it cannot, the value is
    # dropped rather than faked, and the form reports the control as dead.
    if capabilities.response_inclusion:
        kwargs["response_inclusion"] = options.response_inclusion
    # Omitting allowed_callers is what selects the API default
    # (code_execution_20260120), so it is only sent when the form asks for
    # something else and the plugin can express it.
    if options.allowed_callers != DYNAMIC_FILTERING_CALLER:
        if not capabilities.allowed_callers:
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


def _verify(options: ChatOptions, kwargs: dict, capabilities: ModelCapabilities) -> None:
    """Refuse to stream a request that contradicts the form."""
    if not options.system.strip() and "system" in kwargs:
        raise ValueError("system is empty but the request carries a system prompt")
    if options.system.strip() and kwargs.get("system") != options.system:
        raise ValueError("the request system prompt does not match the form")

    if kwargs.get("model") != capabilities.api_model_id:
        raise ValueError("the request model does not match the form")
    if kwargs.get("max_tokens") != options.max_tokens:
        raise ValueError("the request max_tokens does not match the form")
    _verify_effort(options, kwargs, capabilities)
    _verify_cache(options, kwargs)

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
    if capabilities.response_inclusion:
        if tool.get("response_inclusion") != options.response_inclusion:
            raise ValueError("the request response_inclusion does not match the form")
    expected = "active" if options.allowed_callers == DYNAMIC_FILTERING_CALLER else "disabled"
    state = dynamic_filtering_state(kwargs)
    if state != expected:
        raise ValueError(f"dynamic filtering is {state} but the form asked for {expected}")


def _verify_cache(options: ChatOptions, kwargs: dict) -> None:
    """Prompt caching has to land where the form said, or not at all."""
    blocks = [
        block
        for message in kwargs.get("messages", ())
        for block in message.get("content", ())
        if isinstance(block, dict)
    ]
    marked = [block for block in blocks if "cache_control" in block]
    if options.cache_control and not marked:
        raise ValueError("cache_control is on but no content block carries it")
    if not options.cache_control and marked:
        raise ValueError("cache_control is off but the request still carries it")


def _verify_effort(options: ChatOptions, kwargs: dict, capabilities: ModelCapabilities) -> None:
    """Effort has to land on the request the way the form promised."""
    sent_effort = kwargs.get("output_config", {}).get("effort")
    thinking = kwargs.get("thinking")
    if options.effort == DEFAULT_EFFORT:
        if sent_effort is not None:
            raise ValueError("effort=default must leave effort off the request")
    elif options.effort == DISABLED_EFFORT:
        # "off" is only honest if thinking really is disabled; for models that
        # always think, llm-anthropic refuses before we get here.
        if sent_effort is not None:
            raise ValueError("effort=off must not send an effort level")
        if capabilities.supports_thinking and thinking != {"type": "disabled"}:
            raise ValueError("effort=off must send thinking={'type': 'disabled'}")
    else:
        if sent_effort != options.effort:
            raise ValueError(f"the request effort is {sent_effort}, not {options.effort}")


def final_message_text(response: Any) -> str:
    """Assemble the reply from the final accumulated Message.

    Streaming yields fragments; the one thing that is safe to show and to keep
    is the Message the provider finished with, so both panes and the recorded
    conversation agree on it.
    """
    message = getattr(response, "response_json", None) or {}
    blocks = message.get("content") or []
    text = "".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    )
    return text or response.text_or_raise()


def form_schema(model_id: str) -> dict:
    """Everything the chat form needs: defaults, limits, capabilities."""
    capabilities = capabilities_for(model_id)
    defaults = ChatOptions(model=model_id)
    fallback = profile()
    return {
        "models": list(model_ids()),
        "default_model": DEFAULT_MODEL,
        # Where the numbers on this page came from: the Models API when it
        # answered, the labelled fallback profile when it did not.
        "model_data": {
            **provenance(),
            "profile_version": fallback["profile_version"],
            "profile_source": fallback["profile_source"],
            "fallback_models": list(fallback_models()),
        },
        "model": {
            "id": capabilities.id,
            "sends": capabilities.api_model_id,
            "max_tokens": capabilities.max_output_tokens,
            "context_window": capabilities.context_window,
            "web_search_type": capabilities.web_search_type,
        },
        "defaults": {
            "model": model_id,
            "max_tokens": defaults.max_tokens,
            "system": defaults.system,
            "effort": defaults.effort,
            "web_search": defaults.web_search,
            "web_search_type": capabilities.web_search_type or defaults.web_search_type,
            "allowed_callers": defaults.allowed_callers,
            "response_inclusion": defaults.response_inclusion,
            "max_uses": defaults.max_uses,
            "cache_control": defaults.cache_control,
            "stream": defaults.stream,
        },
        "transport": {
            "sdk_method": plugin_transport(llm.get_model(resolve_model_id(model_id))),
            "streaming_note": STREAMING_TRANSPORT_NOTE,
        },
        "capabilities": capabilities.as_dict(),
        "web_search_types": list(WEB_SEARCH_TYPES),
        "response_inclusions": list(RESPONSE_INCLUSIONS),
        "allowed_callers": {
            "options": list(ALLOWED_CALLERS),
            "api_default": DYNAMIC_FILTERING_CALLER,
            "sent": capabilities.allowed_callers,
        },
        "cache_ttl": {
            "supported": capabilities.cache_ttl,
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
    transport: str


class ChatSession:
    """One browser conversation backed by an ``llm.Conversation``."""

    def __init__(self, options: ChatOptions | None = None):
        self.options = options or ChatOptions()
        self.capabilities = capabilities_for(self.options.model)
        if self.options.web_search_type is None:
            self.options = replace(
                self.options, web_search_type=self.capabilities.web_search_type
            )
        self.model = llm.get_model(resolve_model_id(self.options.model))
        self._check_max_tokens()
        self._check_effort()
        self._check_web_search()
        self.conversation = llm.Conversation(model=self.model)

    def _check_max_tokens(self) -> None:
        ceiling = self.capabilities.max_output_tokens
        if ceiling and self.options.max_tokens > ceiling:
            raise ValueError(
                f"max_tokens must be at most {ceiling} for {self.options.model}"
            )

    def _check_effort(self) -> None:
        effort = self.options.effort
        if effort == DEFAULT_EFFORT:
            return
        if effort == DISABLED_EFFORT:
            if not self.capabilities.can_disable_thinking:
                raise UnsupportedOptionError(
                    f"effort cannot be turned off for {self.options.model}: "
                    "llm-anthropic marks this model as always thinking"
                )
            return
        if not self.capabilities.supports_effort:
            raise UnsupportedOptionError(
                f"{self.options.model} does not support effort "
                f"({', '.join(self.capabilities.effort_levels) or 'no effort levels'})"
            )
        if effort not in self.capabilities.effort_levels:
            raise ValueError(
                f"effort must be one of {self.capabilities.effort_levels} "
                f"for {self.options.model}"
            )

    def _check_web_search(self) -> None:
        if self.options.web_search and not self.capabilities.supports_web_search:
            raise UnsupportedOptionError(
                f"{self.options.model} does not support the web search tool"
            )

    def _tools(self) -> list:
        if not self.options.web_search:
            return []
        return [_build_web_search_tool(self.model, self.options, self.capabilities)]

    def _option_dict(self) -> dict:
        options: dict[str, Any] = {
            "max_tokens": self.options.max_tokens,
            "cache": self.options.cache_control,
        }
        # "default" leaves both off so the model's own default effort applies.
        if self.options.effort == DISABLED_EFFORT:
            options["thinking"] = False
        elif self.options.effort != DEFAULT_EFFORT:
            options["thinking_effort"] = self.options.effort
        return options

    def _transport(self) -> str:
        """The SDK method the installed plugin will really call."""
        return plugin_transport(self.model)

    def _transport_note(self) -> str:
        if self._transport() != "stream":
            return ""
        if self.options.stream:
            return (
                "llm-anthropic sends this with the streaming transport, and the "
                "UI shows each chunk as it arrives."
            )
        return (
            "The UI asked for a buffered response, so it waits for the final "
            "message instead of showing chunks. " + STREAMING_TRANSPORT_NOTE
        )

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
        transport = self._transport()
        return PreparedTurn(
            response=response,
            kwargs=kwargs,
            code=render_kwargs(kwargs, transport=transport, note=self._transport_note()),
            dynamic_filtering=dynamic_filtering_state(kwargs),
            transport=transport,
        )

    def _require_key(self) -> None:
        """Fail locally rather than send a request nobody authorised."""
        try:
            self.model.get_key()
        except llm.NeedsKeyException as ex:
            raise MissingKeyError(str(ex)) from ex

    def stream_turn(self, text: str) -> Iterator[dict]:
        """Yield events: the prepared request, then text, then the final text.

        With ``stream`` off nothing is emitted until the turn is complete, which
        is what "buffered" means here: the plugin still streams internally, and
        the completed response is what the UI receives.
        """
        prepared = self.prepare(text)
        self._require_key()
        yield {
            "type": "prepared",
            "code": prepared.code,
            "kwargs": prepared.kwargs,
            "dynamic_filtering": prepared.dynamic_filtering,
            "transport": prepared.transport,
        }
        if self.options.stream:
            for event in prepared.response.stream_events():
                if event.type == "text":
                    yield {"type": "text", "text": event.chunk}
        else:
            # Consume the whole response so it is accumulated and recorded on
            # the conversation, without showing any of it yet.
            prepared.response.text()
        yield {"type": "done", "text": final_message_text(prepared.response)}

    def run_turn(self, text: str) -> dict:
        events = list(self.stream_turn(text))
        text_events = [event["text"] for event in events if event["type"] == "text"]
        return {
            "text": events[-1]["text"],
            "code": events[0]["code"],
            "kwargs": events[0]["kwargs"],
            "dynamic_filtering": events[0]["dynamic_filtering"],
            "transport": events[0]["transport"],
            "chunks": text_events,
        }
