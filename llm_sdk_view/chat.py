"""The real conversation path.

    Browser -> llm-sdk-view -> LLM Python API -> llm-anthropic -> Anthropic SDK

This module deliberately owns no request-shaping logic of its own. The
parameters shown in the right pane are the dictionary ``llm_anthropic`` hands to
the Anthropic SDK: they come from ``model.build_kwargs()``, which is the last
thing the plugin calls before sending. There is no second, hand-maintained copy
of the request anywhere in this project.

The form values below are *requests*, not facts: every one of them is checked
against the built request, and against :mod:`llm_sdk_view.capabilities`, before
a turn is allowed to stream. When the installed plugin cannot express a value
the form asks for, the turn is refused instead of quietly sending something
else.

Each control also reports a status, so the page can say which of four things a
value means: what the Anthropic API supports, what the selected model supports,
what ``llm-anthropic`` exposes, and what this request actually sends.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from math import ceil
from typing import Any

import llm

from .capabilities import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    EFFORT_LEVELS_NEEDING_THINKING,
    THINKING_OFF,
    THINKING_ON,
    THINKING_STATES,
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

# The Anthropic API default for the newer web search tool: the tool may also be
# called from code execution, which is what enables dynamic content filtering.
# allowed_callers=["direct"] is the value that turns filtering off.
DYNAMIC_FILTERING_CALLER = "code_execution_20260120"

# Earlier tool versions default to direct calls and have no dynamic filtering.
BASIC_SEARCH_CALLER = "direct"

# 0 means "no limit" in the form. The provider expresses that by omitting the
# field and rejects max_uses=0, so 0 must never be forwarded.
UNLIMITED_MAX_USES = 0

# Why llm-anthropic never calls messages.create(), quoted from its execute():
# "The Anthropic SDK rejects non-streaming requests with large max_tokens
# values because they may take longer than ten minutes."
STREAMING_TRANSPORT_NOTE = (
    "llm-anthropic always opens client.messages.stream(): the Anthropic API "
    "rejects non-streaming requests whose max_tokens could run past ten "
    "minutes. Streaming is therefore fixed by the runtime, not chosen here."
)

# The status vocabulary. Every control reports one of these, so a greyed-out
# value always says *why* it is greyed out.
EDITABLE = "Editable"
RUNTIME_FIXED = "API supported · runtime fixed"
UNSUPPORTED_BY_MODEL = "Unsupported by selected model"
UNSUPPORTED_BY_TOOL = "Unsupported by current tool version"
PROVIDER_DEFAULT = "Provider default"
FALLBACK_DATA = "Fallback capability data"

# No tokenizer ships with this project, and counting tokens would be a paid
# API call, so a pre-send context figure is an estimate and says so.
CHARS_PER_TOKEN = 4

ESTIMATE_NOTE = (
    f"estimated before sending (~{CHARS_PER_TOKEN} characters per token, no "
    "tokenizer and no API call)"
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
    # None means "the official default for this model", resolved per model.
    thinking: str | None = None
    effort: str = DEFAULT_EFFORT
    web_search: bool = True
    # None means "whatever tool version llm-anthropic gives this model".
    web_search_type: str | None = None
    allowed_callers: str = DYNAMIC_FILTERING_CALLER
    response_inclusion: str = "excluded"
    max_uses: int = 1
    cache_control: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if self.max_uses < 0:
            raise ValueError("max_uses must be 0 (unlimited) or a positive integer")
        if self.thinking is not None and self.thinking not in THINKING_STATES:
            raise ValueError(f"thinking must be one of {THINKING_STATES}")
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
    # llm-anthropic has no allowed_callers parameter at all: it always emits
    # the bare tool, and the API default caller applies. Anything else would be
    # a second, fake request, so it is refused.
    if options.allowed_callers != DYNAMIC_FILTERING_CALLER:
        raise UnsupportedOptionError(
            "the installed llm-anthropic cannot send allowed_callers, so "
            "dynamic filtering cannot be turned off from this UI"
        )
    return tool_class(**kwargs)


def _find_web_search_tool(kwargs: dict) -> dict | None:
    for tool in kwargs.get("tools", ()):
        if tool.get("name") == "web_search":
            return tool
    return None


def dynamic_filtering_state(kwargs: dict) -> str:
    """Read dynamic filtering off the request, never off the form.

    ``web_search_20260318`` runs from code execution unless told otherwise, so
    an omitted ``allowed_callers`` means filtering is active. The older tool
    has no dynamic filtering at all, which is a different answer from "off".
    """
    tool = _find_web_search_tool(kwargs)
    if tool is None:
        return "off"
    if tool.get("type") != "web_search_20260318":
        return "not-supported"
    callers = tool.get("allowed_callers")
    if callers is None:
        # Omitted means the API default, code_execution_20260120, applies.
        return "active"
    return "active" if DYNAMIC_FILTERING_CALLER in callers else "disabled"


def effective_allowed_callers(kwargs: dict) -> str | None:
    """The caller the API will really use, read off the request."""
    tool = _find_web_search_tool(kwargs)
    if tool is None:
        return None
    callers = tool.get("allowed_callers")
    if callers:
        return list(callers)[0]
    return (
        DYNAMIC_FILTERING_CALLER
        if tool.get("type") == "web_search_20260318"
        else BASIC_SEARCH_CALLER
    )


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
    _verify_thinking(options, kwargs, capabilities)
    _verify_effort(options, kwargs)
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
    if tool.get("allowed_callers") is not None:
        raise ValueError("llm-anthropic cannot send allowed_callers, but the request has it")


def _verify_thinking(
    options: ChatOptions, kwargs: dict, capabilities: ModelCapabilities
) -> None:
    """Thinking has to land as the official value the form promised."""
    thinking = kwargs.get("thinking")
    if options.thinking == THINKING_OFF:
        if capabilities.thinking_off_request == "disabled":
            if thinking != {"type": "disabled"}:
                raise ValueError("thinking is off but the request does not disable it")
        elif thinking is not None:
            raise ValueError(
                "thinking is off, so the request must carry no thinking field"
            )
        return
    if thinking is None:
        raise ValueError("thinking is on but the request carries no thinking field")
    if capabilities.thinking_mode == "adaptive":
        if thinking.get("type") != "adaptive":
            raise ValueError(f"thinking is {thinking.get('type')}, not adaptive")
        return
    # Extended (manual) thinking: the budget is set by the runtime, not by us,
    # so the request has to carry exactly the value the plugin hard-codes.
    if thinking.get("type") != "enabled":
        raise ValueError(f"thinking is {thinking.get('type')}, not enabled")
    if thinking.get("budget_tokens") != capabilities.budget_tokens:
        raise ValueError(
            "the request budget_tokens is "
            f"{thinking.get('budget_tokens')}, not {capabilities.budget_tokens}"
        )


def _verify_effort(options: ChatOptions, kwargs: dict) -> None:
    """Effort has to land on the request the way the form promised."""
    sent_effort = (kwargs.get("output_config") or {}).get("effort")
    if options.effort == DEFAULT_EFFORT:
        if sent_effort is not None:
            raise ValueError("effort=default must leave effort off the request")
        return
    if sent_effort != options.effort:
        raise ValueError(f"the request effort is {sent_effort}, not {options.effort}")
    if options.thinking == THINKING_OFF and sent_effort in EFFORT_LEVELS_NEEDING_THINKING:
        raise ValueError(
            f"effort={sent_effort} is rejected when thinking is disabled"
        )


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


# --- context ----------------------------------------------------------------


def _estimate_text(text: str) -> int:
    return max(1, ceil(len(text) / CHARS_PER_TOKEN)) if text else 0


def estimate_request_tokens(kwargs: dict) -> int | None:
    """A labelled estimate of what this request will cost in input tokens.

    Returns ``None`` when the request holds something a character count cannot
    speak for - an image, an attachment, a block this project does not model -
    because showing a number derived from nothing would be worse than showing
    nothing.
    """
    total = 0
    for message in kwargs.get("messages", ()):
        content = message.get("content")
        if isinstance(content, str):
            total += _estimate_text(content)
            continue
        if not isinstance(content, list):
            return None
        for block in content:
            if not isinstance(block, dict):
                return None
            block_type = block.get("type")
            if block_type == "text":
                total += _estimate_text(block.get("text", ""))
            elif block_type in ("thinking", "redacted_thinking"):
                total += _estimate_text(
                    block.get("thinking") or block.get("data") or ""
                )
            elif block_type in ("tool_use", "server_tool_use"):
                total += _estimate_text(json.dumps(block.get("input", {})))
            elif block_type.endswith("_tool_result"):
                total += _estimate_text(json.dumps(block.get("content", "")))
            else:
                return None
    total += _estimate_text(kwargs.get("system") or "")
    total += _estimate_text(json.dumps(kwargs.get("tools", [])))
    return total


@dataclass(frozen=True)
class ContextState:
    """What the context meter shows, and how much of it is a fact."""

    tokens: int | None
    limit: int | None
    source: str
    limit_source: str
    limit_data_source: str
    percent: float | None
    reserved: int
    fits: bool | None

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "limit": self.limit,
            "source": self.source,
            "limit_source": self.limit_source,
            "limit_data_source": self.limit_data_source,
            "percent": self.percent,
            "reserved": self.reserved,
            "fits": self.fits,
        }


def context_state(
    kwargs: dict,
    capabilities: ModelCapabilities,
    usage: dict | None = None,
    reserved: int = 0,
) -> ContextState:
    """Build the context figure: real usage when there is any, else an estimate."""
    limit = capabilities.context_window
    if usage is not None:
        tokens = (usage.get("input") or 0) + (usage.get("output") or 0)
        source = "API usage"
    else:
        tokens = estimate_request_tokens(kwargs)
        source = "estimated" if tokens is not None else "unknown"

    if limit is None or tokens is None:
        percent = None
        fits = None
    else:
        percent = round(100.0 * tokens / limit, 2) if limit else None
        fits = tokens + reserved <= limit

    return ContextState(
        tokens=tokens,
        limit=limit,
        source=source,
        limit_source=capabilities.context_window_source,
        limit_data_source=(
            PROVIDER_DEFAULT if capabilities.data_source == "models-api" else FALLBACK_DATA
        ),
        percent=percent,
        reserved=reserved,
        fits=fits,
    )


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


# --- the form surface --------------------------------------------------------


def _thinking_control(capabilities: ModelCapabilities, thinking: str) -> dict:
    if not capabilities.supports_thinking:
        return {
            "status": UNSUPPORTED_BY_MODEL,
            "value": THINKING_OFF,
            "options": _option_list(THINKING_STATES, disabled=THINKING_STATES),
            "note": "this model has no thinking parameter",
        }
    if not capabilities.thinking_editable:
        return {
            "status": RUNTIME_FIXED,
            "value": THINKING_ON,
            "options": _option_list(THINKING_STATES, disabled=(THINKING_OFF,)),
            "note": (
                "this model always thinks, and both the API and llm-anthropic "
                "reject thinking={'type': 'disabled'}"
            ),
        }
    return {
        "status": EDITABLE,
        "value": thinking,
        "options": _option_list(THINKING_STATES, disabled=()),
        "note": (
            "ON sends thinking={'type': 'adaptive'}"
            if capabilities.thinking_mode == "adaptive"
            else "ON sends thinking={'type': 'enabled'} with the budget below"
        ),
    }


def _option_list(values: tuple[str, ...], disabled: tuple[str, ...]) -> list[dict]:
    return [{"value": value, "disabled": value in disabled} for value in values]


def _effort_control(capabilities: ModelCapabilities) -> dict:
    if not capabilities.supports_effort:
        return {
            "status": UNSUPPORTED_BY_MODEL,
            "value": DEFAULT_EFFORT,
            "options": _option_list(capabilities.effort_options(), disabled=()),
            "levels": [],
            "note": "this model has no effort parameter",
        }
    # Only meaningful where thinking can actually be turned off.
    blocked = capabilities.effort_disabled_with(THINKING_OFF)
    options = []
    for value in capabilities.effort_options():
        options.append(
            {
                "value": value,
                "disabled": False,
                "blocked_without_thinking": value in blocked,
            }
        )
    note = "sent as output_config.effort"
    if blocked:
        note += "; " + ", ".join(blocked) + " are rejected when thinking is off"
    return {
        "status": EDITABLE,
        "value": DEFAULT_EFFORT,
        "options": options,
        "levels": list(capabilities.effort_levels),
        "note": note,
    }


def _budget_control(capabilities: ModelCapabilities, thinking: str) -> dict:
    if capabilities.thinking_mode != "extended" or capabilities.budget_tokens is None:
        return {
            "status": UNSUPPORTED_BY_MODEL,
            "value": None,
            "editable": False,
            "note": (
                "only extended thinking uses budget_tokens; this model uses "
                + capabilities.thinking_mode
                + " thinking"
            ),
        }
    return {
        "status": RUNTIME_FIXED,
        "value": capabilities.budget_tokens,
        "editable": capabilities.budget_tokens_editable,
        "note": (
            "llm-anthropic hard-codes DEFAULT_THINKING_TOKENS and exposes no "
            "option for it, so the value is fixed by the runtime and is only "
            "sent while thinking is ON"
        ),
        "active": thinking == THINKING_ON,
    }


def _response_inclusion_control(capabilities: ModelCapabilities) -> dict:
    if not capabilities.web_search_type:
        return {
            "status": UNSUPPORTED_BY_MODEL,
            "note": "this model has no web search tool",
        }
    if not capabilities.plugin_response_inclusion:
        return {
            "status": RUNTIME_FIXED,
            "note": "llm-anthropic exposes no response_inclusion parameter",
        }
    if capabilities.web_search_type != "web_search_20260318":
        return {
            "status": UNSUPPORTED_BY_TOOL,
            "note": f"{capabilities.web_search_type} has no response_inclusion",
        }
    return {"status": EDITABLE, "note": f"sent on {capabilities.web_search_type}"}


def _allowed_callers_control(capabilities: ModelCapabilities) -> dict:
    options = []
    for value in ALLOWED_CALLERS:
        if value == capabilities.effective_allowed_callers:
            options.append(
                {
                    "value": value,
                    "disabled": False,
                    "note": "API default · runtime fixed",
                }
            )
        else:
            options.append(
                {
                    "value": value,
                    "disabled": True,
                    "note": "API supported · not exposed by llm-anthropic",
                }
            )
    return {
        "status": RUNTIME_FIXED,
        "value": capabilities.effective_allowed_callers,
        "editable": capabilities.allowed_callers,
        "options": options,
        "note": (
            "the field is not sent, so the API default applies; llm-anthropic "
            "has no allowed_callers parameter, so the alternative cannot be "
            "chosen without changing what the request really contains"
        ),
    }


def form_schema(model_id: str) -> dict:
    """Everything the chat form needs: defaults, limits, capabilities, status."""
    capabilities = capabilities_for(model_id)
    defaults = ChatOptions(model=model_id)
    thinking = capabilities.thinking_default
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
            "thinking": thinking,
            "effort": defaults.effort,
            "web_search": defaults.web_search,
            "web_search_type": capabilities.web_search_type or defaults.web_search_type,
            "allowed_callers": defaults.allowed_callers,
            "response_inclusion": defaults.response_inclusion,
            "max_uses": defaults.max_uses,
            "cache_control": defaults.cache_control,
        },
        "transport": {
            "sdk_method": plugin_transport(llm.get_model(resolve_model_id(model_id))),
            "streaming_note": STREAMING_TRANSPORT_NOTE,
        },
        "controls": {
            "model": {"status": EDITABLE},
            "max_tokens": {
                "status": EDITABLE,
                "min": 1,
                "max": capabilities.max_output_tokens,
                "min_by_thinking": {
                    THINKING_ON: capabilities.min_max_tokens(THINKING_ON),
                    THINKING_OFF: capabilities.min_max_tokens(THINKING_OFF),
                },
                "note": (
                    f"at most {capabilities.max_output_tokens} for {capabilities.id}"
                    if capabilities.max_output_tokens
                    else "no output ceiling is known for this model"
                ),
            },
            "system": {"status": EDITABLE, "note": "omitted from the request when empty"},
            "thinking": _thinking_control(capabilities, thinking),
            "budget_tokens": _budget_control(capabilities, thinking),
            "effort": _effort_control(capabilities),
            "web_search": {"status": EDITABLE},
            "web_search_type": {
                "status": RUNTIME_FIXED,
                "value": capabilities.web_search_type,
                "options": _option_list(
                    WEB_SEARCH_TYPES,
                    disabled=tuple(
                        value
                        for value in WEB_SEARCH_TYPES
                        if value != capabilities.web_search_type
                    ),
                ),
                "note": "llm-anthropic derives the tool version from the model",
            },
            "allowed_callers": _allowed_callers_control(capabilities),
            "dynamic_filtering": {
                "status": (
                    UNSUPPORTED_BY_TOOL
                    if capabilities.dynamic_filtering == "not-supported"
                    else RUNTIME_FIXED
                ),
                "value": capabilities.dynamic_filtering,
                "note": {
                    "active": "the API default caller runs dynamic filtering",
                    "not-supported": (
                        f"{capabilities.web_search_type} has no dynamic "
                        "filtering; it is called directly"
                    ),
                    "off": "no web search tool is sent",
                }[capabilities.dynamic_filtering],
            },
            "response_inclusion": _response_inclusion_control(capabilities),
            "max_uses": {
                "status": EDITABLE,
                "note": "0 means unlimited and is expressed by omitting the field",
            },
            "cache_control": {
                "status": EDITABLE,
                "ttl": {
                    "status": PROVIDER_DEFAULT,
                    "value": "5m",
                    "editable": False,
                    "supported": capabilities.cache_ttl,
                    "note": (
                        "llm-anthropic hard-codes cache_control="
                        "{'type': 'ephemeral'} and exposes no TTL option, so the "
                        "form offers no TTL control"
                    ),
                },
            },
            "stream": {
                "status": RUNTIME_FIXED,
                "value": "ON",
                "editable": False,
                "sdk_method": "client.messages.stream(...)",
                "options": [
                    {"value": "ON", "disabled": False},
                    {
                        "value": "OFF",
                        "disabled": True,
                        "note": "API supported · not used by llm-anthropic",
                    },
                ],
                "note": STREAMING_TRANSPORT_NOTE,
            },
        },
        "context": context_state({}, capabilities).as_dict(),
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
    allowed_callers: str | None
    transport: str
    context: ContextState


class ChatSession:
    """One browser conversation backed by an ``llm.Conversation``."""

    def __init__(self, options: ChatOptions | None = None):
        self.options = options or ChatOptions()
        self.capabilities = capabilities_for(self.options.model)
        if self.options.thinking is None:
            self.options = replace(self.options, thinking=self.capabilities.thinking_default)
        if self.options.web_search_type is None:
            self.options = replace(
                self.options, web_search_type=self.capabilities.web_search_type
            )
        self.model = llm.get_model(resolve_model_id(self.options.model))
        self._check_thinking()
        self._check_effort()
        self._check_max_tokens()
        self._check_web_search()
        self.conversation = llm.Conversation(model=self.model)
        self._last_kwargs: dict = {}
        self._usage: dict | None = None

    def _check_thinking(self) -> None:
        thinking = self.options.thinking
        if thinking == THINKING_OFF and not self.capabilities.can_disable_thinking:
            raise UnsupportedOptionError(
                f"thinking cannot be turned off for {self.options.model}: "
                "both the API and llm-anthropic reject thinking="
                "{'type': 'disabled'} on this model"
            )
        if thinking == THINKING_ON and not self.capabilities.supports_thinking:
            raise UnsupportedOptionError(
                f"{self.options.model} has no thinking parameter"
            )

    def _check_effort(self) -> None:
        effort = self.options.effort
        if effort == DEFAULT_EFFORT:
            return
        if not self.capabilities.supports_effort:
            raise UnsupportedOptionError(
                f"{self.options.model} does not support effort"
            )
        if effort not in self.capabilities.effort_levels:
            raise ValueError(
                f"effort must be one of {self.capabilities.effort_levels} "
                f"for {self.options.model}"
            )
        if effort in self.capabilities.effort_disabled_with(self.options.thinking):
            raise ValueError(
                f"effort={effort} is rejected when thinking is off: Anthropic "
                "accepts thinking={'type': 'disabled'} only at "
                + self._highest_effort_without_thinking()
                + " or below"
            )

    def _highest_effort_without_thinking(self) -> str:
        allowed = [
            level
            for level in self.capabilities.effort_levels
            if level not in EFFORT_LEVELS_NEEDING_THINKING
        ]
        return allowed[-1] if allowed else "high"

    def _check_max_tokens(self) -> None:
        ceiling = self.capabilities.max_output_tokens
        if ceiling and self.options.max_tokens > ceiling:
            raise ValueError(
                f"max_tokens must be at most {ceiling} for {self.options.model}"
            )
        minimum = self.capabilities.min_max_tokens(self.options.thinking)
        if self.options.max_tokens < minimum:
            raise ValueError(
                f"max_tokens must be greater than the thinking budget "
                f"({self.capabilities.budget_tokens}) for {self.options.model}: "
                "Anthropic requires budget_tokens to be smaller than max_tokens"
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
        # Thinking and effort are separate controls and separate fields.
        if self.options.thinking == THINKING_ON:
            options["thinking"] = True
        elif self.capabilities.thinking_off_request == "disabled":
            # Adaptive models: the only legal way to stop thinking.
            options["thinking"] = False
        # "omitted": leave the option off, which is the model's own default.
        if self.options.effort != DEFAULT_EFFORT:
            options["thinking_effort"] = self.options.effort
        return options

    def _transport(self) -> str:
        """The SDK method the installed plugin will really call."""
        return plugin_transport(self.model)

    def context(self) -> ContextState:
        """The context figure for the state this session is in."""
        return context_state(
            self._last_kwargs,
            self.capabilities,
            self._usage,
            reserved=self.options.max_tokens,
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
        self._last_kwargs = kwargs
        before = context_state(
            kwargs, self.capabilities, usage=None, reserved=self.options.max_tokens
        )
        if before.fits is False:
            raise ValueError(
                f"this request does not fit: about {before.tokens} input tokens "
                f"(estimated) plus {self.options.max_tokens} reserved for the "
                f"reply exceeds the {before.limit} token context window. Lower "
                "max_tokens or start a new conversation - this UI never trims "
                "the history for you"
            )
        transport = self._transport()
        return PreparedTurn(
            response=response,
            kwargs=kwargs,
            code=render_kwargs(kwargs, transport=transport, note=STREAMING_TRANSPORT_NOTE),
            dynamic_filtering=dynamic_filtering_state(kwargs),
            allowed_callers=effective_allowed_callers(kwargs),
            transport=transport,
            context=before,
        )

    def _require_key(self) -> None:
        """Fail locally rather than send a request nobody authorised."""
        try:
            self.model.get_key()
        except llm.NeedsKeyException as ex:
            raise MissingKeyError(str(ex)) from ex

    def _record_usage(self, response: Any) -> None:
        """Prefer the API's own token counts once they exist."""
        try:
            usage = response.usage()
        except Exception:  # noqa: BLE001 - an unknown usage shape is not fatal
            return
        self._usage = {
            "input": getattr(usage, "input", None),
            "output": getattr(usage, "output", None),
        }

    def stream_turn(self, text: str) -> Iterator[dict]:
        """Yield events: the prepared request, then text, then the final text.

        There is no buffered mode. ``llm-anthropic`` always opens a stream, so
        the chunks it produces are the only honest thing to show.
        """
        prepared = self.prepare(text)
        self._require_key()
        yield {
            "type": "prepared",
            "code": prepared.code,
            "kwargs": prepared.kwargs,
            "dynamic_filtering": prepared.dynamic_filtering,
            "allowed_callers": prepared.allowed_callers,
            "transport": prepared.transport,
            "context": prepared.context.as_dict(),
        }
        for event in prepared.response.stream_events():
            if event.type == "text":
                yield {"type": "text", "text": event.chunk}
        self._record_usage(prepared.response)
        yield {
            "type": "done",
            "text": final_message_text(prepared.response),
            "context": self.context().as_dict(),
        }

    def run_turn(self, text: str) -> dict:
        events = list(self.stream_turn(text))
        return {
            "text": events[-1]["text"],
            "code": events[0]["code"],
            "kwargs": events[0]["kwargs"],
            "dynamic_filtering": events[0]["dynamic_filtering"],
            "allowed_callers": events[0]["allowed_callers"],
            "transport": events[0]["transport"],
            "context": events[-1]["context"],
            "chunks": [event["text"] for event in events if event["type"] == "text"],
        }
