"""Anthropic through ``llm-anthropic``, as the first provider behind the seam.

Nothing here is new behaviour. It is the assembly this project has always
done, written down in one place now that it is no longer the only one:
``llm-anthropic`` builds the whole request in ``build_kwargs()``, and it always
opens ``client.messages.stream()`` because the API rejects non-streaming
requests whose ``max_tokens`` could run past ten minutes.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, replace
from typing import Any

from ..capabilities import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    EFFORT_LEVELS_NEEDING_THINKING,
    THINKING_OFF,
    THINKING_ON,
    THINKING_STATES,
    ModelCapabilities,
    capabilities_for,
    fallback_models,
    model_catalog,
    profile,
    provenance,
    resolve_model_id,
)
from ..turn import (
    EDITABLE,
    PROVIDER_DEFAULT,
    RUNTIME_FIXED,
    UNSUPPORTED_BY_MODEL,
    UNSUPPORTED_BY_TOOL,
    TurnOptions,
    UnsupportedOptionError,
    estimate_text,
    option_list,
)
from .base import Transport, first_int, system_as_sent

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
# Shown as the status of the greyed-out stream control and in /api/form. It is
# deliberately NOT rendered into the code pane: that pane is the request, and a
# request carries no commentary.
STREAMING_TRANSPORT_NOTE = (
    "llm-anthropic always opens client.messages.stream(): the Anthropic API "
    "rejects non-streaming requests whose max_tokens could run past ten "
    "minutes. Streaming is therefore fixed by the runtime, not chosen here."
)

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


@dataclass(frozen=True)
class AnthropicOptions(TurnOptions):
    """Every value the Anthropic chat form can set.

    These defaults are the defaults the form shows. There is intentionally no
    second copy of them in the template: the UI reads them from the form
    schema.
    """

    model: str = DEFAULT_MODEL
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
        super().__post_init__()
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


def _build_web_search_tool(
    model, options: AnthropicOptions, capabilities: ModelCapabilities
):
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
    # the bare tool, and the API default caller applies. The only value that
    # can honestly reach this point is the caller that will really be in
    # effect for the selected model; anything else would be a second, fake
    # request, so it is refused.
    if options.allowed_callers != capabilities.effective_allowed_callers:
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


def _verify(
    options: AnthropicOptions, kwargs: dict, capabilities: ModelCapabilities
) -> None:
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
    options: AnthropicOptions, kwargs: dict, capabilities: ModelCapabilities
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


def _verify_effort(options: AnthropicOptions, kwargs: dict) -> None:
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


def _verify_cache(options: AnthropicOptions, kwargs: dict) -> None:
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
            total += estimate_text(content)
            continue
        if not isinstance(content, list):
            return None
        for block in content:
            if not isinstance(block, dict):
                return None
            block_type = block.get("type")
            if block_type == "text":
                total += estimate_text(block.get("text", ""))
            elif block_type in ("thinking", "redacted_thinking"):
                total += estimate_text(
                    block.get("thinking") or block.get("data") or ""
                )
            elif block_type in ("tool_use", "server_tool_use"):
                total += estimate_text(json.dumps(block.get("input", {})))
            elif block_type.endswith("_tool_result"):
                total += estimate_text(json.dumps(block.get("content", "")))
            else:
                return None
    total += estimate_text(kwargs.get("system") or "")
    total += estimate_text(json.dumps(kwargs.get("tools", [])))
    return total


# --- the form -----------------------------------------------------------
#
# One control per API field, each saying whether it may be edited and why
# not when it may not. These live here rather than beside the shared form
# skeleton because every one of them is a fact about Anthropic's API or
# about llm-anthropic, and a form assembled from another provider's
# vocabulary is how a page ends up offering a control that does nothing.


def _thinking_control(capabilities: ModelCapabilities, thinking: str) -> dict:
    if not capabilities.supports_thinking:
        return {
            "status": UNSUPPORTED_BY_MODEL,
            "value": THINKING_OFF,
            "options": option_list(THINKING_STATES, disabled=THINKING_STATES),
            "note": "this model has no thinking parameter",
        }
    if not capabilities.thinking_editable:
        return {
            "status": RUNTIME_FIXED,
            "value": THINKING_ON,
            "options": option_list(THINKING_STATES, disabled=(THINKING_OFF,)),
            "note": (
                "this model always thinks, and both the API and llm-anthropic "
                "reject thinking={'type': 'disabled'}"
            ),
        }
    return {
        "status": EDITABLE,
        "value": thinking,
        "options": option_list(THINKING_STATES, disabled=()),
        "note": (
            "ON sends thinking={'type': 'adaptive'}"
            if capabilities.thinking_mode == "adaptive"
            else "ON sends thinking={'type': 'enabled'} with the budget below"
        ),
    }


def _effort_control(capabilities: ModelCapabilities) -> dict:
    if not capabilities.supports_effort:
        return {
            "status": UNSUPPORTED_BY_MODEL,
            "value": DEFAULT_EFFORT,
            "options": option_list(capabilities.effort_options(), disabled=()),
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

class AnthropicProvider:
    """Claude through ``llm-anthropic`` and Anthropic's official Python SDK."""

    id = "anthropic"
    label = "Anthropic"
    # Which SDK the rendered code is for. Named rather than assumed, because
    # the page says it out loud above the pane.
    sdk = "anthropic-python"
    # messages.count_tokens is Anthropic's own, free, and the reason the
    # context figure can be exact before a turn is sent.
    counts_tokens = True
    # The counts that occupy context, and they add up because Anthropic
    # reports the uncached input, the writes and the reads as three separate
    # counts of disjoint tokens. The cached prefix occupies context exactly
    # like fresh tokens do, so leaving the cache out under-reports the moment
    # caching starts working.
    context_counts = ("input", "cache_creation", "cache_read", "output")

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

    def unavailable(self, model_id: str, error: Exception) -> Exception:
        """``llm-anthropic`` registers its models with or without a key, so an
        id it does not know really is one this tool version cannot send."""
        return error

    def stop_reason(self, message: dict) -> str | None:
        """Anthropic's own word for it, on the Message where it already sits."""
        return message.get("stop_reason")

    def blocks(self, message: dict) -> list[dict]:
        """Anthropic's Message already is a list of typed blocks.

        This is the shape the record's vocabulary was taken from, so there is
        nothing to translate - only the check that a list is what arrived.
        """
        content = message.get("content")
        return list(content) if isinstance(content, list) else []

    # ---- what the model can do -------------------------------------------

    def resolve_model_id(self, model_id: str) -> str:
        return resolve_model_id(model_id)

    def capabilities_for(self, model_id: str) -> ModelCapabilities:
        return capabilities_for(model_id)

    def catalog(self) -> dict:
        return model_catalog()

    # ---- the turn --------------------------------------------------------

    def options_from(self, payload: dict) -> AnthropicOptions:
        """The chat form, mapped onto this provider's options.

        Unknown or malformed values raise, and the routes turn that into a
        400, so a bad form never falls back to a different request than the
        one the page was showing.
        """
        defaults = AnthropicOptions()
        thinking = payload.get("thinking", defaults.thinking)
        return AnthropicOptions(
            model=payload.get("model", defaults.model),
            max_tokens=int(payload.get("max_tokens", defaults.max_tokens)),
            system=system_as_sent(payload.get("system", defaults.system)),
            # None means "the official default for this model"; anything else
            # is validated against the model's real thinking capability.
            thinking=str(thinking) if thinking is not None else None,
            # The form always sends the effort key; an empty value is
            # "nothing selected" (a greyed-out select after a model switch),
            # i.e. the default, never a level to validate against the model.
            effort=str(payload.get("effort") or defaults.effort),
            web_search=bool(payload.get("web_search", defaults.web_search)),
            web_search_type=payload.get("web_search_type", defaults.web_search_type),
            allowed_callers=payload.get("allowed_callers", defaults.allowed_callers),
            response_inclusion=payload.get(
                "response_inclusion", defaults.response_inclusion
            ),
            max_uses=int(payload.get("max_uses", defaults.max_uses)),
            cache_control=bool(payload.get("cache_control", defaults.cache_control)),
        )

    def accept(
        self, options: AnthropicOptions, capabilities: ModelCapabilities
    ) -> AnthropicOptions:
        """These options, resolved - or a refusal saying which one and why.

        Resolution first, then the checks, and in that order for a reason:
        "use this model's default" arrives as ``None``, and a check run
        against ``None`` would read it as "thinking is off" and approve a
        request that disables thinking on a model that is always thinking.
        """
        settled = replace(
            options,
            thinking=(
                options.thinking
                if options.thinking is not None
                else capabilities.thinking_default
            ),
            web_search_type=(
                options.web_search_type
                if options.web_search_type is not None
                else capabilities.web_search_type
            ),
            # The default caller is the newer tool's API default; a model whose
            # tool version is always called directly gets its own fact instead.
            allowed_callers=(
                capabilities.effective_allowed_callers
                if options.allowed_callers == DYNAMIC_FILTERING_CALLER
                and capabilities.effective_allowed_callers != DYNAMIC_FILTERING_CALLER
                else options.allowed_callers
            ),
        )
        _check_thinking(settled, capabilities)
        _check_effort(settled, capabilities)
        _check_max_tokens(settled, capabilities)
        _check_web_search(settled, capabilities)
        return settled

    def plugin_options(
        self, options: AnthropicOptions, capabilities: ModelCapabilities
    ) -> dict:
        built: dict[str, Any] = {
            "max_tokens": options.max_tokens,
            "cache": options.cache_control,
        }
        # Thinking and effort are separate controls and separate fields.
        if options.thinking == THINKING_ON:
            built["thinking"] = True
        elif capabilities.thinking_off_request == "disabled":
            # Adaptive models: the only legal way to stop thinking.
            built["thinking"] = False
        # "omitted": leave the option off, which is the model's own default.
        if options.effort != DEFAULT_EFFORT:
            built["thinking_effort"] = options.effort
        return built

    def tools(
        self, model: Any, options: AnthropicOptions, capabilities: ModelCapabilities
    ) -> list:
        if not options.web_search:
            return []
        return [_build_web_search_tool(model, options, capabilities)]

    def verify(
        self, options: AnthropicOptions, kwargs: dict, capabilities: ModelCapabilities
    ) -> None:
        _verify(options, kwargs, capabilities)

    # ---- the form ---------------------------------------------------------

    default_model = DEFAULT_MODEL

    def form(self, capabilities: ModelCapabilities, transport: Transport) -> dict:
        """This provider's own half of the chat form.

        The shared half is three controls - a model, a reply ceiling and a
        system prompt - and everything below is Anthropic's vocabulary:
        thinking and effort are separate API fields, the web search tool
        version is derived from the model, and prompt caching is a parameter
        that only this API has. Assembling it here is what lets the page ask
        one question and get a form the selected model will actually accept.
        """
        defaults = AnthropicOptions(model=capabilities.id)
        thinking = capabilities.thinking_default
        fallback = profile()
        return {
            "model": {
                "sends": capabilities.api_model_id,
                "web_search_type": capabilities.web_search_type,
            },
            # Where the numbers on this page came from: the Models API when it
            # answered, the labelled fallback profile when it did not.
            "model_data": {
                **provenance(),
                "profile_version": fallback["profile_version"],
                "profile_source": fallback["profile_source"],
                "fallback_models": list(fallback_models()),
            },
            "defaults": {
                "thinking": thinking,
                "effort": defaults.effort,
                "web_search": defaults.web_search,
                "web_search_type": (
                    capabilities.web_search_type or defaults.web_search_type
                ),
                "allowed_callers": defaults.allowed_callers,
                "response_inclusion": defaults.response_inclusion,
                "max_uses": defaults.max_uses,
                "cache_control": defaults.cache_control,
            },
            "transport": {"streaming_note": STREAMING_TRANSPORT_NOTE},
            "controls": {
                "max_tokens": {
                    "min_by_thinking": {
                        THINKING_ON: capabilities.min_max_tokens(THINKING_ON),
                        THINKING_OFF: capabilities.min_max_tokens(THINKING_OFF),
                    }
                },
                "thinking": _thinking_control(capabilities, thinking),
                "budget_tokens": _budget_control(capabilities, thinking),
                "effort": _effort_control(capabilities),
                "web_search": {"status": EDITABLE},
                "web_search_type": {
                    "status": RUNTIME_FIXED,
                    "value": capabilities.web_search_type,
                    "options": option_list(
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
                            "{'type': 'ephemeral'} and exposes no TTL option, so "
                            "the form offers no TTL control"
                        ),
                    },
                },
                "stream": {
                    "status": RUNTIME_FIXED,
                    "value": "ON",
                    "editable": False,
                    "sdk_method": f"{transport.call}(...)",
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
            # Vocabularies the page needs whole, to label a value it did not
            # pick itself.
            "vocabulary": {
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
                        "llm-anthropic hard-codes cache_control="
                        "{'type': 'ephemeral'} and exposes no TTL option, so the "
                        "form offers no TTL control."
                    ),
                },
            },
        }

    def request_facts(self, kwargs: dict) -> dict:
        return {
            "dynamic_filtering": dynamic_filtering_state(kwargs),
            "allowed_callers": effective_allowed_callers(kwargs),
        }

    def estimate_input_tokens(self, kwargs: dict) -> int | None:
        return estimate_request_tokens(kwargs)

    def usage_from(self, document: dict) -> dict:
        """The provider's own counts, under the names this app reports.

        ``input_tokens`` is the **uncached** part of the prompt; the cache
        writes and reads are counted beside it, not inside it. That is what
        makes :attr:`context_counts` a sum of all four here and only two on
        OpenRouter, where the prompt arrives whole.

        Read off the document rather than off llm's ``token_details``, which
        is the same object after ``simplify_usage_dict`` has deleted every
        zero and every dict that flattened to nothing - a turn that reported
        no cache activity and a turn that reported nothing at all come out of
        it identical.
        """
        return {
            "input": first_int(document, "input_tokens"),
            "output": first_int(document, "output_tokens"),
            "cache_creation": first_int(document, "cache_creation_input_tokens"),
            "cache_read": first_int(document, "cache_read_input_tokens"),
        }


def _check_thinking(options: AnthropicOptions, capabilities: ModelCapabilities) -> None:
    thinking = options.thinking
    if thinking == THINKING_OFF and not capabilities.can_disable_thinking:
        raise UnsupportedOptionError(
            f"thinking cannot be turned off for {options.model}: "
            "both the API and llm-anthropic reject thinking="
            "{'type': 'disabled'} on this model"
        )
    if thinking == THINKING_ON and not capabilities.supports_thinking:
        raise UnsupportedOptionError(f"{options.model} has no thinking parameter")


def _check_effort(options: AnthropicOptions, capabilities: ModelCapabilities) -> None:
    effort = options.effort
    if effort == DEFAULT_EFFORT:
        return
    if not capabilities.supports_effort:
        raise UnsupportedOptionError(f"{options.model} does not support effort")
    if effort not in capabilities.effort_levels:
        raise ValueError(
            f"effort must be one of {capabilities.effort_levels} for {options.model}"
        )
    if effort in capabilities.effort_disabled_with(options.thinking):
        raise ValueError(
            f"effort={effort} is rejected when thinking is off: Anthropic "
            "accepts thinking={'type': 'disabled'} only at "
            + _highest_effort_without_thinking(capabilities)
            + " or below"
        )


def _highest_effort_without_thinking(capabilities: ModelCapabilities) -> str:
    allowed = [
        level
        for level in capabilities.effort_levels
        if level not in EFFORT_LEVELS_NEEDING_THINKING
    ]
    return allowed[-1] if allowed else "high"


def _check_max_tokens(options: AnthropicOptions, capabilities: ModelCapabilities) -> None:
    ceiling = capabilities.max_output_tokens
    if ceiling and options.max_tokens > ceiling:
        raise ValueError(f"max_tokens must be at most {ceiling} for {options.model}")
    minimum = capabilities.min_max_tokens(options.thinking)
    if options.max_tokens < minimum:
        raise ValueError(
            f"max_tokens must be greater than the thinking budget "
            f"({capabilities.budget_tokens}) for {options.model}: "
            "Anthropic requires budget_tokens to be smaller than max_tokens"
        )


def _check_web_search(options: AnthropicOptions, capabilities: ModelCapabilities) -> None:
    if options.web_search and not capabilities.supports_web_search:
        raise UnsupportedOptionError(
            f"{options.model} does not support the web search tool"
        )
