"""The real conversation path, for whichever provider is sending it.

    Browser -> native-api-chat -> LLM Python API -> a provider plugin -> its SDK

This module deliberately owns no request-shaping logic of its own, and since
there has been more than one provider it owns no option-shaping logic either.
What is left here is the part that is genuinely the same either way: run an
``llm.Conversation``, build the turn without sending it, measure the context,
stream it, and record it once.

Everything that differs between providers is asked of the provider -
capabilities, which options exist, what reaches the plugin, which tools are
built, and what the built request is checked against. The session never
branches on a model id or a provider name; if something here needed to know
which provider it was talking to, that would be the seam leaking.

The form values are *requests*, not facts: every one of them is checked
against the request that was actually built before a turn is allowed to
stream. When the installed plugin cannot express a value the form asks for,
the turn is refused instead of quietly sending something else.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

import llm

from .capabilities import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    THINKING_OFF,
    THINKING_ON,
    THINKING_STATES,
    ModelCapabilities,
    capabilities_for,
    fallback_models,
    profile,
    provenance,
)
from .codegen import render_kwargs
from .providers import Transport, provider_for
from .providers.anthropic import (
    ALLOWED_CALLERS,
    DYNAMIC_FILTERING_CALLER,
    RESPONSE_INCLUSIONS,
    STREAMING_TRANSPORT_NOTE,
    WEB_SEARCH_TYPES,
    AnthropicOptions,
    dynamic_filtering_state,
    effective_allowed_callers,
)
from .records import TurnRecord, build_response_view
from .turn import (
    EDITABLE,
    FALLBACK_DATA,
    PROVIDER_DEFAULT,
    RUNTIME_FIXED,
    UNSUPPORTED_BY_MODEL,
    UNSUPPORTED_BY_TOOL,
    MissingKeyError,
    TurnOptions,
    UnsupportedOptionError,
    estimate_text,
    option_list,
)

# The Anthropic option set, under the name it has had since this project had
# only one provider. The form work that gives each provider its own controls
# renames it at the call sites; until then this keeps one spelling in use.
ChatOptions = AnthropicOptions

# How the context figure is named, in the order it is trusted. The counter and
# the usage object are the provider's own numbers; the estimate is ours, and
# says so in the one word the page checks.
COUNTED = "API count"
USAGE = "API usage"
USAGE_WITH_ESTIMATE = "API usage + estimated draft"
ESTIMATED = "estimated"
UNKNOWN = "unknown"

__all__ = [
    "ALLOWED_CALLERS",
    "COUNTED",
    "EDITABLE",
    "ESTIMATED",
    "FALLBACK_DATA",
    "PROVIDER_DEFAULT",
    "RESPONSE_INCLUSIONS",
    "RUNTIME_FIXED",
    "STREAMING_TRANSPORT_NOTE",
    "UNKNOWN",
    "UNSUPPORTED_BY_MODEL",
    "UNSUPPORTED_BY_TOOL",
    "USAGE",
    "USAGE_WITH_ESTIMATE",
    "WEB_SEARCH_TYPES",
    "AnthropicOptions",
    "ChatOptions",
    "ChatSession",
    "ContextState",
    "MissingKeyError",
    "PreparedTurn",
    "TurnOptions",
    "UnsupportedOptionError",
    "context_state",
    "dynamic_filtering_state",
    "effective_allowed_callers",
    "final_message_text",
    "form_schema",
]


# --- context ----------------------------------------------------------------


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
    # True while the API's counter is being asked about this exact request, so
    # the page knows to look again instead of keeping a fallback figure.
    pending: bool = False

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
            "pending": self.pending,
        }


def context_state(
    kwargs: dict,
    capabilities: ModelCapabilities,
    usage: dict | None = None,
    reserved: int = 0,
    counted: int | None = None,
    draft: str = "",
) -> ContextState:
    """Build the context figure from the best source that can answer.

    In order: the API's own count of this exact request, the API's own usage
    for the conversation so far, and only then a labelled estimate. The
    estimate is the only one of the three that cannot see what the provider
    adds for a server tool, which is why it is never preferred.

    ``draft`` is the text that is not yet in ``usage`` - the message being
    typed. It is estimated and added when there is no count to replace both.

    Which provider's request shape this is comes from the capability record's
    own model id rather than from an argument, so a caller cannot hand the
    Anthropic estimator an OpenRouter request and be told a number for it.
    """
    limit = capabilities.context_window
    if counted is not None:
        tokens = counted
        source = COUNTED
    elif usage is not None:
        # Total input = uncached + cache write + cache read (the API's own
        # accounting); the cached prefix occupies context exactly like fresh
        # tokens do. The reply counts too: it is part of the next request.
        tokens = (
            (usage.get("input") or 0)
            + (usage.get("cache_creation") or 0)
            + (usage.get("cache_read") or 0)
            + (usage.get("output") or 0)
        )
        added = estimate_text(draft)
        tokens += added
        source = USAGE_WITH_ESTIMATE if added else USAGE
    else:
        tokens = provider_for(capabilities.id).estimate_input_tokens(kwargs)
        source = ESTIMATED if tokens is not None else UNKNOWN

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
        # The record says how sure it is of its own ceiling. Deciding that
        # here would mean knowing each provider's data sources by name, which
        # is exactly the branch the seam exists to delete.
        limit_data_source=capabilities.limit_status,
        percent=percent,
        reserved=reserved,
        fits=fits,
    )


def final_message_text(response: Any, provider: Any = None) -> str:
    """Assemble the reply from the final accumulated Message.

    Streaming yields fragments; the one thing that is safe to show and to keep
    is the Message the provider finished with, so both panes and the recorded
    conversation agree on it.

    Which shape that Message has is the sending provider's business. Reading it
    here without asking assumed Anthropic's typed blocks, which on OpenRouter's
    Chat Completions path is a plain string - and iterating a string as blocks
    raises on the first character.
    """
    message = getattr(response, "response_json", None) or {}
    blocks = provider.blocks(message) if provider else _anthropic_blocks(message)
    text = "".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    )
    return text or response.text_or_raise()


def _anthropic_blocks(message: dict) -> list[dict]:
    """The shape this project read before it had more than one provider."""
    content = message.get("content")
    return list(content) if isinstance(content, list) else []


# --- the form surface --------------------------------------------------------


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


def form_schema(model_id: str) -> dict:
    """Everything the chat form needs: defaults, limits, capabilities, status."""
    provider = provider_for(model_id)
    capabilities = capabilities_for(model_id)
    defaults = ChatOptions(model=model_id)
    thinking = capabilities.thinking_default
    fallback = profile()
    # Read once, from the provider, so the transport named in the schema and
    # the one the streaming control describes cannot be two different facts.
    transport = provider.transport(
        llm.get_model(provider.resolve_model_id(model_id)), defaults
    )
    catalog = provider.catalog()
    return {
        "models": catalog["models"],
        # What the narrowing left out. Hidden models are still listed here so
        # the page can say so: a short list that silently dropped nine models
        # would read like lost data rather than like a rule.
        "superseded": catalog["superseded"],
        "series_rule": catalog["rule"],
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
            "sdk_method": transport.name,
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
                        "{'type': 'ephemeral'} and exposes no TTL option, so the "
                        "form offers no TTL control"
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
    transport: Transport
    context: ContextState
    # What this provider can say about the request that the form cannot -
    # Anthropic's dynamic filtering and effective caller, OpenRouter's
    # reasoning block and routing. A dict rather than named fields, because
    # naming them here would put one provider's vocabulary in the shared type.
    facts: dict


class ChatSession:
    """One conversation, through whichever provider owns its model."""

    def __init__(
        self,
        options: TurnOptions | None = None,
        conversation_id: str | None = None,
        history: list[Any] | None = None,
        baseline_usage: dict | None = None,
    ):
        self.options = options or ChatOptions()
        self.provider = provider_for(self.options.model)
        self.capabilities = self.provider.capabilities_for(self.options.model)
        # Resolution and the checks in one call, and in that order: "use this
        # model's default" arrives as None, and a check run against None
        # would read it as a value the user chose.
        self.options = self.provider.accept(self.options, self.capabilities)
        self.model = self._get_model(self.provider.resolve_model_id(self.options.model))
        self.conversation = self._new_conversation(conversation_id, history)
        self.last_response: Any = None
        self._last_kwargs: dict = {}
        # The provider's counts for the last turn of this conversation, when
        # the caller already knows them (a stored conversation being reopened).
        # It is the same field a finished turn writes to, so the context figure
        # has one baseline whether the turn was sent a second ago or last week.
        self._usage: dict | None = baseline_usage

    def _get_model(self, resolved_id: str) -> Any:
        """llm's model, or the real reason there is not one.

        "Unknown model" is the wrong answer often enough to be worth asking
        about: ``llm-openrouter`` registers nothing at all without a key, so a
        keyless machine reports every OpenRouter model as unknown. The model is
        not unknown, the key is missing, and only one of those the user can
        act on.
        """
        try:
            return llm.get_model(resolved_id)
        except llm.UnknownModelError as ex:
            raise self.provider.unavailable(self.options.model, ex) from ex

    def _new_conversation(
        self, conversation_id: str | None, history: list[Any] | None
    ) -> Any:
        """A conversation identified by llm's own ULID.

        Reusing the id means this app does not mint a second identifier space,
        and passing the stored messages back in means a resumed conversation
        carries the original content blocks rather than history rebuilt from
        chat text.
        """
        extra: dict[str, Any] = {}
        if conversation_id:
            extra["id"] = conversation_id
        if history:
            extra["loaded_messages"] = list(history)
        return llm.Conversation(model=self.model, **extra)

    @property
    def conversation_id(self) -> str:
        return self.conversation.id

    def update_options(self, options: TurnOptions) -> None:
        """Apply a form change to a session that already holds history.

        It goes through the same resolution and the same checks as a fresh
        session, and a refusal leaves the session exactly as it was: a
        half-applied form would build a request neither the page nor the user
        asked for.
        """
        previous = self.options
        try:
            self.options = self.provider.accept(options, self.capabilities)
        except Exception:
            self.options = previous
            raise

    def _transport(self) -> Transport:
        """The SDK call this turn will really make."""
        return self.provider.transport(self.model, self.options)

    def context(self) -> ContextState:
        """The context figure for the state this session is in."""
        return context_state(
            self._last_kwargs,
            self.capabilities,
            self._usage,
            reserved=self.options.max_tokens,
        )

    @property
    def baseline_usage(self) -> dict | None:
        """The provider's own counts for the last turn in this conversation.

        None until a turn has been sent here, or until the conversation was
        opened with the counts it was stored with. It is what makes the context
        figure exact the moment a stored conversation is reopened.
        """
        return self._usage

    def measure(
        self, kwargs: dict, text: str = "", counted: int | None = None
    ) -> ContextState:
        """The context figure for a prepared request.

        The API's own count of this exact request wins; then its own usage for
        this conversation plus the message being typed; then a labelled
        estimate. :meth:`prepare` measures through this method too, so the
        figure under the prompt and the figure inside the fit check cannot
        disagree with each other.
        """
        return context_state(
            kwargs,
            self.capabilities,
            usage=self._usage,
            reserved=self.options.max_tokens,
            counted=counted,
            draft=text,
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
            options=self.provider.plugin_options(self.options, self.capabilities),
            tools=self.provider.tools(self.model, self.options, self.capabilities),
            stream=True,
        )
        kwargs = self.provider.assemble(
            self.model, response.prompt, self.conversation, self.options
        )
        self.provider.verify(self.options, kwargs, self.capabilities)
        self._last_kwargs = kwargs
        before = self.measure(kwargs, text)
        if before.fits is False:
            raise ValueError(
                f"this request does not fit: about {before.tokens} input tokens "
                f"({before.source}) plus {self.options.max_tokens} reserved for the "
                f"reply exceeds the {before.limit} token context window. Lower "
                "max_tokens or start a new conversation - this UI never trims "
                "the history for you"
            )
        transport = self._transport()
        return PreparedTurn(
            response=response,
            kwargs=kwargs,
            code=render_kwargs(kwargs, transport=transport),
            transport=transport,
            context=before,
            facts=self.provider.request_facts(kwargs),
        )

    def _require_key(self) -> None:
        """Fail locally rather than send a request nobody authorised."""
        try:
            self.model.get_key()
        except llm.NeedsKeyException as ex:
            raise MissingKeyError(str(ex)) from ex

    def _record_usage(self, response: Any) -> None:
        """Prefer the provider's own token counts once they exist.

        Which details hold the cache counters is the provider's business:
        ``llm`` reports uncached input and output, and each plugin keeps the
        rest under its own names.
        """
        try:
            usage = response.usage()
        except Exception:  # noqa: BLE001 - an unknown usage shape is not fatal
            return
        details = getattr(response, "token_details", None)
        if not isinstance(details, dict):
            details = {}
        self._usage = self.provider.usage_from(usage, details)

    def stream_turn(self, text: str) -> Iterator[dict]:
        """Yield events: the prepared request, then text, then the final text.

        There is no buffered mode. ``llm-anthropic`` always opens a stream, so
        the chunks it produces are the only honest thing to show. Reasoning
        chunks travel as their own event type: the chat presents them in a
        collapsible thinking strip rather than mixing them into the answer.
        """
        prepared = self.prepare(text)
        self._require_key()
        yield {
            "type": "prepared",
            "code": prepared.code,
            "kwargs": prepared.kwargs,
            "transport": prepared.transport.name,
            "context": prepared.context.as_dict(),
            **prepared.facts,
        }
        # Time to the first text chunk, measured client-side. llm records
        # the whole-call duration on the response; this is the "how long
        # until words appear" half of the latency story.
        started = time.monotonic()
        ttft_ms: int | None = None
        for event in prepared.response.stream_events():
            if event.type == "text":
                if ttft_ms is None:
                    ttft_ms = int((time.monotonic() - started) * 1000)
                yield {"type": "text", "text": event.chunk}
            elif event.type == "reasoning" and event.chunk:
                # Signature and redacted markers arrive as empty reasoning
                # chunks; only text is worth an event.
                yield {"type": "reasoning", "text": event.chunk}
        self._record_usage(prepared.response)
        # Only now does a turn exist: before the stream ends there is no
        # complete Message, so there is nothing to look at or to store.
        self.last_response = prepared.response
        record = self.record(text, prepared, ttft_ms=ttft_ms)
        yield {
            "type": "record",
            "record": record.as_dict(),
        }
        yield {
            "type": "done",
            "text": final_message_text(prepared.response, self.provider),
            # The record's own reading of the finished Message, so the
            # streamed reasoning preview and the stored thinking can never
            # disagree about what the model said.
            "thinking": record.response.thinking,
            "context": self.context().as_dict(),
        }

    def record(self, text: str, prepared: PreparedTurn, ttft_ms: int | None = None) -> TurnRecord:
        """The one object every other view of this turn is derived from."""
        return TurnRecord(
            conversation_id=self.conversation_id,
            turn_id=getattr(prepared.response, "id", None),
            user_input=text,
            options=asdict(self.options),
            request_kwargs=prepared.kwargs,
            rendered_code=prepared.code,
            response=build_response_view(
                prepared.response,
                ttft_ms=ttft_ms,
                blocks=self.provider.blocks(
                    getattr(prepared.response, "response_json", None) or {}
                ),
            ),
            context=self.context().as_dict(),
        )

    def run_turn(self, text: str) -> dict:
        events = list(self.stream_turn(text))
        record = next(event["record"] for event in events if event["type"] == "record")
        prepared = events[0]
        return {
            "text": events[-1]["text"],
            "code": prepared["code"],
            "kwargs": prepared["kwargs"],
            "transport": prepared["transport"],
            "context": events[-1]["context"],
            "chunks": [event["text"] for event in events if event["type"] == "text"],
            "record": record,
            # Whatever this provider reported about the request. Listing the
            # keys here would make the shared path know one provider's words.
            **{
                key: value
                for key, value in prepared.items()
                if key not in ("type", "code", "kwargs", "transport", "context")
            },
        }
