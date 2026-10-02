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

from .capabilities import ModelCapabilities
from .codegen import render_kwargs
from .providers import PROVIDERS, Transport, provider_for
from .providers.anthropic import (
    ALLOWED_CALLERS,
    RESPONSE_INCLUSIONS,
    STREAMING_TRANSPORT_NOTE,
    WEB_SEARCH_TYPES,
    AnthropicOptions,
    dynamic_filtering_state,
    effective_allowed_callers,
)
from .records import TurnRecord, build_response_view, usage_document
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
    The same id decides how its counters are added up, and that is not
    pedantry: Anthropic counts the uncached input beside its cache counters,
    while OpenRouter counts a prompt that already contains them, so one sum
    for both would add a cached slice to itself on one provider's turns and
    report a conversation as fuller than it is.
    """
    provider = provider_for(capabilities.id)
    limit = capabilities.context_window
    if counted is not None:
        tokens = counted
        source = COUNTED
    elif usage is not None:
        # The counts that occupy context, in the sending provider's own
        # arithmetic, plus the reply - which becomes part of the next
        # request. Which those are is the provider's list to name.
        tokens = sum(usage.get(name) or 0 for name in provider.context_counts)
        added = estimate_text(draft)
        tokens += added
        source = USAGE_WITH_ESTIMATE if added else USAGE
    else:
        tokens = provider.estimate_input_tokens(kwargs)
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


#: The counts :func:`context_state` adds up. A usage carrying none of them
#: cannot produce a figure, however many other fields it has.
COUNT_FIELDS = ("input", "cache_creation", "cache_read", "output")


def _has_counts(usage: dict) -> bool:
    return any(isinstance(usage.get(field), int) for field in COUNT_FIELDS)


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


def form_schema(model_id: str) -> dict:
    """Everything the chat form needs: defaults, limits, capabilities, status.

    Only the skeleton is here. Three controls are genuinely shared - a model,
    a reply ceiling and a system prompt - and the rest of the form is the
    provider's own vocabulary, assembled by the provider. A form built from a
    union of both providers' fields would offer the selected model controls it
    has never heard of, which is the one thing a form must not do.
    """
    provider = provider_for(model_id)
    capabilities = provider.capabilities_for(model_id)
    defaults = provider.options_from({"model": model_id})
    # Read once, from the provider, so the transport named in the schema and
    # the one the streaming control describes cannot be two different facts.
    transport = provider.transport(
        llm.get_model(provider.resolve_model_id(model_id)), defaults
    )
    catalog = provider.catalog()
    schema = {
        "provider": provider_choices(provider),
        "models": catalog["models"],
        # What the narrowing left out. Hidden models are still listed here so
        # the page can say so: a short list that silently dropped nine models
        # would read like lost data rather than like a rule.
        "superseded": catalog["superseded"],
        "series_rule": catalog["rule"],
        "default_model": provider.default_model,
        "model": {
            "id": capabilities.id,
            "max_tokens": capabilities.max_output_tokens,
            "context_window": capabilities.context_window,
        },
        "defaults": {
            "model": model_id,
            "max_tokens": defaults.max_tokens,
            "system": defaults.system,
        },
        "transport": {"sdk_method": transport.name},
        "controls": {
            "model": {"status": EDITABLE},
            "max_tokens": {
                "status": EDITABLE,
                "min": 1,
                "max": capabilities.max_output_tokens,
                "note": (
                    f"at most {capabilities.max_output_tokens} for {capabilities.id}"
                    if capabilities.max_output_tokens
                    else "no output ceiling is known for this model"
                ),
            },
            "system": {"status": EDITABLE, "note": "omitted from the request when empty"},
        },
        "context": context_state({}, capabilities).as_dict(),
        "capabilities": capabilities.as_dict(),
    }
    return _merge_form(schema, provider.form(capabilities, transport))


#: Sections of the schema a provider adds to rather than replaces. The shared
#: skeleton put something in each of them, and a provider that returned a whole
#: section would silently drop the three controls every form has.
MERGED_SECTIONS = ("model", "defaults", "transport", "controls")


def _merge_form(schema: dict, own: dict) -> dict:
    """Fold a provider's own form into the shared skeleton.

    ``controls`` is merged one control at a time, so a provider may add its
    own fields to a shared control - Anthropic's reply ceiling has a minimum
    that depends on thinking - without restating the parts it did not change.
    """
    for section in MERGED_SECTIONS:
        for key, value in (own.get(section) or {}).items():
            if section == "controls" and key in schema[section]:
                schema[section][key] = {**schema[section][key], **value}
            else:
                schema[section][key] = value
    schema["model_data"] = own.get("model_data") or {}
    schema.update(own.get("vocabulary") or {})
    return schema


def provider_choices(selected: Any) -> dict:
    """Who this form belongs to, and who else the user could switch to.

    A provider with nothing to offer says so instead of being hidden: an
    empty model list looks like a broken page, while "no key configured" is
    something the user can act on. Availability is asked of the provider
    rather than assumed from a key name, because what makes a provider usable
    is its own business - llm-openrouter has no models at all without a key,
    llm-anthropic registers its own either way.
    """
    return {
        "id": selected.id,
        "label": selected.label,
        "sdk": selected.sdk,
        "options": [_provider_choice(provider) for provider in PROVIDERS],
    }


def _provider_choice(provider: Any) -> dict:
    """One entry in the switcher, with the model a switch to it would land on.

    The catalogue is read once and the default taken from what it returned,
    rather than asked for separately: a provider whose list is live would
    otherwise build it twice per page load to answer two questions about it.
    """
    models = provider.catalog()["models"]
    return {
        "id": provider.id,
        "label": provider.label,
        "models": len(models),
        "available": bool(models),
        # The provider's own answer, not the head of the list: two readings of
        # "the default model" in one response is two things that can disagree,
        # and this response already carries the other one.
        "default_model": provider.default_model,
        "unavailable_note": (
            "" if models else f"no models are available from {provider.label}"
        ),
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

    def _record_usage(self, document: dict) -> None:
        """Prefer the provider's own token counts once they exist.

        Which document holds the cache counters, and what they are called
        there, is the provider's business: ``llm`` flattens every plugin's
        usage into one shape and drops the zeros on the way, so the reading
        is taken off what the provider really sent.

        A usage that arrives with no counts in it is not kept, because
        downstream "there is a usage" means "the provider measured this". It
        happens for real: a reply truncated at the token ceiling ends the
        Responses stream with ``response.incomplete``, which ``llm`` does not
        read, so every count comes back None. Keeping that dict made the
        context meter add four Nones to nothing and report 0 tokens as the
        provider's own figure - the one kind of wrong answer this meter must
        never give, since being measured is the whole reason it is trusted.
        """
        reported = self.provider.usage_from(document)
        self._usage = reported if _has_counts(reported) else None

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
        self._record_usage(usage_document(prepared.response))
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
                provider=self.provider,
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
