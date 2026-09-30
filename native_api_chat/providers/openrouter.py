"""OpenRouter through ``llm-openrouter``, reached with the OpenAI SDK.

OpenRouter is usually the example of the thing this project refuses to be - a
lowest-common-denominator layer in front of everyone else's models - and it is
worth being exact about why it belongs here anyway. It is treated as a
provider in its own right, with a native surface of its own: provider routing
preferences, the four reasoning controls, its own hosted server tools, and
usage that reports what a turn actually cost rather than what it probably
cost. What is *not* claimed is that ``openrouter/anthropic/claude-sonnet-5``
is Anthropic's Claude. It is a different entry, with different capabilities
and a different SDK call, and the day those two collapse into one row this
project has started lying.

Two transports, one model. ``llm-openrouter`` registers only the Responses
classes, and the ``chat_completions`` option makes ``execute()`` delegate to
``OpenRouterChat`` for that one request - so the call a turn makes is decided
per turn, and :meth:`OpenRouterProvider.transport` has to read the option
rather than the model. The delegation is also where a real constraint lives:
the plugin refuses server-side tools on the Chat Completions path, so
switching transport takes web search away. That is reported, never worked
around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from .. import openrouter_api, turn
from ..model_series import OPENROUTER_RULE, newest_per_series, openrouter_series_for
from ..turn import (
    PROVIDER_DEFAULT,
    UNKNOWN_CAPABILITY,
    MissingKeyError,
    TurnOptions,
    UnsupportedOptionError,
    estimate_text,
    option_list,
)
from .base import Transport

MODEL_PREFIX = "openrouter/"

API_BASE = "https://openrouter.ai/api/v1"

# The headers llm-openrouter really attaches when it registers a model. They
# identify the caller to OpenRouter's rankings, they are on the wire, and a
# rendered call that left them out would not be the request that was sent.
HEADERS = {
    "HTTP-Referer": "https://llm.datasette.io/",
    "X-OpenRouter-Title": "LLM",
}

# No key is written into the generated code: the SDK reads OPENROUTER_KEY from
# the environment, and this project never puts a secret in a pane the user is
# about to copy.
HEADER = (
    "import os\n"
    "\n"
    "import openai\n"
    "\n"
    "client = openai.OpenAI(\n"
    f'    base_url="{API_BASE}",\n'
    '    api_key=os.environ["OPENROUTER_KEY"],\n'
    "    default_headers={\n"
    + "".join(f'        "{key}": "{value}",\n' for key, value in HEADERS.items())
    + "    },\n"
    ")\n"
)

# How each stream is really read. `responses.create(stream=True)` hands back an
# iterator of typed events, while chat completions yield choice deltas; both
# are printed the way the SDK's own documentation reads them.
RESPONSES_BODY = (
    "for event in stream:\n"
    '    if event.type == "response.output_text.delta":\n'
    '        print(event.delta, end="", flush=True)\n'
)

CHAT_BODY = (
    "for chunk in completion:\n"
    "    delta = chunk.choices[0].delta.content if chunk.choices else None\n"
    "    if delta:\n"
    '        print(delta, end="", flush=True)\n'
)

RESPONSES = Transport(
    name="responses.create",
    header=HEADER,
    call="client.responses.create",
    binding="stream",
    context=False,
    body=RESPONSES_BODY,
)

CHAT_COMPLETIONS = Transport(
    name="chat.completions.create",
    header=HEADER,
    call="client.chat.completions.create",
    binding="completion",
    context=False,
    body=CHAT_BODY,
)


# Options llm-openrouter declares but does not put on the wire, measured by
# building a request and looking for them rather than by reading the code.
# `reasoning_summary` is offered on both Options objects, but the mixin's
# chat_completions branch never copies it into `reasoning`, so asking for it
# there is asking for something that silently will not happen.
NOT_SENT: dict[str, dict[str, str]] = {
    CHAT_COMPLETIONS.name: {
        "reasoning_summary": (
            "llm-openrouter does not copy reasoning_summary into the "
            "chat.completions request; only effort, max_tokens and enabled "
            "reach the wire"
        ),
    },
}

# Names the option and the catalogue do not share. Everything else is matched
# by name, so a parameter OpenRouter adds later needs no change here.
PARAMETER_ALIASES = {
    "reasoning_effort": "reasoning_effort",
    "reasoning_max_tokens": "reasoning",
    "reasoning_enabled": "reasoning",
    "reasoning_summary": "reasoning",
    "json_object": "response_format",
}

# How OpenRouter's hosted web search names itself in a built request. The
# prefix is what separates it from a function tool, which is named instead.
WEB_SEARCH_TYPE = "openrouter:web_search"

# Why llm-openrouter never makes the non-streaming call. Both of its models
# subclass llm's OpenAI handlers, which open a stream whenever `stream=True`,
# and this project always asks for one - the chunks are the only honest thing
# to show while a reply is arriving. Shown as the status of the greyed-out
# stream control, and deliberately not rendered into the code pane.
STREAMING_NOTE = (
    "llm-openrouter streams: this project asks for stream=True, so the call "
    "rendered is the one that happens. Streaming is fixed here, not chosen."
)

# Options that are OpenRouter's own routing surface rather than a per-model
# parameter: the catalogue never lists them, and every model accepts them.
ALWAYS_AVAILABLE = frozenset({"provider", "chat_completions", "image_detail", "max_tokens"})

# Where a capability record's facts came from, in the two states that exist.
CATALOGUE_SOURCE = "openrouter-models"
UNKNOWN_SOURCE = "unknown"

EDITABLE = "editable"
UNSUPPORTED_BY_MODEL = "unsupported-by-model"
NOT_SENT_BY_PLUGIN = "not-sent-by-plugin"
# The catalogue is the only source for what a model accepts, so when it has
# not been read there is no answer - and "no answer" is not "no". A machine
# that is offline the first time it opens the page would otherwise be told
# every control is unsupported by the model, which is a different fact and a
# false one.
UNKNOWN_TO_CATALOGUE = "unknown-to-catalogue"

# The four states, in the words the page greys a control out with. They are
# shared with Anthropic deliberately: a control that is refused has to say why
# in the same words on both sides, or the page teaches two dialects of one
# idea. The codes above stay internal, where the distinctions are decided.
STATUS_WORDS = {
    EDITABLE: turn.EDITABLE,
    UNSUPPORTED_BY_MODEL: turn.UNSUPPORTED_BY_MODEL,
    # "The plugin will not send this" is exactly what the shared vocabulary
    # calls unsupported by the current tool version.
    NOT_SENT_BY_PLUGIN: turn.UNSUPPORTED_BY_TOOL,
    UNKNOWN_TO_CATALOGUE: turn.UNKNOWN_CAPABILITY,
}


# OpenRouter's reasoning surface is four separate fields, and they are kept
# four separate controls for the same reason thinking and effort are kept
# apart on the Anthropic side: they are separate API fields, and a single
# slider that wrote to whichever one seemed to fit would be inventing a
# parameter the API does not have.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
REASONING_SUMMARIES = ("auto", "concise", "detailed")

# "default" means the field is left off the request, so OpenRouter's own
# default applies. Absence on the wire means the provider default applied -
# never that a value was lost.
OMITTED = "default"

SEARCH_CONTEXT_SIZES = ("low", "medium", "high")

# Why these three are absent from OpenRouterOptions rather than refused by it.
# The Responses path rejects them outright, quoted from the plugin: "The
# OpenRouter Responses API does not support these options: ...". It raises
# rather than dropping them, and the transport is chosen per request, so a
# control for any of them would be one the runtime cannot honour on the
# default path - the rule is to not offer it, and this is the record of that
# decision. Adding a field here means adding a refusal to `accept` too.
UNAVAILABLE_ON_RESPONSES = ("stop", "logit_bias", "seed")

# What the plugin calls each form field. Only the ones that differ are here.
# `provider` is OpenRouter's routing object, and the form calls it `routing`
# because "provider" already means something else in every other module of
# this project - the mapping happens here and nowhere else, so the pane can
# still print the field by its real name.
PLUGIN_OPTION_NAMES = {"routing": "provider"}


def uses_chat_completions(options: Any) -> bool:
    """Whether this turn takes the Chat Completions path.

    The option lives on the model's own Options object, so it is read by name
    from whatever the form produced rather than from a copy kept here.
    """
    return bool(getattr(options, "chat_completions", False))


@dataclass(frozen=True)
class OpenRouterOptions(TurnOptions):
    """Every value OpenRouter's chat form can set.

    Not a superset of the Anthropic form and not a subset of it. The overlap
    is the three fields in :class:`~native_api_chat.turn.TurnOptions`; the
    rest of this is OpenRouter's own API, which is the whole reason it is here
    as a provider rather than as a row in somebody else's list.
    """

    # Which of the two calls this turn makes. It is a request-level choice,
    # not a model-level one, and it changes what else is allowed.
    chat_completions: bool = False
    reasoning_effort: str = OMITTED
    reasoning_max_tokens: int | None = None
    reasoning_enabled: bool | None = None
    reasoning_summary: str = OMITTED
    web_search: bool = False
    # None means "omit", which is the tool's own default rather than a limit
    # this project invented.
    max_uses: int | None = None
    search_context_size: str | None = None
    routing: dict | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reasoning_effort != OMITTED and self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be {OMITTED} or one of {REASONING_EFFORTS}"
            )
        if self.reasoning_summary != OMITTED and self.reasoning_summary not in REASONING_SUMMARIES:
            raise ValueError(
                f"reasoning_summary must be {OMITTED} or one of {REASONING_SUMMARIES}"
            )
        if self.reasoning_max_tokens is not None and self.reasoning_max_tokens < 1:
            raise ValueError("reasoning_max_tokens must be a positive integer")
        # The plugin's own floor, so the refusal happens on the form rather
        # than inside the tool's constructor.
        if self.max_uses is not None and self.max_uses < 1:
            raise ValueError("max_uses must be a positive integer, or absent for no limit")
        if (
            self.search_context_size is not None
            and self.search_context_size not in SEARCH_CONTEXT_SIZES
        ):
            raise ValueError(f"search_context_size must be one of {SEARCH_CONTEXT_SIZES}")
        if self.routing is not None and not isinstance(self.routing, dict):
            raise ValueError("routing must be a JSON object")


def slug_for(model_id: str) -> str:
    """The id OpenRouter knows, with llm's routing prefix removed."""
    return model_id[len(MODEL_PREFIX) :] if model_id.startswith(MODEL_PREFIX) else model_id


# How the Responses output names the two parts a record shows. Reasoning text
# arrives under either key depending on whether a summary was asked for.
TEXT_PART = "output_text"
REASONING_PART_KEYS = ("summary", "content")


def _blocks_from_output(output: Any) -> list[dict]:
    """The reply and the reasoning, out of a Responses ``output`` list.

    Tool calls and the opaque ``encrypted_content`` are deliberately not
    translated: they are the plugin's business for the next request, and a
    record that showed them as content would be showing the model's private
    scratch space as part of its answer.
    """
    if not isinstance(output, list):
        return []
    blocks: list[dict] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            text = "".join(_part_texts(item.get(key)) for key in REASONING_PART_KEYS)
            if text:
                blocks.append({"type": "thinking", "thinking": text})
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == TEXT_PART:
                block = {"type": "text", "text": part.get("text") or ""}
                # OpenRouter returns web search sources as annotations; the
                # record reads citations, and they are the same fact.
                if part.get("annotations"):
                    block["citations"] = list(part["annotations"])
                blocks.append(block)
    return blocks


def _part_texts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    return "".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


# Where the cache read hides on each path. Both are read, because the
# transport is chosen per request and a turn's counters must not depend on
# this project having guessed which one it was.
CACHED_TOKEN_PARENTS = ("input_tokens_details", "prompt_tokens_details")


def _cached_tokens(details: dict) -> int | None:
    """The cached input tokens this turn reused, or None if none was reported.

    Zero is a real answer and is reported as zero: ``llm`` removes zero-valued
    keys from the details, so a parent that is present with no ``cached_tokens``
    inside it means the provider reported no cache hits, while a parent that is
    absent means it reported nothing at all.
    """
    for parent in CACHED_TOKEN_PARENTS:
        nested = details.get(parent)
        if isinstance(nested, dict):
            return nested.get("cached_tokens") or 0
    return None


def _optional_int(value: Any) -> int | None:
    """A number the form may simply not have set.

    An empty string is a cleared field, which means "omit", not zero - and
    zero is a value some of these fields reject outright.
    """
    if value is None or value == "":
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    """A tri-state checkbox: on, off, or never touched.

    ``None`` leaves the field off the request, which is not the same request
    as sending ``false``: one takes OpenRouter's default and the other
    overrides it.
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value.lower() not in ("false", "0", "off", "no")
    return bool(value)


def _estimate_content(content: Any) -> int | None:
    """One message's content, estimated - or None if it cannot be.

    Both transports carry text as either a bare string or a list of typed
    parts, and the part names differ between them (``text`` on one,
    ``input_text`` and ``output_text`` on the other), so both vocabularies
    are read and anything else stops the count rather than being skipped.
    """
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_text(content)
    if not isinstance(content, list):
        return None
    total = 0
    for part in content:
        if isinstance(part, str):
            total += estimate_text(part)
            continue
        if not isinstance(part, dict):
            return None
        if part.get("type") in ("text", "input_text", "output_text"):
            total += estimate_text(part.get("text", ""))
        else:
            return None
    return total


def _wire_value(value: Any) -> Any:
    """An option's value as the wire carries it, not as Python spells it.

    The plugin's options are ``str`` enums, so ``json.dumps`` already writes
    ``"high"`` - but ``str()`` writes ``ReasoningEffortEnum.high``, and a
    record is read back by more than one thing. What is reported is the value
    that is really in the request.
    """
    return getattr(value, "value", value)


def _server_tool_class(model: Any, name: str):
    """The plugin's own tool class, looked up on the model that will send it."""
    for tool_class in getattr(model, "supported_server_side_tools", ()):
        if getattr(tool_class, "name", None) == name:
            return tool_class
    raise UnsupportedOptionError(f"{model.model_id} has no {name} server tool")


def _find_web_search_tool(kwargs: dict) -> dict | None:
    """The web search tool in a built request, on either transport.

    Both paths put server tools in ``tools``, but the Responses path names
    them by ``type`` while a function tool is named by ``name``, so the type
    prefix is what identifies OpenRouter's hosted one.
    """
    for tool in kwargs.get("tools", ()):
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "openrouter:web_search":
            return tool
    return None


def _verify_system(options: OpenRouterOptions, kwargs: dict, on_chat: bool) -> None:
    """The system prompt has to land in whichever field this path uses."""
    wanted = options.system.strip()
    if on_chat:
        first = (kwargs.get("messages") or [{}])[0]
        carried = first.get("content") if first.get("role") == "system" else None
    else:
        carried = kwargs.get("instructions")
    if not wanted:
        if carried:
            raise ValueError("system is empty but the request carries a system prompt")
        return
    if carried != options.system:
        raise ValueError("the request system prompt does not match the form")


def _verify_reasoning(options: OpenRouterOptions, kwargs: dict, on_chat: bool) -> None:
    """Each of the four reasoning fields, where this path really puts it.

    The Chat Completions path nests the block under ``extra_body`` while the
    Responses path has it at the top level, and ``summary`` never appears on
    the first one at all - which is why asking for it there is refused before
    a request is ever built.
    """
    block = (
        (kwargs.get("extra_body") or {}).get("reasoning")
        if on_chat
        else kwargs.get("reasoning")
    ) or {}
    for field, wanted, absent in (
        ("effort", options.reasoning_effort, OMITTED),
        ("max_tokens", options.reasoning_max_tokens, None),
        ("enabled", options.reasoning_enabled, None),
        ("summary", options.reasoning_summary, OMITTED),
    ):
        if wanted == absent:
            if field in block:
                raise ValueError(
                    f"reasoning {field} was not asked for but the request carries it"
                )
        elif block.get(field) != wanted:
            raise ValueError(
                f"the request reasoning {field} is {block.get(field)!r}, not {wanted!r}"
            )


def available_model_ids() -> tuple[str, ...]:
    """OpenRouter models ``llm`` has actually registered.

    ``llm-openrouter`` registers nothing without a key, so this is empty on a
    machine that has not configured one - which is the truth the page should
    tell, rather than a catalogue of models it cannot send to.
    """
    import llm

    return tuple(
        model.model_id
        for model in llm.get_models()
        if model.model_id.startswith(MODEL_PREFIX)
    )


@dataclass(frozen=True)
class OpenRouterCapabilities:
    """What one OpenRouter model can do, as its own catalogue describes it.

    Deliberately not the same shape as the Anthropic record. Thinking modes,
    budget tokens and caller allow-lists are Anthropic's vocabulary; forcing
    them onto a model that has never heard of them would put controls on the
    page that cannot do anything.
    """

    id: str
    slug: str
    name: str
    created: int | None
    context_window: int | None
    max_output_tokens: int | None
    supported_parameters: frozenset[str]
    pricing: dict[str, str]
    input_modalities: tuple[str, ...]
    data_source: str
    provenance: dict

    @property
    def described(self) -> bool:
        """Whether the catalogue had an entry for this model at all."""
        return self.data_source == CATALOGUE_SOURCE

    @property
    def context_window_source(self) -> str:
        """Where the context figure's ceiling came from, in words."""
        if not self.described:
            return "no OpenRouter catalogue entry for this model"
        return "OpenRouter catalogue context_length"

    @property
    def limit_status(self) -> str:
        """How sure this record is of its own context window.

        A catalogue that could not be read leaves the ceiling unknown, and
        unknown is its own status: reporting it as a provider default would
        dress a missing number up as a deliberate one.
        """
        return PROVIDER_DEFAULT if self.described else UNKNOWN_CAPABILITY

    def supports(self, option: str) -> bool:
        """Whether the model accepts the parameter behind a form control."""
        if option in ALWAYS_AVAILABLE:
            return True
        return PARAMETER_ALIASES.get(option, option) in self.supported_parameters

    def option_status(self, option: str, transport: Transport) -> tuple[str, str]:
        """This control's status on this transport, and why.

        Four answers are possible and each is a different fact: the plugin
        will not send it, the model does not take it, nobody knows because the
        catalogue was not readable, or it works. Collapsing any of the first
        three into "disabled" would hide which one the user would have to fix
        - and reporting the third as the second would be an outright lie.
        """
        blocked = NOT_SENT.get(transport.name, {}).get(option)
        if blocked:
            return NOT_SENT_BY_PLUGIN, blocked
        if option in ALWAYS_AVAILABLE:
            return EDITABLE, ""
        if not self.described:
            return (
                UNKNOWN_TO_CATALOGUE,
                f"the OpenRouter catalogue does not describe {self.slug}"
                + (
                    f" ({self.provenance['error']})"
                    if self.provenance.get("error")
                    else ""
                ),
            )
        if not self.supports(option):
            return (
                UNSUPPORTED_BY_MODEL,
                f"{self.slug} does not list "
                f"{PARAMETER_ALIASES.get(option, option)} as a supported parameter",
            )
        return EDITABLE, ""

    def price(self, key: str) -> float | None:
        """A per-token price, or None when the catalogue does not give one.

        A missing or zero price is returned as "no figure" rather than as
        free: OpenRouter writes "0" for models it is not pricing as well as
        for models that really cost nothing, and guessing which is which
        would put an invented number on the page.
        """
        raw = self.pricing.get(key)
        if raw in (None, "", "0", "-1"):
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value or None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "slug": self.slug,
            "name": self.name,
            "created": self.created,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supported_parameters": sorted(self.supported_parameters),
            "input_modalities": list(self.input_modalities),
            "data_source": self.data_source,
        }


class _Facts:
    """The catalogue, indexed by llm model id."""

    def __init__(self, snapshot: dict):
        self.provenance = snapshot["provenance"]
        self._by_slug = {entry["id"]: entry for entry in snapshot["models"]}

    def entry(self, model_id: str) -> dict | None:
        return self._by_slug.get(slug_for(model_id))

    def created_at(self, model_id: str) -> int | None:
        entry = self.entry(model_id)
        return entry.get("created") if entry else None


def _facts_by_id() -> _Facts:
    return _Facts(openrouter_api.snapshot())


def _newest_of(model_id: str, newest: tuple[str, ...]) -> str | None:
    """Which visible model superseded this one, by series."""
    key = openrouter_series_for(model_id).key
    for candidate in newest:
        if openrouter_series_for(candidate).key == key:
            return candidate
    return None


class OpenRouterProvider:
    """OpenRouter's own API, reached through the OpenAI SDK."""

    id = "openrouter"
    label = "OpenRouter"
    # OpenRouter is reached through the OpenAI SDK pointed at openrouter.ai,
    # so that is the library the rendered code imports and the page names.
    sdk = "openai-python"
    # OpenRouter publishes no token counter, so there is no exact figure to be
    # had before a turn is sent. Asking Anthropic's would be worse than not
    # asking: the Chat Completions request carries `messages`, which is the
    # field count_tokens wants, so the call would be well-formed enough to
    # send this conversation to a provider that has nothing to do with it.
    counts_tokens = False

    def owns(self, model_id: str) -> bool:
        return model_id.startswith(MODEL_PREFIX)

    def transport(self, model: Any, options: Any = None) -> Transport:
        """Which call this turn makes, decided by the request, not the model.

        ``llm-openrouter`` registers Responses models only and delegates to
        ``OpenRouterChat`` inside ``execute()`` when ``chat_completions`` is
        set, so the same model id can produce either call.
        """
        return CHAT_COMPLETIONS if uses_chat_completions(options) else RESPONSES

    def assemble(
        self, model: Any, prompt: Any, conversation: Any, options: Any = None
    ) -> dict:
        """The complete request, in whichever shape this turn will send.

        Neither path hands back a finished request: ``llm`` adds ``model``,
        the input and ``stream`` at the call site. Putting them back here is
        not a second copy of the request - it is the only place the request
        exists whole, which is exactly what the pane has to print.
        """
        if uses_chat_completions(prompt.options):
            return self._chat_completions_kwargs(model, prompt, conversation)
        return self._responses_kwargs(model, prompt)

    def _responses_kwargs(self, model: Any, prompt: Any) -> dict:
        image_detail = getattr(prompt.options, "image_detail", None)
        if image_detail is not None:
            image_detail = getattr(image_detail, "value", image_detail)
        input_items, instructions = model._build_responses_input(
            prompt, image_detail=image_detail
        )
        kwargs = model._finalize_responses_kwargs(prompt, True, instructions)
        return {
            "model": model.model_name or model.model_id,
            "input": input_items,
            "stream": True,
            **kwargs,
        }

    def _chat_completions_kwargs(
        self, model: Any, prompt: Any, conversation: Any
    ) -> dict:
        """Built through the delegate the plugin itself would build.

        ``execute()`` does not send this request from the Responses model: it
        constructs an ``OpenRouterChat`` from ``_delegate_chat_kwargs()`` and
        hands the prompt to that. Asking the same object for the request is
        what keeps the pane and the wire in step.
        """
        chat = self._delegate(model)
        image_detail = getattr(prompt.options, "image_detail", None)
        if image_detail is not None:
            image_detail = getattr(image_detail, "value", image_detail)
        messages = chat.build_messages(prompt, conversation, image_detail=image_detail)
        kwargs = chat.build_kwargs(prompt, True)
        return {
            "model": chat.model_name or chat.model_id,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

    @staticmethod
    def _delegate(model: Any):
        from llm_openrouter import OpenRouterChat

        return OpenRouterChat(**model._delegate_chat_kwargs())

    # ---- the turn --------------------------------------------------------

    def options_from(self, payload: dict) -> OpenRouterOptions:
        """The chat form, mapped onto this provider's options.

        The tri-state fields are read with a sentinel rather than with
        ``or``: ``reasoning_enabled=False`` and "the user did not touch
        reasoning_enabled" are different requests, and ``or`` cannot tell
        them apart. Likewise an empty string from a cleared number field is
        absence, not zero.
        """
        return OpenRouterOptions(
            model=payload["model"],
            max_tokens=int(payload.get("max_tokens", 16384)),
            system=str(payload.get("system") or ""),
            chat_completions=bool(payload.get("chat_completions", False)),
            reasoning_effort=str(payload.get("reasoning_effort") or OMITTED),
            reasoning_max_tokens=_optional_int(payload.get("reasoning_max_tokens")),
            reasoning_enabled=_optional_bool(payload.get("reasoning_enabled")),
            reasoning_summary=str(payload.get("reasoning_summary") or OMITTED),
            web_search=bool(payload.get("web_search", False)),
            max_uses=_optional_int(payload.get("max_uses")),
            search_context_size=payload.get("search_context_size") or None,
            routing=payload.get("routing") or None,
        )

    def accept(
        self, options: OpenRouterOptions, capabilities: OpenRouterCapabilities
    ) -> OpenRouterOptions:
        """These options, resolved - or a refusal saying which one and why.

        Two of the refusals below are transport constraints rather than model
        ones, which is what makes ``chat_completions`` more than a rendering
        choice: flipping it takes web search away and makes
        ``reasoning_summary`` a field that never reaches the wire. Both are
        refused rather than silently dropped, because a form that showed a
        summary setting on a request that cannot carry one is describing a
        request nobody sent.
        """
        transport = CHAT_COMPLETIONS if options.chat_completions else RESPONSES
        if options.chat_completions:
            if options.web_search:
                raise UnsupportedOptionError(
                    "llm-openrouter refuses server-side tools on the "
                    "chat.completions path, so web search is unavailable "
                    "while that transport is selected"
                )
            if options.reasoning_summary != OMITTED:
                raise UnsupportedOptionError(
                    NOT_SENT[CHAT_COMPLETIONS.name]["reasoning_summary"]
                )
        for option, value in (
            ("reasoning_effort", options.reasoning_effort != OMITTED),
            ("reasoning_max_tokens", options.reasoning_max_tokens is not None),
            ("reasoning_enabled", options.reasoning_enabled is not None),
            ("reasoning_summary", options.reasoning_summary != OMITTED),
        ):
            if not value:
                continue
            status, reason = capabilities.option_status(option, transport)
            if status == UNSUPPORTED_BY_MODEL:
                raise UnsupportedOptionError(reason)
        if options.web_search and not capabilities.supports("tools"):
            # Only when the catalogue actually said so. It cannot refuse on
            # behalf of a model it has never described.
            if capabilities.described:
                raise UnsupportedOptionError(
                    f"{capabilities.slug} does not list tools as a supported "
                    "parameter, so it cannot be given a server-side tool"
                )
        ceiling = capabilities.max_output_tokens
        if ceiling and options.max_tokens > ceiling:
            raise ValueError(
                f"max_tokens must be at most {ceiling} for {capabilities.slug}: "
                "that is the ceiling the serving endpoint reports"
            )
        return replace(options)

    def plugin_options(
        self, options: OpenRouterOptions, capabilities: OpenRouterCapabilities
    ) -> dict:
        """The option dict ``llm`` is given, in the plugin's own names.

        Every "default" is expressed by leaving the field out, so the request
        carries no field the user did not ask for and the provider's own
        default is what applies.
        """
        built: dict[str, Any] = {"max_tokens": options.max_tokens}
        if options.chat_completions:
            built["chat_completions"] = True
        if options.reasoning_effort != OMITTED:
            built["reasoning_effort"] = options.reasoning_effort
        if options.reasoning_max_tokens is not None:
            built["reasoning_max_tokens"] = options.reasoning_max_tokens
        if options.reasoning_enabled is not None:
            built["reasoning_enabled"] = options.reasoning_enabled
        if options.reasoning_summary != OMITTED:
            built["reasoning_summary"] = options.reasoning_summary
        if options.routing:
            built[PLUGIN_OPTION_NAMES["routing"]] = dict(options.routing)
        return built

    def tools(
        self,
        model: Any,
        options: OpenRouterOptions,
        capabilities: OpenRouterCapabilities,
    ) -> list:
        """The plugin's own WebSearch, never a description of it.

        Building the spec by hand here would be the second copy of a request
        this project exists to avoid: the tool's ``tool_spec()`` is what ends
        up on the wire, so the tool object is what has to be constructed.
        """
        if not options.web_search:
            return []
        tool_class = _server_tool_class(model, "web_search")
        kwargs: dict[str, Any] = {}
        if options.max_uses is not None:
            kwargs["max_uses"] = options.max_uses
        if options.search_context_size is not None:
            kwargs["search_context_size"] = options.search_context_size
        return [tool_class(**kwargs)]

    def verify(
        self,
        options: OpenRouterOptions,
        kwargs: dict,
        capabilities: OpenRouterCapabilities | None = None,
    ) -> None:
        """Refuse to stream a request that contradicts the form.

        The two transports name the same settings differently, so this reads
        whichever pair of names the turn actually built rather than one shape
        it assumed.
        """
        on_chat = options.chat_completions
        ceiling_field = "max_tokens" if on_chat else "max_output_tokens"
        if kwargs.get(ceiling_field) != options.max_tokens:
            raise ValueError(
                f"the request {ceiling_field} does not match the form"
            )
        _verify_system(options, kwargs, on_chat)
        _verify_reasoning(options, kwargs, on_chat)

        tool = _find_web_search_tool(kwargs)
        if options.web_search and tool is None:
            raise ValueError("web_search is on but no web search tool was built")
        if not options.web_search:
            if tool is not None:
                raise ValueError("web_search is off but a web search tool was built")
            return
        parameters = tool.get("parameters") or {}
        if options.max_uses is None:
            if "max_uses" in parameters:
                raise ValueError("max_uses was not set but the request limits uses")
        elif parameters.get("max_uses") != options.max_uses:
            raise ValueError("the request max_uses does not match the form")
        if (
            options.search_context_size is not None
            and parameters.get("search_context_size") != options.search_context_size
        ):
            raise ValueError("the request search_context_size does not match the form")

    def unavailable(self, model_id: str, error: Exception) -> Exception:
        """Missing key or missing model - they look identical from llm.

        ``llm-openrouter`` registers no models at all until it has a key, so
        an empty registry is the signal: with no key every id is unknown, and
        with a key an unknown id is genuinely not on offer.
        """
        if not available_model_ids():
            return MissingKeyError(
                "No key for OpenRouter, so no OpenRouter models are available. "
                "Set one with 'llm keys set openrouter' or OPENROUTER_KEY."
            )
        return error

    def stop_reason(self, message: dict) -> str | None:
        """Why the reply ended, from whichever of the two paths finished.

        Chat Completions reports ``finish_reason``. Responses reports a
        ``status``, and the reason only when that status is ``incomplete`` -
        where it is the one worth showing, because it is the truncation the
        user would otherwise mistake for a complete answer. Both are reported
        as ``status: reason`` so the two halves cannot be read as one word.
        """
        finish = message.get("finish_reason")
        if isinstance(finish, str) and finish:
            return finish
        status = message.get("status")
        reason = (message.get("incomplete_details") or {}).get("reason")
        if status and reason:
            return f"{status}: {reason}"
        return status or reason or None

    def blocks(self, message: dict) -> list[dict]:
        """Translate whichever of the two shapes finished into record blocks.

        Chat Completions ends with ``content`` as one plain string. Responses
        ends with an ``output`` list whose message items hold ``output_text``
        parts, and whose reasoning items hold their text under ``summary`` or
        ``content``. Neither is the typed-block list the record reads, and the
        string is the dangerous one: iterating it yields characters, so a
        reader written for blocks sees a block per letter.
        """
        content = message.get("content")
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if isinstance(content, list):
            return list(content)
        return _blocks_from_output(message.get("output"))

    # ---- the form ---------------------------------------------------------

    @property
    def default_model(self) -> str | None:
        """The model a switch to OpenRouter lands on: its newest standard one.

        There is no constant to use here, and committing one would be wrong
        rather than merely arbitrary: the catalogue is live, so an id pinned in
        this repo would eventually name a model OpenRouter no longer offers,
        and a release here cannot be what fixes that.

        "Newest by publication date" is a fact the catalogue already reports,
        which is why it is the rule rather than a recommendation - this project
        has no business ranking 356 models by merit. The pricing tiers are
        excluded: ``:batch`` is an asynchronous endpoint and ``:free`` is rate
        limited by whoever else is using it, so neither is what "just open
        OpenRouter" should mean, though both stay selectable.

        None when there are no models - a keyless machine - which the switcher
        reports rather than presenting an empty list as a broken page.
        """
        models = self.catalog()["models"]
        standard = [model_id for model_id in models if ":" not in slug_for(model_id)]
        facts = _facts_by_id()
        ranked = sorted(
            standard or models,
            key=lambda model_id: (facts.created_at(model_id) or 0, model_id),
            reverse=True,
        )
        return ranked[0] if ranked else None

    def form(
        self, capabilities: OpenRouterCapabilities, transport: Transport
    ) -> dict:
        """This provider's own half of the chat form.

        Every control below is settled the same way: ask the catalogue whether
        this model takes the parameter, ask the plugin whether it would send it
        on this transport, and report whichever of the four answers came back.
        None of it is decided by what looks reasonable - ``chat_completions``
        silently drops the reasoning summary and raises on server tools, and
        both were measured rather than read.

        ``reasoning`` is four controls because it is four API fields. One
        slider writing to whichever seemed to fit would be inventing a
        parameter OpenRouter does not have.
        """
        defaults = OpenRouterOptions(model=capabilities.id)
        return {
            "model": {
                "sends": capabilities.slug,
                "name": capabilities.name,
                "input_modalities": sorted(capabilities.input_modalities),
            },
            "model_data": capabilities.provenance,
            "defaults": {
                "chat_completions": defaults.chat_completions,
                "reasoning_effort": defaults.reasoning_effort,
                "reasoning_max_tokens": defaults.reasoning_max_tokens,
                "reasoning_enabled": defaults.reasoning_enabled,
                "reasoning_summary": defaults.reasoning_summary,
                "web_search": defaults.web_search,
                "max_uses": defaults.max_uses,
                "search_context_size": defaults.search_context_size,
                "routing": defaults.routing,
            },
            "transport": {"streaming_note": STREAMING_NOTE},
            "controls": {
                "chat_completions": {
                    **self._control("chat_completions", capabilities, transport),
                    "value": defaults.chat_completions,
                    "note": (
                        "the same model on the other API: chat.completions.create "
                        "instead of responses.create. Chosen per request, and it "
                        "changes which controls can be sent at all"
                    ),
                },
                "reasoning_effort": {
                    **self._control("reasoning_effort", capabilities, transport),
                    "value": OMITTED,
                    "options": option_list((OMITTED, *REASONING_EFFORTS)),
                    "note": "sent as reasoning.effort; default leaves the field out",
                },
                "reasoning_max_tokens": {
                    **self._control("reasoning_max_tokens", capabilities, transport),
                    "value": None,
                    "min": 1,
                    "note": "sent as reasoning.max_tokens; empty leaves the field out",
                },
                "reasoning_enabled": {
                    **self._control("reasoning_enabled", capabilities, transport),
                    "value": None,
                    "note": (
                        "sent as reasoning.enabled. Three states, not two: "
                        "off is a request, and untouched is not"
                    ),
                },
                "reasoning_summary": {
                    **self._control("reasoning_summary", capabilities, transport),
                    "value": OMITTED,
                    "options": option_list((OMITTED, *REASONING_SUMMARIES)),
                    "note": "sent as reasoning.summary",
                },
                "web_search": {
                    **self._control("web_search", capabilities, transport),
                    "value": defaults.web_search,
                    "note": (
                        "the plugin's own web search tool, sent as "
                        f"{WEB_SEARCH_TYPE}"
                    ),
                },
                "max_uses": {
                    **self._control("web_search", capabilities, transport),
                    "value": None,
                    "note": "a ceiling on the search tool; empty leaves it out",
                },
                "search_context_size": {
                    **self._control("web_search", capabilities, transport),
                    "value": None,
                    "options": option_list((OMITTED, *SEARCH_CONTEXT_SIZES)),
                    "note": "how much of each result the search tool returns",
                },
                "routing": {
                    **self._control("provider", capabilities, transport),
                    "value": None,
                    "note": (
                        "OpenRouter's own provider routing object, sent as "
                        "`provider`. Which upstream serves the model, in its own "
                        "vocabulary rather than a simplified one"
                    ),
                },
                "stream": {
                    "status": turn.RUNTIME_FIXED,
                    "value": "ON",
                    "editable": False,
                    "sdk_method": f"{transport.call}(...)",
                    "options": [
                        {"value": "ON", "disabled": False},
                        {
                            "value": "OFF",
                            "disabled": True,
                            "note": "API supported · not used by llm-openrouter",
                        },
                    ],
                    "note": STREAMING_NOTE,
                },
            },
            "vocabulary": {
                "reasoning_efforts": list(REASONING_EFFORTS),
                "reasoning_summaries": list(REASONING_SUMMARIES),
                "search_context_sizes": list(SEARCH_CONTEXT_SIZES),
                "omitted": OMITTED,
                "unavailable_on_responses": list(UNAVAILABLE_ON_RESPONSES),
                "pricing": capabilities.pricing,
            },
        }

    def _control(
        self,
        option: str,
        capabilities: OpenRouterCapabilities,
        transport: Transport,
    ) -> dict:
        """One control's status, in the words the page greys it out with.

        The internal code is deliberately not passed on. It is what decides
        which of the four answers this is, and a page given it would sooner or
        later branch on it - putting a capability decision in a UI conditional,
        where the next model's rules cannot reach it.
        """
        status, note = capabilities.option_status(option, transport)
        control = {"status": STATUS_WORDS[status]}
        if note:
            control["unavailable_note"] = note
        return control

    def request_facts(self, kwargs: dict) -> dict:
        """What the pane can say about this request that the form cannot.

        Read off the built request, never off the options: the routing object
        and the reasoning block are what the endpoint will act on, and the
        form is only what asked for them.
        """
        extra = kwargs.get("extra_body") or {}
        reasoning = kwargs.get("reasoning") or extra.get("reasoning") or {}
        tool = _find_web_search_tool(kwargs)
        return {
            "reasoning": {key: _wire_value(value) for key, value in reasoning.items()}
            or None,
            "routing": dict(extra.get("provider") or {}) or None,
            "web_search": tool.get("type") if tool else None,
        }

    def estimate_input_tokens(self, kwargs: dict) -> int | None:
        """A labelled estimate, walking whichever request shape was built.

        ``None`` when the request holds something a character count cannot
        speak for - an image, a file, a block this project does not model -
        because a number derived from nothing is worse than no number.
        """
        total = 0
        for message in kwargs.get("messages", ()):
            counted = _estimate_content(message.get("content"))
            if counted is None:
                return None
            total += counted
        for item in kwargs.get("input", ()):
            if not isinstance(item, dict):
                return None
            counted = _estimate_content(item.get("content"))
            if counted is None:
                return None
            total += counted
        total += estimate_text(kwargs.get("instructions") or "")
        total += estimate_text(json.dumps(kwargs.get("tools", [])))
        return total

    def usage_from(self, usage: Any, details: dict) -> dict:
        """OpenRouter's own counts, under the names this app reports.

        The cache read is nested, and under a different parent on each path:
        ``input_tokens_details.cached_tokens`` on Responses and
        ``prompt_tokens_details.cached_tokens`` on Chat Completions. Reading
        it off the top level - which is where Anthropic's counters sit - finds
        nothing and reports every turn as a cache miss.

        ``llm`` strips zeros out of the details, so the parent's presence is
        what separates "nothing was cached" from "nothing was reported". Both
        are honest answers and they are not the same one: only the first can
        be shown as a 0% hit rate.

        There is no cache *write* count on either path. OpenRouter prices one
        for some models but does not report one here, so it stays unknown
        rather than being inferred from the read.
        """
        return {
            "input": getattr(usage, "input", None),
            "output": getattr(usage, "output", None),
            "cache_creation": None,
            "cache_read": _cached_tokens(details),
        }

    # ---- what the model can do -------------------------------------------

    def resolve_model_id(self, model_id: str) -> str:
        """``llm`` registers OpenRouter models under the id the form uses.

        Nothing to map: the plugin builds ``openrouter/<slug>`` from the
        catalogue entry, which is exactly what reaches this project.
        """
        return model_id

    def catalog(self) -> dict:
        """The OpenRouter models this machine can actually send to.

        Two different lists could be shown here and only one of them is
        honest. ``/api/v1/models`` describes 464 models, but
        ``llm-openrouter`` registers none of them without a key, so every one
        of those rows would be a model the runtime cannot honour. The ids
        therefore come from ``llm``'s registry - what is really usable - and
        the catalogue is used only to describe them.
        """
        registered = tuple(sorted(available_model_ids()))
        facts = _facts_by_id()
        if not registered:
            return {
                "models": [],
                "superseded": [],
                "rule": OPENROUTER_RULE.description,
                "note": (
                    "no OpenRouter models are registered; run "
                    "`llm keys set openrouter` and they will appear"
                ),
                "provenance": facts.provenance,
            }
        newest = newest_per_series(
            registered,
            created_at=lambda model_id: facts.created_at(model_id),
            rule=OPENROUTER_RULE,
        )
        hidden = tuple(model_id for model_id in registered if model_id not in newest)
        return {
            # Ids, not descriptions: the dropdown shows what goes on the wire,
            # and the selected model's facts come from /api/form. Both
            # providers answer in the same shape, because the page reads one
            # list and must not learn which provider filled it.
            "models": list(newest),
            "superseded": [
                {
                    "id": model_id,
                    "series": openrouter_series_for(model_id).key,
                    "kept_by": _newest_of(model_id, newest),
                }
                for model_id in hidden
            ],
            "rule": OPENROUTER_RULE.description,
            "note": "",
            "provenance": facts.provenance,
        }

    def capabilities_for(
        self, model_id: str, facts: _Facts | None = None
    ) -> OpenRouterCapabilities:
        """What this model can do, and where each fact came from."""
        facts = facts if facts is not None else _facts_by_id()
        entry = facts.entry(model_id)
        top = (entry.get("top_provider") or {}) if entry else {}
        parameters = frozenset(entry.get("supported_parameters") or ()) if entry else frozenset()
        architecture = (entry.get("architecture") or {}) if entry else {}
        return OpenRouterCapabilities(
            id=model_id,
            slug=slug_for(model_id),
            name=(entry or {}).get("name") or slug_for(model_id),
            created=(entry or {}).get("created"),
            context_window=(entry or {}).get("context_length") or top.get("context_length"),
            max_output_tokens=top.get("max_completion_tokens"),
            supported_parameters=parameters,
            pricing=dict((entry or {}).get("pricing") or {}),
            input_modalities=tuple(architecture.get("input_modalities") or ()),
            data_source=CATALOGUE_SOURCE if entry else UNKNOWN_SOURCE,
            provenance=facts.provenance,
        )
