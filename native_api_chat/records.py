"""One record per turn, and nothing else that can drift from it.

Every surface shares these two objects:

``ResponseView``
    Read off the Message the provider finished with. It is never reassembled
    from the text that was streamed into the chat bubble, because a
    reassembly loses thinking blocks, citations, tool results and usage, and
    then silently agrees with itself about things that were never there.

``TurnRecord``
    One turn: what the form asked for, the request ``build_kwargs()``
    produced, the code the right pane rendered, and the response above. The
    chat bubble, the Response pane and the rows written to storage are all
    projections of this one object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .pricing import cost_breakdown

SOURCE = "native-api-chat"

# Titles come from the first user message, not from a model call.
NAME_MIN_CHARS = 40
NAME_MAX_CHARS = 60

THINKING_BLOCKS = ("thinking", "redacted_thinking")
SERVER_TOOL_BLOCKS = ("server_tool_use", "web_search_tool_result", "code_execution_tool_result")


def conversation_name(user_input: str) -> str:
    """The first user message, trimmed to a stable length.

    A model is never asked to write a title: that would cost tokens and add a
    request nobody asked for. Empty input still gets a usable name.
    """
    text = " ".join((user_input or "").split())
    if not text:
        return "Untitled conversation"
    if len(text) <= NAME_MAX_CHARS:
        return text
    cut = text[:NAME_MAX_CHARS]
    # Do not cut inside a word when there is room to step back.
    space = cut.rfind(" ")
    if space >= NAME_MIN_CHARS:
        cut = cut[:space]
    return cut.rstrip(" ,.:;") + "…"


def _as_blocks(message: dict | None) -> list[dict]:
    if not message:
        return []
    blocks = message.get("content")
    return list(blocks) if isinstance(blocks, list) else []


def _block_text(blocks: list[dict], kinds: tuple[str, ...]) -> str | None:
    keys = ("text", "data", "thinking", "summary")
    parts = [
        _first_of(block, keys)
        for block in blocks
        if block.get("type") in kinds
    ]
    text = "".join(part for part in parts if isinstance(part, str))
    return text or None


def _first_of(block: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = block.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _streamed_text(response: Any) -> str:
    """What llm accumulated from the stream, or "" if the turn never finished."""
    try:
        return response.text_or_raise()
    except Exception:  # noqa: BLE001 - an unfinished response has no text yet
        return ""


def _citations(blocks: list[dict]) -> list[dict]:
    found: list[dict] = []
    for block in blocks:
        citations = block.get("citations")
        if isinstance(citations, list):
            found.extend(citation for citation in citations if isinstance(citation, dict))
    return found


def _server_tool_blocks(blocks: list[dict]) -> list[dict]:
    return [block for block in blocks if block.get("type") in SERVER_TOOL_BLOCKS]


def _usage(response: Any, message: dict | None) -> dict:
    """The counts the provider reported, with nothing invented.

    llm-anthropic moves ``usage`` off the Message and onto the Response as it
    consumes the stream, so the Response - not the Message - is where the
    numbers live by the time a turn is finished.
    """
    usage: dict[str, Any] = {}
    raw = (message or {}).get("usage")
    if isinstance(raw, dict):
        usage.update(raw)
    usage.setdefault("input_tokens", getattr(response, "input_tokens", None))
    usage.setdefault("output_tokens", getattr(response, "output_tokens", None))
    details = getattr(response, "token_details", None)
    if isinstance(details, dict):
        for key, value in details.items():
            usage.setdefault(key, value)
    # The count Anthropic reports nested inside server_tool_use; pulled up so
    # the pane can show it next to the tokens it cost.
    server_tool = usage.get("server_tool_use")
    if isinstance(server_tool, dict):
        usage.setdefault("web_search_requests", server_tool.get("web_search_requests"))
    else:
        usage.setdefault("web_search_requests", None)
    return usage


def _number(value: Any) -> Any:
    return value if isinstance(value, int) else None


def _duration_ms(response: Any) -> int | None:
    """llm's own wall-clock measurement for this response, or None.

    llm times every call from dispatch to stream end and exposes it as
    ``duration_ms``. The Anthropic Console reports a server-side figure the
    API never returns, so this client-side measurement is the honest one
    available here; it includes streaming drain time and network.
    """
    getter = getattr(response, "duration_ms", None)
    if getter is None:
        return None
    try:
        value = getter() if callable(getter) else getter
    except Exception:
        # A response that never finished has no duration to report.
        return None
    return value if isinstance(value, int) and value >= 0 else None


def format_duration(duration_ms: int | None) -> str | None:
    """0.34 s style, matching the Console's Latency column."""
    if not isinstance(duration_ms, int) or duration_ms < 0:
        return None
    return f"{duration_ms / 1000:.2f} s"


def usage_summary(usage: dict, counts: dict | None = None) -> str:
    """One line: what this turn cost, including cache and search.

    The cache figures come from ``counts`` - the sending provider's own
    reading - rather than from the raw document, because only the provider
    knows where its API puts them. OpenRouter nests the read one level down
    and under a different parent per transport, so Anthropic's spelling found
    nothing and reported "unreported" for a turn that reported 0.
    """
    if counts:
        creation = counts.get("cache_creation")
        read = counts.get("cache_read")
    else:
        # A record stored before the counters were read per provider. Every
        # one of those is an Anthropic turn, so its figures really are in the
        # document under Anthropic's own names - this is the stored shape,
        # not a guess at an unknown one.
        creation = usage.get("cache_creation_input_tokens")
        read = usage.get("cache_read_input_tokens")
    parts = [
        f"input {_fmt_number(usage.get('input_tokens'))}",
        f"output {_fmt_number(usage.get('output_tokens'))}",
        f"cache creation {_fmt_number(creation)}",
        f"cache read {_fmt_number(read)}",
        f"web searches {_fmt_number(usage.get('web_search_requests'))}",
    ]
    return " · ".join(parts)


def _fmt_number(value: Any) -> str:
    return str(value) if isinstance(value, int) else "unreported"


# --- the official Raw shape -------------------------------------------------
#
# The Response pane shows the Message the way the API returned it, so these
# orders exist to put it back the way the official Playground prints it. They
# are presentation only: a key the API adds later is appended, never dropped,
# and a key it does not send is never invented.

RAW_MESSAGE_KEYS = (
    "model",
    "id",
    "type",
    "role",
    "content",
    "container",
    "stop_reason",
    "stop_sequence",
    "stop_details",
    "usage",
    "diagnostics",
)

RAW_USAGE_KEYS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cache_creation",
    "output_tokens",
    "service_tier",
    "inference_geo",
)

RAW_CACHE_CREATION_KEYS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")

RAW_TEXT_BLOCK_KEYS = ("type", "text", "citations")

# Counts this app hoists out of the API's own usage object. They are ours, so
# they must not appear in a view that claims to be the provider's Raw JSON.
DERIVED_USAGE_KEYS = ("web_search_requests",)


def _ordered(source: dict, order: tuple[str, ...]) -> dict:
    """Known keys in ``order`` first, then anything the API added later."""
    known = [key for key in order if key in source]
    extra = [key for key in source if key not in order]
    return {key: source[key] for key in known + extra}


def raw_message(message: dict | None, usage: dict) -> dict | None:
    """The Message as the API returned it, with ``usage`` back on it.

    ``llm-anthropic`` moves ``usage`` off the Message and onto its own Response
    while it consumes the stream, so a Raw view has to put it back. Anything
    this app derived is left out; anything the API did not report stays absent
    rather than being filled in with a zero.
    """
    if not message:
        return None
    rebuilt = dict(message)
    if usage:
        cleaned = {
            key: value for key, value in usage.items() if key not in DERIVED_USAGE_KEYS
        }
        cache_creation = cleaned.get("cache_creation")
        if isinstance(cache_creation, dict):
            cleaned["cache_creation"] = _ordered(cache_creation, RAW_CACHE_CREATION_KEYS)
        rebuilt["usage"] = _ordered(cleaned, RAW_USAGE_KEYS)
    content = rebuilt.get("content")
    if isinstance(content, list):
        rebuilt["content"] = [
            _ordered(block, RAW_TEXT_BLOCK_KEYS)
            if isinstance(block, dict)
            else block
            for block in content
        ]
    return _ordered(rebuilt, RAW_MESSAGE_KEYS)


@dataclass(frozen=True)
class ResponseView:
    """What the provider answered, read off the finished Message."""

    # llm's own id for the call: the one the turn is stored under.
    response_id: str | None
    # The provider's id for the Message itself, when the response carried one.
    message_id: str | None
    model: str | None
    content_blocks: list[dict]
    text: str
    thinking: str | None
    citations: list[dict]
    server_tool_blocks: list[dict]
    stop_reason: str | None
    usage: dict
    response_json: dict | None
    # The same counters, read by the provider that sent the turn and reduced
    # to this app's four names. ``usage`` above is the provider's own
    # document, which is what the Response pane shows; a figure is read from
    # here, because "the cache read" is spelt differently on every API - and
    # is nested on both OpenRouter paths, where Anthropic's spelling finds
    # nothing and every turn reads as a cache miss.
    counts: dict = field(default_factory=dict)
    # Client-side latency. duration_ms is llm's own measurement (dispatch to
    # stream end); ttft_ms is this app's time to the first text chunk.
    # Older stored records predate both, so None means "not measured".
    duration_ms: int | None = None
    ttft_ms: int | None = None

    def summary(self) -> str:
        return usage_summary(self.usage, self.counts)

    def latency_summary(self) -> str | None:
        duration = format_duration(self.duration_ms)
        if duration is None:
            return None
        return f"latency {duration} (client-measured, includes streaming)"

    @property
    def raw(self) -> dict | None:
        """The Message as the API returned it: official keys in official order.

        This is what the Response pane shows. Everything else on this object -
        ``text``, ``thinking``, ``citations``, ``summary``, the latency figures -
        is this app's reading of the same record and stays out of the JSON.
        """
        return raw_message(self.response_json, self.usage)

    @property
    def cost(self) -> dict | None:
        """The turn's cost, line by line, from published rates.

        Derived like ``raw``: recomputed on read, so a rate-table update
        re-prices every stored conversation. None when the model's rates are
        unknown - an estimate is never invented.
        """
        return cost_breakdown(self.model, self.usage)

    def as_dict(self) -> dict:
        return {
            "response_id": self.response_id,
            "message_id": self.message_id,
            "model": self.model,
            "content_blocks": self.content_blocks,
            "text": self.text,
            "thinking": self.thinking,
            "citations": self.citations,
            "server_tool_blocks": self.server_tool_blocks,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "counts": self.counts,
            "summary": self.summary(),
            "response_json": self.response_json,
            "duration_ms": self.duration_ms,
            "ttft_ms": self.ttft_ms,
            # Derived, so it is recomputed rather than read back: a stored
            # copy would go stale the moment the order or the rules change.
            "raw": self.raw,
            # Derived for the same reason: rates are versioned, and a stored
            # total would silently keep yesterday's prices.
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResponseView:
        return cls(
            response_id=data.get("response_id"),
            message_id=data.get("message_id"),
            model=data.get("model"),
            content_blocks=data.get("content_blocks") or [],
            text=data.get("text") or "",
            thinking=data.get("thinking"),
            citations=data.get("citations") or [],
            server_tool_blocks=data.get("server_tool_blocks") or [],
            stop_reason=data.get("stop_reason"),
            usage=data.get("usage") or {},
            counts=data.get("counts") or {},
            response_json=data.get("response_json"),
            duration_ms=data.get("duration_ms"),
            ttft_ms=data.get("ttft_ms"),
        )


def build_response_view(
    response: Any,
    ttft_ms: int | None = None,
    provider: Any = None,
    counts: dict | None = None,
) -> ResponseView:
    """Read the Answer off the Message the SDK finished with.

    ``response_json`` is the accumulated Message, so citations, thinking and
    server tool blocks are still whole; nothing here rebuilds them from text.

    ``provider`` is the one that sent it, and it is asked about every part of
    that Message whose shape differs between providers. Without it the
    Anthropic shape is assumed: right for Anthropic, and on either OpenRouter
    path it finds no text and no stop reason at all.

    ``counts`` is that provider's reading of its own counters, passed in
    rather than read again here: the session has already asked for it, and a
    second reading is a second answer that can disagree with the first.
    """
    message = getattr(response, "response_json", None) or {}
    blocks = provider.blocks(message) if provider else _as_blocks(message)
    stop_reason = (
        provider.stop_reason(message) if provider else message.get("stop_reason")
    )
    return ResponseView(
        response_id=getattr(response, "id", None),
        message_id=message.get("id"),
        model=message.get("model") or getattr(getattr(response, "model", None), "model_id", None),
        content_blocks=blocks,
        # The blocks are the answer; llm's own accumulated text is the fallback
        # for a Message this reader could not make sense of, so an unfamiliar
        # shape costs the citations and the thinking but never the reply.
        text=_block_text(blocks, ("text",)) or _streamed_text(response),
        thinking=_block_text(blocks, THINKING_BLOCKS),
        citations=_citations(blocks),
        server_tool_blocks=_server_tool_blocks(blocks),
        stop_reason=stop_reason,
        usage=_usage(response, message),
        counts=counts or {},
        response_json=message or None,
        duration_ms=_duration_ms(response),
        ttft_ms=ttft_ms,
    )


@dataclass(frozen=True)
class TurnRecord:
    """Everything one turn produced, in the shape it is stored in."""

    conversation_id: str
    turn_id: str | None
    user_input: str
    options: dict
    request_kwargs: dict
    rendered_code: str
    response: ResponseView
    context: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "options": dict(self.options),
            "request_kwargs": self.request_kwargs,
            "rendered_code": self.rendered_code,
            "response": self.response.as_dict(),
            "context": dict(self.context),
            "timestamp": self.timestamp,
            "source": SOURCE,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TurnRecord:
        return cls(
            conversation_id=data["conversation_id"],
            turn_id=data.get("turn_id"),
            user_input=data.get("user_input", ""),
            options=data.get("options") or {},
            request_kwargs=data.get("request_kwargs") or {},
            rendered_code=data.get("rendered_code", ""),
            response=ResponseView.from_dict(data.get("response") or {}),
            context=data.get("context") or {},
            timestamp=data.get("timestamp", ""),
        )
