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

SOURCE = "llm-sdk-view"

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


def usage_summary(usage: dict) -> str:
    """One line: what this turn cost, including cache and search."""
    parts = [
        f"input {_fmt_number(usage.get('input_tokens'))}",
        f"output {_fmt_number(usage.get('output_tokens'))}",
        f"cache creation {_fmt_number(usage.get('cache_creation_input_tokens'))}",
        f"cache read {_fmt_number(usage.get('cache_read_input_tokens'))}",
        f"web searches {_fmt_number(usage.get('web_search_requests'))}",
    ]
    return " · ".join(parts)


def _fmt_number(value: Any) -> str:
    return str(value) if isinstance(value, int) else "unreported"


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

    def summary(self) -> str:
        return usage_summary(self.usage)

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
            "summary": self.summary(),
            "response_json": self.response_json,
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
            response_json=data.get("response_json"),
        )


def build_response_view(response: Any) -> ResponseView:
    """Read the Answer off the Message the SDK finished with.

    ``response_json`` is the accumulated Message, so citations, thinking and
    server tool blocks are still whole; nothing here rebuilds them from text.
    """
    message = getattr(response, "response_json", None) or {}
    blocks = _as_blocks(message)
    return ResponseView(
        response_id=getattr(response, "id", None),
        message_id=message.get("id"),
        model=message.get("model") or getattr(getattr(response, "model", None), "model_id", None),
        content_blocks=blocks,
        text=_block_text(blocks, ("text",)) or "",
        thinking=_block_text(blocks, THINKING_BLOCKS),
        citations=_citations(blocks),
        server_tool_blocks=_server_tool_blocks(blocks),
        stop_reason=message.get("stop_reason"),
        usage=_usage(response, message),
        response_json=message or None,
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
