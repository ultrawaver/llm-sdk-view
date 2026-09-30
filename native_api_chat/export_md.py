"""One conversation as one Markdown document, for reading elsewhere.

The export is built from the same stored records the chat panes read, so
what leaves the app is what the app showed: the user's messages, the
finished answers, thinking when the model produced any, and the per-turn
figures (latency, tokens, cache activity, cost) as a light metadata line -
enough for another AI to follow the conversation without drowning it in
provider JSON.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from urllib.parse import quote


def _stamp(iso: str) -> str:
    """A stored UTC stamp in the machine's own zone, second precision."""
    try:
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    return when.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _money(cost: dict | None) -> str | None:
    if not cost or cost.get("total") is None:
        return None
    return f"${cost['total']:.4f}"


def _turn_meta(turn: Any) -> str:
    """The one-line figure strip under an answer; only what exists appears."""
    response = turn.response
    usage = response.usage or {}
    parts = []
    if response.duration_ms is not None:
        parts.append(f"{response.duration_ms / 1000:.1f}s")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    write = usage.get("cache_creation_input_tokens") or 0
    read = usage.get("cache_read_input_tokens") or 0
    tokens = []
    if input_tokens is not None:
        tokens.append(f"in {input_tokens:,}")
    if write:
        tokens.append(f"{write:,} cache write")
    if read:
        tokens.append(f"{read:,} cache read")
    # The input side sums (uncached + cache traffic); output stands alone.
    if tokens:
        parts.append(" + ".join(tokens))
    if output_tokens is not None:
        parts.append(f"out {output_tokens:,}")
    price = _money(response.cost)
    if price:
        parts.append(price)
    return " · ".join(part for part in parts if part)


def _tool_lines(blocks: list[dict]) -> list[str]:
    """Server-side tool use, one line per call - the query, not the payload."""
    lines = []
    for block in blocks:
        if block.get("type") != "server_tool_use":
            continue
        name = block.get("name") or "tool"
        query = (block.get("input") or {}).get("query")
        lines.append(f'{name}: "{query}"' if query else name)
    return lines


def _source_line(citations: list[dict]) -> str | None:
    links = []
    for citation in citations:
        url = citation.get("url")
        title = citation.get("title") or url
        if url:
            links.append(f"[{title}]({url})")
    return "Sources: " + "; ".join(links) if links else None


def conversation_markdown(loaded: dict) -> str:
    """The Markdown document for one loaded conversation.

    ``loaded`` is store.load_conversation's shape: id, name, model and the
    turns in order. Sections that never happened (thinking, tools, sources)
    leave no trace in the document.
    """
    turns = loaded["turns"]
    header = [f"# {loaded['name']}", ""]
    if loaded.get("model"):
        header.append(f"- **Model:** {loaded['model']}")
    header.append(f"- **Turns:** {len(turns)}")
    priced = [turn.response.cost for turn in turns]
    if priced and all(priced):
        header.append(f"- **Total cost:** ${sum(c['total'] for c in priced):.4f}")
    elif any(priced):
        total = sum(c["total"] for c in priced if c)
        header.append(f"- **Total cost:** ${total:.4f} (partial: not every turn is priced)")
    system = next(
        (turn.options.get("system") for turn in turns if turn.options.get("system")),
        None,
    )
    if system:
        header.append(f"- **System:** {system}")
    header.append("")
    header.append("---")

    body: list[str] = []
    previous_system = system
    for index, turn in enumerate(turns, start=1):
        body.append("")
        body.append(f"## Turn {index} · {_stamp(turn.timestamp)}")
        body.append("")
        body.append("**User**")
        body.append("")
        body.append(turn.user_input)
        body.append("")
        body.append("**Assistant**")
        body.append("")
        response = turn.response
        if response.thinking:
            body.append("<thinking>")
            body.append(response.thinking)
            body.append("</thinking>")
            body.append("")
        if response.text:
            body.append(response.text)
            body.append("")
        notes = []
        current_system = turn.options.get("system") or ""
        if index > 1 and current_system != (previous_system or ""):
            notes.append(f"System prompt changed this turn: {current_system or '(cleared)'}")
        previous_system = current_system
        notes.extend(_tool_lines(response.server_tool_blocks))
        sources = _source_line(response.citations)
        if sources:
            notes.append(sources)
        meta = _turn_meta(turn)
        if meta:
            notes.append(meta)
        for note in notes:
            body.append(f"> {note}")
        body.append("")
        body.append("---")
    return "\n".join(header + body).rstrip() + "\n"


# What a filename may not carry: path separators and shell/control bytes.
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def export_filename(name: str, turns: list[Any]) -> str:
    """A download name from the conversation's own title and last day.

    The title is kept as-is (CJK included) - the Content-Disposition header
    carries it per RFC 5987 - with only the genuinely unsafe bytes removed.
    """
    slug = _UNSAFE.sub("", (name or "").strip())[:60].strip(". ") or "conversation"
    day = ""
    if turns:
        try:
            day = datetime.fromisoformat(
                turns[-1].timestamp.replace("Z", "+00:00")
            ).strftime("-%Y-%m-%d")
        except ValueError:
            day = ""
    return f"{slug}{day}.md"


_ASCII_RUNS = re.compile(r"[^A-Za-z0-9]+")


def content_disposition(filename: str) -> str:
    """attachment header: ASCII fallback plus the UTF-8 name per RFC 5987.

    The fallback is what a client that ignores ``filename*`` would save. A
    CJK title reduces to its punctuation and date - "-2026-09-29.md" - which
    is not a name, so when nothing but digits and separators survives, the
    file is named for what it is instead of left a mystery slug.
    """
    ascii_only = filename.encode("ascii", "ignore").decode()
    stem, dot, ext = ascii_only.rpartition(".")
    if not dot:
        stem, ext = ascii_only, ""
    stem = "-".join(part for part in _ASCII_RUNS.split(stem) if part).strip("-")
    if not re.search(r"[A-Za-z]", stem):
        stem = "-".join(filter(None, ["conversation", stem]))
    fallback = (f"{stem}.{ext}" if ext else stem).replace('"', "")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"
