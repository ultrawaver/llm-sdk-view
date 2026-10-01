"""What a turn costs, from the rates the provider that sent it publishes.

Anthropic publishes no API for its prices, so those come from the pricing page
(read in the background, cached on disk, labelled with where they came from -
see ``rates_page``). OpenRouter publishes them as a field of the model
catalogue it already caches - see ``rates_openrouter``. Either way a price is
fetched rather than shipped, so an upstream change needs no commit here.

What is still true and still matters:

- The counters are the APIs' own, the rates are the publishers', and the
  product of the two is an estimate - it knows nothing about an account's
  credits or a discount negotiated outside the catalogue.
- A model the rates do not cover gets no estimate at all rather than an
  invented one, and a rate that could not be fetched is labelled as a cache or
  as unavailable rather than presented as live.
- The two providers do not spell the same thing the same way. Anthropic
  reports the uncached part of the prompt as ``input_tokens`` and bills cache
  writes and reads beside it; OpenRouter reports the whole prompt, cached part
  included, and reports no cache write at all. Pricing one provider's counters
  with the other's arithmetic reads like a working number, which is why the
  two are separate functions below.
"""

from __future__ import annotations

from typing import Any

from . import rates_openrouter, rates_page
from .providers import provider_for

# The row type and the page URL live in rates_page (the module that reads the
# page); they are re-exported here because pricing is where callers look for
# them. Importing them from rates_page also keeps the two modules acyclic.
from .rates_page import RATES_URL, ModelRates

__all__ = [
    "RATES_SOURCE",
    "RATES_URL",
    "ModelRates",
    "WEB_SEARCH_PER_CALL_USD",
    "cost_breakdown",
    "provider_of",
    "rates_for",
    "rates_status",
]

RATES_SOURCE = "Anthropic pricing"

# $10 per 1,000 searches, per the pricing page. A search that errors is not
# billed; the count here comes from the API's own web_search_requests.
WEB_SEARCH_PER_CALL_USD = 0.01

# How each source of rates is described where an estimate is shown.
_SOURCE_LABEL = {
    "pricing-page": "Anthropic pricing",
    "disk-cache": "Anthropic pricing (cached)",
    "unavailable": "Anthropic pricing (unavailable)",
}


def provider_of(model: str | None) -> str | None:
    """Which provider sends this model, by the one place that decides it.

    None for an id no provider claims - a conversation stored by a plugin that
    is no longer installed. No provider is then answerable for its rates.
    """
    if not model:
        return None
    try:
        return provider_for(model).id
    except ValueError:
        return None


def _match(table: dict[str, ModelRates], model: str) -> ModelRates | None:
    """The row for this model, matched by longest id prefix first.

    A dated id like claude-haiku-4-5-20251001 matches its family prefix, and
    claude-fable-5-1 must win over claude-fable-5.
    """
    for prefix in sorted(table, key=len, reverse=True):
        if model.startswith(prefix):
            return table[prefix]
    return None


def rates_for(model: str | None) -> ModelRates | None:
    """The current published rates for this model, or None when there are none."""
    if not model:
        return None
    return _match(rates_page.table(), model)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _line(
    key: str,
    label: str,
    group: str,
    quantity: int,
    rate: float,
    rate_label: str,
    note: str | None = None,
    reported: bool = True,
    per_mtok: bool = True,
) -> dict:
    per_unit = rate / 1_000_000 if per_mtok else rate
    return {
        "key": key,
        "label": label,
        "group": group,
        "quantity": quantity,
        "rate": rate,
        "rate_label": rate_label,
        "amount": round(quantity * per_unit, 6),
        "note": note,
        # False when the provider's counter was dropped on the way here and
        # the zero is "no cache activity possible", not a reported zero.
        "reported": reported,
    }


def rates_status(provider: str = "anthropic") -> dict:
    """Where the current rates came from, in the terms the UI shows.

    Prices are fetched rather than shipped, so "no estimate" has more than one
    honest reason: the source could not be read, or this model is not on it.
    Which source that is depends on the provider, so the question is asked of
    one at a time and the page asks about the provider it is showing.
    """
    if provider == "openrouter":
        return rates_openrouter.status()
    return _rates_provenance(rates_page.snapshot()["provenance"])


def _rates_provenance(provenance: dict) -> dict:
    source = provenance.get("source", "unavailable")
    fetched_at = provenance.get("fetched_at")
    return {
        "rates_source": _SOURCE_LABEL.get(source, RATES_SOURCE),
        "rates_date": fetched_at[:10] if isinstance(fetched_at, str) else "unknown",
        "rates_url": RATES_URL,
        "rates_state": {
            "pricing-page": "live",
            "disk-cache": "cached",
            "unavailable": "unavailable",
        }.get(source, "unavailable"),
        "rates_error": provenance.get("error"),
    }


def cost_breakdown(
    model: str | None,
    usage: dict | None,
    counts: dict | None = None,
    provider: str | None = None,
) -> dict | None:
    """One turn's cost, line by line, from the usage the API reported.

    Returns None when the model's rates are unknown or the two counters every
    estimate needs (input, output) are missing.

    ``counts`` is the sending provider's own reading of its counters, passed
    in rather than read again here. It matters most on the OpenRouter path:
    the cache read is nested there, and under a different parent per
    transport, so Anthropic's top-level spelling finds nothing and every turn
    reads as a cache miss - a number, so nothing downstream can tell it is a
    wrong one.

    ``provider`` says which price list applies. It is taken from the turn when
    the turn knows it, because an id cannot always say: OpenRouter answers
    with its catalogue slug (``openai/gpt-5.4``), which carries none of llm's
    routing prefix, so an id that reached this app from a response would be
    read as nobody's and priced from the wrong document.
    """
    which = provider or provider_of(model)
    if which == "openrouter":
        return _openrouter_cost(model, usage, counts or {})
    return _anthropic_cost(model, usage)


def _anthropic_cost(model: str | None, usage: dict | None) -> dict | None:
    """Anthropic's counters against Anthropic's rates.

    Anthropic reports the uncached input, the cache writes and the cache reads
    as three separate counts, so the turn's total input is their sum - and a
    cache counter llm-anthropic dropped (caching off, no server tools) is
    treated as zero, which with caching off is a fact rather than a guess.
    """
    snapshot = rates_page.snapshot()
    rates = _match(snapshot["rates"], model) if model else None
    if rates is None or not usage:
        return None
    uncached = _int(usage.get("input_tokens"))
    output = _int(usage.get("output_tokens"))
    if uncached is None or output is None:
        return None

    cache_creation = usage.get("cache_creation")
    creation_total = _int(usage.get("cache_creation_input_tokens"))
    read = _int(usage.get("cache_read_input_tokens"))
    counters_reported = creation_total is not None or read is not None

    write_5m = write_1h = None
    if isinstance(cache_creation, dict):
        write_5m = _int(cache_creation.get("ephemeral_5m_input_tokens"))
        write_1h = _int(cache_creation.get("ephemeral_1h_input_tokens"))
    if creation_total is not None and (write_5m is not None or write_1h is not None):
        # The split and the total must agree, and the API has been seen
        # disagreeing with itself: a web-search turn reported a 0 + 0 split
        # against a total of 8,940. The total is what the countdown shows,
        # so the total wins and the split only carves out its 1h part -
        # llm-anthropic only ever writes a 5-minute entry, so whatever the
        # 1h detail does not claim is a 5m write.
        write_1h = write_1h or 0
        write_5m = max(creation_total - write_1h, 0)
    elif write_5m is None and creation_total is not None:
        # No 5m/1h split, but llm-anthropic only ever writes a 5-minute
        # entry, so an unsplit write total is a 5m write.
        write_5m = creation_total
        write_1h = write_1h or 0

    server_tool = usage.get("server_tool_use")
    searches = _int(usage.get("web_search_requests"))
    if searches is None and isinstance(server_tool, dict):
        searches = _int(server_tool.get("web_search_requests"))
    searches = searches or 0

    lines = [
        _line(
            "uncached_input", "Uncached input", "input", uncached, rates.input,
            f"${rates.input:.2f} / MTok", note="after the last cache breakpoint",
        ),
        _line(
            "cache_write_5m", "Cache write · 5m TTL", "input", write_5m or 0,
            rates.cache_write_5m, f"${rates.cache_write_5m:.2f} / MTok",
            reported=counters_reported,
        ),
        _line(
            "cache_read", "Cache read (hit)", "input", read or 0,
            rates.cache_read, f"${rates.cache_read:.2f} / MTok",
            reported=counters_reported,
        ),
        _line(
            "output", "Output", "output", output, rates.output,
            f"${rates.output:.2f} / MTok",
        ),
        _line(
            "web_search", "Web search", "tools", searches,
            WEB_SEARCH_PER_CALL_USD, "$0.01 / search", per_mtok=False,
        ),
    ]
    if write_1h:
        lines.insert(
            2,
            _line(
                "cache_write_1h", "Cache write · 1h TTL", "input", write_1h,
                rates.cache_write_1h, f"${rates.cache_write_1h:.2f} / MTok",
                reported=counters_reported,
            ),
        )

    total = round(sum(line["amount"] for line in lines), 6)
    write_total = (write_5m or 0) + (write_1h or 0)
    input_total = uncached + write_total + (read or 0)

    no_cache_total = round(
        input_total * rates.input / 1_000_000
        + output * rates.output / 1_000_000
        + searches * WEB_SEARCH_PER_CALL_USD,
        6,
    )
    input_cost = sum(line["amount"] for line in lines if line["group"] == "input")

    notes = []
    if not counters_reported:
        notes.append("cache counters were not reported (caching off) — "
                     "the cache lines read $0")

    return {
        "currency": "USD",
        "estimated": True,
        "model": model,
        "lines": lines,
        "total": total,
        "input_total_tokens": input_total,
        "cache_counters_reported": counters_reported,
        # cache_read / (uncached + writes + reads): the token-level prompt
        # cache hit rate the observability tools report.
        "cache_hit_rate": (read or 0) / input_total if input_total else None,
        # What the same turn would have cost with caching off entirely.
        "no_cache_total": no_cache_total,
        "savings": round(no_cache_total - total, 6),
        "effective_input_rate": (
            round(input_cost / input_total * 1_000_000, 2) if input_total else None
        ),
        "notes": notes,
        **_rates_provenance(snapshot["provenance"]),
    }


def _openrouter_cost(model: str | None, usage: dict, counts: dict) -> dict | None:
    """OpenRouter's counters against OpenRouter's own catalogue prices.

    Three things differ from the Anthropic path, and each one is a fact about
    the API rather than a preference:

    - ``input_tokens`` is the **whole prompt**, cached part included, because
      llm reads OpenRouter's ``prompt_tokens``. The cached tokens are a slice
      of it, so charging all of it at the input rate *and* the cache read at
      its own rate would bill the cached tokens twice.
    - **No cache write count exists on either transport**, so no write is
      priced - rather than a write being assumed at some rate.
    - **No search count reaches this app either**, so a priced search is not
      in the total; the receipt says so where the model prices one.
    """
    row = rates_openrouter.rates_for(model)
    if row is None or not usage:
        return None
    prompt = _int(usage.get("input_tokens"))
    output = _int(usage.get("output_tokens"))
    if prompt is None or output is None:
        return None

    cached = _int(counts.get("cache_read"))
    if cached is not None:
        # A floor and a ceiling, because a counter larger than the prompt it
        # is a part of is a broken reading and not a bigger bill.
        cached = min(max(cached, 0), prompt)
    rates = row.at(prompt)
    if cached and rates.cache_read is None:
        # A cached read this model does not price: there is no rate to use,
        # and the input rate would be an invention.
        return None

    uncached = prompt - (cached or 0)
    cache_read_rate = rates.cache_read or 0.0
    uncached_note = "after the last cache breakpoint"
    if cached is None:
        uncached_note = "cache read unreported"

    lines = [
        _line(
            "uncached_input", "Uncached input", "input", uncached, rates.input,
            f"${rates.input:.2f} / MTok", note=uncached_note,
        ),
        _line(
            "cache_read", "Cache read (hit)", "input", cached or 0,
            cache_read_rate, f"${cache_read_rate:.2f} / MTok",
            reported=cached is not None,
        ),
        _line(
            "output", "Output", "output", output, rates.output,
            f"${rates.output:.2f} / MTok",
        ),
    ]

    total = round(sum(line["amount"] for line in lines), 6)
    # The prompt *is* the total here: the cached tokens are already counted
    # inside it.
    input_total = prompt
    no_cache_total = round(
        prompt * rates.input / 1_000_000 + output * rates.output / 1_000_000, 6
    )
    input_cost = sum(line["amount"] for line in lines if line["group"] == "input")

    notes = []
    if rates.long_context_from:
        notes.append(
            "long-context rate: this model costs more above "
            + f"{rates.long_context_from:,} prompt tokens"
        )
    if cached is None:
        # Worded without "this turn": the same sentence is shown under a
        # conversation total whenever every turn in it agreed on it.
        notes.append("OpenRouter did not report a cache read — "
                     "the whole prompt is priced at the input rate")
    unpriced = []
    if rates.cache_write_5m is not None or rates.cache_write_1h is not None:
        unpriced.append("cache writes")
    if row.web_search is not None:
        unpriced.append("searches")
    if len(unpriced) == 1:
        notes.append(f"{unpriced[0]} are not in this total — OpenRouter "
                     "reports no count for them")
    elif unpriced:
        notes.append(" and ".join(unpriced)
                     + " are not in this total — OpenRouter reports no count "
                     "for either")

    return {
        "currency": "USD",
        "estimated": True,
        "model": model,
        "lines": lines,
        "total": total,
        "input_total_tokens": input_total,
        "cache_counters_reported": cached is not None,
        "cache_hit_rate": (cached or 0) / input_total if input_total else None,
        "no_cache_total": no_cache_total,
        "savings": round(no_cache_total - total, 6),
        "effective_input_rate": (
            round(input_cost / input_total * 1_000_000, 2) if input_total else None
        ),
        "notes": notes,
        **rates_openrouter.provenance(),
    }
