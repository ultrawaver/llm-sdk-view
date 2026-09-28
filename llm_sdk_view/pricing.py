"""What a turn costs, from rates Anthropic publishes.

No API exposes prices, so the rates come from the pricing page: fetched in the
background, cached on disk, and labelled with where they came from
(see ``rates_page``). They used to be transcribed into this file by hand, which
meant every upstream price change needed a commit; fetching them is the same
honesty with less drift.

What is still true and still matters:

- The counters are the API's own, the rates are the page's, and the product of
  the two is an estimate - it knows nothing about batch discounts, long-context
  premiums or an account's credits.
- A model the current rates do not cover gets no estimate at all rather than an
  invented one, and a rate that could not be fetched is labelled as a cache or
  as unavailable rather than presented as live.
"""

from __future__ import annotations

from typing import Any

from . import rates_page

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


def rates_status() -> dict:
    """Where the current rates came from, in the terms the UI shows.

    Prices are fetched rather than shipped, so "no estimate" has more than one
    honest reason: the page could not be read, or this model is not on it.
    """
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


def cost_breakdown(model: str | None, usage: dict | None) -> dict | None:
    """One turn's cost, line by line, from the usage the API reported.

    Returns None when the model's rates are unknown or the two counters every
    estimate needs (input, output) are missing. Cache counters that
    llm-anthropic dropped (caching off, no server tools) are treated as zero -
    with caching off no cache activity is possible, so that zero is a fact,
    not a guess - and flagged with ``cache_counters_reported: False`` so the
    UI can say why the cache lines read zero.
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
    if write_5m is None and creation_total is not None:
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
        **_rates_provenance(snapshot["provenance"]),
    }
