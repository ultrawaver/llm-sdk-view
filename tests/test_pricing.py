"""The cost estimate: honest rates, honest arithmetic, honest absences.

The rates come from the pricing page at runtime, so these tests pin the three
things that must never drift: the arithmetic against known rates (total input
= uncached + writes + reads, hit rate = reads / total input), the rule that an
unknown model or a missing counter yields no estimate rather than an invented
one, and the rule that a rate which could not be fetched is labelled as stale
or absent rather than shown as live.
"""

from datetime import datetime, timedelta, timezone

from native_api_chat import pricing, rates_page
from native_api_chat.pricing import (
    RATES_SOURCE,
    RATES_URL,
    cost_breakdown,
    rates_for,
)

FABLE = "claude-fable-5-1"

# The Fable example from the pricing research: 900 uncached + 8,300 cache
# read + 900 output + 1 web search.
USAGE = {
    "input_tokens": 900,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 8300,
    "cache_creation": {
        "ephemeral_5m_input_tokens": 0,
        "ephemeral_1h_input_tokens": 0,
    },
    "output_tokens": 900,
    "server_tool_use": {"web_search_requests": 1},
    "web_search_requests": 1,
}


def _line(cost, key):
    return next(line for line in cost["lines"] if line["key"] == key)


# --- the rates themselves -----------------------------------------------------


def test_rates_carry_their_source_and_the_day_they_were_read():
    cost = cost_breakdown(FABLE, USAGE)

    assert cost["rates_source"] == RATES_SOURCE == "Anthropic pricing"
    assert cost["rates_url"] == RATES_URL
    assert cost["rates_date"] == datetime.now(timezone.utc).date().isoformat()
    assert cost["rates_state"] == "live"
    assert cost["rates_error"] is None
    assert cost["estimated"] is True


def test_a_stale_cache_is_shown_as_a_cache_not_as_live(monkeypatch):
    stale = datetime.now(timezone.utc) - timedelta(days=3)
    monkeypatch.setattr(
        rates_page,
        "load_cache",
        lambda: {
            "fetched_at": stale.isoformat(),
            "rates": {"claude-fable-5-1": {"input": 10.0, "output": 50.0,
                                           "cache_write_5m": 12.5,
                                           "cache_write_1h": 20.0,
                                           "cache_read": 0.25}},
        },
    )

    cost = cost_breakdown(FABLE, USAGE)

    assert cost["rates_state"] == "cached"
    assert "cached" in cost["rates_source"]
    assert cost["rates_date"] == stale.date().isoformat()
    # The estimate still stands - it just says what it is standing on.
    assert cost["total"] > 0


def test_without_prices_there_is_no_estimate_at_all(monkeypatch):
    monkeypatch.setattr(rates_page, "load_cache", lambda: None)

    assert cost_breakdown(FABLE, USAGE) is None


def test_fable_rates_match_the_pricing_page():
    rates = rates_for(FABLE)

    assert rates is not None
    assert (rates.input, rates.output) == (10.0, 50.0)
    assert (rates.cache_write_5m, rates.cache_write_1h) == (12.50, 20.0)
    # Fable 5.1's cache read is 0.025x base input, not the usual 0.1x.
    assert rates.cache_read == 0.25


def test_a_dated_model_id_matches_its_family():
    assert rates_for("claude-haiku-4-5-20251001") == rates_for("claude-haiku-4-5")


def test_an_unknown_model_gets_no_estimate():
    assert rates_for("claude-imaginary-9") is None
    assert cost_breakdown("claude-imaginary-9", USAGE) is None
    assert cost_breakdown(None, USAGE) is None


# --- the arithmetic ------------------------------------------------------------


def test_each_line_is_tokens_times_unit_rate():
    cost = cost_breakdown(FABLE, USAGE)

    assert _line(cost, "uncached_input")["amount"] == 0.009
    assert _line(cost, "cache_read")["amount"] == round(8300 * 0.25 / 1e6, 6)
    assert _line(cost, "output")["amount"] == 0.045
    assert _line(cost, "web_search")["amount"] == 0.01
    assert _line(cost, "web_search")["rate_label"] == "$0.01 / search"


def test_every_line_shows_its_unit_rate():
    cost = cost_breakdown(FABLE, USAGE)

    for line in cost["lines"]:
        assert line["rate_label"].startswith("$")


def test_total_input_counts_uncached_writes_and_reads():
    cost = cost_breakdown(FABLE, USAGE)

    assert cost["input_total_tokens"] == 900 + 0 + 8300


def test_the_hit_rate_is_token_level_not_request_level():
    cost = cost_breakdown(FABLE, USAGE)

    assert cost["cache_hit_rate"] == 8300 / 9200


def test_savings_compare_against_the_no_cache_turn():
    cost = cost_breakdown(FABLE, USAGE)

    # No-cache: everything billed as plain input.
    expected_no_cache = 9200 * 10 / 1e6 + 900 * 50 / 1e6 + 0.01
    assert cost["no_cache_total"] == round(expected_no_cache, 6)
    assert cost["savings"] == round(expected_no_cache - cost["total"], 6)


def test_a_cache_rewrite_can_cost_more_than_it_saves():
    # TTL expired: the whole prefix is written again at 1.25x.
    usage = dict(
        USAGE,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=8300,
        cache_creation={
            "ephemeral_5m_input_tokens": 8300,
            "ephemeral_1h_input_tokens": 0,
        },
    )
    cost = cost_breakdown(FABLE, usage)

    assert cost["savings"] < 0
    assert cost["cache_hit_rate"] == 0


# --- honest absences -------------------------------------------------------------


def test_missing_cache_counters_are_zeroed_and_flagged_not_invented():
    # Caching off: llm-anthropic drops the counters, so the zero here is
    # "no cache activity possible" - a fact, but a flagged one.
    usage = {"input_tokens": 100, "output_tokens": 50}
    cost = cost_breakdown(FABLE, usage)

    assert cost["cache_counters_reported"] is False
    assert _line(cost, "cache_read")["amount"] == 0
    assert cost["cache_hit_rate"] == 0


def test_no_tokens_no_estimate():
    assert cost_breakdown(FABLE, {}) is None
    assert cost_breakdown(FABLE, {"input_tokens": 10}) is None


def test_an_unsplit_write_total_is_a_five_minute_write():
    # llm-anthropic only ever writes a 5m entry, so when the 5m/1h split is
    # absent the total lands on the 5m line.
    usage = dict(USAGE, cache_creation=None, cache_creation_input_tokens=500)
    cost = cost_breakdown(FABLE, usage)

    assert _line(cost, "cache_write_5m")["quantity"] == 500
    assert all(line["key"] != "cache_write_1h" for line in cost["lines"])


def test_a_zero_split_cannot_hide_a_real_write_total():
    # Seen in the wild: a web-search turn reported a 0 + 0 split against a
    # total of 8,940, and the cost panel showed "Cache write 0" while the
    # countdown ticked on the same 8,940. The total is the fact; the split
    # is a detail that lost the right to override it.
    usage = dict(
        USAGE,
        cache_creation_input_tokens=8940,
        cache_creation={
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    )
    cost = cost_breakdown(FABLE, usage)

    assert _line(cost, "cache_write_5m")["quantity"] == 8940
    assert all(line["key"] != "cache_write_1h" for line in cost["lines"])


def test_a_partial_split_still_answers_to_the_total():
    # Even when the split is partly plausible, the total wins: the 1h detail
    # keeps its claim, the unaccounted remainder lands on the 5m line.
    usage = dict(
        USAGE,
        cache_creation_input_tokens=8940,
        cache_creation={
            "ephemeral_5m_input_tokens": 100,
            "ephemeral_1h_input_tokens": 60,
        },
    )
    cost = cost_breakdown(FABLE, usage)

    assert _line(cost, "cache_write_1h")["quantity"] == 60
    assert _line(cost, "cache_write_5m")["quantity"] == 8880


def test_a_one_hour_write_gets_its_own_line_and_rate():
    usage = dict(
        USAGE,
        cache_creation_input_tokens=100,
        cache_creation={
            "ephemeral_5m_input_tokens": 40,
            "ephemeral_1h_input_tokens": 60,
        },
    )
    cost = cost_breakdown(FABLE, usage)

    assert _line(cost, "cache_write_1h")["amount"] == round(60 * 20 / 1e6, 6)


# --- the footer surface ---------------------------------------------------------


def test_the_footer_has_the_bar_and_the_popover(static_page):
    assert 'id="costBar"' in static_page
    assert 'id="costPop"' in static_page
    assert 'id="costToggle"' in static_page
    # The TTL countdown keeps its slot in the bar.
    assert 'id="cacheState"' in static_page


def test_the_old_fact_rows_are_gone(static_page):
    # Model data and dynamic filtering duplicated the settings panel.
    assert "modelDataSource" not in static_page
    assert "dynamicFiltering" not in static_page
    assert "class=\"facts\"" not in static_page


def test_the_popover_speaks_the_industry_terms(static_page):
    assert "prompt cache hit rate" in static_page
    assert "vs no-cache" in static_page


def test_the_page_can_say_why_there_are_no_rates(monkeypatch):
    monkeypatch.setattr(rates_page, "load_cache", lambda: None)

    status = pricing.rates_status()

    assert status["rates_state"] == "unavailable"
    assert status["rates_error"]
    assert status["rates_url"] == RATES_URL
