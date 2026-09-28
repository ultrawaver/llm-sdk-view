"""The pricing page as the source of unit prices.

Rates used to be transcribed into the source, which meant every upstream
price change needed a commit. They are fetched now, so the two things worth
pinning are: a well-formed page parses into the rates we expect, and every
way of not having current prices is reported honestly - a cache is labelled
as a cache, and no prices at all means no estimate rather than a guess.

Nothing here touches the network; the fixture markdown is a copy of the
page's own shape.
"""

import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_sdk_view import rates_page
from llm_sdk_view.pricing import ModelRates
from llm_sdk_view.rates_page import (
    RatesParseError,
    RatesUnavailableError,
    fetch_pricing_markdown,
    fetch_rates,
    parse_model_pricing,
    snapshot,
    snapshot_from_cache,
)

# The page's own shape, kept in a file because a pricing table is wider than
# any line length worth enforcing.
PAGE = (Path(__file__).parent / "fixtures" / "pricing-page.md").read_text("utf-8")

HAIKU = ModelRates(1.0, 5.0, 1.25, 2.0, 0.10)


def _haiku_row() -> dict:
    return {
        "input": 1.0,
        "output": 5.0,
        "cache_write_5m": 1.25,
        "cache_write_1h": 2.0,
        "cache_read": 0.10,
    }


def _stale(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


# --- parsing ----------------------------------------------------------------


def test_the_page_table_parses_into_rates():
    parsed = parse_model_pricing(PAGE)

    assert parsed["claude-haiku-4-5"] == HAIKU
    assert parsed["claude-fable-5-1"] == ModelRates(10.0, 50.0, 12.50, 20.0, 0.25)
    assert parsed["claude-opus-5-5"].cache_read == 0.20


def test_a_name_with_a_link_and_a_parenthetical_becomes_a_clean_id():
    parsed = parse_model_pricing(PAGE)

    # "[limited availability](url)" and "(retired, except on Bedrock)" are
    # page furniture, not part of the model id.
    assert "claude-mythos-5-1" in parsed
    assert "claude-opus-4-1" in parsed


def test_only_the_model_pricing_section_is_read():
    parsed = parse_model_pricing(PAGE)

    # The cloud table below it has different columns and higher prices.
    assert parsed["claude-opus-5-5"].input == 4.0


# --- a page that changed shape ------------------------------------------------


def test_a_page_without_the_pricing_section_is_an_error():
    with pytest.raises(RatesParseError, match="Model pricing"):
        parse_model_pricing("# Pricing\n\nNothing here yet.\n")


def test_a_table_missing_a_column_is_an_error():
    page = PAGE.replace("| 1h cache writes |", "| Something else |")

    with pytest.raises(RatesParseError, match="1h cache writes"):
        parse_model_pricing(page)


def test_a_truncated_table_is_an_error_not_a_smaller_table():
    head, _, _ = PAGE.partition("| Claude Sonnet 5")

    with pytest.raises(RatesParseError, match="only 3 models"):
        parse_model_pricing(head)


def test_a_row_with_no_readable_price_is_an_error():
    page = PAGE.replace("$1 / MTok         | $1.25", "n/a              | $1.25")

    with pytest.raises(RatesParseError, match="no price could be read"):
        parse_model_pricing(page)


def test_an_unreachable_page_is_a_reason_not_a_guess(monkeypatch):
    def explode(*_args, **_kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", explode)

    with pytest.raises(RatesUnavailableError, match="could not read"):
        fetch_pricing_markdown()


def test_fetch_rates_parses_what_it_read(monkeypatch):
    monkeypatch.setattr(rates_page, "fetch_pricing_markdown", lambda **_k: PAGE)

    assert fetch_rates()["claude-haiku-4-5"] == HAIKU


# --- the cache -----------------------------------------------------------------


def test_a_fetch_is_written_to_disk_and_reads_back(known_rates):
    known_rates["use_real_cache"]()

    rates_page.save_cache({"claude-haiku-4-5": HAIKU})
    cached = rates_page.load_cache()

    assert cached is not None
    assert rates_page._as_rates(cached["rates"])["claude-haiku-4-5"] == HAIKU


def test_a_half_written_cache_is_never_read_as_empty(known_rates):
    known_rates["use_real_cache"]()
    path = rates_page.cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"fetched_at": "2026-09-27T00:00:00+00:00", "rat', "utf-8")

    assert rates_page.load_cache() is None


def test_a_row_that_will_not_parse_costs_only_that_row(known_rates):
    known_rates["use_real_cache"]()
    rates_page.save_cache({"claude-haiku-4-5": HAIKU})
    cached = rates_page.load_cache()
    cached["rates"]["claude-broken-9"] = {"input": "free"}

    assert list(rates_page._as_rates(cached["rates"])) == ["claude-haiku-4-5"]


# --- what the app is told --------------------------------------------------------


def test_a_fresh_cache_is_live():
    result = snapshot()

    assert result["provenance"]["source"] == "pricing-page"
    assert "claude-haiku-4-5" in result["rates"]


def test_a_stale_cache_is_labelled_as_one(monkeypatch):
    monkeypatch.setattr(
        rates_page,
        "load_cache",
        lambda: {"fetched_at": _stale(3), "rates": {"claude-haiku-4-5": _haiku_row()}},
    )

    result = snapshot()

    assert result["provenance"]["source"] == "disk-cache"
    # Still usable, still priced - just not presented as current.
    assert result["rates"]["claude-haiku-4-5"] == HAIKU


def test_no_cache_at_all_means_no_prices_and_a_reason(monkeypatch):
    monkeypatch.setattr(rates_page, "load_cache", lambda: None)

    result = snapshot_from_cache(error="no cached prices")

    assert result["rates"] == {}
    assert result["provenance"]["source"] == "unavailable"
    assert result["provenance"]["error"] == "no cached prices"


def test_a_failed_refresh_falls_back_to_the_cache(monkeypatch):
    monkeypatch.setattr(rates_page, "load_cache", lambda: None)
    monkeypatch.setattr(
        rates_page,
        "fetch_rates",
        lambda **_k: (_ for _ in ()).throw(RatesUnavailableError("offline")),
    )

    result = rates_page.refresh()

    assert result["provenance"]["source"] == "unavailable"
    assert "offline" in result["provenance"]["error"]


def test_the_table_puts_longest_prefixes_first(monkeypatch):
    monkeypatch.setattr(
        rates_page,
        "load_cache",
        lambda: {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "rates": {
                "claude-fable-5": {"input": 10.0, "output": 50.0, "cache_write_5m": 12.5,
                                   "cache_write_1h": 20.0, "cache_read": 1.0},
                "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_write_5m": 12.5,
                                     "cache_write_1h": 20.0, "cache_read": 0.25},
            },
        },
    )

    assert list(rates_page.table()) == ["claude-fable-5-1", "claude-fable-5"]
