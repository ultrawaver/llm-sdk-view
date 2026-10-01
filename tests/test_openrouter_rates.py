"""OpenRouter's catalogue, read as a price list.

The catalogue is the source of these prices, so what these tests pin is the
reading of it: a per-token string becomes a rate per million tokens, a
long-context override is a tier chosen by the prompt's own size, the two
tiers of one model are their own rows, and a zero is never read as "free"
without the slug saying so.
"""

import json
from datetime import datetime, timedelta, timezone

from native_api_chat import openrouter_api, rates_openrouter


def row(model_id, **pricing):
    """One catalogue entry, in the shape OpenRouter sends it."""
    return {"id": model_id, "pricing": pricing}


def test_a_per_token_price_is_kept_per_million_tokens(openrouter_catalogue):
    """The catalogue writes "0.0000025"; the receipt prints $2.50 / MTok."""
    openrouter_catalogue(
        row("vendor/model", prompt="0.0000025", completion="0.000015")
    )

    rates = rates_openrouter.rates_for("openrouter/vendor/model")

    assert rates.input == 2.5
    assert rates.output == 15.0


def test_a_free_model_is_marked_by_its_slug_and_not_by_its_price(
    openrouter_catalogue,
):
    """Zero is what OpenRouter writes for a model it is not pricing, too.

    Four models in the live catalogue are priced at zero without being on the
    free tier. Reading a zero as "free" would price those turns at $0.00 and
    be wrong by the whole bill, so only the slug - which names the tier - is
    read as free.
    """
    openrouter_catalogue(
        row("vendor/not-priced", prompt="0", completion="0"),
        row("vendor/generous-1:free", prompt="0", completion="0"),
    )

    assert rates_openrouter.rates_for("openrouter/vendor/not-priced") is None
    free = rates_openrouter.rates_for("openrouter/vendor/generous-1:free")
    assert free is not None
    assert (free.input, free.output) == (0.0, 0.0)


def test_each_pricing_tier_is_its_own_row(openrouter_catalogue):
    """Matching ":batch" against its base model would charge double."""
    openrouter_catalogue(
        row("vendor/model", prompt="0.000002", completion="0.00001"),
        row("vendor/model:batch", prompt="0.000001", completion="0.000005"),
    )

    base = rates_openrouter.rates_for("openrouter/vendor/model")
    batch = rates_openrouter.rates_for("openrouter/vendor/model:batch")

    assert (base.input, base.output) == (2.0, 10.0)
    assert (batch.input, batch.output) == (1.0, 5.0)


def test_a_long_prompt_is_priced_at_the_tier_it_falls_in(openrouter_catalogue):
    """The threshold is the catalogue's own, and the prompt size is known.

    A 300,000-token prompt on a model that doubles above 272,000 is charged
    the doubled rate; pricing it at the base rate is wrong by half.
    """
    openrouter_catalogue(
        row(
            "vendor/model",
            prompt="0.0000025",
            completion="0.000015",
            input_cache_read="0.00000025",
            overrides=[
                {
                    "min_prompt_tokens": 272000,
                    "prompt": "0.000005",
                    "completion": "0.0000225",
                    "input_cache_read": "0.0000005",
                }
            ],
        )
    )

    rates = rates_openrouter.rates_for("openrouter/vendor/model")

    assert (rates.at(1000).input, rates.at(1000).output) == (2.5, 15.0)
    assert rates.at(272000).input == 5.0
    assert rates.at(300000).output == 22.5
    assert rates.at(300000).cache_read == 0.5
    # And it says which tier it moved to, so a receipt can explain the rate.
    assert rates.at(300000).long_context_from == 272000
    assert rates.at(1000).long_context_from is None


def test_an_override_changes_only_the_rates_it_names(openrouter_catalogue):
    """The catalogue writes an override as a patch, not as a whole row."""
    openrouter_catalogue(
        row(
            "vendor/model",
            prompt="0.000002",
            completion="0.00001",
            web_search="0.01",
            overrides=[{"min_prompt_tokens": 1000, "prompt": "0.000004"}],
        )
    )

    rates = rates_openrouter.rates_for("openrouter/vendor/model").at(5000)

    assert rates.input == 4.0
    assert rates.output == 10.0
    assert rates.web_search == 0.01


def test_a_row_that_is_not_a_price_list_is_skipped(openrouter_catalogue):
    """One malformed entry must not cost the whole table."""
    openrouter_catalogue(
        row("vendor/model", prompt="0.000002", completion="0.00001"),
        {"id": "vendor/no-pricing"},
        {"id": "vendor/bad-prompt", "pricing": {"prompt": "free", "completion": "0.1"}},
        {"pricing": {"prompt": "0.1", "completion": "0.1"}},
    )

    table = rates_openrouter.snapshot()["rates"]

    assert set(table) == {"vendor/model"}


def test_the_routing_prefix_is_not_part_of_the_models_name(openrouter_catalogue):
    """llm builds `openrouter/<slug>`; the catalogue has never heard of it."""
    openrouter_catalogue(row("openai/gpt-5.4", prompt="0.0000025", completion="0.000015"))

    assert rates_openrouter.slug_of("openrouter/openai/gpt-5.4") == "openai/gpt-5.4"
    assert rates_openrouter.rates_for("openrouter/openai/gpt-5.4") is not None
    # A model from another provider, or none at all, is simply not here.
    assert rates_openrouter.rates_for("claude-fable-5-1") is None
    assert rates_openrouter.rates_for(None) is None


def test_the_prices_are_read_again_when_the_catalogue_is(openrouter_catalogue):
    """The rows are built once and kept until the file they came from changes.

    The catalogue is most of a megabyte, and a receipt recomputes its own cost
    on every render, so re-parsing it per turn is not affordable - but a
    background refresh has to be picked up, or yesterday's prices would stand
    for the life of the process.
    """
    openrouter_catalogue(row("vendor/model", prompt="0.000002", completion="0.00001"))
    first = rates_openrouter.rates_for("openrouter/vendor/model")

    openrouter_catalogue(row("vendor/model", prompt="0.000003", completion="0.00002"))

    assert first.input == 2.0
    assert rates_openrouter.rates_for("openrouter/vendor/model").input == 3.0


def test_the_status_names_the_catalogue_and_the_day_it_was_read(
    openrouter_catalogue,
):
    openrouter_catalogue(row("vendor/model", prompt="0.000002", completion="0.00001"))

    status = rates_openrouter.status()

    assert status["rates_source"] == rates_openrouter.RATES_SOURCE
    assert status["rates_state"] == "live"
    assert status["rates_date"] == datetime.now(timezone.utc).date().isoformat()
    assert status["models"] == 1


def test_a_catalogue_past_its_window_is_shown_as_a_cache(openrouter_catalogue):
    """A stale catalogue is labelled, and says why it is the one in use."""
    entries = openrouter_catalogue(
        row("vendor/model", prompt="0.000002", completion="0.00001")
    )
    openrouter_api.cache_path().write_text(
        json.dumps({
            "fetched_at": (
                datetime.now(timezone.utc) - timedelta(days=3)
            ).isoformat(),
            "models": entries,
        }),
        "utf-8",
    )

    status = rates_openrouter.status()

    assert status["rates_state"] == "cached"
    assert status["rates_source"] == f"{rates_openrouter.RATES_SOURCE} (cached)"
    assert status["rates_error"] == "the cached OpenRouter catalogue is stale"


def test_a_machine_with_no_catalogue_says_so_rather_than_guessing(
    no_openrouter_catalogue,
):
    """No catalogue is not an empty price list: it is an answer, and it is
    "unavailable"."""
    status = rates_openrouter.status()

    assert status["rates_state"] == "unavailable"
    assert status["rates_error"] == "no cached OpenRouter catalogue"
    assert status["models"] == 0
    assert rates_openrouter.rates_for("openrouter/vendor/model") is None
