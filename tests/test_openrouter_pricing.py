"""An OpenRouter turn's cost, from OpenRouter's own catalogue prices.

The prices themselves are pinned in ``test_openrouter_rates``. What is pinned
here is the arithmetic over them, and the three places OpenRouter's counters
differ from Anthropic's - each of which turns a plausible-looking number into
a wrong one if it is assumed away:

- ``input_tokens`` is the **whole prompt**, so the cached tokens are a slice
  of it and not a second count on top;
- there is **no cache-write count** on either transport;
- **no search count** reaches this app either.
"""

from native_api_chat import openrouter_api, pricing, rates_openrouter
from native_api_chat.pricing import cost_breakdown

MODEL = "openrouter/vendor/model"

#: The catalogue row every test below prices against: $2.50 in, $15 out,
#: $0.25 a cached read, $2.50 a cache write, a cent a search.
PRICES = {
    "prompt": "0.0000025",
    "completion": "0.000015",
    "input_cache_read": "0.00000025",
    "input_cache_write": "0.0000025",
    "web_search": "0.01",
}


def install(openrouter_catalogue, model_id="vendor/model", **pricing):
    """One priced catalogue row, and the model id llm builds from it."""
    openrouter_catalogue({"id": model_id, "pricing": {**PRICES, **pricing}})
    return f"openrouter/{model_id}"


def line(cost, key):
    return next((row for row in cost["lines"] if row["key"] == key), None)


def test_a_turn_is_priced_from_the_catalogue_that_lists_its_model(
    openrouter_catalogue,
):
    model = install(openrouter_catalogue)

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900},
                          {"cache_read": 1024})

    assert cost["rates_source"] == rates_openrouter.RATES_SOURCE
    assert cost["rates_url"] == openrouter_api.MODELS_URL
    assert cost["rates_state"] == "live"
    assert cost["estimated"] is True
    assert line(cost, "uncached_input")["quantity"] == 176
    assert line(cost, "uncached_input")["amount"] == 0.00044
    assert line(cost, "cache_read")["quantity"] == 1024
    assert line(cost, "cache_read")["amount"] == 0.000256
    assert line(cost, "output")["amount"] == 0.0135
    assert cost["total"] == 0.014196


def test_the_cached_tokens_are_not_billed_twice(openrouter_catalogue):
    """llm reads OpenRouter's ``prompt_tokens``, which already include them.

    Charging all 1,200 at the input rate *and* the 1,024 cached ones at their
    own rate would bill the cached tokens twice - a number that looks like a
    number, which is why it is worth a test rather than a comment.
    """
    model = install(openrouter_catalogue)

    with_cache = cost_breakdown(
        model, {"input_tokens": 1200, "output_tokens": 900}, {"cache_read": 1024}
    )
    all_uncached = cost_breakdown(
        model, {"input_tokens": 1200, "output_tokens": 900}, {"cache_read": 0}
    )

    assert with_cache["input_total_tokens"] == 1200
    assert all_uncached["input_total_tokens"] == 1200
    # Caching the same prompt is cheaper, and by the cached tokens' own share.
    assert with_cache["total"] < all_uncached["total"]
    assert with_cache["savings"] == 0.002304
    assert with_cache["cache_hit_rate"] == 1024 / 1200


def test_a_broken_cache_counter_cannot_inflate_the_bill(openrouter_catalogue):
    """A cache read larger than the prompt it is a part of is a broken
    reading, not a bigger turn."""
    model = install(openrouter_catalogue)

    cost = cost_breakdown(
        model, {"input_tokens": 100, "output_tokens": 10}, {"cache_read": 9999}
    )

    assert line(cost, "uncached_input")["quantity"] == 0
    assert line(cost, "cache_read")["quantity"] == 100
    assert cost["input_total_tokens"] == 100


def test_no_cache_write_is_priced_because_none_is_reported(
    openrouter_catalogue,
):
    """OpenRouter prices a write for some models and reports one for none.

    A write line at some assumed quantity would be an invention, so the line
    is absent - and the receipt says the row priced a write, so its absence is
    explained rather than left to look like a free one.
    """
    model = install(openrouter_catalogue)

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900},
                          {"cache_read": 1024})

    assert line(cost, "cache_write_5m") is None
    assert line(cost, "cache_write_1h") is None
    assert line(cost, "web_search") is None
    assert any("cache writes" in note and "searches" in note
               for note in cost["notes"]), cost["notes"]


def test_a_model_that_prices_neither_gets_no_note_about_either(
    openrouter_catalogue,
):
    """A receipt says why a row is missing, and says nothing when none is."""
    model = install(
        openrouter_catalogue, "vendor/plain",
        input_cache_write=None, web_search=None,
    )

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900},
                          {"cache_read": 1024})

    assert cost["notes"] == []


def test_an_unreported_cache_read_is_flagged_and_not_assumed(
    openrouter_catalogue,
):
    """An OpenRouter turn can report no cache counter at all.

    With nothing reported, the whole prompt is priced at the input rate -
    which overstates if anything was cached - so the line says it was not
    reported and the receipt says which way the estimate leans.
    """
    model = install(openrouter_catalogue)

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900}, {})

    assert cost["cache_counters_reported"] is False
    assert line(cost, "cache_read")["quantity"] == 0
    assert line(cost, "cache_read")["reported"] is False
    assert line(cost, "uncached_input")["quantity"] == 1200
    assert line(cost, "uncached_input")["note"] == "cache read unreported"
    assert any("whole prompt is priced at the input rate" in note
               for note in cost["notes"]), cost["notes"]


def test_a_reported_zero_is_not_an_unreported_one(openrouter_catalogue):
    """llm strips zeros out of the details, so the parent's presence is the
    whole signal - and only a reported zero is a hit rate of 0%."""
    model = install(openrouter_catalogue)

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900},
                          {"cache_read": 0})

    assert cost["cache_counters_reported"] is True
    assert line(cost, "cache_read")["reported"] is True
    assert cost["cache_hit_rate"] == 0.0


def test_a_long_prompt_is_priced_at_the_tier_it_fell_in(openrouter_catalogue):
    """The tier is named on the receipt, because the rate is not the model's
    headline rate."""
    model = install(
        openrouter_catalogue,
        overrides=[{
            "min_prompt_tokens": 272000,
            "prompt": "0.000005",
            "completion": "0.0000225",
        }],
    )

    cost = cost_breakdown(model, {"input_tokens": 300000, "output_tokens": 100},
                          {"cache_read": 0})

    assert line(cost, "uncached_input")["rate"] == 5.0
    assert line(cost, "output")["rate"] == 22.5
    assert any("272,000 prompt tokens" in note for note in cost["notes"]), \
        cost["notes"]


def test_a_free_model_costs_nothing_and_is_not_an_invention(
    openrouter_catalogue,
):
    """The slug names the free tier, so $0.00 is a fact rather than a guess
    from a price of zero."""
    model = install(openrouter_catalogue, "vendor/generous-1:free",
                    prompt="0", completion="0",
                    input_cache_read=None, input_cache_write=None,
                    web_search=None)

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900},
                          {"cache_read": 0})

    assert cost["total"] == 0.0
    assert cost["no_cache_total"] == 0.0
    assert cost["rates_state"] == "live"


def test_a_model_the_catalogue_is_not_pricing_gets_no_estimate(
    openrouter_catalogue,
):
    """Four models in the live catalogue are priced at zero without being
    free. A $0.00 estimate for them would be wrong by the whole bill."""
    install(openrouter_catalogue, "vendor/not-priced", prompt="0", completion="0")

    assert cost_breakdown(
        "openrouter/vendor/not-priced", {"input_tokens": 10, "output_tokens": 5}, {}
    ) is None


def test_a_cached_read_this_model_does_not_price_gets_no_estimate(
    openrouter_catalogue,
):
    """There is no rate to use, and the input rate would be an invention."""
    model = install(openrouter_catalogue, "vendor/no-cache-rate",
                    input_cache_read=None)

    assert cost_breakdown(
        model, {"input_tokens": 1200, "output_tokens": 900}, {"cache_read": 1024}
    ) is None
    # With nothing cached there is nothing to price, so the turn still costs.
    assert cost_breakdown(
        model, {"input_tokens": 1200, "output_tokens": 900}, {"cache_read": 0}
    ) is not None


def test_a_turn_without_its_own_counters_gets_no_estimate(
    openrouter_catalogue,
):
    model = install(openrouter_catalogue)

    assert cost_breakdown(model, {"input_tokens": 1200}, {}) is None
    assert cost_breakdown(model, None, {}) is None
    assert cost_breakdown(None, {"input_tokens": 1, "output_tokens": 1}, {}) is None


def test_a_machine_with_no_catalogue_prices_nothing(
    openrouter_catalogue, no_openrouter_catalogue,
):
    """No catalogue is not a free turn: it is no estimate, and it says so."""
    assert cost_breakdown(
        "openrouter/vendor/model", {"input_tokens": 10, "output_tokens": 5}, {}
    ) is None


def test_each_provider_answers_the_rates_question_for_itself(
    openrouter_catalogue,
):
    """Two sources, two labels - and the page is shown the one that applies.

    Telling an OpenRouter turn that the pricing page could not be read sends
    the reader to a document nobody asked.
    """
    install(openrouter_catalogue)

    openrouter = pricing.rates_status("openrouter")
    anthropic = pricing.rates_status()

    assert openrouter["rates_source"] == rates_openrouter.RATES_SOURCE
    assert openrouter["rates_url"] == openrouter_api.MODELS_URL
    assert openrouter["models"] == 1
    assert anthropic["rates_url"] == pricing.RATES_URL
    assert anthropic["rates_source"] == pricing.RATES_SOURCE


def test_an_anthropic_turn_is_still_priced_by_the_pricing_page(
    openrouter_catalogue, key,
):
    """The seam is a dispatch, not a replacement: the other provider's path
    must not move when OpenRouter's arrives."""
    openrouter_catalogue({"id": "vendor/model", "pricing": PRICES})

    cost = cost_breakdown("claude-fable-5-1", {
        "input_tokens": 900,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 8300,
        "output_tokens": 900,
        "web_search_requests": 1,
    })

    assert cost["rates_source"] == "Anthropic pricing"
    assert cost["rates_url"] == pricing.RATES_URL
    # Anthropic's own four counters, and its own price per search.
    assert line(cost, "web_search")["rate"] == 0.01
    assert line(cost, "cache_write_5m") is not None


def test_a_model_no_provider_claims_gets_no_estimate(openrouter_catalogue):
    install(openrouter_catalogue)

    assert pricing.provider_of("gpt-4o") is None
    assert cost_breakdown("gpt-4o", {"input_tokens": 10, "output_tokens": 5}) is None
