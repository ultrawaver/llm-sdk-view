"""An OpenRouter turn's cost, from OpenRouter's own catalogue prices.

The prices themselves are pinned in ``test_openrouter_rates``. What is pinned
here is the arithmetic over them, and the four places OpenRouter's counters
differ from Anthropic's - each of which turns a plausible-looking number into
a wrong one if it is assumed away:

- ``input_tokens`` is the **whole prompt**, so the cached and written tokens
  are slices of it and not a second count on top;
- a **cache write is reported**, under a field sent only for models that
  price one, so a turn that wrote says so and a turn that did not is not
  read as a turn that wrote nothing;
- a **search is reported** too, in the same ``server_tool_use`` count
  Anthropic uses;
- the write count **carries no TTL**, so a model that prices two of them
  differently cannot have its write charged at either.
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


def test_a_reported_cache_write_is_charged_and_not_counted_twice(
    openrouter_catalogue,
):
    """OpenRouter does report a write, and it is a slice of the prompt too.

    So the written tokens leave the uncached figure and are charged at the
    write rate. Dropping the line priced them at the input rate, which for a
    write is the wrong one by however much the catalogue says a write costs;
    leaving them in the uncached figure as well would bill them twice.
    """
    model = install(openrouter_catalogue, "vendor/writes",
                    input_cache_write="0.000005")

    cost = cost_breakdown(
        model,
        {"input_tokens": 1200, "output_tokens": 900},
        {"cache_read": 0, "cache_creation": 1000},
    )

    assert line(cost, "cache_write")["quantity"] == 1000
    assert line(cost, "cache_write")["rate"] == 5.0
    assert line(cost, "cache_write")["amount"] == 0.005
    assert line(cost, "cache_write")["reported"] is True
    assert line(cost, "uncached_input")["quantity"] == 200
    assert cost["total"] == 0.019
    # The prompt is still the whole input: a write adds nothing to it.
    assert cost["input_total_tokens"] == 1200
    # A write is dearer than reading the same tokens as fresh input, so the
    # comparison against an uncached turn is a cost rather than a saving.
    assert cost["savings"] < 0


def test_a_write_count_with_two_ttl_rates_is_counted_but_not_charged(
    openrouter_catalogue,
):
    """OpenRouter reports one write count and states no TTL for it.

    The catalogue may price two, and does - differently - for every row that
    carries both, so there is no rate the count belongs to. Charging it at
    either would be picking a price the API never named.
    """
    model = install(openrouter_catalogue, "vendor/two-ttls",
                    input_cache_write="0.0000025",
                    input_cache_write_1h="0.000004")

    cost = cost_breakdown(
        model,
        {"input_tokens": 1200, "output_tokens": 900},
        {"cache_read": 0, "cache_creation": 1000},
    )

    write = line(cost, "cache_write")
    assert write["quantity"] == 1000
    assert write["reported"] is True
    assert write["rate"] is None
    assert write["amount"] == 0
    assert any("no TTL" in note for note in cost["notes"]), cost["notes"]
    # Counted, so still out of the uncached figure - it did happen.
    assert line(cost, "uncached_input")["quantity"] == 200


def test_an_unreported_write_is_not_read_as_a_write_of_nothing(
    openrouter_catalogue,
):
    """The field is sent only for models with a cache-write price.

    Its absence is "not reported": the row says so rather than printing a
    confident zero, and the receipt says which way the estimate leans - any
    written tokens are in the uncached figure, priced at the input rate.
    """
    model = install(openrouter_catalogue)

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900},
                          {"cache_read": 0})

    write = line(cost, "cache_write")
    assert write["reported"] is False
    assert write["quantity"] == 0
    assert write["note"] == "unreported"
    assert any("did not report a cache write" in note for note in cost["notes"])


def test_a_write_row_exists_on_every_receipt(openrouter_catalogue):
    """The row the receipt was missing.

    Cache writes read as an absent line, and an absent line reads as a free
    one - which is how a receipt could price an input whose explanation it
    did not have.
    """
    plain = install(openrouter_catalogue, "vendor/plain",
                    input_cache_write=None, web_search=None)

    for counts in ({}, {"cache_read": 0}, {"cache_read": 512}):
        cost = cost_breakdown(
            plain, {"input_tokens": 1200, "output_tokens": 900}, counts
        )
        assert line(cost, "cache_write") is not None, counts


def test_no_receipt_claims_openrouter_reports_no_count(openrouter_catalogue):
    """The sentence that shipped was about the API and was not true.

    "OpenRouter reports no count for them" was printed beside cache writes
    and searches. Both are reported - the write under a field sent only for
    models with a write price, the search in the same ``server_tool_use``
    count Anthropic uses. A note may say a count did not arrive; it may not
    say the provider has none to send.
    """
    model = install(openrouter_catalogue)

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900}, {})

    assert not any("reports no count" in note for note in cost["notes"])
    assert any("did not report a cache write" in note for note in cost["notes"])
    assert any("no search count" in note for note in cost["notes"])


def test_a_search_is_charged_where_the_catalogue_prices_one(
    openrouter_catalogue,
):
    """OpenRouter counts a search in the count the shared reader already
    hoists, and the catalogue prices one per request rather than per token."""
    model = install(openrouter_catalogue)

    cost = cost_breakdown(
        model,
        {"input_tokens": 1200, "output_tokens": 900, "web_search_requests": 2},
        {"cache_read": 0, "cache_creation": 0},
    )

    search = line(cost, "web_search")
    assert search["quantity"] == 2
    assert search["rate"] == 0.01
    assert search["amount"] == 0.02
    assert search["group"] == "tools"
    assert not any("no search count" in note for note in cost["notes"])


def test_a_rate_the_catalogue_never_published_is_not_printed_as_zero(
    openrouter_catalogue,
):
    """A price of nothing and no price at all are different claims.

    Most rows carry no cache-read rate. Printing ``$0.00 / MTok`` beside one
    of them tells the reader the model charges nothing for a cache read, and
    the catalogue never said that - it said nothing.
    """
    model = install(openrouter_catalogue, "vendor/no-read-rate",
                    input_cache_read=None)

    cost = cost_breakdown(model, {"input_tokens": 1200, "output_tokens": 900},
                          {"cache_read": 0})

    read = line(cost, "cache_read")
    assert read["quantity"] == 0
    assert read["rate"] is None
    assert read["rate_label"] == pricing.NO_RATE
    assert "$0.00" not in read["rate_label"]


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
