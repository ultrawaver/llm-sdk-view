"""What is known about an OpenRouter model, and how sure we are of it.

The request tests live next door in ``test_openrouter.py``; this file is about
the other half - the catalogue that says what a model accepts and costs, and
the rule that a fact nobody could read must never be reported as a fact.

Nothing here reaches the network. ``no_openrouter_catalogue`` makes the fetch
a refusal, so the default state in these tests is the state of a machine that
has just opened the page for the first time.
"""

import json

import llm
import pytest

from native_api_chat import openrouter_api
from native_api_chat.providers.openrouter import (
    CHAT_COMPLETIONS,
    EDITABLE,
    NOT_SENT_BY_PLUGIN,
    RESPONSES,
    UNKNOWN_TO_CATALOGUE,
    UNSUPPORTED_BY_MODEL,
    OpenRouterProvider,
    available_model_ids,
    slug_for,
)

pytest.importorskip("llm_openrouter")

MODEL_ID = "openrouter/anthropic/claude-sonnet-5"
PROVIDER = OpenRouterProvider()


# --- the catalogue as a source ----------------------------------------------


def test_no_cache_reads_as_no_catalogue_and_says_why():
    """Never an empty success: "I have no models" and "there are no models"
    are different answers, and only one of them is true here."""
    snapshot = openrouter_api.snapshot()

    assert snapshot["models"] == []
    assert snapshot["provenance"]["source"] == "none"
    assert snapshot["provenance"]["cache"] == "none"
    assert snapshot["provenance"]["error"] == "no cached OpenRouter catalogue"


def test_a_fresh_cache_is_labelled_as_disk(openrouter_catalogue):
    openrouter_catalogue("anthropic/claude-sonnet-5")
    snapshot = openrouter_api.snapshot()

    assert [entry["id"] for entry in snapshot["models"]] == ["anthropic/claude-sonnet-5"]
    assert snapshot["provenance"]["cache"] == "disk"
    assert snapshot["provenance"]["error"] is None
    assert snapshot["provenance"]["fetched_at"]


def test_a_stale_cache_is_still_used_but_labelled_stale(openrouter_catalogue):
    """A price from this morning is worth having; presenting it as live is
    not. The figure is kept and the label carries the doubt."""
    openrouter_catalogue("anthropic/claude-sonnet-5")
    payload = json.loads(openrouter_api.cache_path().read_text("utf-8"))
    payload["fetched_at"] = "2020-01-01T00:00:00+00:00"
    openrouter_api.cache_path().write_text(json.dumps(payload), "utf-8")

    snapshot = openrouter_api.snapshot()

    assert [entry["id"] for entry in snapshot["models"]] == ["anthropic/claude-sonnet-5"]
    assert snapshot["provenance"]["cache"] == "stale-disk"
    assert "stale" in snapshot["provenance"]["error"]


def test_a_failed_refresh_falls_back_to_the_cache_it_could_not_replace(
    openrouter_catalogue,
):
    openrouter_catalogue("anthropic/claude-sonnet-5")
    result = openrouter_api.refresh()

    assert [entry["id"] for entry in result["models"]] == ["anthropic/claude-sonnet-5"]
    assert "tests never read the catalogue" in result["provenance"]["error"]


def test_a_refresh_that_succeeds_is_labelled_live(monkeypatch):
    monkeypatch.setattr(
        openrouter_api, "fetch_models", lambda timeout=5.0: [{"id": "a/b"}]
    )
    result = openrouter_api.refresh()

    assert result["provenance"]["cache"] == "live"
    assert result["provenance"]["error"] is None


def test_a_half_written_cache_never_reads_as_a_catalogue(openrouter_catalogue):
    """The background refresh writes while a request may be reading, so the
    write is atomic and no partial file is left behind to be found."""
    openrouter_catalogue("anthropic/claude-sonnet-5")
    directory = openrouter_api.cache_path().parent

    assert not list(directory.glob("*.tmp"))


def test_a_corrupt_cache_is_no_cache(openrouter_catalogue):
    openrouter_catalogue("anthropic/claude-sonnet-5")
    openrouter_api.cache_path().write_text("{ not json", "utf-8")

    assert openrouter_api.load_cache() is None
    assert openrouter_api.snapshot()["models"] == []


# --- what the catalogue says about one model --------------------------------


def test_the_facts_come_from_the_catalogue(openrouter_catalogue):
    openrouter_catalogue("anthropic/claude-sonnet-5")
    capabilities = PROVIDER.capabilities_for(MODEL_ID)

    assert capabilities.slug == "anthropic/claude-sonnet-5"
    assert capabilities.context_window == 1_000_000
    assert capabilities.max_output_tokens == 128_000
    assert capabilities.created == 1782843083
    assert capabilities.input_modalities == ("text", "image", "file")
    assert capabilities.data_source == "openrouter-models"
    assert capabilities.described


def test_the_output_ceiling_comes_from_the_serving_provider(openrouter_catalogue):
    """``context_length`` is the model's; ``max_completion_tokens`` is the
    endpoint's, and the endpoint is what the request has to fit."""
    openrouter_catalogue(
        {
            "id": "vendor/model-1",
            "context_length": 200_000,
            "top_provider": {"context_length": 200_000, "max_completion_tokens": 8_192},
            "supported_parameters": ["max_tokens"],
            "pricing": {},
        }
    )
    capabilities = PROVIDER.capabilities_for("openrouter/vendor/model-1")

    assert capabilities.context_window == 200_000
    assert capabilities.max_output_tokens == 8_192


def test_a_model_the_catalogue_does_not_cover_is_unknown_not_unsupported(
    openrouter_catalogue,
):
    """The distinction this whole module exists for. A machine that could not
    read the catalogue knows nothing about the model; telling the user every
    control is unsupported by it would be an invented fact."""
    openrouter_catalogue("anthropic/claude-sonnet-5")
    capabilities = PROVIDER.capabilities_for("openrouter/vendor/never-heard-of-it")

    assert not capabilities.described
    assert capabilities.context_window is None
    status, reason = capabilities.option_status("reasoning_effort", RESPONSES)
    assert status == UNKNOWN_TO_CATALOGUE
    assert "does not describe vendor/never-heard-of-it" in reason


def test_an_unreadable_catalogue_does_not_disable_every_control():
    """With no catalogue at all, a control is unknown rather than refused,
    and the reason carries the fetch failure that caused it."""
    capabilities = PROVIDER.capabilities_for(MODEL_ID)
    status, reason = capabilities.option_status("reasoning_effort", RESPONSES)

    assert status == UNKNOWN_TO_CATALOGUE
    assert "no cached OpenRouter catalogue" in reason


# --- the three ways a control can be unavailable ----------------------------


def test_a_parameter_the_model_lists_is_editable(openrouter_catalogue):
    openrouter_catalogue("anthropic/claude-sonnet-5")
    capabilities = PROVIDER.capabilities_for(MODEL_ID)

    assert capabilities.option_status("reasoning_effort", RESPONSES) == (EDITABLE, "")


def test_a_parameter_the_model_does_not_list_is_refused(openrouter_catalogue):
    openrouter_catalogue(
        {
            "id": "anthropic/claude-sonnet-5",
            "supported_parameters": ["max_tokens"],
            "pricing": {},
            "top_provider": {},
        }
    )
    status, reason = PROVIDER.capabilities_for(MODEL_ID).option_status(
        "reasoning_effort", RESPONSES
    )

    assert status == UNSUPPORTED_BY_MODEL
    assert "does not list reasoning_effort" in reason


def test_an_option_the_plugin_drops_is_reported_as_not_sent(openrouter_catalogue):
    """Measured in test_openrouter.py against the real request: on the Chat
    Completions path llm-openrouter never copies reasoning_summary into
    `reasoning`. A form that offered it there would be offering nothing."""
    openrouter_catalogue("anthropic/claude-sonnet-5")
    capabilities = PROVIDER.capabilities_for(MODEL_ID)

    on_responses, _ = capabilities.option_status("reasoning_summary", RESPONSES)
    on_chat, reason = capabilities.option_status("reasoning_summary", CHAT_COMPLETIONS)

    assert on_responses == EDITABLE
    assert on_chat == NOT_SENT_BY_PLUGIN
    assert "reasoning_summary" in reason


def test_openrouters_own_routing_surface_needs_no_catalogue_entry():
    """`provider` and `chat_completions` are OpenRouter's, not the model's:
    the catalogue never lists them and every model takes them, so they stay
    editable even on a machine that has never read the catalogue."""
    capabilities = PROVIDER.capabilities_for(MODEL_ID)

    assert capabilities.option_status("provider", RESPONSES) == (EDITABLE, "")
    assert capabilities.option_status("chat_completions", RESPONSES) == (EDITABLE, "")


# --- prices -----------------------------------------------------------------


def test_prices_come_with_the_models(openrouter_catalogue):
    openrouter_catalogue("anthropic/claude-sonnet-5")
    capabilities = PROVIDER.capabilities_for(MODEL_ID)

    assert capabilities.price("prompt") == 0.000002
    assert capabilities.price("completion") == 0.00001
    assert capabilities.price("input_cache_read") == 0.0000002


@pytest.mark.parametrize("written", ["0", "", "-1", None, "nonsense"])
def test_a_missing_or_zero_price_is_no_estimate_never_free(
    openrouter_catalogue, written
):
    """OpenRouter writes "0" both for models it is not pricing and for models
    that really are free. Guessing which would put an invented number on the
    page, so neither is reported as a rate."""
    pricing = {} if written is None else {"prompt": written}
    openrouter_catalogue(
        {"id": "anthropic/claude-sonnet-5", "pricing": pricing, "top_provider": {}}
    )

    assert PROVIDER.capabilities_for(MODEL_ID).price("prompt") is None


def test_a_price_is_never_borrowed_from_a_neighbour(openrouter_catalogue):
    openrouter_catalogue("anthropic/claude-sonnet-5")

    assert PROVIDER.capabilities_for("openrouter/anthropic/claude-opus-5").price(
        "prompt"
    ) is None


# --- which models are offered -----------------------------------------------


def test_a_keyless_machine_offers_no_openrouter_models_and_says_so():
    """llm-openrouter registers nothing without a key, so every row would be
    a model this machine cannot send to."""
    assert available_model_ids() == ()
    catalog = PROVIDER.catalog()

    assert catalog["models"] == []
    assert "llm keys set openrouter" in catalog["note"]


def test_the_models_offered_are_the_ones_llm_registered(openrouter_registry):
    """Not the 464 the catalogue describes: the ids come from llm's registry,
    which is what the runtime can actually honour."""
    openrouter_registry("anthropic/claude-sonnet-5", "openai/gpt-5.4")
    catalog = PROVIDER.catalog()

    assert set(catalog["models"]) == {
        "openrouter/anthropic/claude-sonnet-5",
        "openrouter/openai/gpt-5.4",
    }
    assert catalog["note"] == ""


def test_a_catalogue_model_that_is_not_registered_is_not_offered(
    openrouter_registry, openrouter_catalogue
):
    openrouter_registry("anthropic/claude-sonnet-5")
    openrouter_catalogue("anthropic/claude-sonnet-5", "vendor/unregistered-1")

    assert PROVIDER.catalog()["models"] == [
        "openrouter/anthropic/claude-sonnet-5"
    ]


def test_only_the_newest_of_a_series_is_offered_and_the_rest_are_named(
    openrouter_registry,
):
    """Superseded models are listed rather than dropped: a list that silently
    lost two entries reads like lost data instead of like a rule."""
    openrouter_registry(
        {"id": "openai/gpt-5.2", "created": 1_700_000_000, "pricing": {}},
        {"id": "openai/gpt-5.4", "created": 1_800_000_000, "pricing": {}},
    )
    catalog = PROVIDER.catalog()

    assert catalog["models"] == ["openrouter/openai/gpt-5.4"]
    assert catalog["superseded"] == [
        {
            "id": "openrouter/openai/gpt-5.2",
            "series": "family:openai/gpt",
            "kept_by": "openrouter/openai/gpt-5.4",
        }
    ]
    assert "most recently published" in catalog["rule"]


def test_the_catalogue_is_read_once_for_the_whole_list(openrouter_registry, monkeypatch):
    """The catalogue is an 800KB document. Reading it per model turned a list
    of fifty into fifty parses of it."""
    openrouter_registry("anthropic/claude-sonnet-5", "openai/gpt-5.4", "x-ai/grok-4.3")
    reads = []
    real = openrouter_api.snapshot
    monkeypatch.setattr(
        openrouter_api, "snapshot", lambda: (reads.append(1), real())[1]
    )

    PROVIDER.catalog()

    assert len(reads) == 1


def test_the_registered_model_is_one_llm_can_actually_build(openrouter_registry):
    """The registry fixture is only worth trusting if what it registers is a
    real plugin model with real options, not a name in a list."""
    (model_id,) = openrouter_registry("anthropic/claude-sonnet-5")
    model = llm.get_model(model_id)

    assert model.model_name == slug_for(model_id)
    assert model.headers == {
        "HTTP-Referer": "https://llm.datasette.io/",
        "X-OpenRouter-Title": "LLM",
    }
    assert "chat_completions" in model.Options.model_fields
