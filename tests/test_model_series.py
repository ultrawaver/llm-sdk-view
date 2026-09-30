"""Reading a family and a version out of a model id.

This is where "show only the newest of each series" is decided, so it gets
tested on its own, with nothing else loaded: an id nobody has thought of yet
has to be grouped correctly, and nothing here may touch the network.
"""

from dataclasses import replace

import pytest

from native_api_chat.model_series import (
    OPENROUTER_RULE,
    Series,
    newest_per_series,
    openrouter_series_for,
    series_for,
    series_label,
)

MODERN = {
    "claude-sonnet-5-5": ("sonnet", (5, 5)),
    "claude-sonnet-5": ("sonnet", (5, 0)),
    "claude-opus-5-5": ("opus", (5, 5)),
    "claude-opus-4-8": ("opus", (4, 8)),
    "claude-fable-5-1": ("fable", (5, 1)),
    "claude-mythos-5": ("mythos", (5, 0)),
}

LEGACY = {
    "claude-3-5-sonnet-20240620": ("sonnet", (3, 5)),
    "claude-3-7-sonnet-20250219": ("sonnet", (3, 7)),
    "claude-3-opus-20240229": ("opus", (3, 0)),
    "claude-3-haiku-20240307": ("haiku", (3, 0)),
}

SNAPSHOTS = {
    "claude-opus-4-5-20251101": ("opus", (4, 5)),
    "claude-haiku-4-5-20251001": ("haiku", (4, 5)),
    "claude-sonnet-4-20250514": ("sonnet", (4, 0)),
    "claude-3-5-sonnet-latest": ("sonnet", (3, 5)),
    "claude-3-opus-latest": ("opus", (3, 0)),
}


@pytest.mark.parametrize(
    "model_id,series", [*MODERN.items(), *LEGACY.items(), *SNAPSHOTS.items()]
)
def test_family_and_version_come_out_of_the_id(model_id, series):
    parsed = series_for(model_id)

    assert (parsed.family, parsed.version) == series


@pytest.mark.parametrize(
    "model_id", ["claude-mythos-preview", "claude-not-registered-anywhere", "gpt-4o"]
)
def test_an_unreadable_version_gets_a_series_of_its_own(model_id):
    """Ordering something is not the same as being allowed to hide it: an id
    this module cannot compare still has to survive the narrowing."""
    parsed = series_for(model_id)

    assert parsed.version is None
    kept = newest_per_series([model_id, "claude-sonnet-5", "claude-sonnet-5-5"])
    assert model_id in kept
    assert "claude-sonnet-5-5" in kept


def test_a_date_is_a_snapshot_not_a_version():
    """claude-haiku-4-5 and its dated snapshot are the same model."""
    kept = newest_per_series(["claude-haiku-4-5-20251001", "claude-haiku-4-5"])

    assert len(kept) == 1


def test_the_highest_version_wins_regardless_of_arrival_order():
    kept = newest_per_series(
        ["claude-sonnet-5", "claude-sonnet-5-5", "claude-sonnet-4-6"]
    )

    assert kept == ("claude-sonnet-5-5",)


def test_a_tie_is_broken_by_created_at():
    """Same version from two snapshots: the newer one is the model to keep."""
    created = {"claude-opus-4-5": "2025-11-01", "claude-opus-4-5-20251101": ""}

    kept = newest_per_series(created, created_at=created.get)

    assert kept == ("claude-opus-4-5",)


def test_two_different_versions_of_one_family_collapse_to_one_row():
    kept = newest_per_series(
        ["claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6"]
    )

    assert kept == ("claude-opus-5",)


def test_distinct_families_are_all_kept():
    kept = newest_per_series(
        ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"]
    )

    assert len(kept) == 3
    assert set(kept) == {
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-haiku-4-5-20251001",
    }


def test_how_the_ui_names_a_series():
    assert series_label("claude-sonnet-5-5") == "sonnet 5.5"
    assert series_label("claude-3-5-sonnet-20240620") == "sonnet 3.5"
    assert series_label("claude-mythos-preview") == "mythos-preview"


def test_a_family_that_does_not_exist_yet_still_groups():
    """The rules read where the numbers sit, so a brand-new line needs no
    entry anywhere to be grouped with its own future releases."""
    kept = newest_per_series(["claude-nova-2", "claude-nova-2-1", "claude-nova-2-2"])

    assert kept == ("claude-nova-2-2",)


# --- OpenRouter ids ---------------------------------------------------------


def test_an_openrouter_id_reads_vendor_family_and_version():
    assert openrouter_series_for("openrouter/anthropic/claude-sonnet-5.5") == Series(
        "anthropic/claude-sonnet", (5, 5)
    )
    assert openrouter_series_for("openrouter/openai/gpt-5.4-mini") == Series(
        "openai/gpt-mini", (5, 4)
    )
    assert openrouter_series_for("openrouter/google/gemini-3.8-flash") == Series(
        "google/gemini-flash", (3, 8)
    )


def test_a_pricing_tier_is_its_own_series():
    """`:free` is a different offering, not an older version of the paid one,
    so neither may knock the other out of the list."""
    paid = openrouter_series_for("openrouter/qwen/qwen3.8-27b")
    free = openrouter_series_for("openrouter/qwen/qwen3.8-27b:free")

    assert paid.key != free.key
    assert newest_per_series(
        ["openrouter/qwen/qwen3.8-27b", "openrouter/qwen/qwen3.8-27b:free"],
        rule=OPENROUTER_RULE,
    ) == ("openrouter/qwen/qwen3.8-27b:free", "openrouter/qwen/qwen3.8-27b")


def test_openrouter_leads_with_the_publication_date_not_the_version():
    """The case the rule exists for, measured against the live catalogue:
    x-ai/grok-4.20 was published 2026-03-31 and x-ai/grok-4.3 a month later,
    so "4.20" is 4.2.0. Ranking by version tuple offers the older model."""
    created = {
        "openrouter/x-ai/grok-4.20": "1774915200",
        "openrouter/x-ai/grok-4.3": "1777593600",
    }

    by_version = replace(OPENROUTER_RULE, created_first=False)

    assert newest_per_series(
        list(created), created_at=created.get, rule=OPENROUTER_RULE
    ) == ("openrouter/x-ai/grok-4.3",)
    # Same grouping, version as the leading key: the superseded model wins.
    assert newest_per_series(
        list(created), created_at=created.get, rule=by_version
    ) == ("openrouter/x-ai/grok-4.20",)


def test_an_openrouter_snapshot_date_is_not_a_version():
    assert openrouter_series_for("openrouter/openai/gpt-4.1-mini-2024-07-18") == Series(
        "openai/gpt-mini", (4, 1)
    )


def test_an_unreadable_openrouter_version_keeps_its_own_key():
    """Being unable to order something is not grounds for dropping it."""
    kept = newest_per_series(
        ["openrouter/openai/gpt-4o", "openrouter/openai/gpt-4o-mini"],
        rule=OPENROUTER_RULE,
    )

    assert set(kept) == {"openrouter/openai/gpt-4o", "openrouter/openai/gpt-4o-mini"}
