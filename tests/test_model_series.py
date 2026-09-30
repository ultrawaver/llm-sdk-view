"""Reading a family and a version out of a model id.

This is where "show only the newest of each series" is decided, so it gets
tested on its own, with nothing else loaded: an id nobody has thought of yet
has to be grouped correctly, and nothing here may touch the network.
"""

import pytest

from native_api_chat.model_series import newest_per_series, series_for, series_label

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
