"""Which members of a model family belong in the dropdown, decided by rule.

Every series offers exactly one model: its newest member. That has to be
derived rather than listed, or every Anthropic release needs an edit here -
which is the maintenance cost this module exists to remove. The family and
the version are already in the id:

    claude-sonnet-5-5            sonnet 5.5
    claude-sonnet-5              sonnet 5.0
    claude-haiku-4-5-20251001    haiku 4.5      (the date is a snapshot, not a version)
    claude-3-5-sonnet-20240620   sonnet 3.5     (3.x puts the family last)

Two generations of naming are recognised. Neither depends on knowing the
family names: where the numbers sit decides which pattern applies, so a
series that does not exist yet is grouped correctly by accident.

A model whose version cannot be read keeps a key of its own. It is offered
rather than hidden: being unable to order something is not grounds for
dropping it, and a wrong guess would be worse than one extra row.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# claude-sonnet-5-5, claude-haiku-4-5, claude-mythos-5-1
_MODERN = re.compile(
    r"^claude-(?P<family>[a-z][a-z0-9]*)-(?P<major>\d+)(?:-(?P<minor>\d{1,2}))?$"
)
# claude-3-5-sonnet, claude-3-opus, claude-3-7-sonnet
_LEGACY = re.compile(
    r"^claude-(?P<major>\d+)(?:-(?P<minor>\d{1,2}))?-(?P<family>[a-z]+)$"
)
# -20250514 pinned a snapshot; -latest is a moving alias. Neither is a version.
_SNAPSHOT = re.compile(r"-(?:\d{8}|latest)$")

_CLAUDE = "claude-"


@dataclass(frozen=True)
class Series:
    """A family and the version of that family one id belongs to."""

    family: str
    version: tuple[int, ...] | None

    @property
    def key(self) -> str:
        """Models sharing a key are the same series at different versions.

        The version chooses *which* member survives; it does not pick the
        group. Keying on the version instead puts every release in a series
        of its own, and then nothing is ever superseded. An id whose version
        cannot be read gets a group of its own, so it is never knocked out
        by a model this module merely failed to compare it with.
        """
        if self.version is None:
            return f"unversioned:{self.family}"
        return f"family:{self.family}"


def strip_snapshot(model_id: str) -> str:
    """Remove trailing date or alias markers, however many are stacked."""
    previous = None
    while previous != model_id:
        previous = model_id
        model_id = _SNAPSHOT.sub("", model_id)
    return model_id


def series_for(model_id: str) -> Series:
    """Read the family and version out of one Anthropic model id."""
    bare = strip_snapshot(model_id)
    if not bare.startswith(_CLAUDE):
        return Series(bare or model_id, None)
    modern = _MODERN.match(bare)
    if modern:
        return Series(modern["family"], _version(modern))
    legacy = _LEGACY.match(bare)
    if legacy:
        return Series(legacy["family"], _version(legacy))
    # A shape this module does not know. Keep it visible and ungrouped.
    return Series(bare[len(_CLAUDE):], None)


def _version(match: re.Match) -> tuple[int, ...]:
    major = int(match["major"])
    minor = match["minor"]
    return (major, int(minor)) if minor else (major, 0)


def newest_per_series(
    model_ids: Iterable[str],
    created_at: Callable[[str], str | None] | None = None,
) -> tuple[str, ...]:
    """One id per series: the highest version, ties broken by ``created_at``.

    Ordering is newest first, so the list the user sees opens on this week's
    release rather than on whatever the API happened to return first.
    """
    chosen: dict[str, str] = {}
    for model_id in model_ids:
        incumbent = chosen.get(series_for(model_id).key)
        if incumbent is None or _beats(model_id, incumbent, created_at):
            chosen[series_for(model_id).key] = model_id
    ordered = sorted(chosen.values(), key=lambda mid: _rank(mid, created_at), reverse=True)
    return tuple(ordered)


def _beats(candidate: str, incumbent: str, created_at) -> bool:
    """True when the candidate should replace the incumbent in its series."""
    return _rank(candidate, created_at) > _rank(incumbent, created_at)


def _rank(model_id: str, created_at) -> tuple:
    """Sort newest last, so ``reverse=True`` puts the newest at the front."""
    version = series_for(model_id).version or ()
    created = created_at(model_id) if created_at else None
    # A missing timestamp sorts oldest; equal versions then fall back to the
    # id, which keeps the order deterministic without inventing a date.
    return (version, created or "", model_id)


def series_label(model_id: str) -> str:
    """How the UI names the series this model heads: ``sonnet 5.5``."""
    series = series_for(model_id)
    if series.version is None:
        return series.family
    return f"{series.family} " + ".".join(str(part) for part in series.version)
