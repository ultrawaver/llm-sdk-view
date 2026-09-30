"""Which members of a model family belong in the dropdown, decided by rule.

Every series offers exactly one model: its newest member. That has to be
derived rather than listed, or every provider release needs an edit here -
which is the maintenance cost this module exists to remove. The family and
the version are already in the id:

    claude-sonnet-5-5            sonnet 5.5
    claude-sonnet-5              sonnet 5.0
    claude-haiku-4-5-20251001    haiku 4.5      (the date is a snapshot, not a version)
    claude-3-5-sonnet-20240620   sonnet 3.5     (3.x puts the family last)

Two generations of Anthropic naming are recognised. Neither depends on
knowing the family names: where the numbers sit decides which pattern
applies, so a series that does not exist yet is grouped correctly by accident.

OpenRouter names models ``vendor/name`` with the version inside the name and
an optional ``:tier`` after it, so it gets a rule of its own:

    anthropic/claude-sonnet-5.5  anthropic/claude-sonnet 5.5
    openai/gpt-5.4-mini          openai/gpt-mini 5.4
    google/gemini-3.8-flash      google/gemini-flash 3.8
    qwen/qwen3-max:free          a tier of its own, never superseding the paid one

A model whose version cannot be read keeps a key of its own. It is offered
rather than hidden: being unable to order something is not grounds for
dropping it, and a wrong guess would be worse than one extra row.

Which of the two keys decides the winner is not a preference, it is measured.
Anthropic's ids are versioned consistently enough to lead with the version.
OpenRouter's are not, and the counter-example is ``x-ai/grok-4.20``, published
on 2026-03-31, a month *before* ``x-ai/grok-4.3``: read as a version tuple
``(4, 20)`` beats ``(4, 3)``, and the dropdown would offer a superseded model
as the newest one. OpenRouter reports ``created`` for every model, so that is
what its rule leads with.
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

# openrouter/anthropic/claude-sonnet-5.5 -> the part this module reads.
_OPENROUTER_PREFIX = "openrouter/"
# ":free" and ":batch" are pricing tiers of the same weights, not versions.
_TIER = re.compile(r":(?P<tier>[a-z0-9-]+)$")
# -2024-07-18 and -20240718 both pin a snapshot.
_DATED = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$")
# 5, 5.4, v2 - a version token. "4o" is deliberately not one: it is a name.
_NUMBERED = re.compile(r"^v?(?P<major>\d+)(?:\.(?P<minor>\d+))?$")


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


def openrouter_series_for(model_id: str) -> Series:
    """Read the family and version out of one OpenRouter model id.

    The version is whichever token first reads as a number, and the family is
    the vendor plus everything else in order - so ``openai/gpt-5.4-mini`` and
    ``openai/gpt-4.1-mini`` share a series while ``openai/gpt-5.4-nano`` does
    not. A pricing tier stays in the family, because ``:free`` is a different
    offering from the paid entry rather than an older version of it, and one
    must never knock the other out of the list.
    """
    bare = model_id[len(_OPENROUTER_PREFIX):] if model_id.startswith(
        _OPENROUTER_PREFIX
    ) else model_id
    tier_match = _TIER.search(bare)
    tier = tier_match["tier"] if tier_match else ""
    bare = _TIER.sub("", bare)
    previous = None
    while previous != bare:
        previous = bare
        bare = _DATED.sub("", bare)
    vendor, _, name = bare.partition("/")
    if not name:
        return Series(bare or model_id, None)
    version: tuple[int, ...] | None = None
    rest: list[str] = []
    for token in name.split("-"):
        numbered = _NUMBERED.match(token)
        if numbered and version is None:
            version = _version(numbered)
        else:
            rest.append(token)
    family = f"{vendor}/" + "-".join(rest)
    return Series(f"{family}:{tier}" if tier else family, version)


def _version(match: re.Match) -> tuple[int, ...]:
    major = int(match["major"])
    minor = match["minor"]
    return (major, int(minor)) if minor else (major, 0)


@dataclass(frozen=True)
class SeriesRule:
    """One provider's answer to "same series?" and "which member wins?".

    ``created_first`` is the whole reason this is a rule rather than a
    function: see the module docstring for the measurement behind it.
    """

    series_for: Callable[[str], Series]
    created_first: bool
    description: str


ANTHROPIC_RULE = SeriesRule(
    series_for=series_for,
    created_first=False,
    description="one model per series: the highest version, ties broken by created_at",
)

OPENROUTER_RULE = SeriesRule(
    series_for=openrouter_series_for,
    created_first=True,
    description=(
        "one model per series and pricing tier: the most recently published, "
        "ties broken by version - OpenRouter's version numbers do not always "
        "order (grok-4.20 predates grok-4.3)"
    ),
)


def newest_per_series(
    model_ids: Iterable[str],
    created_at: Callable[[str], str | None] | None = None,
    rule: SeriesRule = ANTHROPIC_RULE,
) -> tuple[str, ...]:
    """One id per series, newest first.

    Ordering is newest first, so the list the user sees opens on this week's
    release rather than on whatever the API happened to return first.
    """
    chosen: dict[str, str] = {}
    for model_id in model_ids:
        key = rule.series_for(model_id).key
        incumbent = chosen.get(key)
        if incumbent is None or _rank(model_id, created_at, rule) > _rank(
            incumbent, created_at, rule
        ):
            chosen[key] = model_id
    return tuple(
        sorted(chosen.values(), key=lambda mid: _rank(mid, created_at, rule), reverse=True)
    )


def _rank(model_id: str, created_at, rule: SeriesRule = ANTHROPIC_RULE) -> tuple:
    """Sort newest last, so ``reverse=True`` puts the newest at the front."""
    version = rule.series_for(model_id).version or ()
    created = created_at(model_id) if created_at else None
    # A missing timestamp sorts oldest; equal leading keys then fall back to
    # the id, which keeps the order deterministic without inventing a date.
    if rule.created_first:
        return (created or "", version, model_id)
    return (version, created or "", model_id)


def series_label(model_id: str, rule: SeriesRule = ANTHROPIC_RULE) -> str:
    """How the UI names the series this model heads: ``sonnet 5.5``."""
    series = rule.series_for(model_id)
    if series.version is None:
        return series.family
    return f"{series.family} " + ".".join(str(part) for part in series.version)
