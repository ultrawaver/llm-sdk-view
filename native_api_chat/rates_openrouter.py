"""OpenRouter's own catalogue, as the live source of unit prices.

The Anthropic side has to read a page (``rates_page``) because no API
publishes its prices. OpenRouter publishes them, in the same document that
lists its models: one request returns all four hundred and sixty-four
entries, each carrying per-token ``prompt``, ``completion``,
``input_cache_read``, ``input_cache_write`` and a per-request ``web_search``.
So there is nothing to scrape here and nothing to commit - the price is a
field of the catalogue this app already caches on disk and refreshes in the
background (``openrouter_api``), and this module is only the reading of it.

What the catalogue says, and what it does not:

- **A price is per token, written as a string.** ``"0.0000025"`` is
  $2.50 / MTok. Rates are kept per million tokens, the unit the receipt
  prints, because that is the only unit in which these numbers are readable.
- **A model may carry long-context overrides.** Above a ``min_prompt_tokens``
  the same model costs more, so one rate for the whole model would be wrong
  by a factor of two on a long turn. The tier comes from the turn's own
  prompt size, which is known, and never from a guess.
- **``:batch`` and ``:free`` are rows of their own**, each carrying its own
  prices - the batch row already holds the halved rate - so a slug is matched
  exactly and no tier arithmetic happens here. Matching ``:batch`` against
  its base model would quietly charge double.
- **Zero is not "free".** OpenRouter writes ``0`` both for what costs nothing
  and for what it is not pricing, and four models in the current catalogue
  are priced at zero without being on the free tier. Such a row is dropped
  rather than priced at $0, so the turn gets "no estimate" instead of a
  confident nothing; only a slug that *names* the free tier is read as free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from . import openrouter_api
from .providers.openrouter import MODEL_PREFIX

RATES_SOURCE = "OpenRouter catalogue"
RATES_URL = openrouter_api.MODELS_URL

#: The tier whose own name says it costs nothing. Read off the slug and never
#: off a zero price - see the module docstring.
FREE_TIER_SUFFIX = ":free"

#: USD per million tokens, the unit every rate in this app is kept in.
_PER_MTOK = 1_000_000

#: The catalogue's own field names, and the rate each one is.
_RATE_FIELDS = {
    "prompt": "input",
    "completion": "output",
    "input_cache_read": "cache_read",
    "input_cache_write": "cache_write_5m",
    "input_cache_write_1h": "cache_write_1h",
}


@dataclass(frozen=True)
class ModelRates:
    """One catalogue row's prices, USD per million tokens, plus its tiers."""

    slug: str
    input: float
    output: float
    cache_read: float | None
    cache_write_5m: float | None
    cache_write_1h: float | None
    #: Per request rather than per token - OpenRouter bills a search, not the
    #: tokens a search returned.
    web_search: float | None
    overrides: tuple[tuple[int, dict], ...] = ()
    #: Set by :meth:`at` when an override applied: the prompt size the rate
    #: this row now carries starts at, so a receipt can say why it moved.
    long_context_from: int | None = None

    def at(self, prompt_tokens: int | None) -> ModelRates:
        """The same row at the tier this prompt size falls in.

        OpenRouter states the threshold in the entry itself, so the tier is
        read rather than guessed: a 300,000-token prompt on a model that
        doubles above 272,000 is charged the doubled rate, and pricing it at
        the base rate would be wrong by half.
        """
        if prompt_tokens is None:
            return self
        applied = self
        for threshold, changed in sorted(self.overrides, key=lambda item: item[0]):
            if prompt_tokens >= threshold:
                applied = replace(self, long_context_from=threshold, **changed)
        return applied


def _per_mtok(pricing: dict, field: str) -> float | None:
    """One per-token price from the catalogue, as USD per million tokens."""
    value = pricing.get(field)
    if value is None:
        return None
    try:
        # Rounded, because 0.00000005 * 1e6 lands on 0.049999999999999996 and
        # a rate that prints as 0.05 should be 0.05 wherever it is read.
        return round(float(value) * _PER_MTOK, 6)
    except (TypeError, ValueError):
        return None


def _usd(value: Any) -> float | None:
    """One per-request price, which is already in dollars."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _overrides(pricing: dict) -> tuple[tuple[int, dict], ...]:
    """The long-context tiers, as a threshold and the rates it changes.

    An override names only what it changes, so the rest of the row stands;
    that merge is left to :meth:`ModelRates.at`, where ``replace`` does it.
    """
    raw = pricing.get("overrides")
    if not isinstance(raw, list):
        return ()
    found: list[tuple[int, dict]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        threshold = entry.get("min_prompt_tokens")
        if not isinstance(threshold, int) or threshold <= 0:
            continue
        changed = {
            name: rate
            for field, name in _RATE_FIELDS.items()
            if (rate := _per_mtok(entry, field)) is not None
        }
        if changed:
            found.append((threshold, changed))
    return tuple(found)


def _row(entry: dict) -> ModelRates | None:
    """One catalogue entry as a price row, or None when it is not priced."""
    if not isinstance(entry, dict):
        return None
    slug = entry.get("id")
    pricing = entry.get("pricing")
    if not isinstance(slug, str) or not slug or not isinstance(pricing, dict):
        return None
    # Both token rates, or this is not a price list. One malformed entry must
    # not cost the whole table.
    input_rate = _per_mtok(pricing, "prompt")
    output_rate = _per_mtok(pricing, "completion")
    if input_rate is None or output_rate is None:
        return None
    if input_rate == 0.0 and output_rate == 0.0 and not slug.endswith(FREE_TIER_SUFFIX):
        # Zero on both is what OpenRouter writes for a model it simply is not
        # pricing, and it is also what a free model costs. The slug is the
        # only thing that tells the two apart.
        return None
    return ModelRates(
        slug=slug,
        input=input_rate,
        output=output_rate,
        cache_read=_per_mtok(pricing, "input_cache_read"),
        cache_write_5m=_per_mtok(pricing, "input_cache_write"),
        cache_write_1h=_per_mtok(pricing, "input_cache_write_1h"),
        web_search=_usd(pricing.get("web_search")),
        overrides=_overrides(pricing),
    )


def _table(models: list[dict]) -> dict[str, ModelRates]:
    table: dict[str, ModelRates] = {}
    for entry in models:
        row = _row(entry)
        if row is not None:
            table[row.slug] = row
    return table


# --- the snapshot the app reads -------------------------------------------------

#: The catalogue is most of a megabyte of JSON, and a receipt recomputes its
#: own cost on every render - so the rows are built once and kept until the
#: cache file itself changes, which is exactly when a background refresh has
#: landed. The file's identity rather than its timestamp: a test installs a
#: catalogue by writing one, and two writes inside the same clock tick would
#: otherwise look like no change at all.
_built: tuple[tuple | None, dict] | None = None


def _cache_identity() -> tuple | None:
    path = openrouter_api.cache_path()
    try:
        info = path.stat()
    except OSError:
        # The path on its own: a machine with no catalogue still has to
        # notice the day one arrives, and two runs must not share one empty
        # table between them.
        return (str(path), None)
    # The inode as well as the clock: the cache is written by replacing the
    # file, so every save is a new file - which two writes inside the same
    # clock tick would otherwise hide.
    return (str(path), info.st_ino, info.st_size, info.st_mtime_ns)


def snapshot() -> dict:
    """Best-known prices plus where they came from. Returns immediately."""
    global _built
    identity = _cache_identity()
    if _built is not None and _built[0] == identity:
        return _built[1]
    catalog = openrouter_api.snapshot()
    payload = {"rates": _table(catalog["models"]), "provenance": catalog["provenance"]}
    _built = (identity, payload)
    return payload


def slug_of(model_id: str) -> str:
    """``openrouter/openai/gpt-5.4`` -> ``openai/gpt-5.4``.

    llm's routing prefix is not part of the model's name; the catalogue has
    never heard of it.
    """
    if model_id.startswith(MODEL_PREFIX):
        return model_id[len(MODEL_PREFIX) :]
    return model_id


def rates_for(model_id: str | None) -> ModelRates | None:
    """The catalogue's row for this model id, or None when it has none."""
    if not model_id:
        return None
    return snapshot()["rates"].get(slug_of(model_id))


def provenance() -> dict:
    """Where these prices came from, in the terms the UI shows.

    The catalogue is cached with its own timestamp, and a cache inside its
    window is the current catalogue rather than a stale copy of an old one -
    the same reading ``rates_page`` gives a fresh pricing page.
    """
    source = snapshot()["provenance"]
    fetched_at = source.get("fetched_at")
    cache = source.get("cache")
    if not fetched_at or cache in (None, "none"):
        label, state = f"{RATES_SOURCE} (unavailable)", "unavailable"
    elif cache == "stale-disk":
        label, state = f"{RATES_SOURCE} (cached)", "cached"
    else:
        label, state = RATES_SOURCE, "live"
    return {
        "rates_source": label,
        "rates_date": fetched_at[:10] if isinstance(fetched_at, str) else "unknown",
        "rates_url": RATES_URL,
        "rates_state": state,
        "rates_error": source.get("error"),
    }


def status() -> dict:
    """:func:`provenance`, plus how many models these prices actually cover."""
    return {**provenance(), "models": len(snapshot()["rates"])}


__all__ = [
    "FREE_TIER_SUFFIX",
    "ModelRates",
    "RATES_SOURCE",
    "RATES_URL",
    "provenance",
    "rates_for",
    "slug_of",
    "snapshot",
    "status",
]
