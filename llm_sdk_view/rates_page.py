"""Anthropic's pricing page, as the live source of unit prices.

This mirrors ``model_api.py`` on purpose: prices are facts with a source, and
the same rules apply to both.

- The page is read in the background and cached on disk. Nothing here blocks
  the page load.
- A cache is labelled as a cache, and carries when it was fetched. A price
  that could not be fetched is never presented as a live one.
- When there is no page and no cache there are simply no prices, and the
  caller says so instead of estimating against an invented rate.

Prices used to be transcribed into the source by hand, which meant every
upstream price change needed a commit. Fetching them keeps the numbers honest
without editing code - and the disk cache means an offline start still has
yesterday's prices rather than nothing.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

RATES_URL = "https://platform.claude.com/docs/en/about-claude/pricing"

# The docs site serves every page as markdown at the same path plus ".md",
# which beats scraping HTML for a number that has to be exactly right.
PRICING_MD_URL = f"{RATES_URL}.md"
FETCH_TIMEOUT_SECONDS = 4.0

# Prices move far less often than the model catalog does.
CACHE_TTL = timedelta(hours=12)

# Below this the "table" is almost certainly not the real pricing table.
MIN_ROWS = 5

SECTION_HEADING = "## Model pricing"

# The page's column order: base input, 5m write, 1h write, cache hit, output.
# ModelRates keeps a different order, so the two are mapped explicitly.
COLUMN_MATCHERS = (
    "base input",
    "5m cache writes",
    "1h cache writes",
    "cache hits",
    "output",
)
RATE_FIELDS = (
    "input",
    "output",
    "cache_write_5m",
    "cache_write_1h",
    "cache_read",
)
# Index into the page's column order for each ModelRates field.
_PAGE_INDEX_FOR_FIELD = (0, 4, 1, 2, 3)

_refresh_lock = threading.Lock()
_refresh_thread: threading.Thread | None = None


@dataclass(frozen=True)
class ModelRates:
    """USD per million tokens, one row of the pricing table."""

    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


class RatesUnavailableError(RuntimeError):
    """The pricing page could not be read. The caller should fall back."""


class RatesParseError(RuntimeError):
    """The page was read but does not look like the pricing page."""


# --- the cache -----------------------------------------------------------------


def cache_path() -> Path:
    """Disk cache location. Overridable so tests never touch the real one."""
    override = os.environ.get("LLM_SDK_VIEW_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "llm-sdk-view"
    return base / "rates.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_cache() -> dict | None:
    path = cache_path()
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("rates"):
        return None
    return payload


def _as_rates(raw: dict) -> dict[str, ModelRates]:
    table: dict[str, ModelRates] = {}
    for model, values in raw.items():
        if not isinstance(values, dict):
            continue
        try:
            table[model] = ModelRates(
                **{field: float(values[field]) for field in RATE_FIELDS}
            )
        except (KeyError, TypeError, ValueError):
            # One malformed row must not cost us the whole table.
            continue
    return table


def save_cache(rates: dict[str, ModelRates]) -> dict:
    payload = {
        "fetched_at": _now().isoformat(),
        "rates": {
            model: {field: getattr(row, field) for field in RATE_FIELDS}
            for model, row in rates.items()
        },
    }
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: a background refresh can land while a request is
    # reading the cache, and a half-written file must never read as "no cache".
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), "utf-8")
    os.replace(temporary, path)
    return payload


def _is_fresh(payload: dict) -> bool:
    try:
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
    except (KeyError, ValueError):
        return False
    return _now() - fetched_at < CACHE_TTL


# --- reading the page ------------------------------------------------------------


def fetch_pricing_markdown(
    url: str = PRICING_MD_URL, timeout: float = FETCH_TIMEOUT_SECONDS
) -> str:
    """The pricing page as markdown, or a reason it could not be read."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "llm-sdk-view rates"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as ex:
        raise RatesUnavailableError(f"could not read {url}: {ex}") from ex
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as ex:
        raise RatesParseError(f"{url} is not UTF-8 text: {ex}") from ex


def _section(markdown: str) -> str:
    start = markdown.find(SECTION_HEADING)
    if start == -1:
        raise RatesParseError(f"no '{SECTION_HEADING}' section on the page")
    rest = markdown[start + len(SECTION_HEADING) :]
    following = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: following.start()] if following else rest


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell != "")


def _price(cell: str) -> str | None:
    match = re.search(r"\$([\d,]+(?:\.\d+)?)", cell)
    return match.group(1).replace(",", "") if match else None


def _model_id(name: str) -> str:
    """'Claude Mythos 5.1 ([limited availability](url))' -> claude-mythos-5-1."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", name)
    text = re.sub(r"\s*\([^)]*\)", "", text)
    return text.strip().lower().replace(" ", "-").replace(".", "-")


def parse_model_pricing(markdown: str) -> dict[str, ModelRates]:
    """The Model pricing table, keyed by the model id prefix it should match."""
    rows = [
        line for line in _section(markdown).splitlines() if line.lstrip().startswith("|")
    ]
    if len(rows) < 2:
        raise RatesParseError("the Model pricing section has no table")

    header = [cell.lower() for cell in _cells(rows[0])]
    columns: list[int] = []
    for matcher in COLUMN_MATCHERS:
        index = next((i for i, cell in enumerate(header) if matcher in cell), None)
        if index is None:
            raise RatesParseError(f"the pricing table has no '{matcher}' column")
        columns.append(index)

    parsed: dict[str, ModelRates] = {}
    for row in rows[1:]:
        cells = _cells(row)
        if _is_separator(cells):
            continue
        if len(cells) <= max(columns):
            raise RatesParseError(f"a pricing row is missing cells: {row[:60]}")
        name = cells[0]
        if not name or name.lower() == "model":
            continue
        printed = [_price(cells[index]) for index in columns]
        if any(value is None for value in printed):
            raise RatesParseError(f"no price could be read for '{name}'")
        numbers = [float(value) for value in printed if value is not None]
        ordered = [numbers[index] for index in _PAGE_INDEX_FOR_FIELD]
        parsed[_model_id(name)] = ModelRates(*ordered)

    if len(parsed) < MIN_ROWS:
        raise RatesParseError(
            f"only {len(parsed)} models parsed, expected at least {MIN_ROWS}"
        )
    return parsed


def fetch_rates(timeout: float = FETCH_TIMEOUT_SECONDS) -> dict[str, ModelRates]:
    """Read the page now. Raises rather than returning a partial guess."""
    return parse_model_pricing(fetch_pricing_markdown(timeout=timeout))


# --- the snapshot the app reads ---------------------------------------------------


def provenance(
    source: str, cache: str, fetched_at: str | None, error: str | None
) -> dict:
    return {
        "source": source,
        "cache": cache,
        "fetched_at": fetched_at,
        "error": error,
    }


def snapshot_from_cache(error: str | None = None) -> dict:
    cached = load_cache()
    if cached is None:
        return {"rates": {}, "provenance": provenance("unavailable", "none", None, error)}
    fresh = _is_fresh(cached)
    return {
        "rates": _as_rates(cached["rates"]),
        "provenance": provenance(
            "pricing-page" if fresh else "disk-cache",
            "disk",
            cached.get("fetched_at"),
            None if fresh else (error or "the cached prices are out of date"),
        ),
    }


def refresh() -> dict:
    """Fetch now, cache the result, and return the snapshot."""
    try:
        save_cache(fetch_rates())
    except (RatesUnavailableError, RatesParseError) as ex:
        return snapshot_from_cache(error=str(ex))
    return snapshot_from_cache()


def kick_refresh(force: bool = False) -> bool:
    """Refresh in the background so the next snapshot is live. Never blocks."""
    global _refresh_thread
    with _refresh_lock:
        if _refresh_thread is not None and _refresh_thread.is_alive():
            return False
        cached = load_cache()
        if not force and cached and _is_fresh(cached):
            return False

        def run() -> None:
            try:
                refresh()
            except Exception:  # noqa: BLE001 - a background refresh must not raise
                pass

        _refresh_thread = threading.Thread(
            target=run, name="llm-sdk-view-rates", daemon=True
        )
        _refresh_thread.start()
        return True


def snapshot() -> dict:
    """Best-known prices plus where they came from.

    Returns immediately. A stale or missing cache triggers a background
    refresh instead of making the caller wait for the network.
    """
    cached = load_cache()
    if cached and _is_fresh(cached):
        return {
            "rates": _as_rates(cached["rates"]),
            "provenance": provenance("pricing-page", "disk", cached.get("fetched_at"), None),
        }
    reason = (
        "no cached prices" if cached is None else "the cached prices are out of date"
    )
    kick_refresh()
    return snapshot_from_cache(error=reason)


def table() -> dict[str, ModelRates]:
    """Current prices, longest prefixes first so prefix matching is stable."""
    rates = snapshot()["rates"]
    return dict(sorted(rates.items(), key=lambda item: (-len(item[0]), item[0])))
