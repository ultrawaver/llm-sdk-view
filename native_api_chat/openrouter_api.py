"""OpenRouter's model catalogue, as the live source of its facts and prices.

The same rules the Anthropic Models API layer follows apply here, for the same
reasons: never block the page, cache on disk, refresh in the background, and
say which of those a number came from rather than presenting a stale read as a
live one.

Two things differ, and both are improvements:

- **No key is needed.** ``/api/v1/models`` is public, so the catalogue is
  readable on a machine that has never configured OpenRouter - the dropdown
  and the capability matrix work before the user has a key, and only sending
  needs one.
- **Prices come with the models.** Each entry carries per-token ``prompt``,
  ``completion``, ``web_search``, ``input_cache_read`` and
  ``input_cache_write`` figures, so there is no page to scrape and still
  nothing to commit. A model the catalogue does not price gets no estimate,
  never a rate borrowed from a neighbour.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

MODELS_URL = "https://openrouter.ai/api/v1/models"

CACHE_TTL = timedelta(hours=6)
FETCH_TIMEOUT_SECONDS = 5.0

_refresh_lock = threading.Lock()
_refresh_thread: threading.Thread | None = None


class CatalogUnavailable(RuntimeError):
    """The catalogue could not be read. The caller reports it, never guesses."""


def cache_path() -> Path:
    """Disk cache location. Overridable so tests never touch the real one."""
    override = os.environ.get("NATIVE_API_CHAT_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "native-api-chat"
    return base / "openrouter-models.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_cache() -> dict | None:
    try:
        payload = json.loads(cache_path().read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("models"):
        return None
    return payload


def save_cache(models: list[dict]) -> dict:
    payload = {"fetched_at": _now().isoformat(), "models": models}
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written atomically: a background refresh can land while a request is
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


def fetch_models(timeout: float = FETCH_TIMEOUT_SECONDS) -> list[dict]:
    """Read the public catalogue. Raises rather than returning a guess.

    ``httpx`` is used rather than ``urllib`` because it honours the proxy
    environment by default, and a developer behind one would otherwise see
    this time out while every other tool on the machine works.
    """
    import httpx

    try:
        response = httpx.get(MODELS_URL, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        models = response.json()["data"]
    except Exception as ex:  # noqa: BLE001 - every failure is the same answer
        raise CatalogUnavailable(f"{type(ex).__name__}: {ex}") from ex
    if not isinstance(models, list) or not models:
        raise CatalogUnavailable("the catalogue held no models")
    return models


def provenance(source: str, cache: str, fetched_at: str | None, error: str | None) -> dict:
    return {"source": source, "cache": cache, "fetched_at": fetched_at, "error": error}


def refresh() -> dict:
    """Fetch now, cache the result, and return the snapshot."""
    with _refresh_lock:
        try:
            payload = save_cache(fetch_models())
        except CatalogUnavailable as ex:
            return snapshot_from_cache(stale_ok=True, error=str(ex))
        return {
            "models": payload["models"],
            "provenance": provenance(
                "openrouter-models", "live", payload["fetched_at"], None
            ),
        }


def snapshot_from_cache(stale_ok: bool, error: str | None = None) -> dict:
    cached = load_cache()
    if cached is None:
        return {"models": [], "provenance": provenance("none", "none", None, error)}
    fresh = _is_fresh(cached)
    if not fresh and not stale_ok:
        return {"models": [], "provenance": provenance("none", "none", None, error)}
    return {
        "models": cached["models"],
        "provenance": provenance(
            "openrouter-models",
            "disk" if fresh else "stale-disk",
            cached.get("fetched_at"),
            error,
        ),
    }


def kick_refresh() -> bool:
    """Refresh in the background so the next snapshot is live. Never blocks."""
    global _refresh_thread
    if _refresh_thread is not None and _refresh_thread.is_alive():
        return False
    cached = load_cache()
    if cached and _is_fresh(cached):
        return False

    def run():
        try:
            refresh()
        except Exception:  # noqa: BLE001 - a background refresh must not raise
            pass

    _refresh_thread = threading.Thread(
        target=run, name="native-api-chat-openrouter-models", daemon=True
    )
    _refresh_thread.start()
    return True


def snapshot() -> dict:
    """Best-known catalogue plus where it came from. Returns immediately."""
    cached = load_cache()
    if cached and _is_fresh(cached):
        return {
            "models": cached["models"],
            "provenance": provenance(
                "openrouter-models", "disk", cached.get("fetched_at"), None
            ),
        }
    reason = (
        "no cached OpenRouter catalogue"
        if cached is None
        else "the cached OpenRouter catalogue is stale"
    )
    kick_refresh()
    return snapshot_from_cache(stale_ok=True, error=reason)
