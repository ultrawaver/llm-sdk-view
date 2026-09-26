"""The Anthropic Models API, as an optional and cached source of model facts.

Rules this module exists to enforce:

- It only ever calls ``client.models.list()``. That endpoint is free, but a
  Messages call is not, so nothing here is allowed near the request path.
- It is called directly through the Anthropic SDK, not through ``llm``, so the
  result never lands in ``llm``'s prompt/response log.
- It never blocks the page. :func:`snapshot` returns the best data already on
  disk and refreshes in the background.
- When it cannot run - no key, no network, an error - the caller falls back to
  a versioned profile and says so. A fallback is never presented as a live API
  result.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import llm

CACHE_TTL = timedelta(hours=6)
FETCH_TIMEOUT_SECONDS = 3.0
MAX_PAGES = 5

_refresh_lock = threading.Lock()
_refresh_thread: threading.Thread | None = None


class ModelsApiUnavailable(RuntimeError):
    """The Models API could not be read. The caller should fall back."""


def cache_path() -> Path:
    """Disk cache location. Overridable so tests never touch the real one."""
    override = os.environ.get("LLM_SDK_VIEW_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "llm-sdk-view"
    return base / "models-api.json"


def api_key() -> str | None:
    """An Anthropic key, if one is already configured. Never prompts."""
    return llm.get_key(alias="anthropic", env="ANTHROPIC_API_KEY")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def load_cache() -> dict | None:
    path = cache_path()
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("models"):
        return None
    return payload


def save_cache(models: list[dict]) -> dict:
    payload = {"fetched_at": _now().isoformat(), "models": models}
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


def fetch_models(timeout: float = FETCH_TIMEOUT_SECONDS) -> list[dict]:
    """Read the Models API. Raises ModelsApiUnavailable rather than guessing."""
    key = api_key()
    if not key:
        raise ModelsApiUnavailable("no Anthropic key configured")
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=key)
        page = client.models.list(timeout=timeout)
        models = [_as_dict(item) for item in page.data]
        pages = 1
        while page.has_next_page() and pages < MAX_PAGES:
            page = page.get_next_page()
            models.extend(_as_dict(item) for item in page.data)
            pages += 1
    except Exception as ex:  # noqa: BLE001 - any failure means "fall back"
        raise ModelsApiUnavailable(str(ex)) from ex
    if not models:
        raise ModelsApiUnavailable("the Models API returned no models")
    return models


def _as_dict(item: Any) -> dict:
    dump = getattr(item, "model_dump", None)
    return dump(mode="json") if callable(dump) else dict(item)


def refresh() -> dict:
    """Fetch now, cache the result, and return the snapshot."""
    with _refresh_lock:
        try:
            payload = save_cache(fetch_models())
        except ModelsApiUnavailable as ex:
            return snapshot_from_cache(stale_ok=True, error=str(ex))
        return {
            "models": payload["models"],
            "provenance": provenance("models-api", "live", payload["fetched_at"], None),
        }


def provenance(source: str, cache: str, fetched_at: str | None, error: str | None) -> dict:
    return {
        "source": source,
        "cache": cache,
        "fetched_at": fetched_at,
        "error": error,
    }


def snapshot_from_cache(stale_ok: bool, error: str | None = None) -> dict:
    cached = load_cache()
    if cached is None:
        return {"models": [], "provenance": provenance("fallback", "none", None, error)}
    fresh = _is_fresh(cached)
    if not fresh and not stale_ok:
        return {"models": [], "provenance": provenance("fallback", "none", None, error)}
    return {
        "models": cached["models"],
        "provenance": provenance(
            "models-api",
            "disk" if fresh else "stale-disk",
            cached.get("fetched_at"),
            error,
        ),
    }


def kick_refresh() -> bool:
    """Refresh in the background so the next snapshot is live. Never blocks."""
    global _refresh_thread
    if not api_key():
        return False
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

    _refresh_thread = threading.Thread(target=run, name="llm-sdk-view-models-api", daemon=True)
    _refresh_thread.start()
    return True


def snapshot() -> dict:
    """Best-known Models API catalog plus where it came from.

    Returns immediately. A stale or missing cache triggers a background
    refresh instead of making the caller wait for the network.
    """
    cached = load_cache()
    if cached and _is_fresh(cached):
        return {
            "models": cached["models"],
            "provenance": provenance("models-api", "disk", cached.get("fetched_at"), None),
        }
    reasons = [
        "no cached Models API response" if cached is None
        else "the cached Models API response is stale"
    ]
    if api_key():
        kick_refresh()
    else:
        reasons.append("no Anthropic key configured")
    return snapshot_from_cache(stale_ok=True, error="; ".join(reasons))
