"""The API's own token counter: free, exact, and never on the request path.

Why the context meter needs a network call at all: this app's requests contain
a cost no character count can see.

- A server tool is *declared* by us and *expanded* by Anthropic. This app sends
  ``{"type": "web_search_...", "name": "web_search", "max_uses": 1}`` - 70
  characters of JSON - and the API bills roughly 2,200 input tokens for it.
  Measured with :func:`count_now` on 2026-09-27: ``claude-haiku-4-5`` went from
  12 tokens to 2,220 when that stub was added, ``claude-sonnet-5`` from 12 to
  2,806. None of those tokens are in the request body, so no local count can
  find them.
- CJK text is about one token per character, not one per four. The
  four-characters-per-token rule this module's estimate replaces was an English
  rule; on the conversation that exposed this it was out by a factor of 47.

So the figure comes from the provider whenever it can, in this order:

1. ``client.messages.count_tokens()`` - documented as free to use - run in the
   background and cached by request signature. :func:`lookup` returns what is
   already known and :func:`kick` asks for the rest.
2. the ``usage`` object on the previous turn, for a conversation that has
   already been sent (``chat.context_state``).
3. only then, an estimate that is labelled as one.

Rules this module keeps:

- It only ever calls ``count_tokens``. A Messages call costs money; none is
  made here.
- It counts exactly the payload ``prepare()`` built - the same dict the right
  pane renders - never a second model of the request.
- It never blocks the page and never raises into a request path. No key, no
  network or an error leaves the caller with the labelled estimate.
- The cache is in memory only. A count is a fact about one exact request, and
  keeping facts about requests that no longer exist on disk buys nothing; the
  ``usage`` on a stored turn already survives restarts.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

import llm

# The counter answers in well under a second when it answers at all; this is
# how long a background thread is allowed to wait before giving up.
COUNT_TIMEOUT_SECONDS = 5.0

# After a failure (no network, no key, rate limit) stop asking for a while.
# A dead network must not mean one stalled call per keystroke, and a reason
# that changes in a minute - a laptop rejoining wifi - is worth retrying.
FAILURE_COOLDOWN = timedelta(seconds=60)

# Bounded so a long session cannot grow the cache without limit. One entry per
# distinct request, and a request changes on every keystroke, so this is a few
# minutes of typing.
MAX_CACHED = 256

# The fields that change the input token count. Everything else in the request
# (max_tokens, temperature, stream) is output or transport, and the counter
# rejects some of them outright.
COUNTED_FIELDS = ("model", "messages", "system", "tools", "tool_choice", "thinking")


class TokenCountUnavailable(RuntimeError):
    """The API's counter could not be read. The caller should fall back."""


_lock = threading.Lock()
_cache: dict[str, int] = {}
_queued: dict[str, dict] = {}
_worker: threading.Thread | None = None
_last_failure: tuple[datetime, str] | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def reset() -> None:
    """Forget every count and reason. Tests call this; the app never does."""
    global _worker, _last_failure
    with _lock:
        _cache.clear()
        _queued.clear()
        _worker = None
        _last_failure = None


def api_key() -> str | None:
    """An Anthropic key, if one is already configured. Never prompts."""
    return llm.get_key(alias="anthropic", env="ANTHROPIC_API_KEY")


def counted_payload(kwargs: dict) -> dict:
    """The part of a request the counter is asked about.

    Dropped rather than passed through as None: an empty ``system`` or an empty
    ``tools`` list is not what the request carries, and counting something the
    request does not contain would be a second model of the request.
    """
    payload: dict[str, Any] = {}
    for field in COUNTED_FIELDS:
        value = kwargs.get(field)
        if value or value == 0:
            payload[field] = value
    return payload


def signature(payload: dict) -> str:
    """A stable name for one exact request."""
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def client(api_key: str) -> Any:
    """The Anthropic client, built in one place.

    Imported here rather than at module scope, and built through this function
    rather than inline, so the suite can replace it and prove the plumbing
    without a network call.
    """
    from anthropic import Anthropic

    return Anthropic(api_key=api_key)


def count_now(payload: dict, timeout: float = COUNT_TIMEOUT_SECONDS) -> int:
    """Ask the API how many input tokens this payload is.

    Raises :class:`TokenCountUnavailable` rather than guessing, so a caller can
    tell "the provider says 2,225" from "we could not ask".
    """
    key = api_key()
    if not key:
        raise TokenCountUnavailable("no Anthropic key configured")
    try:
        result = client(key).messages.count_tokens(**payload, timeout=timeout)
    except Exception as ex:  # noqa: BLE001 - any failure means "fall back"
        raise TokenCountUnavailable(str(ex)) from ex
    tokens = getattr(result, "input_tokens", None)
    if not isinstance(tokens, int) or tokens < 0:
        raise TokenCountUnavailable("the counter reported no input_tokens")
    return tokens


def last_failure() -> str | None:
    """Why the counter last failed, or None. Read by tests and diagnostics."""
    with _lock:
        return _last_failure[1] if _last_failure else None


def lookup(kwargs: dict) -> int | None:
    """The already-known count for this exact request, if there is one.

    Returns immediately and never touches the network, so the page can render
    an exact figure from cache while :func:`kick` fetches the next one.
    """
    payload = counted_payload(kwargs)
    if not payload:
        return None
    with _lock:
        return _cache.get(signature(payload))


def _cooling_down() -> bool:
    return _last_failure is not None and _now() - _last_failure[0] < FAILURE_COOLDOWN


def _drain() -> None:
    """Count queued requests, newest first, until there is nothing left.

    Newest first because the newest request is the one on screen: a user who
    kept typing wants the current draft counted, not the one they abandoned
    three keystrokes ago.

    ``_worker`` is the queue's owner: it is set to None under the lock on the
    way out, and only when the queue is empty, so a queued request can never
    outlive the thread that was supposed to count it.
    """
    global _worker
    while True:
        with _lock:
            if not _queued:
                _worker = None
                return
            sig, payload = _queued.popitem()
        try:
            tokens = count_now(payload)
        except Exception as ex:  # noqa: BLE001 - a background count must not raise
            _record_failure(str(ex))
            continue
        _record_count(sig, tokens)


def _record_count(sig: str, tokens: int) -> None:
    global _last_failure
    with _lock:
        _cache[sig] = tokens
        # A reason from an earlier request is not this request's answer.
        _last_failure = None
        while len(_cache) > MAX_CACHED:
            _cache.pop(next(iter(_cache)))


def _record_failure(reason: str) -> None:
    global _last_failure
    with _lock:
        _last_failure = (_now(), reason)


def kick(kwargs: dict) -> bool:
    """Ask for a count of this exact request, in the background.

    Returns True while a count for this request is coming - queued now or
    already running - which is what the page needs to know to look again. False
    means no count is on its way: it is already known, there is no key, or the
    counter is in its failure cooldown.
    """
    global _worker
    payload = counted_payload(kwargs)
    if not payload or not api_key():
        return False
    sig = signature(payload)
    with _lock:
        if sig in _cache:
            return False
        if sig in _queued:
            return True
        if _cooling_down():
            return False
        _queued[sig] = payload
        if _worker is None:
            # Started while the lock is held, and judged by ``_worker`` rather
            # than ``is_alive()``: a worker that has found the queue empty is
            # on its way out, and "almost dead" must not pass for "will pick
            # this up".
            _worker = threading.Thread(
                target=_drain, name="native-api-chat-token-count", daemon=True
            )
            _worker.start()
    return True
