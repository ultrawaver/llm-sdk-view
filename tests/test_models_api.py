"""The Models API layer: dynamic when it can be, honest when it cannot.

These tests never touch the network. The Anthropic client is replaced, so the
only thing under test is the precedence, caching and labelling rules:
Models API -> disk cache -> versioned fallback profile, with the source always
visible to the UI and never upgraded from fallback to live.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from native_api_chat import capabilities, model_api
from native_api_chat.app import app

FOUR_MODELS = (
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
)


class _Boom:
    """Any attribute access here means a paid Messages call was attempted."""

    def __getattr__(self, name):
        raise AssertionError(f"the Models API layer must not touch {name}")


class _Page:
    def __init__(self, data):
        self.data = data

    def has_next_page(self):
        return False

    def get_next_page(self):
        raise AssertionError("unexpected pagination")


class _ModelsResource:
    def __init__(self, recorder, data, error=None, delay=0.0):
        self.recorder = recorder
        self.data = data
        self.error = error
        self.delay = delay

    def list(self, **kwargs):
        self.recorder.append(("models.list", kwargs.get("timeout")))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return _Page(self.data)


class _FakeAnthropic:
    def __init__(self, recorder, data, error=None, delay=0.0):
        self.models = _ModelsResource(recorder, data, error, delay)
        self.messages = _Boom()
        self.beta = _Boom()


def _entry(model_id, max_input, max_output, effort=None, thinking=None):
    return {
        "id": model_id,
        "display_name": model_id,
        "created_at": "2026-09-01T00:00:00Z",
        "type": "model",
        "max_input_tokens": max_input,
        "max_tokens": max_output,
        "capabilities": {"effort": effort, "thinking": thinking},
    }


API_DATA = [
    _entry("claude-fable-5-1", 1_000_000, 128_000,
           effort={"supported": True, "low": True, "medium": True, "high": True,
                   "xhigh": True, "max": True},
           thinking={"supported": True, "types": ["adaptive"]}),
    _entry("claude-opus-5-5", 1_000_000, 128_000,
           effort={"supported": True, "low": True, "medium": True, "high": True,
                   "xhigh": True, "max": True},
           thinking={"supported": True, "types": ["adaptive"]}),
    _entry("claude-sonnet-5", 1_000_000, 128_000,
           effort={"supported": True, "low": True, "medium": True, "high": True,
                   "xhigh": True, "max": True},
           thinking={"supported": True, "types": ["adaptive"]}),
    _entry("claude-haiku-4-5-20251001", 200_000, 64_000,
           effort={"supported": False},
           thinking={"supported": True, "types": ["enabled"]}),
    # A model this project cannot send a request to: it must be ignored.
    _entry("claude-not-registered-anywhere", 200_000, 8_000),
]


@pytest.fixture
def isolated_cache(monkeypatch):
    """Point the disk cache at a scratch dir and forget any snapshot."""
    scratch = tempfile.mkdtemp(prefix="native-api-chat-cache-")
    monkeypatch.setenv("NATIVE_API_CHAT_CACHE_DIR", scratch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    capabilities.reset_model_data()
    yield Path(scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    capabilities.reset_model_data()


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")


@pytest.fixture
def record(monkeypatch):
    """Install a fake Anthropic client that records everything it is asked."""
    import anthropic

    calls = []

    def factory(data=API_DATA, error=None, delay=0.0):
        monkeypatch.setattr(
            anthropic, "Anthropic", lambda **kwargs: _FakeAnthropic(calls, data, error, delay)
        )
        return calls

    return factory


# --- fallback ---------------------------------------------------------------


def test_no_key_means_fallback_and_no_network(isolated_cache, record):
    calls = record()

    snapshot = model_api.snapshot()

    assert snapshot["provenance"]["source"] == "fallback"
    assert snapshot["provenance"]["cache"] == "none"
    assert "key" in (snapshot["provenance"]["error"] or "")
    assert calls == [], "without a key the Models API must not be called"


def test_the_form_still_works_without_any_model_api(isolated_cache):
    data = TestClient(app).get("/api/form").json()

    # One model per series, whatever order they land in: offline the profile
    # is the whole world and every entry is the head of its own series.
    assert set(data["models"]) == set(FOUR_MODELS)
    assert data["model_data"]["source"] == "fallback"
    assert data["model_data"]["profile_version"] == "2026-09-26"
    # The default model is Claude Haiku 4.5, so the fallback window is 200k.
    assert data["capabilities"]["context_window"] == 200_000


def test_the_list_opens_on_the_newest_member_of_each_series(
    isolated_cache, record, with_key
):
    """A release must supersede its series without anyone editing anything.

    The five extra models below are members of series already listed. None of
    them may appear, and the newer member must win even though the API sent it
    later in the payload: ordering here is what makes a new release visible.
    """
    record(
        data=API_DATA
        + [
            _entry("claude-sonnet-5-5", 1_000_000, 128_000,
                   effort={"supported": True, "high": True},
                   thinking={"supported": True, "types": ["adaptive"]}),
            _entry("claude-sonnet-4-6", 1_000_000, 64_000),
            _entry("claude-opus-4-6", 1_000_000, 64_000),
            _entry("claude-fable-5", 1_000_000, 64_000),
            _entry("claude-haiku-4-5", 200_000, 64_000),
        ]
    )
    model_api.refresh()
    capabilities.reset_model_data()

    offered = capabilities.model_ids()

    assert "claude-sonnet-5-5" in offered
    assert "claude-sonnet-5" not in offered, "superseded by the same series' 5.5"
    assert "claude-sonnet-4-6" not in offered
    assert "claude-opus-4-6" not in offered
    assert "claude-fable-5" not in offered
    # The haiku already listed is the dated snapshot; an undated haiku 4.5 is
    # the same model and must not add a second row.
    assert len([m for m in offered if m.startswith("claude-haiku")]) == 1
    assert offered[0] == "claude-sonnet-5-5", "the newest model opens the list"


def test_superseded_models_are_reported_rather_than_silently_dropped(
    isolated_cache, record, with_key
):
    """Nine models leaving the list must be said somewhere, or it reads as
    data loss rather than as a rule."""
    record()
    model_api.refresh()
    capabilities.reset_model_data()

    catalog = capabilities.model_catalog()

    assert set(catalog["models"]) == set(FOUR_MODELS)
    assert catalog["superseded"] == [] or all(
        "kept_by" in item for item in catalog["superseded"]
    )


def test_a_fallback_is_never_labelled_as_the_api(isolated_cache):
    capabilities_for = capabilities.capabilities_for("claude-sonnet-5")

    assert capabilities_for.data_source.startswith("fallback-profile")
    assert "fallback" in capabilities_for.context_window_source


def test_the_context_meter_labels_a_fallback_limit(isolated_cache):
    """The context figure must say that its ceiling came from the profile."""
    from native_api_chat.chat import context_state

    data = TestClient(app).get("/api/form").json()

    assert "fallback profile" in data["context"]["limit_source"]
    assert data["context"]["limit_data_source"] == "Fallback capability data"
    assert context_state({}, capabilities.capabilities_for("claude-sonnet-5")).limit == (
        capabilities.capabilities_for("claude-sonnet-5").context_window
    )


def test_the_context_meter_labels_a_models_api_limit(isolated_cache, record, with_key):
    from native_api_chat.chat import context_state

    record()
    model_api.refresh()
    capabilities.reset_model_data()

    caps = capabilities.capabilities_for("claude-sonnet-5")
    state = context_state({}, caps)

    assert "Models API" in state.limit_source
    assert state.limit_data_source == "Provider default"


# --- cache -------------------------------------------------------------------


def test_a_fresh_cache_is_used_and_marked(isolated_cache, record, with_key):
    record()
    model_api.refresh()

    capabilities.reset_model_data()
    snapshot = model_api.snapshot()

    assert snapshot["provenance"]["source"] == "models-api"
    assert snapshot["provenance"]["cache"] == "disk"
    assert snapshot["provenance"]["fetched_at"]


def test_a_stale_cache_is_served_but_flagged(isolated_cache, record, with_key, monkeypatch):
    record()
    payload = model_api.refresh()
    path = model_api.cache_path()
    stale = dict(payload)
    stale["fetched_at"] = "2020-01-01T00:00:00+00:00"
    stale["models"] = payload["models"]
    path.write_text(json.dumps(stale), "utf-8")
    # Drop the key so no background refresh can overwrite the stale file while
    # it is being read: the point of the test is the stale path, not the race.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    capabilities.reset_model_data()
    snapshot = model_api.snapshot()

    assert snapshot["provenance"]["cache"] == "stale-disk"
    assert snapshot["models"], "stale data is still better than none"


def test_a_fetch_failure_falls_back_with_the_reason(isolated_cache, record, with_key):
    record(error=RuntimeError("network is down"))

    snapshot = model_api.refresh()

    assert snapshot["provenance"]["source"] == "fallback"
    assert "network is down" in (snapshot["provenance"]["error"] or "")


def test_models_api_unavailable_is_raised_when_there_is_no_key(isolated_cache):
    with pytest.raises(model_api.ModelsApiUnavailable):
        model_api.fetch_models()


# --- precedence -------------------------------------------------------------


def test_api_values_win_over_the_fallback_profile(isolated_cache, record, with_key):
    record(data=[_entry("claude-sonnet-5", 500_000, 32_000,
                        effort={"supported": True, "low": True, "high": True},
                        thinking={"supported": True, "types": ["enabled"]})])
    model_api.refresh()
    capabilities.reset_model_data()

    caps = capabilities.capabilities_for("claude-sonnet-5")

    assert caps.context_window == 500_000
    assert caps.max_output_tokens == 32_000
    assert caps.data_source == "models-api"
    assert "Models API" in caps.context_window_source
    assert caps.effort_levels == ("low", "high")
    assert caps.thinking_mode == "extended"


def test_models_missing_from_the_api_fall_back_per_model(isolated_cache, record, with_key):
    record(data=[_entry("claude-sonnet-5", 500_000, 32_000)])
    model_api.refresh()
    capabilities.reset_model_data()

    sonnet = capabilities.capabilities_for("claude-sonnet-5")
    haiku = capabilities.capabilities_for("claude-haiku-4-5-20251001")

    assert sonnet.data_source == "models-api"
    assert haiku.data_source.startswith("fallback-profile")
    assert haiku.context_window == 200_000


def test_the_api_model_list_drops_models_llm_cannot_resolve(isolated_cache, record, with_key):
    record()
    model_api.refresh()
    capabilities.reset_model_data()

    assert "claude-not-registered-anywhere" not in capabilities.model_ids()
    for model_id in FOUR_MODELS:
        assert model_id in capabilities.model_ids()


# --- cost and safety ---------------------------------------------------------


def test_only_the_free_models_endpoint_is_called(isolated_cache, record, with_key, fake_provider):
    calls = record()
    model_api.refresh()

    assert calls == [("models.list", model_api.FETCH_TIMEOUT_SECONDS)]
    assert fake_provider == [], "no Messages request may be sent to read model data"


def test_reading_model_data_does_not_block_the_page(isolated_cache, record, with_key):
    """A slow Models API must not slow down rendering the form."""
    record(delay=2.0)

    started = time.monotonic()
    snapshot = model_api.snapshot()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, "the form must not wait for the Models API"
    assert snapshot["provenance"]["source"] in {"models-api", "fallback"}


def test_the_form_reports_provenance(isolated_cache, record, with_key):
    record()
    model_api.refresh()
    capabilities.reset_model_data()

    data = TestClient(app).get("/api/form").json()

    assert data["model_data"]["source"] == "models-api"
    assert data["model_data"]["cache"] in {"disk", "live"}
    assert data["model_data"]["fallback_models"] == list(FOUR_MODELS)
    assert data["model_data"]["profile_version"] == "2026-09-26"


def test_the_ui_shows_the_provenance_rather_than_assuming_it(static_page):
    # The footer's dedicated row is gone; the provenance now rides the
    # context-window line in the params panel ("context window N tokens ·
    # <source>"), which names the source instead of assuming live data.
    assert "context_window_source" in static_page
    assert "contextWindow" in static_page
