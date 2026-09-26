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

from llm_sdk_view import capabilities, model_api
from llm_sdk_view.app import app

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
    scratch = tempfile.mkdtemp(prefix="llm-sdk-view-cache-")
    monkeypatch.setenv("LLM_SDK_VIEW_CACHE_DIR", scratch)
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

    assert data["models"] == list(FOUR_MODELS)
    assert data["model_data"]["source"] == "fallback"
    assert data["model_data"]["profile_version"] == "2026-09-26"
    assert data["capabilities"]["context_window"] == 1_000_000


def test_a_fallback_is_never_labelled_as_the_api(isolated_cache):
    capabilities_for = capabilities.capabilities_for("claude-sonnet-5")

    assert capabilities_for.data_source.startswith("fallback-profile")
    assert "fallback" in capabilities_for.context_window_source


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


def test_the_ui_shows_the_provenance_rather_than_assuming_it():
    import llm_sdk_view

    html = (Path(llm_sdk_view.__file__).parent / "static" / "index.html").read_text("utf-8")

    assert "model_data" in html
    assert "fallback profile" in html
