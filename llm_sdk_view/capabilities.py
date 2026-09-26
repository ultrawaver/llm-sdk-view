"""Model capabilities, in exactly one place, from three ranked sources.

1. The Anthropic **Models API** - model list, ``max_input_tokens`` (context
   window), ``max_tokens`` (output ceiling) and the ``thinking`` / ``effort``
   capabilities. Read live when a key exists, otherwise from a disk cache, by
   :mod:`llm_sdk_view.model_api`. It is never allowed to block the page.
2. The **fallback profile** in ``models.json`` - a versioned snapshot of
   Anthropic's published specs, used only for the four current models and only
   when the API is unavailable. It is always labelled as a fallback.
3. The **installed llm / llm-anthropic** - what the plugin can actually put on
   the wire: which web search tool version a model emits, whether thinking can
   be disabled, and which tool options exist at all. Read every time.

Only narrow rules the Models API does not expose are hard-coded: default effort
per model, whether thinking can be switched off, and how dynamic filtering maps
onto ``allowed_callers``. Nothing else in this project branches on a model id:
the form, the request builder and the validation all read
:class:`ModelCapabilities`.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import llm

from . import model_api

MODELS_JSON = Path(__file__).parent / "models.json"

# "default" means the field is left off the request, so the model's own
# default effort applies. "off" means thinking is disabled outright.
DEFAULT_EFFORT = "default"
DISABLED_EFFORT = "off"

DEFAULT_MODEL = "claude-sonnet-5"

EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")

# How long an in-process Models API snapshot may be reused before the disk
# cache is consulted again.
SNAPSHOT_TTL_SECONDS = 300
_snapshot_state = [0.0]


@dataclass(frozen=True)
class ModelCapabilities:
    """What one model can do, and where each fact came from."""

    id: str
    llm_id: str
    api_model_id: str
    context_window: int | None
    context_window_source: str
    max_output_tokens: int | None
    max_output_source: str
    thinking_mode: str
    thinking_always_on: bool
    supports_thinking: bool
    can_disable_thinking: bool
    supports_effort: bool
    effort_levels: tuple[str, ...]
    default_effort: str | None
    supports_web_search: bool
    web_search_type: str | None
    plugin_response_inclusion: bool
    response_inclusion: bool
    allowed_callers: bool
    cache_ttl: bool
    data_source: str
    verified: str

    def effort_options(self) -> tuple[str, ...]:
        return (DEFAULT_EFFORT, *self.effort_levels, DISABLED_EFFORT)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "llm_id": self.llm_id,
            "api_model_id": self.api_model_id,
            "context_window": self.context_window,
            "context_window_source": self.context_window_source,
            "max_output_tokens": self.max_output_tokens,
            "max_output_source": self.max_output_source,
            "thinking_mode": self.thinking_mode,
            "thinking_always_on": self.thinking_always_on,
            "supports_thinking": self.supports_thinking,
            "can_disable_thinking": self.can_disable_thinking,
            "supports_effort": self.supports_effort,
            "effort_levels": list(self.effort_levels),
            "effort_options": list(self.effort_options()),
            "default_effort": self.default_effort,
            "supports_web_search": self.supports_web_search,
            "web_search_type": self.web_search_type,
            "response_inclusion": self.response_inclusion,
            "plugin_response_inclusion": self.plugin_response_inclusion,
            "allowed_callers": self.allowed_callers,
            "cache_ttl": self.cache_ttl,
            "data_source": self.data_source,
            "verified": self.verified,
        }


def profile() -> dict:
    """The fallback profile, straight off disk."""
    return json.loads(MODELS_JSON.read_text("utf-8"))


def fallback_models() -> tuple[str, ...]:
    return tuple(profile()["models"])


@lru_cache(maxsize=1)
def _api_models_cached() -> tuple[tuple[dict, ...], dict]:
    """One Models API snapshot per process, so the page never pays twice."""
    snapshot = model_api.snapshot()
    return tuple(snapshot["models"]), snapshot["provenance"]


def _api_models() -> tuple[tuple[dict, ...], dict]:
    """Cached, but not forever: re-read the disk cache every few minutes."""
    now = time.monotonic()
    if now - _snapshot_state[0] > SNAPSHOT_TTL_SECONDS:
        reset_model_data()
        _snapshot_state[0] = now
    return _api_models_cached()


def api_models() -> tuple[dict, ...]:
    return _api_models()[0]


def provenance() -> dict:
    """Where the model data on this page came from."""
    return dict(_api_models()[1])


def refresh_model_data() -> dict:
    """Force a live Models API read and drop the cached snapshot."""
    result = model_api.refresh()
    _api_models_cached.cache_clear()
    return result


def reset_model_data() -> None:
    """Forget the in-process snapshot so the next read re-reads the disk."""
    _api_models_cached.cache_clear()
    _snapshot_state[0] = time.monotonic()


def _api_entry(model_id: str) -> dict | None:
    for entry in api_models():
        if entry.get("id") == model_id:
            return entry
    return None


def model_ids() -> tuple[str, ...]:
    """The model dropdown: the Models API list, curated models first.

    Only models the installed plugin can actually resolve are offered, so the
    form never lists a model this project cannot send a request to. With no
    Models API data the versioned fallback profile is used instead.
    """
    curated = fallback_models()
    if not api_models():
        return curated
    known = {_api_model_id(model_id): model_id for model_id in curated}
    discovered = []
    for entry in api_models():
        form_id = known.get(entry.get("id"))
        if form_id and form_id not in discovered:
            discovered.append(form_id)
    return tuple(discovered) + tuple(
        model_id for model_id in curated if model_id not in discovered
    )


def _api_model_id(model_id: str) -> str:
    entry = profile()["models"].get(model_id)
    return entry["llm_id"].split("/", 1)[-1] if entry else model_id


def resolve_model_id(model_id: str) -> str:
    """Map a form model id onto the id ``llm.get_model()`` resolves."""
    entry = profile()["models"].get(model_id)
    return entry["llm_id"] if entry else model_id


def _effort_levels_from_api(entry: dict, fallback: tuple[str, ...]) -> tuple[str, ...]:
    effort = (entry.get("capabilities") or {}).get("effort") or {}
    if not effort:
        return fallback
    levels = tuple(level for level in EFFORT_ORDER if effort.get(level))
    return levels or fallback


def _thinking_mode_from_api(entry: dict, fallback: str) -> str:
    thinking = (entry.get("capabilities") or {}).get("thinking") or {}
    types = thinking.get("types") or []
    if "adaptive" in types:
        return "adaptive"
    if "enabled" in types:
        return "extended"
    return fallback


def _plugin_effort_levels(model) -> tuple[str, ...]:
    """Read the effort enum off the installed plugin's own options model."""
    field = getattr(model.Options, "model_fields", {}).get("thinking_effort")
    if field is None:
        return ()
    annotation = field.annotation
    for candidate in (annotation, *getattr(annotation, "__args__", ())):
        members = getattr(candidate, "__members__", None)
        if members:
            return tuple(member.value for member in members.values() if member.value)
    return ()


def _web_search_type(model) -> str | None:
    """Ask the plugin which tool version this model gets, rather than guess."""
    for tool_class in getattr(model, "supported_server_side_tools", ()):
        if getattr(tool_class, "name", None) == "web_search":
            return tool_class().tool_spec(model)["type"]
    return None


def plugin_transport(model) -> str:
    """Which Anthropic SDK method the installed plugin actually calls.

    ``llm-anthropic`` opens ``messages.stream()`` even when LLM asked for a
    buffered response: the API rejects non-streaming requests whose
    ``max_tokens`` could run past ten minutes. So a buffered turn is a
    presentation choice, not a different SDK call, and the right pane must say
    so rather than render a ``messages.create()`` that never happens.
    """
    source = inspect.getsource(type(model).execute)
    if ".create(" in source:
        return "create"
    if ".stream(" in source:
        return "stream"
    return "unknown"


def _plugin_tool_capabilities(model) -> dict:
    parameters: set[str] = set()
    for tool_class in getattr(model, "supported_server_side_tools", ()):
        if getattr(tool_class, "name", None) == "web_search":
            parameters = set(inspect.signature(tool_class.__init__).parameters)
            break
    option_fields = getattr(model.Options, "model_fields", {})
    return {
        "response_inclusion": "response_inclusion" in parameters,
        "allowed_callers": "allowed_callers" in parameters,
        # A prompt cache TTL would be a prompt option, not a tool field.
        "cache_ttl": any("ttl" in name.lower() for name in option_fields),
    }


def capabilities_for(model_id: str) -> ModelCapabilities:
    """Everything the form and the request builder need about one model."""
    data = profile()
    entry = data["models"].get(model_id)
    if entry is None:
        raise ValueError(f"unknown model: {model_id}")
    model = llm.get_model(entry["llm_id"])
    tool_caps = _plugin_tool_capabilities(model)
    web_search_type = _web_search_type(model)
    api = _api_entry(model_id) or {}

    # Context window and output ceiling: the Models API wins when it answers.
    if api.get("max_input_tokens") is not None:
        context_window = api["max_input_tokens"]
        context_source = "Anthropic Models API max_input_tokens"
    else:
        context_window = entry["context_window"]
        context_source = (
            f"fallback profile {data['profile_version']} "
            f"(context window is not available offline)"
        )
    if api.get("max_tokens") is not None:
        max_output = api["max_tokens"]
        max_output_source = "Anthropic Models API max_tokens"
    else:
        max_output = getattr(model, "default_max_tokens", None) or entry["max_output_tokens"]
        max_output_source = (
            "llm-anthropic default_max_tokens, cross-checked against the "
            f"fallback profile {data['profile_version']}"
        )

    # Effort and thinking: capabilities from the API, narrow rules from the
    # profile and the plugin.
    plugin_levels = _plugin_effort_levels(model)
    supports_effort = bool(getattr(model, "supports_thinking_effort", False))
    effort_levels = _effort_levels_from_api(api, plugin_levels)
    api_effort = (api.get("capabilities") or {}).get("effort") or {}
    if api_effort and "supported" in api_effort:
        supports_effort = bool(api_effort["supported"]) and bool(effort_levels)

    thinking_mode = _thinking_mode_from_api(api, entry["thinking_mode"])
    always_thinks = bool(getattr(model, "always_thinks", False))

    return ModelCapabilities(
        id=model_id,
        llm_id=entry["llm_id"],
        api_model_id=model.claude_model_id,
        context_window=context_window,
        context_window_source=context_source,
        max_output_tokens=max_output,
        max_output_source=max_output_source,
        thinking_mode=thinking_mode,
        thinking_always_on=always_thinks,
        supports_thinking=bool(getattr(model, "supports_thinking", False)),
        can_disable_thinking=bool(getattr(model, "supports_thinking", False))
        and not always_thinks,
        supports_effort=supports_effort,
        effort_levels=effort_levels,
        default_effort=entry["default_effort"],
        supports_web_search=bool(getattr(model, "supports_web_search", False)),
        web_search_type=web_search_type,
        plugin_response_inclusion=tool_caps["response_inclusion"],
        # response_inclusion only exists on web_search_20260318, and only if the
        # installed plugin can express it at all.
        response_inclusion=tool_caps["response_inclusion"]
        and web_search_type == "web_search_20260318",
        allowed_callers=tool_caps["allowed_callers"],
        cache_ttl=tool_caps["cache_ttl"],
        data_source="models-api" if api else f"fallback-profile-{data['profile_version']}",
        verified=data["profile_version"],
    )
