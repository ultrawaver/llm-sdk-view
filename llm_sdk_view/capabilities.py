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
   be disabled, which tool options exist at all, and the thinking token budget
   the plugin hard-codes. Read every time.

Only narrow rules the Models API does not expose are hard-coded: default effort
per model, whether thinking can be switched off, how dynamic filtering maps
onto ``allowed_callers``, and the effort levels Anthropic documents as rejected
when thinking is disabled. Nothing else in this project branches on a model id:
the form, the request builder and the validation all read
:class:`ModelCapabilities`.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import llm

from . import model_api

MODELS_JSON = Path(__file__).parent / "models.json"

# Thinking is its own control. "on" asks the plugin to think, "off" asks it
# not to; the value the request carries is decided per model below.
THINKING_ON = "on"
THINKING_OFF = "off"
THINKING_STATES = (THINKING_ON, THINKING_OFF)

# "default" means the field is left off the request, so the model's own
# default effort applies. Effort no longer carries a "turn thinking off"
# value: that is the thinking control's job.
DEFAULT_EFFORT = "default"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")

# Anthropic documents that on models where thinking can be disabled, the
# request is rejected when thinking is disabled at these effort levels. It is
# stated for Claude Opus 5; the same generation shares the behaviour, so the
# form refuses the combination rather than risk a 400.
EFFORT_LEVELS_NEEDING_THINKING = ("xhigh", "max")

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
    thinking_default: str
    thinking_always_on: bool
    thinking_editable: bool
    thinking_off_request: str
    supports_thinking: bool
    can_disable_thinking: bool
    budget_tokens: int | None
    budget_tokens_editable: bool
    supports_effort: bool
    effort_levels: tuple[str, ...]
    default_effort: str | None
    supports_web_search: bool
    web_search_type: str | None
    plugin_response_inclusion: bool
    response_inclusion: bool
    allowed_callers: bool
    effective_allowed_callers: str | None
    dynamic_filtering: str
    cache_ttl: bool
    data_source: str
    verified: str

    def effort_options(self) -> tuple[str, ...]:
        """Effort levels the form offers. There is no "off": that is thinking."""
        return (DEFAULT_EFFORT, *self.effort_levels)

    def effort_disabled_with(self, thinking: str) -> tuple[str, ...]:
        """Effort levels that cannot be sent alongside this thinking state."""
        if thinking != THINKING_OFF:
            return ()
        if not self.can_disable_thinking:
            return self.effort_levels
        return tuple(
            level for level in self.effort_levels if level in EFFORT_LEVELS_NEEDING_THINKING
        )

    def thinking_uses_budget(self, thinking: str) -> bool:
        """True when this thinking state puts a budget_tokens on the request."""
        return (
            thinking == THINKING_ON
            and self.thinking_mode == "extended"
            and self.budget_tokens is not None
        )

    def min_max_tokens(self, thinking: str) -> int:
        """Anthropic requires budget_tokens to be smaller than max_tokens."""
        if self.thinking_uses_budget(thinking):
            return self.budget_tokens + 1
        return 1

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
            "thinking_default": self.thinking_default,
            "thinking_always_on": self.thinking_always_on,
            "thinking_editable": self.thinking_editable,
            "thinking_off_request": self.thinking_off_request,
            "supports_thinking": self.supports_thinking,
            "can_disable_thinking": self.can_disable_thinking,
            "budget_tokens": self.budget_tokens,
            "budget_tokens_editable": self.budget_tokens_editable,
            "supports_effort": self.supports_effort,
            "effort_levels": list(self.effort_levels),
            "effort_options": list(self.effort_options()),
            "effort_needs_thinking": list(EFFORT_LEVELS_NEEDING_THINKING),
            "default_effort": self.default_effort,
            "supports_web_search": self.supports_web_search,
            "web_search_type": self.web_search_type,
            "response_inclusion": self.response_inclusion,
            "plugin_response_inclusion": self.plugin_response_inclusion,
            "allowed_callers": self.allowed_callers,
            "effective_allowed_callers": self.effective_allowed_callers,
            "dynamic_filtering": self.dynamic_filtering,
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
    ``max_tokens`` could run past ten minutes. So the form has no streaming
    choice to make, and the right pane must render the call that happens.
    """
    source = inspect.getsource(type(model).execute)
    if ".create(" in source:
        return "create"
    if ".stream(" in source:
        return "stream"
    return "unknown"


def plugin_thinking_budget() -> int | None:
    """The budget_tokens value llm-anthropic hard-codes for extended thinking.

    The plugin accepts no option for it, so this is a fact about the runtime,
    not a setting: it is read from the installed module and shown read-only.
    """
    module = sys.modules.get("llm_anthropic")
    if module is None:
        try:
            module = importlib.import_module("llm_anthropic")
        except ImportError:
            return None
    budget = getattr(module, "DEFAULT_THINKING_TOKENS", None)
    return budget if isinstance(budget, int) else None


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
        # A thinking budget would be a prompt option too.
        "budget_tokens": any("budget" in name.lower() for name in option_fields),
    }


def _effective_caller(web_search_type: str | None) -> tuple[str | None, str]:
    """What the API will really do with the tool the plugin emits.

    ``web_search_20260209`` and later default to the code execution caller,
    which is what makes dynamic filtering run. Earlier versions default to
    ``direct`` and have no dynamic filtering at all.
    """
    if web_search_type is None:
        return None, "off"
    if web_search_type == "web_search_20260318":
        return "code_execution_20260120", "active"
    return "direct", "not-supported"


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
    supports_thinking = bool(getattr(model, "supports_thinking", False))
    can_disable = supports_thinking and not always_thinks

    # "Off" has to be expressed the way the API accepts it for this model.
    # Adaptive models take thinking={"type": "disabled"}; a model whose
    # default is not to think is turned off by leaving the field out, which is
    # the documented default and cannot be rejected.
    if not can_disable:
        thinking_off_request = "unsupported"
    elif thinking_mode == "adaptive":
        thinking_off_request = "disabled"
    else:
        thinking_off_request = "omitted"

    budget = plugin_thinking_budget() if thinking_mode == "extended" else None
    effective_caller, dynamic_filtering = _effective_caller(web_search_type)

    return ModelCapabilities(
        id=model_id,
        llm_id=entry["llm_id"],
        api_model_id=model.claude_model_id,
        context_window=context_window,
        context_window_source=context_source,
        max_output_tokens=max_output,
        max_output_source=max_output_source,
        thinking_mode=thinking_mode,
        thinking_default=THINKING_ON if always_thinks
        or bool(getattr(model, "thinks_by_default", False)) else THINKING_OFF,
        thinking_always_on=always_thinks,
        thinking_editable=can_disable,
        thinking_off_request=thinking_off_request,
        supports_thinking=supports_thinking,
        can_disable_thinking=can_disable,
        budget_tokens=budget,
        budget_tokens_editable=tool_caps["budget_tokens"],
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
        effective_allowed_callers=effective_caller,
        dynamic_filtering=dynamic_filtering,
        cache_ttl=tool_caps["cache_ttl"],
        data_source="models-api" if api else f"fallback-profile-{data['profile_version']}",
        verified=data["profile_version"],
    )
