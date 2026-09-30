# Upstream contributions

Verified against `llm==0.36` and `llm-anthropic==0.30` on 2026-09-29, by reading
the installed plugin source rather than documentation.

The installed plugin is not the PyPI release: it is the fork branch below, built
from source, so it carries `response_inclusion` while stock 0.30 does not. CI
installs plain `llm-anthropic` from PyPI, so CI exercises the same three of four
behaviours a machine without the branch sees.

## Status of the four required Anthropic behaviours

| # | Behaviour | Status in `llm-anthropic` 0.30 (stock) |
|---|---|---|
| 1 | `web_search_20260318` | Already correct. `WebSearch.tool_spec()` emits it when the model has `supports_adaptive_thinking`, which `claude-sonnet-5` does. |
| 2 | Dynamic filtering | Already correct. `tool_spec()` never sends `allowed_callers`, so direct-only calling is never forced. |
| 3 | `response_inclusion="excluded"` | **Missing.** Stock 0.30 contains neither `response_inclusion` nor `allowed_callers`; only the fork branch supplies it. |
| 4 | Prompt caching | Already correct. `options.cache` puts `cache_control={"type": "ephemeral"}` on the last block of the final message. |

Only one of the four needs an upstream change.

## Minimal PR: `WebSearch(response_inclusion=...)`

Desired CLI shape:

```bash
llm -m claude-sonnet-5 \
  -T 'WebSearch(max_uses=5, response_inclusion="excluded")' \
  'Research a current architecture question'
```

### Files to change

1. `llm_anthropic.py`
   - `WebSearch.__init__`: add `response_inclusion: Optional[str] = None`.
   - Validate locally: accept only `"full"` or `"excluded"`, otherwise raise
     `ValueError`. Do not pass unknown values through to the API.
   - `WebSearch.tool_spec()`:
     - emit `response_inclusion` only for `web_search_20260318` (the `modern`
       branch);
     - raise `ValueError` when it is set for a model that resolves to
       `web_search_20250305`, instead of silently dropping the option — the same
       pattern `WebFetch` already uses for `use_cache`.
2. `tests/test_anthropic.py`
   - Offline test that `build_kwargs()` emits
     `{"type": "web_search_20260318", "name": "web_search", "max_uses": 5,
     "response_inclusion": "excluded"}` for `claude-sonnet-4.6`.
   - Offline test that an unset value is omitted, so current behaviour is
     unchanged.
   - Offline test that an unsupported value raises `ValueError`.
   - Offline test that setting it on a pre-4.6 model raises `ValueError`.
   - No VCR-marked (paid) test is required for this change.
3. `README.md`
   - Document the option in the Web Search section.

### Submitted upstream

The patch above was published as a pull request on 2026-09-26:

- upstream PR: <https://github.com/simonw/llm-anthropic/pull/95> (open)
- fork: <https://github.com/ultrawaver/llm-anthropic>
- branch: `web-search-response-inclusion`
- commit: `a3ce43acf9c00a5b5df750e57d544eeaf3e18c00`

The branch tracks upstream releases by rebasing, never by merging: upstream's
`0.30` tag arrived on 2026-09-29 and the single patch commit was replayed onto
it (a clean replay - upstream had not touched `WebSearch` itself). The previous
tip is preserved locally as `backup/pre-0.30-rebase` (`1fe6125`). Rebasing keeps
the commit's parent honest, so the diff Simon reviews stays one commit; repeated
merges instead would pile branch noise into the PR.

Upgrading the dependency must reinstall from this branch, never
`pip install -U llm-anthropic`: the PyPI release would silently replace the
build and drop `response_inclusion`, leaving this project back at three of four
behaviours with no error anywhere.

This repository does not copy that code; when the branch's commit is installed
the project starts emitting `response_inclusion` with no source change of its
own. Until the PR is merged and released, `llm-anthropic` 0.29 remains unable to
express the option, and the corresponding assertions here skip rather than fail.

### What changes here once upstream merges

1. Replace the pinned fork commit in `.github/workflows/test.yml` with the plain
   PyPI dependency, using a lower bound on the first release that carries the
   option.
2. Tighten `tests/test_upstream_contract.py` so a missing `response_inclusion`
   fails instead of skipping.
3. Update this file to record the released version.

Until then nothing in this repository depends on the branch being merged: the
default configuration degrades to three of the four behaviours.

### Out of scope for this PR

- `llm` core: no change is needed. `llm.ServerSideTool` and
  `llm.models._partition_tools` already transport arbitrary server-side tool
  specs.
- `cache_control` TTL (for example `1h`): `llm-anthropic` hard-codes
  `{"type": "ephemeral"}`. A TTL option is a separate, later PR.
- This repository must not fork `WebSearch`; it waits for the upstream option.

### Fields the form shows but the plugin cannot send

Checked on 2026-09-26 against `llm-anthropic` main and against the
`response_inclusion` branch installed here, by reading the installed source
(`WebSearch.__init__`, `WebSearch.tool_spec()`, `ClaudeOptions`,
`build_kwargs()`) and the upstream test suite. Each of these is shown in the UI
with the status `API supported · runtime fixed`, and none of them can be
changed by editing this repository:

| Field | What `llm-anthropic` can do | What this repository does |
|---|---|---|
| `allowed_callers` | Not a `WebSearch` parameter at all. Omitting the field is what leaves the API default caller in place: `code_execution_20260120` on `web_search_20260318` (dynamic filtering on), `direct` on `web_search_20250305` (no dynamic filtering). | Shows the effective caller, sends nothing, and refuses `direct` instead of silently leaving filtering on. `direct` is listed as `API supported · not exposed by llm-anthropic`. Starts sending it automatically if the plugin gains the parameter. |
| `cache_control` TTL | No TTL option exists; `{"type": "ephemeral"}` is hard-coded. | No TTL control at all, only the read-only `Provider default: 5m`. The form must not offer a setting that cannot reach the request. |
| `thinking.budget_tokens` | `DEFAULT_THINKING_TOKENS` (1024) is hard-coded for models without adaptive thinking; there is no option for it. | Reads the value off the installed module and shows it read-only, marked `API supported · runtime fixed`. `max_tokens` is still validated against it, because Anthropic requires `budget_tokens < max_tokens`. |
| A non-streaming `messages.create()` call | `execute()` always opens `messages.stream()`, even when LLM asked for a buffered response: the API rejects non-streaming requests whose `max_tokens` could run past ten minutes. | `stream` is shown as a read-only `ON`, marked `API supported · runtime fixed by llm-anthropic`. There is no OFF: a control that only changes how this page buffers text would be a lie about the SDK call. The right pane renders `client.messages.stream(...)` and nothing else - the "why" lives in that control's status, never as a comment inside the request. |

### Runtime limits the form has to respect

These are not missing upstream features; they are Anthropic API rules that the
form enforces locally so a request cannot be built that the API would reject:

- Thinking cannot be disabled on `claude-fable-5-1` or `claude-opus-5-5`; both
  the API and `llm-anthropic` reject `thinking={"type": "disabled"}`.
- Thinking disabled is not accepted at `xhigh` or `max` effort (documented for
  Claude Opus 5; the form applies it to the same generation).
- `budget_tokens` must be smaller than `max_tokens`.
- `max_tokens` may not exceed the model's output ceiling.
- `response_inclusion` only exists on `web_search_20260318`.

## Candidate PR: `llm` drops a truncated Responses turn

`llm`'s Responses handler (`llm.default_plugins.openai_models._SharedResponses`)
reads the final payload only from `response.completed`. A reply that hits
`max_output_tokens` ends the stream with `response.incomplete` instead, and
that branch does not exist, so the whole final payload is discarded:
`response_json` stays `None`, `set_usage()` is never reached, and the turn has
no counts, no stop reason and no content blocks. Only the streamed text
survives.

This is not OpenRouter-specific - the event is OpenAI's own, and the Responses
API emits it whenever a reply is cut off at the ceiling - which is what makes
it worth sending upstream rather than working around here.

Measured 2026-09-30 against `openrouter/nvidia/nemotron-3-super-120b-a12b:free`
through `client.responses.create(..., max_output_tokens=32)`. The stream ended:

```
response.output_text.done, response.incomplete
```

with `incomplete_details: {"reason": "max_output_tokens"}` and a full `usage`
object on the discarded payload.

The fix is to treat `response.incomplete` as a terminal payload alongside
`response.completed`, keeping the reason available rather than inferred.

What this project does until then: nothing that guesses. `stop_reason` is
reported as unknown, and a usage object carrying no counts is not stored at
all, so the context meter falls back to its labelled estimate. Storing it made
the meter report 0 tokens as `API usage` - a measured figure that was never
measured (`native_api_chat/chat.py`, `_record_usage`).

## Possible later contribution: LLM core

Only after a demonstrated need, propose provider-neutral observation hooks for:

- final provider SDK arguments;
- provider stream events;
- final provider response and usage.

Do not propose UI-specific hooks or Anthropic-specific fields in LLM core.
