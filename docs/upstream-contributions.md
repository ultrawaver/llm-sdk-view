# Upstream contributions

Verified against `llm==0.36` and `llm-anthropic==0.29` on 2026-09-26, by reading
the installed plugin source rather than documentation.

## Status of the four required Anthropic behaviours

| # | Behaviour | Status in `llm-anthropic` 0.29 |
|---|---|---|
| 1 | `web_search_20260318` | Already correct. `WebSearch.tool_spec()` emits it when the model has `supports_adaptive_thinking`, which `claude-sonnet-5` does. |
| 2 | Dynamic filtering | Already correct. `tool_spec()` never sends `allowed_callers`, so direct-only calling is never forced. |
| 3 | `response_inclusion="excluded"` | **Missing.** Not in `WebSearch.__init__` and not in `tool_spec()`. |
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
- commit: `1fe612526d89b1f381a6e480a3d32e05614103dd`

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

## Possible later contribution: LLM core

Only after a demonstrated need, propose provider-neutral observation hooks for:

- final provider SDK arguments;
- provider stream events;
- final provider response and usage.

Do not propose UI-specific hooks or Anthropic-specific fields in LLM core.
