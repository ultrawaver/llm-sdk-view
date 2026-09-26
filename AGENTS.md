# AGENTS.md

This project is an LLM plugin and local web application.

## Development environment

Install the project with test dependencies:

```bash
pip install -e '.[test]'
```

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

## Project boundaries

- Keep the product small: chat on the left, equivalent provider SDK code on the right.
- Reuse LLM and provider plugins instead of rebuilding conversations, key storage, or provider adapters.
- Submit generally useful provider behavior upstream.
- Never make paid API calls in tests.
- Never store, print, serialize, or render API keys.
- Generate SDK code from the request `model.build_kwargs()` returns, never from a second template.
- Read model facts from the Anthropic Models API first; fall back to the versioned profile in `llm_sdk_view/models.json` only when it is unavailable, and always say which one is in use.
- Keep model capability decisions in `llm_sdk_view/capabilities.py`, not in UI conditionals.
- Do not add agents, RAG, memory, MCP, file upload, or additional providers without a separately accepted objective.

## Current Anthropic acceptance gates

1. `web_search_20260318`
2. Dynamic filtering remains enabled
3. `response_inclusion="excluded"`
4. Prompt caching is enabled

Read `docs/architecture.md` and `docs/upstream-contributions.md` before changing
the active implementation objective.
