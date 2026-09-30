# Contributing

Thank you for considering a contribution.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
ruff check .
```

Or use `uv` and `just`:

```bash
just
```

## Scope

Native API Chat is intentionally narrow. The first version has two jobs:

1. provide a practical conversation interface;
2. show equivalent provider SDK code generated from the same canonical request.

Please open an issue before adding a provider, agent workflow, retrieval system, upload pipeline, memory feature, or cloud service.

## Upstream contributions

A change belongs upstream when it is useful without this UI. Examples include a missing Anthropic server-tool option or a provider-neutral observation hook in LLM.

A change belongs here when it concerns the two-pane interface, code rendering, local presentation, or product-specific statistics.

## Tests

- Unit tests must not call paid APIs.
- Request construction tests should assert exact provider-facing fields.
- Code-generation tests should compare complete generated snippets.
- Redact secrets in fixtures and captured failures.

## Pull requests

Keep pull requests focused. Include:

- the user-visible outcome;
- tests proving the behavior;
- any upstream issue or API documentation that defines the behavior;
- an explicit note if no paid API call was used.
