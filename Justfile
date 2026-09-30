# Run tests and linters
@default: test lint

# Run pytest with supplied options. The browser extra is included because a
# UI check that skips is a UI check that is not running; it needs no download
# when this machine has Google Chrome.
@test *options:
    uv run --extra test --extra browser pytest {{options}}

# Run only the checks that drive a real browser
@browser *options:
    uv run --extra test --extra browser pytest -m browser {{options}}

# Run linters
@lint:
    uv run --extra test ruff check .

# Apply automatic fixes
@fix:
    uv run --extra test ruff check . --fix

# Start the local development server
@serve:
    uv run --extra test llm native-chat --reload
