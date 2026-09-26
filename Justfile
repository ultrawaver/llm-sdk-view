# Run tests and linters
@default: test lint

# Run pytest with supplied options
@test *options:
    uv run --extra test pytest {{options}}

# Run linters
@lint:
    uv run --extra test ruff check .

# Apply automatic fixes
@fix:
    uv run --extra test ruff check . --fix

# Start the local development server
@serve:
    uv run --extra test llm sdk-view --reload
