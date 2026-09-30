# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's security advisory feature after the repository is published. Do not open a public issue containing credentials or exploitable details.

## Sensitive data policy

Native API Chat must never persist or display provider API keys. Keys belong in LLM's key store, environment variables, or another provider-supported credential mechanism.

The local server binds to `127.0.0.1` by default. Treat `--host 0.0.0.0` as an explicit trust-boundary change.

Generated SDK examples must use environment variables and must never interpolate live credentials.
