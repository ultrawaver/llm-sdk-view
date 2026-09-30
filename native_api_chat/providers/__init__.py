"""Which provider sends a given model, resolved in exactly one place.

The order matters: the first provider that claims a model id gets it, so the
narrow claims come before the broad ones. Nothing outside this package may
branch on a model id to decide who sends it - that is the conditional this
package exists to delete.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .base import Provider, Transport
from .openrouter import OpenRouterProvider

PROVIDERS: tuple[Provider, ...] = (AnthropicProvider(), OpenRouterProvider())


def provider_for(model_id: str) -> Provider:
    """The provider that will really send ``model_id``."""
    for provider in PROVIDERS:
        if provider.owns(model_id):
            return provider
    raise ValueError(f"no provider claims {model_id!r}")


__all__ = ["PROVIDERS", "Provider", "Transport", "provider_for"]
