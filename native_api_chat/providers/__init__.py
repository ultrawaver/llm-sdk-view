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


def provider_by_id(provider_id: str | None) -> Provider | None:
    """The provider with this id, or None when no installed one answers to it.

    A stored turn records who sent it, which is a better answer than reading
    the model id back, and does not stop being one when the model is a bare
    catalogue slug that carries none of llm's routing prefix. None means the
    plugin that sent it is not installed here any more: nobody is then
    answerable for the vocabulary its counters were written in.
    """
    return next((p for p in PROVIDERS if p.id == provider_id), None)


__all__ = ["PROVIDERS", "Provider", "Transport", "provider_by_id", "provider_for"]
