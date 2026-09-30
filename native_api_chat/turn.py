"""What every provider's turn has in common, and deliberately nothing else.

Three fields and six words. That is genuinely all two providers share once the
compatibility layer is refused: a model, a ceiling on the reply, and a system
prompt. Thinking, effort, prompt caching, provider routing and the reasoning
controls are each one provider's own vocabulary, and the moment they are
hoisted into a shared options object the form starts offering a control the
selected model has never heard of.

This module exists so that :mod:`native_api_chat.chat` and the provider
packages can agree on those three fields and on the status words without
importing each other - the session reaches for a provider, so a provider must
not reach back for the session.

The status vocabulary is shared for the opposite reason to the options: a
greyed-out control has to say *why* in the same words whichever provider it
belongs to, or the page teaches the user two dialects of the same idea.
"""

from __future__ import annotations

from dataclasses import dataclass

# Every control reports one of these, so a value the form cannot set always
# says what would have to change for it to become settable.
EDITABLE = "Editable"
RUNTIME_FIXED = "API supported · runtime fixed"
UNSUPPORTED_BY_MODEL = "Unsupported by selected model"
UNSUPPORTED_BY_TOOL = "Unsupported by current tool version"
PROVIDER_DEFAULT = "Provider default"
FALLBACK_DATA = "Fallback capability data"
# The one status that is not an answer about the control: the source that
# would have answered could not be read. It is not "unsupported", and saying
# so would invent a refusal the provider never made.
UNKNOWN_CAPABILITY = "Capability data unavailable"

STATUSES = (
    EDITABLE,
    RUNTIME_FIXED,
    UNSUPPORTED_BY_MODEL,
    UNSUPPORTED_BY_TOOL,
    PROVIDER_DEFAULT,
    FALLBACK_DATA,
    UNKNOWN_CAPABILITY,
)


class MissingKeyError(RuntimeError):
    """No key is available for the selected provider, so nothing is sent."""


class UnsupportedOptionError(ValueError):
    """The form asked for something the installed plugin cannot send.

    A ``ValueError`` so the routes report it as a bad request: sending the
    request without the value would contradict the form that asked for it.
    """


@dataclass(frozen=True)
class TurnOptions:
    """The three settings no provider can do without.

    Subclassed per provider rather than extended in place. ``max_tokens`` is
    checked here because "reserve a negative reply" is meaningless to every
    API, not because any one of them said so.
    """

    model: str
    max_tokens: int = 16384
    system: str = ""

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")


def option_list(values, disabled=()) -> list[dict]:
    """A control's choices, each carrying whether it can be picked."""
    return [{"value": value, "disabled": value in disabled} for value in values]
