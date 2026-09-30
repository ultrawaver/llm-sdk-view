"""What a provider must answer before a turn can be shown or sent.

There is one rule behind this module, and every field below exists to keep it:
**the right pane renders the call that really happens.** That was cheap to
honour while Anthropic was the only provider - one plugin, one transport, one
`build_kwargs()` that returned the whole request - and it stops being cheap the
moment a second provider arrives, because the three plugins do not agree on
what a request even is:

======================  ==============================================
``llm-anthropic``       ``build_kwargs(prompt, conversation)`` returns
                        the complete request.
``llm`` Chat            ``build_kwargs(prompt, stream)`` returns the
                        options only; ``model``, ``messages`` and
                        ``stream`` are added at the call site.
``llm`` Responses       ``_build_responses_input()`` and
                        ``_finalize_responses_kwargs()`` between them
                        return input and options; ``model``, ``input``
                        and ``stream`` are added at the call site.
======================  ==============================================

So "the request" is not a method any more, it is a per-provider assembly, and
:meth:`Provider.assemble` is where each one is written down. What it returns
must be the dictionary **as it appears at the call site**, because that is what
the pane prints and what the form is checked against. A provider that returned
only the options would render a request that nobody sends.

:class:`Transport` carries the rest of the call for the same reason. The header
is not decoration: OpenRouter is reached through the OpenAI SDK pointed at
``openrouter.ai`` with two headers attached, and a pane that printed a bare
``openai.OpenAI()`` would be describing a request to OpenAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Transport:
    """One SDK call, in the pieces needed to print it back verbatim.

    ``name`` is how the form and the tests refer to the call and is the
    dotted method itself (``messages.stream``, ``responses.create``), so a
    status line naming it cannot drift from the code being rendered.

    ``context`` is the difference between a call that is opened and one that
    returns: ``client.messages.stream()`` is a context manager, while
    ``client.responses.create(stream=True)`` hands back an iterator. Getting
    this wrong would render Python that does not run.
    """

    name: str
    header: str
    call: str
    binding: str
    context: bool
    body: str = ""


@runtime_checkable
class Provider(Protocol):
    """One provider's answer to "what is about to be sent, and how".

    Deliberately narrow. Capability, pricing and token-count sources are not
    here: they are already resolved per model elsewhere, and folding them in
    would make a provider the place every question is asked rather than the
    place the request is built.
    """

    id: str
    label: str

    def owns(self, model_id: str) -> bool:
        """Whether this provider is the one that sends ``model_id``."""

    def transport(self, model: Any, options: Any) -> Transport:
        """The call this turn will really make.

        It takes the options because a provider may switch transport per
        request: OpenRouter's ``chat_completions`` moves the same model from
        ``responses.create`` to ``chat.completions.create``.
        """

    def assemble(
        self, model: Any, prompt: Any, conversation: Any, options: Any
    ) -> dict:
        """The complete request, exactly as it reaches the SDK call."""
