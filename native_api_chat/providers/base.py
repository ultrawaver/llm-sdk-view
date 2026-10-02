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


def first_int(source: dict, *fields: str) -> int | None:
    """The first of these document fields that is really a whole number.

    Two providers call the same counter two names - OpenRouter writes the
    prompt as ``input_tokens`` on one transport and ``prompt_tokens`` on the
    other - so a reader has to try each name before it may say the provider
    reported nothing. It returns None rather than 0 for a missing field,
    because "did not report" and "reported zero" are different answers and
    only the second one is a measurement. ``bool`` is excluded: ``true`` is
    not a token count.
    """
    for field in fields:
        value = source.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def system_as_sent(value: Any) -> str:
    """The system prompt as the request carries it, not as it was typed.

    ``llm`` assembles the system prompt in ``_combine_system()``, which strips
    every fragment and keeps only those that are left - so a prompt typed with
    a trailing newline reaches the API without one, and a prompt of nothing but
    whitespace reaches it as no system prompt at all. An option that kept the
    typed text is then a value no request can contain: the check that compares
    the two refuses the turn over a difference no reader can see, which is how
    "You are helpful\\n" came back as *the request system prompt does not match
    the form*.

    Normalising here makes ``options.system`` the string that is really sent -
    what the pane prints, what the request is checked against, what a stored
    turn records, and what next turn's "system changed" is compared with.
    """
    return str(value or "").strip()


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

    #: Whether this provider's API has a free token counter to ask.
    #:
    #: ``token_count`` is Anthropic's ``messages.count_tokens``, and the only
    #: thing it is given is the request just built. That request is the wrong
    #: shape for anyone else - and the Chat Completions shape is close enough
    #: to be accepted, which would send an OpenRouter conversation to Anthropic
    #: under the user's Anthropic key. A provider without a counter says so and
    #: the context figure falls to the next source down, labelled.
    counts_tokens: bool

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

    def unavailable(self, model_id: str, error: Exception) -> Exception:
        """Why ``llm`` has no model under this id, as the error to raise.

        A plugin that registers nothing without a key makes every one of its
        models look unknown, and "unknown model" sends the user looking for a
        typo instead of for their key. Only the provider knows which of the two
        it is, so only the provider may name it.
        """

    def stop_reason(self, message: dict) -> str | None:
        """Why the reply ended, in whatever word this provider uses.

        The page shows this, and "unreported" is the wrong thing to show when
        the reply was cut off at the token ceiling: a truncated answer that
        does not say it was truncated reads as a complete one. Anthropic calls
        it ``stop_reason``; the two OpenRouter paths call it ``finish_reason``
        and ``incomplete_details.reason``.
        """

    def blocks(self, message: dict) -> list[dict]:
        """The finished reply, as the content blocks a record is made of.

        The wire shapes disagree as much as the requests do. Anthropic ends
        with a list of typed blocks; OpenRouter's Chat Completions path ends
        with ``content`` as a plain string, and its Responses path puts the
        reply under ``output`` and calls a text part ``output_text``. Iterating
        one shape as though it were another either finds nothing or walks a
        string character by character.

        Translating here rather than at each reader keeps the record - and so
        the transcript, the export and the panes - on one block vocabulary,
        which is the only reason those three can be read by the same code.
        """
