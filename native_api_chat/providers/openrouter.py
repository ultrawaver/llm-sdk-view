"""OpenRouter through ``llm-openrouter``, reached with the OpenAI SDK.

OpenRouter is usually the example of the thing this project refuses to be - a
lowest-common-denominator layer in front of everyone else's models - and it is
worth being exact about why it belongs here anyway. It is treated as a
provider in its own right, with a native surface of its own: provider routing
preferences, the four reasoning controls, its own hosted server tools, and
usage that reports what a turn actually cost rather than what it probably
cost. What is *not* claimed is that ``openrouter/anthropic/claude-sonnet-5``
is Anthropic's Claude. It is a different entry, with different capabilities
and a different SDK call, and the day those two collapse into one row this
project has started lying.

Two transports, one model. ``llm-openrouter`` registers only the Responses
classes, and the ``chat_completions`` option makes ``execute()`` delegate to
``OpenRouterChat`` for that one request - so the call a turn makes is decided
per turn, and :meth:`OpenRouterProvider.transport` has to read the option
rather than the model. The delegation is also where a real constraint lives:
the plugin refuses server-side tools on the Chat Completions path, so
switching transport takes web search away. That is reported, never worked
around.
"""

from __future__ import annotations

from typing import Any

from .base import Transport

MODEL_PREFIX = "openrouter/"

API_BASE = "https://openrouter.ai/api/v1"

# The headers llm-openrouter really attaches when it registers a model. They
# identify the caller to OpenRouter's rankings, they are on the wire, and a
# rendered call that left them out would not be the request that was sent.
HEADERS = {
    "HTTP-Referer": "https://llm.datasette.io/",
    "X-OpenRouter-Title": "LLM",
}

# No key is written into the generated code: the SDK reads OPENROUTER_KEY from
# the environment, and this project never puts a secret in a pane the user is
# about to copy.
HEADER = (
    "import os\n"
    "\n"
    "import openai\n"
    "\n"
    "client = openai.OpenAI(\n"
    f'    base_url="{API_BASE}",\n'
    '    api_key=os.environ["OPENROUTER_KEY"],\n'
    "    default_headers={\n"
    + "".join(f'        "{key}": "{value}",\n' for key, value in HEADERS.items())
    + "    },\n"
    ")\n"
)

# How each stream is really read. `responses.create(stream=True)` hands back an
# iterator of typed events, while chat completions yield choice deltas; both
# are printed the way the SDK's own documentation reads them.
RESPONSES_BODY = (
    "for event in stream:\n"
    '    if event.type == "response.output_text.delta":\n'
    '        print(event.delta, end="", flush=True)\n'
)

CHAT_BODY = (
    "for chunk in completion:\n"
    "    delta = chunk.choices[0].delta.content if chunk.choices else None\n"
    "    if delta:\n"
    '        print(delta, end="", flush=True)\n'
)

RESPONSES = Transport(
    name="responses.create",
    header=HEADER,
    call="client.responses.create",
    binding="stream",
    context=False,
    body=RESPONSES_BODY,
)

CHAT_COMPLETIONS = Transport(
    name="chat.completions.create",
    header=HEADER,
    call="client.chat.completions.create",
    binding="completion",
    context=False,
    body=CHAT_BODY,
)


def uses_chat_completions(options: Any) -> bool:
    """Whether this turn takes the Chat Completions path.

    The option lives on the model's own Options object, so it is read by name
    from whatever the form produced rather than from a copy kept here.
    """
    return bool(getattr(options, "chat_completions", False))


class OpenRouterProvider:
    """OpenRouter's own API, reached through the OpenAI SDK."""

    id = "openrouter"
    label = "OpenRouter"

    def owns(self, model_id: str) -> bool:
        return model_id.startswith(MODEL_PREFIX)

    def transport(self, model: Any, options: Any = None) -> Transport:
        """Which call this turn makes, decided by the request, not the model.

        ``llm-openrouter`` registers Responses models only and delegates to
        ``OpenRouterChat`` inside ``execute()`` when ``chat_completions`` is
        set, so the same model id can produce either call.
        """
        return CHAT_COMPLETIONS if uses_chat_completions(options) else RESPONSES

    def assemble(
        self, model: Any, prompt: Any, conversation: Any, options: Any = None
    ) -> dict:
        """The complete request, in whichever shape this turn will send.

        Neither path hands back a finished request: ``llm`` adds ``model``,
        the input and ``stream`` at the call site. Putting them back here is
        not a second copy of the request - it is the only place the request
        exists whole, which is exactly what the pane has to print.
        """
        if uses_chat_completions(prompt.options):
            return self._chat_completions_kwargs(model, prompt, conversation)
        return self._responses_kwargs(model, prompt)

    def _responses_kwargs(self, model: Any, prompt: Any) -> dict:
        image_detail = getattr(prompt.options, "image_detail", None)
        if image_detail is not None:
            image_detail = getattr(image_detail, "value", image_detail)
        input_items, instructions = model._build_responses_input(
            prompt, image_detail=image_detail
        )
        kwargs = model._finalize_responses_kwargs(prompt, True, instructions)
        return {
            "model": model.model_name or model.model_id,
            "input": input_items,
            "stream": True,
            **kwargs,
        }

    def _chat_completions_kwargs(
        self, model: Any, prompt: Any, conversation: Any
    ) -> dict:
        """Built through the delegate the plugin itself would build.

        ``execute()`` does not send this request from the Responses model: it
        constructs an ``OpenRouterChat`` from ``_delegate_chat_kwargs()`` and
        hands the prompt to that. Asking the same object for the request is
        what keeps the pane and the wire in step.
        """
        chat = self._delegate(model)
        image_detail = getattr(prompt.options, "image_detail", None)
        if image_detail is not None:
            image_detail = getattr(image_detail, "value", image_detail)
        messages = chat.build_messages(prompt, conversation, image_detail=image_detail)
        kwargs = chat.build_kwargs(prompt, True)
        return {
            "model": chat.model_name or chat.model_id,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

    @staticmethod
    def _delegate(model: Any):
        from llm_openrouter import OpenRouterChat

        return OpenRouterChat(**model._delegate_chat_kwargs())
