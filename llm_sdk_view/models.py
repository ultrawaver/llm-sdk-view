from dataclasses import dataclass, field
from typing import Literal

ResponseInclusion = Literal["full", "excluded"]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class AnthropicTurn:
    """Canonical Anthropic turn specification.

    The UI form, the generated SDK snippet, and the future execution path all
    derive from this one object. Nothing here is allowed to be a value that
    `llm-anthropic` cannot actually send: a field that the installed plugin has
    no way to express would make the right-hand pane lie about the request.
    """

    model: str = "claude-sonnet-5"
    max_tokens: int = 4096
    messages: tuple[Message, ...] = field(default_factory=tuple)
    web_search: bool = True
    max_searches: int = 5
    response_inclusion: ResponseInclusion = "excluded"
    prompt_cache: bool = True

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.max_searches < 1:
            raise ValueError("max_searches must be at least 1")
        if self.response_inclusion not in ("full", "excluded"):
            raise ValueError("response_inclusion must be 'full' or 'excluded'")
