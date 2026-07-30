from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

MessageRole = Literal["system", "user", "assistant", "tool"]
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


@dataclass(frozen=True)
class ToolCall:
    """A model request to invoke a named tool."""

    id: str
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or len(self.id) > 256:
            raise ValueError("tool call id must contain 1 to 256 characters")
        if not isinstance(self.name, str) or not _TOOL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("tool call name contains invalid characters or is too long")
        if not isinstance(self.arguments, Mapping) or not all(
            isinstance(key, str) for key in self.arguments
        ):
            raise ValueError("tool call arguments must be an object with string keys")


@dataclass(frozen=True, eq=False)
class Message:
    """Provider-neutral conversation message."""

    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Message):
            return NotImplemented
        return (
            self.role,
            self.content,
            self.tool_calls,
            self.tool_call_id,
        ) == (
            other.role,
            other.content,
            other.tool_calls,
            other.tool_call_id,
        )

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role}")
        if self.role in {"system", "user"}:
            if self.content is None:
                raise ValueError(f"{self.role} message content is required")
            if self.tool_calls or self.tool_call_id is not None:
                raise ValueError(f"{self.role} messages cannot contain tool linkage")
        elif self.role == "assistant":
            if self.tool_call_id is not None:
                raise ValueError("assistant messages cannot be tool results")
            if self.content is None and not self.tool_calls:
                raise ValueError("assistant message requires content or tool calls")
            call_ids = [call.id for call in self.tool_calls]
            if len(call_ids) != len(set(call_ids)):
                raise ValueError("assistant tool call ids must be unique")
        elif self.role == "tool":
            if self.content is None:
                raise ValueError("tool result content is required")
            if not self.tool_call_id:
                raise ValueError("tool result requires a tool_call_id")
            if self.tool_calls:
                raise ValueError("tool result cannot request another tool")


class SystemMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="system", content=content)


class UserMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role="user", content=content)


class AssistantMessage(Message):
    def __init__(
        self,
        content: str | None = None,
        *,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> None:
        super().__init__(role="assistant", content=content, tool_calls=tool_calls)


class ToolMessage(Message):
    """Result associated with an assistant tool call by its stable ID."""

    def __init__(self, content: str, *, tool_call_id: str) -> None:
        super().__init__(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass(frozen=True)
class ToolDefinition:
    """JSON-schema description exposed to models that support tool calling."""

    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or value < 0
            for value in (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        ):
            raise ValueError("usage token counts must be non-negative integers")


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    model: str | None = None
    max_tokens: int | None = None

    @classmethod
    def from_prompt(cls, prompt: str) -> ModelRequest:
        return cls(messages=(UserMessage(prompt),))


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    provider_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("model response text must be a string")
        if not isinstance(self.tool_calls, tuple) or not all(
            isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise ValueError("model response tool_calls must be a ToolCall tuple")
        if not isinstance(self.usage, Usage):
            raise ValueError("model response usage is invalid")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise ValueError("model response finish_reason is invalid")
        if not isinstance(self.provider_metadata, Mapping):
            raise ValueError("model response metadata must be an object")


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError("tool call delta index must be a non-negative integer")
        if self.id is not None and (
            not isinstance(self.id, str) or len(self.id) > 256
        ):
            raise ValueError("tool call delta id is invalid")
        if self.name is not None and (
            not isinstance(self.name, str) or len(self.name) > 128
        ):
            raise ValueError("tool call delta name is invalid")
        if not isinstance(self.arguments_delta, str):
            raise ValueError("tool call arguments delta must be text")


@dataclass(frozen=True)
class ModelStreamChunk:
    """One provider-neutral event from a streaming completion."""

    text_delta: str = ""
    tool_call_deltas: tuple[ToolCallDelta, ...] = ()
    usage: Usage | None = None
    finish_reason: str | None = None
    provider_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text_delta, str):
            raise ValueError("model stream text delta must be text")
        if not isinstance(self.tool_call_deltas, tuple) or not all(
            isinstance(delta, ToolCallDelta) for delta in self.tool_call_deltas
        ):
            raise ValueError("model stream tool call deltas are invalid")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise ValueError("model stream usage is invalid")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise ValueError("model stream finish_reason is invalid")
        if not isinstance(self.provider_metadata, Mapping):
            raise ValueError("model stream metadata must be an object")


@dataclass(frozen=True)
class Subsystem:
    name: str
    path: str
    file_count: int
    notes: str


@dataclass(frozen=True)
class PortingModule:
    name: str
    responsibility: str
    source_hint: str
    status: str = "planned"


@dataclass
class PortingBacklog:
    title: str
    modules: list[PortingModule] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        return [
            f"- {module.name} [{module.status}] — {module.responsibility} (from {module.source_hint})"
            for module in self.modules
        ]
