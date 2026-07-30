"""OpenAI Chat Completions compatible provider adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from .models import (
    Message,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
    Usage,
)
from .providers import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderInvalidResponseError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeoutError,
)

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)
DEFAULT_MAX_PROVIDER_RESPONSE_BYTES = 4_000_000
DEFAULT_MAX_STREAM_EVENT_BYTES = 1_000_000


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    timeout: float = 60.0
    max_tokens: int = 4096
    max_response_bytes: int = DEFAULT_MAX_PROVIDER_RESPONSE_BYTES
    max_stream_event_bytes: int = DEFAULT_MAX_STREAM_EVENT_BYTES
    max_retries: int = 2
    retry_base_delay: float = 0.25
    max_retry_delay: float = 5.0

    @classmethod
    def from_app_config(cls, config: AppConfig) -> OpenAICompatibleConfig:
        """Create adapter settings without coupling the config layer to HTTPX."""

        if config.api_key is None:
            raise ProviderConfigurationError("api_key is required")
        if config.model is None:
            raise ProviderConfigurationError("model is required")
        return cls(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            timeout=config.timeout,
            max_tokens=config.max_tokens,
        )

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProviderConfigurationError("api_key is required")
        if not self.base_url:
            raise ProviderConfigurationError("base_url is required")
        if not self.model:
            raise ProviderConfigurationError("model is required")
        if self.timeout <= 0:
            raise ProviderConfigurationError("timeout must be positive")
        if self.max_tokens <= 0:
            raise ProviderConfigurationError("max_tokens must be positive")
        if self.max_response_bytes <= 0 or self.max_stream_event_bytes <= 0:
            raise ProviderConfigurationError("provider response limits must be positive")
        if self.max_stream_event_bytes > self.max_response_bytes:
            raise ProviderConfigurationError(
                "stream event limit cannot exceed the total response limit"
            )
        if self.max_retries < 0:
            raise ProviderConfigurationError("max_retries must not be negative")
        if self.retry_base_delay < 0 or self.max_retry_delay <= 0:
            raise ProviderConfigurationError("provider retry delays are invalid")


class OpenAICompatibleProvider:
    """Provider for APIs implementing OpenAI's Chat Completions shape."""

    name = "openai-compatible"

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            base_url=f"{config.base_url.rstrip('/')}/",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
            transport=transport,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        for attempt in range(self._config.max_retries + 1):
            retry_delay: float | None = None
            try:
                async with self._client.stream(
                    "POST",
                    "chat/completions",
                    json=self._build_payload(request, stream=False),
                ) as response:
                    if response.status_code == 429 and attempt < self._config.max_retries:
                        retry_delay = self._retry_delay(response, attempt)
                    else:
                        self._raise_for_status(response.status_code)
                        logger.debug(
                            "model provider returned HTTP %s", response.status_code
                        )
                        raw_payload = await self._read_bounded_response(response)
                        try:
                            payload = json.loads(raw_payload)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            raise ProviderInvalidResponseError(
                                "model provider returned invalid JSON"
                            ) from None
                        return self._parse_response(payload)
            except httpx.TimeoutException:
                raise ProviderTimeoutError("model request timed out") from None
            except httpx.RequestError:
                raise ProviderConnectionError(
                    "unable to connect to model provider"
                ) from None
            if retry_delay is not None:
                logger.warning(
                    "model provider rate limited request; retrying attempt=%d delay=%.3fs",
                    attempt + 1,
                    retry_delay,
                )
                await self._sleep(retry_delay)
        raise ProviderRateLimitError("model provider rate limit retry budget exhausted")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        for attempt in range(self._config.max_retries + 1):
            retry_delay: float | None = None
            try:
                async with self._client.stream(
                    "POST",
                    "chat/completions",
                    json=self._build_payload(request, stream=True),
                ) as response:
                    if response.status_code == 429 and attempt < self._config.max_retries:
                        retry_delay = self._retry_delay(response, attempt)
                    else:
                        self._raise_for_status(response.status_code)
                        logger.debug(
                            "model provider opened stream with HTTP %s",
                            response.status_code,
                        )
                        async for data in self._iter_sse_data(response):
                            if data == "[DONE]":
                                return
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError:
                                raise ProviderInvalidResponseError(
                                    "model provider returned invalid streaming JSON"
                                ) from None
                            yield self._parse_stream_chunk(payload)
                        return
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException:
                raise ProviderTimeoutError("model stream timed out") from None
            except httpx.RequestError:
                raise ProviderConnectionError("model stream connection failed") from None
            if retry_delay is not None:
                logger.warning(
                    "model provider rate limited stream; retrying attempt=%d delay=%.3fs",
                    attempt + 1,
                    retry_delay,
                )
                await self._sleep(retry_delay)
        raise ProviderRateLimitError("model provider rate limit retry budget exhausted")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _read_bounded_response(self, response: httpx.Response) -> bytes:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = 0
            if declared_length > self._config.max_response_bytes:
                raise ProviderInvalidResponseError(
                    "model provider response exceeded the configured size limit"
                )
        captured = bytearray()
        async for chunk in response.aiter_bytes():
            if len(captured) + len(chunk) > self._config.max_response_bytes:
                raise ProviderInvalidResponseError(
                    "model provider response exceeded the configured size limit"
                )
            captured.extend(chunk)
        return bytes(captured)

    async def _iter_sse_data(self, response: httpx.Response) -> AsyncIterator[str]:
        total_bytes = 0
        pending = bytearray()
        async for chunk in response.aiter_bytes():
            total_bytes += len(chunk)
            if total_bytes > self._config.max_response_bytes:
                raise ProviderInvalidResponseError(
                    "model provider stream exceeded the configured size limit"
                )
            pending.extend(chunk)
            if len(pending) > self._config.max_stream_event_bytes and b"\n" not in pending:
                raise ProviderInvalidResponseError(
                    "model provider stream event exceeded the configured size limit"
                )
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    break
                raw_line = bytes(pending[:newline]).rstrip(b"\r")
                del pending[: newline + 1]
                data = self._decode_sse_line(raw_line)
                if data is not None:
                    yield data
            if len(pending) > self._config.max_stream_event_bytes:
                raise ProviderInvalidResponseError(
                    "model provider stream event exceeded the configured size limit"
                )
        if pending:
            data = self._decode_sse_line(bytes(pending).rstrip(b"\r"))
            if data is not None:
                yield data

    @staticmethod
    def _decode_sse_line(raw_line: bytes) -> str | None:
        if not raw_line.startswith(b"data:"):
            return None
        raw_data = raw_line[5:].strip()
        if not raw_data:
            return None
        try:
            return raw_data.decode("utf-8")
        except UnicodeDecodeError:
            raise ProviderInvalidResponseError(
                "model provider returned invalid streaming text"
            ) from None

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        fallback = self._config.retry_base_delay * (2**attempt)
        header = response.headers.get("Retry-After")
        if header is not None:
            try:
                requested = float(header)
            except ValueError:
                requested = fallback
        else:
            requested = fallback
        return min(max(0.0, requested), self._config.max_retry_delay)

    def _build_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": request.model or self._config.model,
            "messages": [self._message_payload(message) for message in request.messages],
            "max_tokens": request.max_tokens or self._config.max_tokens,
            "stream": stream,
        }
        if request.tools:
            payload["tools"] = [self._tool_payload(tool) for tool in request.tools]
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _message_payload(message: Message) -> dict[str, object]:
        payload: dict[str, object] = {"role": message.role, "content": message.content}
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    @staticmethod
    def _tool_payload(tool: ToolDefinition) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @classmethod
    def _parse_response(cls, payload: object) -> ModelResponse:
        root = cls._mapping(payload, "response")
        choices = root.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderInvalidResponseError("model response has no choices")
        choice = cls._mapping(choices[0], "choice")
        message = cls._mapping(choice.get("message"), "message")
        role = message.get("role")
        if role is not None and role != "assistant":
            raise ProviderInvalidResponseError(
                "model response attempted to forge a non-assistant message"
            )
        if "tool_call_id" in message:
            raise ProviderInvalidResponseError(
                "model response attempted to forge a tool result"
            )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProviderInvalidResponseError("model response content is invalid")

        tool_calls_value = message.get("tool_calls", [])
        if not isinstance(tool_calls_value, list):
            raise ProviderInvalidResponseError("model response tool calls are invalid")
        tool_calls = tuple(cls._parse_tool_call(item) for item in tool_calls_value)

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ProviderInvalidResponseError("model response finish reason is invalid")

        metadata = {
            key: value
            for key in ("id", "model")
            if isinstance((value := root.get(key)), str)
        }
        metadata.update(cls._billing_metadata(root))
        return ModelResponse(
            text=content or "",
            tool_calls=tool_calls,
            usage=cls._parse_usage(root.get("usage")),
            finish_reason=finish_reason,
            provider_metadata=metadata,
        )

    @classmethod
    def _parse_tool_call(cls, value: object) -> ToolCall:
        item = cls._mapping(value, "tool call")
        call_type = item.get("type")
        if call_type is not None and call_type != "function":
            raise ProviderInvalidResponseError("model response tool call type is invalid")
        call_id = item.get("id")
        function = cls._mapping(item.get("function"), "tool function")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, str):
            raise ProviderInvalidResponseError("model response tool call is invalid")
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            raise ProviderInvalidResponseError("tool call arguments are invalid JSON") from None
        if not isinstance(parsed_arguments, dict):
            raise ProviderInvalidResponseError("tool call arguments must be a JSON object")
        try:
            return ToolCall(id=call_id, name=name, arguments=parsed_arguments)
        except ValueError:
            raise ProviderInvalidResponseError(
                "model response tool call violates protocol limits"
            ) from None

    @classmethod
    def _parse_stream_chunk(cls, payload: object) -> ModelStreamChunk:
        root = cls._mapping(payload, "stream chunk")
        usage = cls._parse_usage(root.get("usage")) if root.get("usage") is not None else None
        metadata = cls._billing_metadata(root)
        choices = root.get("choices", [])
        if not isinstance(choices, list):
            raise ProviderInvalidResponseError("stream chunk choices are invalid")
        if not choices:
            return ModelStreamChunk(usage=usage, provider_metadata=metadata)

        choice = cls._mapping(choices[0], "stream choice")
        delta = cls._mapping(choice.get("delta", {}), "stream delta")
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise ProviderInvalidResponseError("stream text delta is invalid")

        raw_tool_calls = delta.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            raise ProviderInvalidResponseError("stream tool call deltas are invalid")
        tool_call_deltas = tuple(cls._parse_tool_call_delta(item) for item in raw_tool_calls)

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ProviderInvalidResponseError("stream finish reason is invalid")
        return ModelStreamChunk(
            text_delta=content or "",
            tool_call_deltas=tool_call_deltas,
            usage=usage,
            finish_reason=finish_reason,
            provider_metadata=metadata,
        )

    @classmethod
    def _parse_tool_call_delta(cls, value: object) -> ToolCallDelta:
        item = cls._mapping(value, "tool call delta")
        index = item.get("index")
        if not isinstance(index, int):
            raise ProviderInvalidResponseError("tool call delta index is invalid")
        call_id = item.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise ProviderInvalidResponseError("tool call delta id is invalid")
        function_value = item.get("function", {})
        function = cls._mapping(function_value, "tool call delta function")
        name = function.get("name")
        arguments = function.get("arguments", "")
        if name is not None and not isinstance(name, str):
            raise ProviderInvalidResponseError("tool call delta name is invalid")
        if not isinstance(arguments, str):
            raise ProviderInvalidResponseError("tool call argument delta is invalid")
        try:
            return ToolCallDelta(
                index=index,
                id=call_id,
                name=name,
                arguments_delta=arguments,
            )
        except ValueError:
            raise ProviderInvalidResponseError(
                "model response tool call delta violates protocol limits"
            ) from None

    @staticmethod
    def _parse_usage(value: object) -> Usage:
        if value is None:
            return Usage()
        if not isinstance(value, Mapping):
            raise ProviderInvalidResponseError("model usage metadata is invalid")
        prompt = value.get("prompt_tokens", 0)
        completion = value.get("completion_tokens", 0)
        total = value.get("total_tokens", 0)
        if not all(isinstance(item, int) and item >= 0 for item in (prompt, completion, total)):
            raise ProviderInvalidResponseError("model usage token counts are invalid")
        return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)

    @staticmethod
    def _billing_metadata(root: Mapping[str, object]) -> dict[str, object]:
        """Extract only explicit provider billing fields, without estimating."""

        values: dict[str, object] = {}
        usage = root.get("usage")
        sources = (root, usage) if isinstance(usage, Mapping) else (root,)
        for source in sources:
            for key in (
                "cost",
                "cost_usd",
                "input_cost",
                "output_cost",
                "total_cost",
                "currency",
            ):
                value = source.get(key)
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    values[key] = value
        return values

    @staticmethod
    def _mapping(value: object, label: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ProviderInvalidResponseError(f"model {label} is invalid")
        return value

    @staticmethod
    def _raise_for_status(status_code: int) -> None:
        if status_code == 401:
            raise ProviderAuthenticationError("model provider rejected authentication (HTTP 401)")
        if status_code == 429:
            raise ProviderRateLimitError("model provider rate limit exceeded (HTTP 429)")
        if 500 <= status_code <= 599:
            raise ProviderServerError(f"model provider server error (HTTP {status_code})")
        if status_code >= 400:
            raise ProviderRequestError(f"model provider rejected the request (HTTP {status_code})")
