from __future__ import annotations

import io
import json
import logging
import unittest

import httpx

from src.models import Message, ModelRequest, ToolDefinition
from src.openai_provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from src.providers import (
    ProviderAuthenticationError,
    ProviderInvalidResponseError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)

SECRET = "test-api-key-that-must-stay-private"


def provider_for(handler: httpx.MockTransport) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key=SECRET,
            base_url="https://mock.invalid/v1",
            model="mock-model",
            timeout=2,
            max_tokens=128,
            max_retries=0,
        ),
        transport=handler,
    )


class OpenAIProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_model_returns_text_and_usage(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], f"Bearer {SECRET}")
            body = json.loads(request.content)
            self.assertEqual(body["messages"][0]["content"], "hello")
            return httpx.Response(
                200,
                json={
                    "id": "response-1",
                    "model": "mock-model",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "hello back"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        provider = provider_for(httpx.MockTransport(handler))
        try:
            response = await provider.complete(ModelRequest.from_prompt("hello"))
        finally:
            await provider.aclose()

        self.assertEqual(response.text, "hello back")
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(response.provider_metadata["id"], "response-1")

    async def test_mock_model_returns_tool_call(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertEqual(body["tools"][0]["function"]["name"], "read_status")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "read_status",
                                            "arguments": '{"path":"README.md"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )

        provider = provider_for(httpx.MockTransport(handler))
        request = ModelRequest(
            messages=(Message(role="user", content="inspect status"),),
            tools=(
                ToolDefinition(
                    name="read_status",
                    description="Read a status value",
                    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                ),
            ),
        )
        try:
            response = await provider.complete(request)
        finally:
            await provider.aclose()

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_calls[0].name, "read_status")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "README.md"})

    async def test_streaming_text_usage_and_tool_call_delta(self) -> None:
        stream_body = "\n\n".join(
            (
                'data: {"choices":[{"delta":{"content":"hel"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":"lo","tool_calls":[{"index":0,"id":"call-1","function":{"name":"status","arguments":"{\\""}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"path\\":\\"README.md\\"}"}}]},"finish_reason":"tool_calls"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}',
                "data: [DONE]",
                "",
            )
        )

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertTrue(body["stream"])
            self.assertEqual(body["stream_options"], {"include_usage": True})
            return httpx.Response(
                200,
                text=stream_body,
                headers={"content-type": "text/event-stream"},
            )

        provider = provider_for(httpx.MockTransport(handler))
        try:
            chunks = [chunk async for chunk in provider.stream(ModelRequest.from_prompt("hi"))]
        finally:
            await provider.aclose()

        self.assertEqual("".join(chunk.text_delta for chunk in chunks), "hello")
        self.assertEqual(chunks[1].tool_call_deltas[0].name, "status")
        self.assertEqual(chunks[-1].usage.total_tokens, 5)

    async def test_stream_can_be_closed_by_consumer(self) -> None:
        stream = _CloseAwareStream()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                stream=stream,
                headers={"content-type": "text/event-stream"},
            )

        provider = provider_for(httpx.MockTransport(handler))
        iterator = provider.stream(ModelRequest.from_prompt("hi"))
        first = await anext(iterator)
        self.assertEqual(first.text_delta, "first")
        await iterator.aclose()
        await provider.aclose()
        self.assertTrue(stream.closed)

    async def test_domain_errors_are_clear_and_do_not_leak_api_key(self) -> None:
        cases = (
            (401, ProviderAuthenticationError),
            (429, ProviderRateLimitError),
            (500, ProviderServerError),
            (503, ProviderServerError),
        )
        for status, error_type in cases:
            with self.subTest(status=status):
                transport = httpx.MockTransport(
                    lambda request, status=status: httpx.Response(
                        status, text=f"upstream included {SECRET}"
                    )
                )
                provider = provider_for(transport)
                try:
                    with self.assertRaises(error_type) as raised:
                        await provider.complete(ModelRequest.from_prompt("hi"))
                finally:
                    await provider.aclose()
                self.assertIn(str(status), str(raised.exception))
                self.assertNotIn(SECRET, str(raised.exception))

    async def test_timeout_is_mapped_without_leaking_details(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout(f"timeout while using {SECRET}", request=request)

        provider = provider_for(httpx.MockTransport(handler))
        try:
            with self.assertRaises(ProviderTimeoutError) as raised:
                await provider.complete(ModelRequest.from_prompt("hi"))
        finally:
            await provider.aclose()
        self.assertNotIn(SECRET, str(raised.exception))

    async def test_invalid_response_has_domain_error(self) -> None:
        provider = provider_for(
            httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []}))
        )
        try:
            with self.assertRaisesRegex(ProviderInvalidResponseError, "no choices"):
                await provider.complete(ModelRequest.from_prompt("hi"))
        finally:
            await provider.aclose()

    async def test_rate_limit_is_retried_with_bounded_policy(self) -> None:
        requests = 0
        delays: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            if requests == 1:
                return httpx.Response(429, headers={"Retry-After": "0.01"})
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "recovered"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key=SECRET,
                base_url="https://mock.invalid/v1",
                model="mock-model",
                max_retries=2,
            ),
            transport=httpx.MockTransport(handler),
            sleep=lambda delay: _record_delay(delays, delay),
        )
        try:
            response = await provider.complete(ModelRequest.from_prompt("retry"))
        finally:
            await provider.aclose()

        self.assertEqual(response.text, "recovered")
        self.assertEqual(requests, 2)
        self.assertEqual(delays, [0.01])

    async def test_oversized_response_and_forged_tool_result_are_rejected(self) -> None:
        oversized = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key=SECRET,
                base_url="https://mock.invalid/v1",
                model="mock-model",
                max_response_bytes=128,
                max_stream_event_bytes=64,
                max_retries=0,
            ),
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"x" * 129)
            ),
        )
        try:
            with self.assertRaisesRegex(ProviderInvalidResponseError, "size limit"):
                await oversized.complete(ModelRequest.from_prompt("large"))
        finally:
            await oversized.aclose()

        forged = provider_for(
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "tool",
                                    "tool_call_id": "invented",
                                    "content": "success",
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )
            )
        )
        try:
            with self.assertRaisesRegex(ProviderInvalidResponseError, "forge"):
                await forged.complete(ModelRequest.from_prompt("forge"))
        finally:
            await forged.aclose()

    async def test_provider_logs_do_not_contain_api_key(self) -> None:
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        provider_logger = logging.getLogger("src.openai_provider")
        old_handlers = provider_logger.handlers
        old_level = provider_logger.level
        old_propagate = provider_logger.propagate
        provider_logger.handlers = [handler]
        provider_logger.setLevel(logging.DEBUG)
        provider_logger.propagate = False
        provider = provider_for(
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
                )
            )
        )
        try:
            await provider.complete(ModelRequest.from_prompt("hi"))
        finally:
            await provider.aclose()
            provider_logger.handlers = old_handlers
            provider_logger.setLevel(old_level)
            provider_logger.propagate = old_propagate
        self.assertNotIn(SECRET, output.getvalue())


class _CloseAwareStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n'

    async def aclose(self) -> None:
        self.closed = True


async def _record_delay(delays: list[float], delay: float) -> None:
    delays.append(delay)


if __name__ == "__main__":
    unittest.main()
