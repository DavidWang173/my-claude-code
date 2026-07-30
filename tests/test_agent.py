from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from src.agent import (
    AgentCancelledError,
    AgentEventKind,
    AgentLimitExceededError,
    AgentLimits,
    AgentLoop,
    AgentProtocolError,
    AgentRepeatedToolCallError,
    AgentRequest,
    AgentTimeoutError,
    CancellationToken,
)
from src.models import (
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from src.harness.models import RunState
from src.permissions import (
    InteractivePermissionPolicy,
    Operation,
    PermissionDecision,
    PermissionRequest,
    ReadOnlyPermissionPolicy,
)
from src.providers import ProviderRegistry
from src.shell_tools import ShellTool
from src.sessions import Session, SessionListResult, SessionNotFoundError
from src.tools import ToolContext, ToolRegistry, ToolResult


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_text_completes_in_one_turn(self) -> None:
        provider = FakeProvider([ModelResponse(text="done", usage=Usage(2, 3, 5))])
        loop, sessions = build_loop(provider)

        result = await loop.run(task("explain the repository"))

        self.assertEqual(result.text, "done")
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(result.usage.total_tokens, 5)
        request = provider.requests[0]
        self.assertEqual([message.role for message in request.messages], ["system", "user"])
        system_prompt = request.messages[0].content or ""
        self.assertIn("Inspect relevant files", system_prompt)
        self.assertIn("Keep changes narrowly scoped", system_prompt)
        self.assertIn("run appropriate verification", system_prompt)
        self.assertIn("Never fabricate", system_prompt)
        self.assertIn(str(Path.cwd().resolve()), system_prompt)
        self.assertGreaterEqual(sessions.save_count, 2)

    async def test_one_tool_call_then_text_completion(self) -> None:
        tool = FakeTool()
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=(call("call-1", path="README.md"),)),
                ModelResponse(text="tool completed"),
            ]
        )
        loop, sessions = build_loop(provider, tools=(tool,))

        result = await loop.run(task("inspect README"))

        self.assertEqual(result.text, "tool completed")
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(tool.calls, [{"path": "README.md"}])
        second_request = provider.requests[1]
        self.assertEqual(second_request.messages[-1].role, "tool")
        self.assertEqual(second_request.messages[-1].tool_call_id, "call-1")
        self.assertGreaterEqual(sessions.save_count, 4)

    async def test_multiple_tool_calls_execute_in_order(self) -> None:
        tool = FakeTool(
            outcomes=(
                ToolResult("first result"),
                ToolResult("second result"),
            )
        )
        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=(
                        call("call-1", path="one.py"),
                        call("call-2", path="two.py"),
                    )
                ),
                ModelResponse(text="both complete"),
            ]
        )
        loop, _ = build_loop(provider, tools=(tool,))

        result = await loop.run(task("inspect both"))

        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(tool.calls, [{"path": "one.py"}, {"path": "two.py"}])
        self.assertEqual(
            [message.tool_call_id for message in provider.requests[1].messages[-2:]],
            ["call-1", "call-2"],
        )

    async def test_tool_failure_becomes_result_and_model_can_correct(self) -> None:
        tool = FakeTool(
            outcomes=(
                RuntimeError("temporary failure"),
                ToolResult("recovered"),
            )
        )
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=(call("call-1", path="bad.py"),)),
                ModelResponse(tool_calls=(call("call-2", path="good.py"),)),
                ModelResponse(text="fixed after retry"),
            ]
        )
        loop, _ = build_loop(provider, tools=(tool,))

        result = await loop.run(task("recover from failure"))

        self.assertEqual(result.text, "fixed after retry")
        failed_result = provider.requests[1].messages[-1]
        self.assertEqual(failed_result.role, "tool")
        self.assertIn("ERROR: RuntimeError: tool execution failed", failed_result.content or "")
        self.assertEqual(tool.calls, [{"path": "bad.py"}, {"path": "good.py"}])

    async def test_invalid_arguments_become_tool_error_without_execution(self) -> None:
        tool = FakeTool()
        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(id="call-1", name="inspect", arguments={"unexpected": True}),
                    )
                ),
                ModelResponse(text="corrected"),
            ]
        )
        loop, _ = build_loop(provider, tools=(tool,))

        result = await loop.run(task("validate arguments"))

        self.assertEqual(result.text, "corrected")
        self.assertEqual(tool.calls, [])
        self.assertIn("ERROR:", provider.requests[1].messages[-1].content or "")
        self.assertIn("missing required field", provider.requests[1].messages[-1].content or "")

    async def test_tool_declared_write_operation_uses_write_permission(self) -> None:
        tool = WriteFakeTool()
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=(call("call-1", path="target.py"),)),
                ModelResponse(text="write was denied"),
            ]
        )
        loop, _ = build_loop(
            provider,
            tools=(tool,),
            permissions=ReadOnlyPermissionPolicy(),
        )

        result = await loop.run(task("attempt write"))

        self.assertEqual(result.text, "write was denied")
        self.assertEqual(tool.calls, [])
        self.assertIn("Permission denied", provider.requests[1].messages[-1].content or "")
        self.assertIn("write operations require approval", provider.requests[1].messages[-1].content or "")

    async def test_repeated_identical_call_is_blocked(self) -> None:
        tool = FakeTool()
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=(call("call-1", path="same.py"),)),
                ModelResponse(tool_calls=(call("call-2", path="same.py"),)),
            ]
        )
        loop, sessions = build_loop(provider, tools=(tool,))

        with self.assertRaisesRegex(AgentRepeatedToolCallError, "repeated identical"):
            await loop.run(task("do not loop"))

        self.assertEqual(tool.calls, [{"path": "same.py"}])
        last_message = next(iter(sessions.sessions.values())).messages[-1]
        self.assertEqual(last_message.tool_call_id, "call-2")
        self.assertIn("Repeated identical", last_message.content or "")

    async def test_repeated_identical_call_with_reused_id_is_blocked(self) -> None:
        tool = FakeTool()
        repeated = call("same-call", path="same.py")
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=(repeated,)),
                ModelResponse(tool_calls=(repeated,)),
            ]
        )
        loop, _ = build_loop(provider, tools=(tool,))

        with self.assertRaises(AgentRepeatedToolCallError):
            await loop.run(task("do not reuse ids"))

        self.assertEqual(tool.calls, [{"path": "same.py"}])

    async def test_same_tool_call_is_allowed_in_a_later_chat_turn(self) -> None:
        tool = FakeTool()
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=(call("call-1", path="same.py"),)),
                ModelResponse(text="first done"),
                ModelResponse(tool_calls=(call("call-2", path="same.py"),)),
                ModelResponse(text="second done"),
            ]
        )
        loop, _ = build_loop(provider, tools=(tool,))

        first = await loop.run(task("inspect once"))
        second_request = task("inspect again")
        second_request = AgentRequest(
            prompt=second_request.prompt,
            provider=second_request.provider,
            model=second_request.model,
            workspace=second_request.workspace,
            session_id=first.session_id,
        )
        second = await loop.run(second_request)

        self.assertEqual(second.text, "second done")
        self.assertEqual(tool.calls, [{"path": "same.py"}, {"path": "same.py"}])

    async def test_max_turns_stops_the_loop(self) -> None:
        tool = FakeTool()
        provider = FakeProvider([ModelResponse(tool_calls=(call("call-1", path="one.py"),))])
        loop, sessions = build_loop(
            provider,
            tools=(tool,),
            limits=AgentLimits(max_turns=1, max_tool_calls=5, timeout_seconds=2),
        )

        with self.assertRaisesRegex(AgentLimitExceededError, "model turns"):
            await loop.run(task("bounded run"))
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(next(iter(sessions.sessions.values())).run_state, RunState.FAILED)

    async def test_max_tool_calls_stops_before_execution(self) -> None:
        tool = FakeTool()
        provider = FakeProvider([ModelResponse(tool_calls=(call("call-1", path="one.py"),))])
        loop, sessions = build_loop(
            provider,
            tools=(tool,),
            limits=AgentLimits(max_turns=2, max_tool_calls=0, timeout_seconds=2),
        )

        with self.assertRaisesRegex(AgentLimitExceededError, "tool calls"):
            await loop.run(task("bounded tools"))
        self.assertEqual(tool.calls, [])
        self.assertEqual(next(iter(sessions.sessions.values())).run_state, RunState.FAILED)

    async def test_cancellation_interrupts_in_flight_provider(self) -> None:
        provider = BlockingProvider()
        loop, sessions = build_loop(provider)
        cancellation = CancellationToken()
        running = asyncio.create_task(loop.run(task("cancel me"), cancellation=cancellation))
        await provider.started.wait()

        cancellation.cancel()

        with self.assertRaises(AgentCancelledError):
            await running
        self.assertTrue(provider.cancelled)
        self.assertEqual(
            next(iter(sessions.sessions.values())).run_state,
            RunState.CANCELLED,
        )

    async def test_total_timeout_interrupts_in_flight_provider(self) -> None:
        provider = BlockingProvider()
        loop, sessions = build_loop(
            provider,
            limits=AgentLimits(max_turns=2, max_tool_calls=2, timeout_seconds=0.02),
        )

        with self.assertRaises(AgentTimeoutError):
            await loop.run(task("time out"))
        self.assertTrue(provider.cancelled)
        self.assertEqual(next(iter(sessions.sessions.values())).run_state, RunState.FAILED)

    async def test_model_and_tool_outputs_are_bounded_before_persistence(self) -> None:
        oversized_provider = FakeProvider([ModelResponse(text="x" * 11)])
        oversized_loop, _ = build_loop(
            oversized_provider,
            limits=AgentLimits(max_model_output_chars=10),
        )
        with self.assertRaisesRegex(AgentProtocolError, "output limit"):
            await oversized_loop.run(task("bounded model output"))

        tool = FakeTool(outcomes=(ToolResult("y" * 20),))
        provider = FakeProvider(
            [
                ModelResponse(tool_calls=(call("large-tool", path="file.py"),)),
                ModelResponse(text="done"),
            ]
        )
        loop, _ = build_loop(
            provider,
            tools=(tool,),
            limits=AgentLimits(max_tool_result_chars=10),
        )
        await loop.run(task("bounded tool output"))
        persisted_result = provider.requests[1].messages[-1].content or ""
        self.assertIn("tool result truncated", persisted_result)
        self.assertLess(len(persisted_result), 50)

    async def test_streaming_boundary_yields_deltas_and_completion(self) -> None:
        provider = FakeProvider(
            [],
            streams=(
                (
                    ModelStreamChunk(text_delta="hel"),
                    ModelStreamChunk(
                        text_delta="lo",
                        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                        finish_reason="stop",
                    ),
                ),
            ),
        )
        loop, _ = build_loop(provider)

        events = [event async for event in loop.run_stream(task("stream"))]

        self.assertEqual(
            [event.text for event in events if event.kind is AgentEventKind.TEXT_DELTA],
            ["hel", "lo"],
        )
        completed = events[-1].result
        self.assertIsNotNone(completed)
        self.assertEqual(completed.text, "hello")
        self.assertEqual(completed.usage.total_tokens, 2)

    async def test_shell_output_streams_through_agent_with_exact_approval(self) -> None:
        arguments = json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr)",
                ]
            }
        )
        provider = FakeProvider(
            [],
            streams=(
                (
                    ModelStreamChunk(
                        tool_call_deltas=(
                            ToolCallDelta(
                                index=0,
                                id="shell-1",
                                name="run_shell",
                                arguments_delta=arguments,
                            ),
                        ),
                        finish_reason="tool_calls",
                    ),
                ),
                (ModelStreamChunk(text_delta="shell complete", finish_reason="stop"),),
            ),
        )
        approvals: list[PermissionRequest] = []
        permissions = InteractivePermissionPolicy(
            lambda request: approvals.append(request) or True
        )
        loop, _ = build_loop(
            provider,
            tools=(ShellTool(),),  # type: ignore[arg-type]
            permissions=permissions,
        )

        events = [event async for event in loop.run_stream(task("run command"))]

        outputs = [event for event in events if event.kind is AgentEventKind.TOOL_OUTPUT]
        self.assertEqual({event.stream for event in outputs}, {"stdout", "stderr"})
        self.assertTrue(any("out" in event.text for event in outputs))
        self.assertTrue(any("err" in event.text for event in outputs))
        self.assertEqual(events[-1].result.text, "shell complete")
        self.assertEqual(len(approvals), 1)
        self.assertEqual(approvals[0].cwd, str(Path.cwd().resolve()))
        self.assertIn(sys.executable, approvals[0].command or "")
        self.assertTrue(approvals[0].risk_reason)

    async def test_streamed_tool_call_executes_before_next_streamed_turn(self) -> None:
        provider = FakeProvider(
            [],
            streams=(
                (
                    ModelStreamChunk(
                        tool_call_deltas=(
                            ToolCallDelta(
                                index=0,
                                id="call-1",
                                name="inspect",
                                arguments_delta='{"path":',
                            ),
                        )
                    ),
                    ModelStreamChunk(
                        tool_call_deltas=(
                            ToolCallDelta(index=0, arguments_delta='"README.md"}'),
                        ),
                        finish_reason="tool_calls",
                    ),
                ),
                (ModelStreamChunk(text_delta="done", finish_reason="stop"),),
            ),
        )
        tool = FakeTool()
        loop, _ = build_loop(provider, tools=(tool,))

        events = [event async for event in loop.run_stream(task("stream a tool"))]

        self.assertEqual(tool.calls, [{"path": "README.md"}])
        self.assertEqual(events[-1].result.text, "done")


class AgentSyncBoundaryTests(unittest.TestCase):
    def test_run_sync_uses_async_runtime(self) -> None:
        provider = FakeProvider([ModelResponse(text="sync done")])
        loop, _ = build_loop(provider)
        result = loop.run_sync(task("sync"))
        self.assertEqual(result.text, "sync done")


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        responses: list[ModelResponse],
        *,
        streams: tuple[tuple[ModelStreamChunk, ...], ...] = (),
    ) -> None:
        self._responses = list(responses)
        self._streams = list(streams)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("fake provider has no response")
        return self._responses.pop(0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        self.requests.append(request)
        if not self._streams:
            raise AssertionError("fake provider has no stream")
        for chunk in self._streams.pop(0):
            await asyncio.sleep(0)
            yield chunk

    async def aclose(self) -> None:
        return None


class BlockingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.cancelled = False

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class FakeTool:
    name = "inspect"
    description = "Inspect one workspace-relative path"
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {"path": {"type": "string", "minLength": 1}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, outcomes: tuple[ToolResult | Exception, ...] = ()) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        del context
        self.calls.append(dict(arguments))
        if self._outcomes:
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return ToolResult(content=f"inspected {arguments['path']}")


class WriteFakeTool(FakeTool):
    operation = Operation.WRITE


class AllowAllPermissions:
    def decide(self, request: PermissionRequest) -> PermissionDecision:
        del request
        return PermissionDecision(True, "allowed by test")


class MemorySessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.save_count = 0

    def create(self, *, workspace: Path, provider: str, model: str) -> Session:
        session = Session.create(workspace=workspace, provider=provider, model=model)
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        self.sessions[session.id] = session
        self.save_count += 1

    def load(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(f"session not found: {session_id}") from None

    def load_latest(self, *, workspace: Path | None = None) -> Session:
        candidates = [
            session
            for session in self.sessions.values()
            if workspace is None or session.workspace == workspace.resolve()
        ]
        if not candidates:
            raise SessionNotFoundError("no sessions")
        return max(candidates, key=lambda item: item.updated_at)

    def list_sessions(self) -> SessionListResult:
        return SessionListResult(())

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


def call(call_id: str, *, path: str) -> ToolCall:
    return ToolCall(id=call_id, name="inspect", arguments={"path": path})


def task(prompt: str) -> AgentRequest:
    return AgentRequest(
        prompt=prompt,
        provider="fake",
        model="fake-model",
        workspace=Path.cwd(),
    )


def build_loop(
    provider: FakeProvider,
    *,
    tools: tuple[FakeTool, ...] = (),
    limits: AgentLimits | None = None,
    permissions: (
        AllowAllPermissions | ReadOnlyPermissionPolicy | InteractivePermissionPolicy | None
    ) = None,
) -> tuple[AgentLoop, MemorySessionStore]:
    sessions = MemorySessionStore()
    return (
        AgentLoop(
            ProviderRegistry((provider,)),
            ToolRegistry(tools),
            sessions,
            permissions or AllowAllPermissions(),
            limits=limits,
        ),
        sessions,
    )


if __name__ == "__main__":
    unittest.main()
