from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from src.agent import (
    AgentLimits,
    AgentLoop,
    AgentRequest,
    AgentVerificationError,
)
from src.harness.models import RunState
from src.models import ModelRequest, ModelResponse, ModelStreamChunk, ToolCall
from src.permissions import (
    InteractivePermissionPolicy,
    Operation,
    PermissionDecision,
    PermissionRequest,
)
from src.providers import ProviderRegistry
from src.sessions import Session, SessionListResult, SessionNotFoundError
from src.tools import ToolContext, ToolRegistry, ToolResult


class Provider:
    name = "fake"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("no response")
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        response = await self.complete(request)
        yield ModelStreamChunk(text_delta=response.text)

    async def aclose(self) -> None:
        return None


class Store:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.state_history: list[RunState | None] = []

    def create(self, *, workspace: Path, provider: str, model: str) -> Session:
        session = Session.create(workspace=workspace, provider=provider, model=model)
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        self.sessions[session.id] = session
        self.state_history.append(session.run_state)

    def load(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(session_id) from None

    def load_latest(self, *, workspace: Path | None = None) -> Session:
        del workspace
        return next(iter(self.sessions.values()))

    def list_sessions(self) -> SessionListResult:
        return SessionListResult(())

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id)


class CreateCodeTool:
    name = "create_file"
    description = "create code"
    operation = Operation.WRITE
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def permission_request(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> PermissionRequest:
        return PermissionRequest(
            operation=Operation.WRITE,
            target=str(arguments["path"]),
        )

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        path = str(arguments["path"])
        (context.working_directory / path).write_text("value = 1\n", encoding="utf-8")
        return ToolResult("created", metadata={"files": [path]})


class TestTool:
    name = "run_shell"
    description = "record a test"
    operation = Operation.EXECUTE
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        del arguments, context
        return ToolResult(
            "OK",
            metadata={
                "command": "python -m unittest",
                "exit_code": 0,
            },
        )


class HarnessIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_verification_repairs_then_reverifies_and_completes(self) -> None:
        provider = Provider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall("write-1", "create_file", {"path": "module.py"}),
                    )
                ),
                ModelResponse(text="candidate without tests"),
                ModelResponse(
                    tool_calls=(ToolCall("test-1", "run_shell", {}),)
                ),
                ModelResponse(text="implemented and tested"),
            ]
        )
        store = Store()
        with tempfile.TemporaryDirectory() as directory:
            result = await self._loop(provider, store).run(
                AgentRequest(
                    prompt="Create module.py and verify it with tests",
                    provider="fake",
                    model="fake",
                    workspace=Path(directory),
                )
            )

        self.assertEqual(result.text, "implemented and tested")
        self.assertTrue(result.verification and result.verification.passed)
        session = next(iter(store.sessions.values()))
        self.assertEqual(session.run_state, RunState.COMPLETED)
        self.assertEqual(session.repair_attempts, 1)
        self.assertEqual(len(provider.requests), 4)
        self.assertIn(RunState.VERIFYING, store.state_history)
        self.assertIn(RunState.REPAIRING, store.state_history)

    async def test_no_tool_candidate_cannot_bypass_modification_gate(self) -> None:
        provider = Provider([ModelResponse(text="claimed done")])
        store = Store()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AgentVerificationError):
                await self._loop(
                    provider,
                    store,
                    limits=AgentLimits(max_repair_attempts=0),
                ).run(
                    AgentRequest(
                        prompt="Create module.py and run tests",
                        provider="fake",
                        model="fake",
                        workspace=Path(directory),
                    )
                )
        self.assertEqual(next(iter(store.sessions.values())).run_state, RunState.FAILED)

    async def test_permission_callback_observes_waiting_approval(self) -> None:
        provider = Provider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall("write-1", "create_file", {"path": "module.py"}),
                    )
                ),
                ModelResponse(text="operation refused"),
            ]
        )
        store = Store()
        observed: list[RunState | None] = []

        def deny(request: PermissionRequest) -> PermissionDecision:
            del request
            observed.append(next(iter(store.sessions.values())).run_state)
            return PermissionDecision(False, "denied")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AgentVerificationError):
                await self._loop(
                    provider,
                    store,
                    permissions=InteractivePermissionPolicy(deny),
                    limits=AgentLimits(max_repair_attempts=0),
                ).run(
                    AgentRequest(
                        prompt="Create module.py",
                        provider="fake",
                        model="fake",
                        workspace=Path(directory),
                    )
                )

        self.assertEqual(observed, [RunState.WAITING_APPROVAL])
        self.assertEqual(next(iter(store.sessions.values())).run_state, RunState.FAILED)

    def _loop(
        self,
        provider: Provider,
        store: Store,
        *,
        permissions: object | None = None,
        limits: AgentLimits | None = None,
    ) -> AgentLoop:
        return AgentLoop(
            ProviderRegistry((provider,)),
            ToolRegistry((CreateCodeTool(), TestTool())),
            store,
            permissions or InteractivePermissionPolicy(lambda request: True),  # type: ignore[arg-type]
            limits=limits,
        )


if __name__ == "__main__":
    unittest.main()
