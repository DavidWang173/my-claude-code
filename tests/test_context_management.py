from __future__ import annotations

import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from src.agent import AgentLoop, AgentRequest
from src.context import ContextManager, ContextState, estimate_request_tokens
from src.models import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from src.permissions import PermissionDecision, PermissionRequest
from src.providers import ProviderRegistry
from src.sessions import Session, SessionListResult, SessionNotFoundError
from src.tools import ToolRegistry, ToolResult


class WorkspaceInitialContextTests(unittest.TestCase):
    def test_small_workspace_contains_names_and_explicit_instructions_not_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "README.md").write_text(
                "Follow the project review checklist.", encoding="utf-8"
            )
            (workspace / "AGENTS.md").write_text(
                "Run tests before reporting completion.", encoding="utf-8"
            )
            (workspace / "src").mkdir()
            (workspace / "src" / "secret.py").write_text(
                "SOURCE_BODY_MUST_BE_READ_WITH_A_TOOL = True", encoding="utf-8"
            )

            prompt = ContextManager().initial_system_prompt(workspace, "Base rules")

        self.assertIn("src/secret.py", prompt)
        self.assertIn("Follow the project review checklist.", prompt)
        self.assertIn("Run tests before reporting completion.", prompt)
        self.assertNotIn("SOURCE_BODY_MUST_BE_READ_WITH_A_TOOL", prompt)

    def test_large_workspace_directory_is_bounded_and_explicitly_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for index in range(300):
                (workspace / f"module_{index:03}.py").write_text("", encoding="utf-8")

            prompt = ContextManager(
                directory_entries=20,
                directory_chars=1_000,
                instruction_chars=1_000,
            ).initial_system_prompt(workspace, "Base rules")

        self.assertIn("additional entries omitted", prompt)
        self.assertLess(len(prompt), 2_000)
        self.assertIn("module_000.py", prompt)


class ContextCompressionTests(unittest.TestCase):
    def test_compression_keeps_constraints_and_tool_groups_intact(self) -> None:
        messages = [SystemMessage("stable system prompt")]
        for index in range(12):
            messages.extend(
                (
                    UserMessage(f"historical goal {index} " + "x" * 120),
                    AssistantMessage(f"historical decision {index} " + "y" * 160),
                )
            )
        patch_call = ToolCall(
            "patch-old",
            "apply_patch",
            {
                "patch": (
                    "--- a/src/legacy.py\n"
                    "+++ b/src/legacy.py\n"
                    "@@ -1 +1 @@\n-old\n+new"
                )
            },
        )
        messages.append(AssistantMessage(tool_calls=(patch_call,)))
        messages.append(ToolMessage("diff applied", tool_call_id=patch_call.id))
        messages.append(AssistantMessage("Kept the compatibility layer."))
        failed_call = ToolCall("failed-old", "read_file", {"path": "missing.py"})
        messages.append(AssistantMessage(tool_calls=(failed_call,)))
        messages.append(ToolMessage("ERROR: file was not found", tool_call_id=failed_call.id))
        messages.append(AssistantMessage("Will inspect the directory before retrying."))
        constraint = "MUST preserve the public API and do not edit generated files."
        messages.append(UserMessage(constraint))
        completed_call = ToolCall("read-1", "read_file", {"path": "src/a.py"})
        messages.append(AssistantMessage(tool_calls=(completed_call,)))
        messages.append(
            ToolMessage("bounded file result " + "z" * 400, tool_call_id=completed_call.id)
        )
        pending_call = ToolCall("pending-1", "read_file", {"path": "src/b.py"})
        messages.append(AssistantMessage(tool_calls=(pending_call,)))
        state = ContextState()
        manager = ContextManager(max_tokens=650, recent_units=1)

        with self.assertLogs("src.context", level="INFO") as captured:
            selection = manager.select(messages, (), state, session_id="a" * 32)

        self.assertTrue(selection.compressed)
        self.assertGreater(state.compression_count, 0)
        self.assertIn(constraint, [message.content for message in selection.messages])
        retained_calls = {
            call.id
            for message in selection.messages
            if message.role == "assistant"
            for call in message.tool_calls
        }
        for message in selection.messages:
            if message.role == "tool":
                self.assertIn(message.tool_call_id, retained_calls)
        self.assertIn("pending-1", retained_calls)
        summary = next(
            message.content or ""
            for message in selection.messages
            if message.role == "system"
            and (message.content or "").startswith("Structured context summary")
        )
        for field in (
            '"goals"',
            '"modified_files"',
            '"key_decisions"',
            '"failed_attempts"',
            '"remaining_tasks"',
        ):
            self.assertIn(field, summary)
        self.assertIn("src/legacy.py", summary)
        self.assertIn("read_file failed", summary)
        self.assertNotIn(constraint, "\n".join(captured.output))

    def test_read_ranges_merge_and_successful_modification_invalidates_version(self) -> None:
        manager = ContextManager()
        state = ContextState()
        read_call = ToolCall("read-1", "read_file", {"path": "src/a.py"})

        manager.observe_tool_result(
            state,
            read_call,
            ToolResult(
                "lines",
                metadata={
                    "path": "src/a.py",
                    "start_line": 10,
                    "end_line": 20,
                    "version": "version-one",
                },
            ),
        )
        manager.observe_tool_result(
            state,
            read_call,
            ToolResult(
                "more lines",
                metadata={
                    "path": "src/a.py",
                    "start_line": 18,
                    "end_line": 30,
                    "version": "version-one",
                },
            ),
        )

        tracked = state.read_files["src/a.py"]
        self.assertEqual([(item.start, item.end) for item in tracked.ranges], [(10, 30)])
        self.assertEqual(tracked.version, "version-one")

        manager.observe_tool_result(
            state,
            read_call,
            ToolResult(
                "changed file",
                metadata={
                    "path": "src/a.py",
                    "start_line": 1,
                    "end_line": 3,
                    "version": "version-two",
                },
            ),
        )
        tracked = state.read_files["src/a.py"]
        self.assertEqual([(item.start, item.end) for item in tracked.ranges], [(1, 3)])
        self.assertEqual(tracked.version, "version-two")

        manager.observe_tool_result(
            state,
            ToolCall("patch-1", "apply_patch", {"patch": "irrelevant"}),
            ToolResult("diff", metadata={"files": ["src/a.py"], "dry_run": False}),
        )
        self.assertNotIn("src/a.py", state.read_files)
        self.assertIn("read-1", state.stale_read_calls)

        selection = manager.select(
            (
                SystemMessage("system"),
                AssistantMessage(tool_calls=(read_call,)),
                ToolMessage("outdated source contents", tool_call_id=read_call.id),
            ),
            (),
            state,
            session_id="b" * 32,
        )
        tool_result = next(
            message for message in selection.messages if message.role == "tool"
        )
        self.assertIn("STALE FILE READ", tool_result.content or "")
        self.assertNotIn("outdated source contents", tool_result.content or "")


class AgentContextIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_session_compresses_before_provider_request(self) -> None:
        workspace = Path.cwd().resolve()
        session = Session.create(workspace=workspace, provider="fake", model="fake-model")
        session.add_message(SystemMessage("stable system prompt"))
        for index in range(30):
            session.add_message(UserMessage(f"old task {index} " + "x" * 150))
            session.add_message(AssistantMessage(f"old answer {index} " + "y" * 150))
        store = MemorySessionStore(session)
        provider = CapturingProvider(ModelResponse(text="done", usage=Usage(9, 2, 11)))
        loop = AgentLoop(
            ProviderRegistry((provider,)),
            ToolRegistry(),
            store,
            AllowPermissions(),
            context=ContextManager(max_tokens=700, recent_units=2),
        )

        result = await loop.run(
            AgentRequest(
                prompt="Keep this exact active constraint.",
                provider="fake",
                model="fake-model",
                workspace=workspace,
                session_id=session.id,
            )
        )

        self.assertEqual(result.usage, Usage(9, 2, 11))
        self.assertEqual(session.context.last_prompt_tokens, 9)
        self.assertFalse(session.context.last_usage_estimated)
        self.assertGreater(session.context.compression_count, 0)
        request = provider.requests[0]
        self.assertLess(len(request.messages), len(session.messages))
        self.assertLessEqual(estimate_request_tokens(request.messages), 700)
        self.assertIn(
            "Keep this exact active constraint.",
            [message.content for message in request.messages],
        )

    async def test_usage_is_estimated_only_when_provider_omits_it(self) -> None:
        workspace = Path.cwd().resolve()
        session = Session.create(workspace=workspace, provider="fake", model="fake-model")
        store = MemorySessionStore(session)
        provider = CapturingProvider(ModelResponse(text="answer without usage"))
        loop = AgentLoop(
            ProviderRegistry((provider,)),
            ToolRegistry(),
            store,
            AllowPermissions(),
            context=ContextManager(max_tokens=10_000),
        )

        result = await loop.run(
            AgentRequest(
                prompt="estimate this request",
                provider="fake",
                model="fake-model",
                workspace=workspace,
                session_id=session.id,
            )
        )

        self.assertGreater(result.usage.prompt_tokens, 0)
        self.assertGreater(result.usage.completion_tokens, 0)
        self.assertEqual(
            result.usage.total_tokens,
            result.usage.prompt_tokens + result.usage.completion_tokens,
        )
        self.assertTrue(session.context.last_usage_estimated)


class CapturingProvider:
    name = "fake"

    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        raise AssertionError("non-streaming test")
        yield ModelStreamChunk()

    async def aclose(self) -> None:
        return None


class MemorySessionStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, *, workspace: Path, provider: str, model: str) -> Session:
        self.session = Session.create(workspace=workspace, provider=provider, model=model)
        return self.session

    def save(self, session: Session) -> None:
        self.session = session

    def load(self, session_id: str) -> Session:
        if self.session.id != session_id:
            raise SessionNotFoundError(session_id)
        return self.session

    def load_latest(self, *, workspace: Path | None = None) -> Session:
        return self.session

    def list_sessions(self) -> SessionListResult:
        return SessionListResult(())

    def delete(self, session_id: str) -> None:
        if self.session.id != session_id:
            raise SessionNotFoundError(session_id)


class AllowPermissions:
    def decide(self, request: PermissionRequest) -> PermissionDecision:
        return PermissionDecision(True, "allowed by test")


if __name__ == "__main__":
    unittest.main()
