from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from src.agent import AgentLoop, AgentRequest
from src.cli import CliDependencies, main
from src.harness.events import (
    EventType,
    JsonlEventStore,
    RunEvent,
    RunTracer,
)
from src.harness.models import (
    FailureCategory,
    RunState,
    TaskType,
    VerificationCheck,
    VerificationResult,
    VerificationStatus,
)
from src.harness.orchestrator import RunOrchestrator
from src.models import ModelRequest, ModelResponse, ModelStreamChunk, ToolCall, Usage
from src.permissions import InteractivePermissionPolicy, Operation
from src.providers import ProviderRegistry
from src.sessions import Session, SessionListResult, SessionNotFoundError
from src.tools import ToolRegistry, ToolResult


class MemorySessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}

    def create(self, *, workspace: Path, provider: str, model: str) -> Session:
        session = Session.create(workspace=workspace, provider=provider, model=model)
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        self.sessions[session.id] = session

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


class TextProvider:
    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        return ModelResponse(
            text=self.text,
            usage=Usage(11, 7, 18),
            provider_metadata={"cost_usd": 0.012},
        )

    async def stream(
        self, request: ModelRequest
    ) -> AsyncIterator[ModelStreamChunk]:
        del request
        if False:
            yield ModelStreamChunk()

    async def aclose(self) -> None:
        return None


def failed_verification() -> VerificationResult:
    return VerificationResult(
        passed=False,
        status=VerificationStatus.FAILED,
        checks=(
            VerificationCheck(
                name="tests",
                required=True,
                executed=True,
                passed=False,
                failure_reason="failed",
            ),
        ),
        evidence=(),
        failure_category=FailureCategory.TEST_FAILURE,
        repairable=True,
        summary="tests failed",
    )


class EventStoreTests(unittest.TestCase):
    def test_parent_child_relationship_and_durations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonlEventStore(Path(directory))
            tracer = RunTracer(
                store,
                run_id="run-parent",
                session_id="session-parent",
                run_state=RunState.EXECUTING,
                provider="fake",
                model="model",
            )
            started = tracer.start(
                "model:1",
                EventType.MODEL_CALL_STARTED,
                input_summary={"message_count": 2},
            )
            completed = tracer.finish(
                "model:1",
                EventType.MODEL_CALL_COMPLETED,
                output_summary={"text_chars": 20},
                success=True,
            )

            events = store.read("run-parent")

        self.assertEqual(completed.parent_event_id, started.event_id)
        self.assertIsNotNone(completed.duration_ms)
        self.assertGreaterEqual(completed.duration_ms or -1, 0)
        event_ids = {event.event_id for event in events}
        self.assertTrue(
            all(
                event.parent_event_id is None or event.parent_event_id in event_ids
                for event in events
            )
        )

    def test_corrupt_jsonl_line_is_skipped_and_future_appends_survive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = JsonlEventStore(root)
            tracer = RunTracer(
                store,
                run_id="run-corrupt",
                session_id="session-corrupt",
                run_state=RunState.EXECUTING,
            )
            tracer.emit(EventType.RUN_STATE_CHANGED, success=True)
            with (root / "run-corrupt.jsonl").open("ab") as stream:
                stream.write(b'{"truncated":')
            tracer.emit(EventType.MODEL_CALL_STARTED)

            with self.assertLogs("src.harness.events", level="WARNING"):
                events = store.read("run-corrupt")

        self.assertEqual(
            [event.event_type for event in events],
            [
                EventType.RUN_STARTED,
                EventType.RUN_STATE_CHANGED,
                EventType.MODEL_CALL_STARTED,
            ],
        )

    def test_sensitive_values_paths_and_private_reasoning_are_not_stored(self) -> None:
        known_secret = "known-secret-value"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = JsonlEventStore(root, secrets=(known_secret,))
            event = RunEvent(
                event_id="event-redact",
                timestamp=datetime.now(UTC),
                run_id="run-redact",
                session_id="session-redact",
                parent_event_id=None,
                event_type=EventType.RUN_STARTED,
                run_state=RunState.PREPARED,
                input_summary={
                    "api_key": known_secret,
                    "cookie": "session=private-cookie",
                    "path": Path("/Users/alice/private/project/file.py"),
                    "note": "Bearer abcdefghijklmnopqrstuvwxyz",
                },
                metadata={
                    "reasoning_content": "private chain of thought",
                    "safe": True,
                },
            )
            self.assertTrue(store.append(event))
            raw = (root / "run-redact.jsonl").read_text(encoding="utf-8")
            restored = store.read("run-redact")[0]

        self.assertNotIn(known_secret, raw)
        self.assertNotIn("private-cookie", raw)
        self.assertNotIn("/Users/alice", raw)
        self.assertNotIn("private chain of thought", raw)
        self.assertNotIn("reasoning_content", restored.metadata)
        self.assertEqual(restored.input_summary["api_key"], "[REDACTED]")
        self.assertEqual(restored.input_summary["path"], "[REDACTED_PATH]")

    def test_trace_write_failure_warns_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / "blocked"
            blocked.write_text("not a directory", encoding="utf-8")
            store = JsonlEventStore(blocked)
            with self.assertLogs("src.harness.events", level="WARNING"):
                tracer = RunTracer(
                    store,
                    run_id="run-warning",
                    session_id="session-warning",
                    run_state=RunState.PREPARED,
                )
                tracer.emit(EventType.RUN_STATE_CHANGED)

    def test_trace_cli_queries_one_run_as_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            store = JsonlEventStore(root / "traces")
            tracer = RunTracer(
                store,
                run_id="run-query",
                session_id="session-query",
                run_state=RunState.EXECUTING,
            )
            tracer.emit(
                EventType.TOOL_CALL_FAILED,
                tool_call_id="call-query",
                success=False,
                error_category="tool_execution_error",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = main(
                ["trace", "run-query", "--failed", "--json"],
                dependencies=CliDependencies(
                    stdin=io.StringIO(),
                    stdout=stdout,
                    stderr=stderr,
                    environ={"CODING_AGENT_SESSIONS_DIR": str(sessions)},
                ),
            )

        self.assertEqual(exit_code, 0)
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["data"]["event_type"], EventType.TOOL_CALL_FAILED.value
        )
        self.assertEqual(records[0]["data"]["tool_call_id"], "call-query")
        self.assertEqual(stderr.getvalue(), "")


class OrchestratorTraceTests(unittest.TestCase):
    def test_tool_failure_trace_is_parented_to_tool_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_store = JsonlEventStore(Path(directory) / "traces")
            sessions = MemorySessionStore()
            session = sessions.create(
                workspace=Path(directory),
                provider="fake",
                model="model",
            )
            run = RunOrchestrator(
                session,
                sessions,
                task_type=TaskType.EXECUTION,
                event_store=event_store,
            )
            run.transition_to(RunState.EXECUTING)
            call = ToolCall("tool-failure", "run_shell", {"argv": ["false"]})
            run.request_tool_call(call)
            run.begin_tool_call(call, operation=Operation.EXECUTE)
            run.record_tool_result(
                call,
                ToolResult("command failed", is_error=True),
                operation=Operation.EXECUTE,
            )

            events = event_store.read(run.run_id)

        started = next(
            event for event in events if event.event_type is EventType.TOOL_CALL_STARTED
        )
        failed = next(
            event for event in events if event.event_type is EventType.TOOL_CALL_FAILED
        )
        self.assertEqual(failed.parent_event_id, started.event_id)
        self.assertFalse(failed.success)
        self.assertEqual(failed.error_category, "tool_execution_error")

    def test_verification_failure_has_timing_and_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_store = JsonlEventStore(Path(directory) / "traces")
            sessions = MemorySessionStore()
            session = sessions.create(
                workspace=Path(directory),
                provider="fake",
                model="model",
            )
            run = RunOrchestrator(
                session,
                sessions,
                task_type=TaskType.INFORMATIONAL,
                event_store=event_store,
            )
            run.transition_to(RunState.EXECUTING)
            run.transition_to(RunState.VERIFYING)
            run.record_verification(failed_verification())

            events = event_store.read(run.run_id)

        started = next(
            event for event in events if event.event_type is EventType.VERIFICATION_STARTED
        )
        completed = next(
            event
            for event in events
            if event.event_type is EventType.VERIFICATION_COMPLETED
        )
        self.assertEqual(completed.parent_event_id, started.event_id)
        self.assertEqual(completed.error_category, FailureCategory.TEST_FAILURE.value)
        self.assertFalse(completed.success)
        self.assertIsNotNone(completed.duration_ms)

    def test_session_resume_keeps_run_id_and_one_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_store = JsonlEventStore(Path(directory) / "traces")
            sessions = MemorySessionStore()
            session = sessions.create(
                workspace=Path(directory),
                provider="fake",
                model="model",
            )
            first = RunOrchestrator(
                session,
                sessions,
                task_type=TaskType.INFORMATIONAL,
                event_store=event_store,
            )
            first.transition_to(RunState.EXECUTING)
            original_run_id = first.run_id

            resumed = RunOrchestrator(
                session,
                sessions,
                task_type=TaskType.INFORMATIONAL,
                event_store=event_store,
            )
            events = event_store.read(original_run_id)

        self.assertEqual(resumed.run_id, original_run_id)
        self.assertEqual(
            sum(event.event_type is EventType.RUN_STARTED for event in events),
            1,
        )


class AgentTraceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_text_and_private_reasoning_are_not_traced(self) -> None:
        private_text = "PRIVATE_CHAIN_OF_THOUGHT_DO_NOT_STORE"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_store = JsonlEventStore(root / "traces")
            sessions = MemorySessionStore()
            loop = AgentLoop(
                ProviderRegistry((TextProvider(private_text),)),
                ToolRegistry(),
                sessions,
                InteractivePermissionPolicy(lambda request: True),
                event_store=event_store,
            )
            result = await loop.run(
                AgentRequest(
                    prompt="explain this",
                    provider="fake",
                    model="model",
                    workspace=root,
                )
            )
            session = sessions.load(result.session_id)
            raw = (event_store.root / f"{session.run_id}.jsonl").read_text(
                encoding="utf-8"
            )
            events = event_store.read(session.run_id or "")

        self.assertNotIn(private_text, raw)
        model_event = next(
            event
            for event in events
            if event.event_type is EventType.MODEL_CALL_COMPLETED
        )
        self.assertEqual(model_event.output_summary["text_chars"], len(private_text))
        self.assertEqual(model_event.metadata["token_usage"]["total_tokens"], 18)
        self.assertEqual(model_event.metadata["cost"]["cost_usd"], 0.012)


if __name__ == "__main__":
    unittest.main()
