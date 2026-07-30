from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.context import ContextManager
from src.models import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from src.sessions import (
    SCHEMA_VERSION,
    JsonSessionStore,
    Session,
    SessionCorruptedError,
    SessionNotFoundError,
    SessionStorageError,
    default_session_directory,
    session_from_dict,
    session_to_dict,
)
from src.tools import ToolResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SessionModelTests(unittest.TestCase):
    def test_messages_and_tool_relationship_round_trip_without_loss(self) -> None:
        call = ToolCall(
            id="call-1",
            name="read_status",
            arguments={"path": "README.md", "options": {"limit": 10}},
        )
        session = Session.create(
            workspace=PROJECT_ROOT,
            provider="mock-provider",
            model="mock-model",
        )
        session.add_message(SystemMessage("You are a coding agent."))
        session.add_message(UserMessage("Inspect the repository."))
        session.add_message(AssistantMessage(tool_calls=(call,)))
        session.add_message(ToolMessage("clean", tool_call_id=call.id))
        session.add_message(AssistantMessage("The repository is clean."))
        session.add_usage(Usage(prompt_tokens=12, completion_tokens=7, total_tokens=19))
        ContextManager().observe_tool_result(
            session.context,
            ToolCall(id="read-2", name="read_file", arguments={"path": "README.md"}),
            ToolResult(
                "lines",
                metadata={
                    "path": "README.md",
                    "start_line": 1,
                    "end_line": 10,
                    "version": "abc123",
                },
            ),
        )

        encoded = json.dumps(session_to_dict(session), ensure_ascii=False)
        restored = session_from_dict(json.loads(encoded))

        self.assertEqual(restored.schema_version, SCHEMA_VERSION)
        self.assertEqual(restored.session_id, session.session_id)
        self.assertEqual(restored.workspace, PROJECT_ROOT.resolve())
        self.assertEqual(restored.messages, session.messages)
        self.assertEqual(restored.usage, session.usage)
        self.assertEqual(restored.context.read_files["README.md"].version, "abc123")
        self.assertEqual(restored.context.read_call_paths["read-2"], "README.md")
        assistant_call = restored.messages[2].tool_calls[0]
        self.assertEqual(restored.messages[3].tool_call_id, assistant_call.id)

    def test_tool_result_must_reference_an_existing_call(self) -> None:
        session = Session.create()
        with self.assertRaisesRegex(ValueError, "unknown call"):
            session.add_message(ToolMessage("result", tool_call_id="missing"))

    def test_schema_one_session_is_upgraded_with_empty_context_state(self) -> None:
        session = Session.create(workspace=PROJECT_ROOT)
        payload = session_to_dict(session)
        payload["schema_version"] = 1
        payload.pop("context")

        restored = session_from_dict(payload)

        self.assertEqual(restored.schema_version, SCHEMA_VERSION)
        self.assertEqual(restored.context.read_files, {})

    def test_default_directory_is_outside_the_workspace(self) -> None:
        directory = default_session_directory({}).resolve()
        self.assertFalse(directory.is_relative_to(PROJECT_ROOT.resolve()))

    def test_store_rejects_a_directory_inside_git_workspace(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside a Git workspace"):
            JsonSessionStore(PROJECT_ROOT / ".coding-agent" / "sessions")


class JsonSessionStoreTests(unittest.TestCase):
    def test_atomic_write_preserves_previous_file_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = JsonSessionStore(root)
            session = store.create(
                workspace=PROJECT_ROOT,
                provider="mock-provider",
                model="mock-model",
            )
            session.add_message(UserMessage("not committed"))

            with patch("src.sessions.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(SessionStorageError):
                    store.save(session)

            restored = store.load(session.session_id)
            self.assertEqual(restored.messages, [])
            self.assertFalse(any(path.suffix == ".tmp" for path in root.iterdir()))

    def test_corrupted_session_does_not_hide_valid_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = JsonSessionStore(root)
            first = store.create(
                workspace=PROJECT_ROOT,
                provider="mock-provider",
                model="model-a",
            )
            second = store.create(
                workspace=PROJECT_ROOT,
                provider="mock-provider",
                model="model-b",
            )
            corrupted_id = "f" * 32
            (root / f"{corrupted_id}.json").write_text("{broken", encoding="utf-8")

            result = store.list_sessions()

            self.assertEqual({item.session_id for item in result.sessions}, {first.id, second.id})
            self.assertEqual(len(result.errors), 1)
            self.assertIn("corrupted", result.errors[0].message)
            self.assertEqual(store.load(first.id).model, "model-a")
            with self.assertRaisesRegex(SessionCorruptedError, "corrupted"):
                store.load(corrupted_id)

    def test_latest_by_id_list_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonSessionStore(Path(directory))
            first = store.create(
                workspace=PROJECT_ROOT,
                provider="provider-a",
                model="model-a",
            )
            second = store.create(
                workspace=PROJECT_ROOT,
                provider="provider-b",
                model="model-b",
            )
            self.assertEqual(store.load_latest(workspace=PROJECT_ROOT).id, second.id)
            self.assertEqual(store.load(first.id).provider, "provider-a")
            self.assertEqual(len(store.list_sessions().sessions), 2)

            store.delete(first.id)
            with self.assertRaises(SessionNotFoundError):
                store.load(first.id)

    def test_session_file_contains_only_versioned_safe_metadata(self) -> None:
        secret = "api-key-must-not-be-persisted"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = JsonSessionStore(root)
            session = store.create(
                workspace=PROJECT_ROOT,
                provider="openai-compatible",
                model="mock-model",
            )
            text = (root / f"{session.id}.json").read_text(encoding="utf-8")
            payload = json.loads(text)

        self.assertNotIn(secret, text)
        self.assertNotIn("api_key", payload)
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "session_id",
                "workspace",
                "created_at",
                "updated_at",
                "provider",
                "model",
                "messages",
                "usage",
                "context",
                "run_id",
                "run_state",
                "task_type",
                "current_plan",
                "current_step_id",
                "repair_attempts",
                "last_verification",
                "pending_approval",
                "checkpoint_version",
                "completed_tool_call_ids",
                "uncertain_tool_call_ids",
                "run_decision_summary",
                "run_failure_reason",
            },
        )

    def test_session_permissions_and_known_secret_redaction(self) -> None:
        secret = "sk-session-secret-value"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sessions"
            store = JsonSessionStore(root, secrets=(secret,))
            session = store.create(
                workspace=PROJECT_ROOT,
                provider="openai-compatible",
                model="mock-model",
            )
            session.add_message(UserMessage(f"do not persist {secret}"))
            store.save(session)
            path = root / f"{session.id}.json"
            persisted = path.read_text(encoding="utf-8")

            self.assertNotIn(secret, persisted)
            self.assertIn("[REDACTED]", persisted)
            if os.name == "posix":
                self.assertEqual(root.stat().st_mode & 0o777, 0o700)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            restored = store.load(session.id)
            self.assertIn("[REDACTED]", restored.messages[-1].content or "")

    def test_session_symlink_is_never_followed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sessions"
            store = JsonSessionStore(root)
            session = store.create(
                workspace=PROJECT_ROOT,
                provider="mock-provider",
                model="mock-model",
            )
            path = root / f"{session.id}.json"
            outside = Path(directory) / "outside-session.json"
            path.replace(outside)
            os.symlink(outside, path)

            with self.assertRaisesRegex(SessionCorruptedError, "corrupted"):
                store.load(session.id)


class SessionCliTests(unittest.TestCase):
    def test_cli_can_create_list_resume_latest_resume_by_id_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sessions_dir = Path(directory) / "sessions"
            environment = os.environ.copy()
            environment.update(
                {
                    "CODING_AGENT_SESSIONS_DIR": str(sessions_dir),
                    "CODING_AGENT_PROVIDER": "mock-provider",
                    "CODING_AGENT_MODEL": "mock-model",
                    "CODING_AGENT_API_KEY": "not-persisted-secret",
                }
            )
            created = self._run(
                "sessions",
                "new",
                "--workspace",
                str(PROJECT_ROOT),
                environment=environment,
            )
            session_id = created.stdout.strip()
            self.assertRegex(session_id, r"^[a-f0-9]{32}$")

            listed = self._run("sessions", "list", environment=environment)
            self.assertIn(session_id, listed.stdout)

            corrupted_id = "e" * 32
            (sessions_dir / f"{corrupted_id}.json").write_text("not-json", encoding="utf-8")
            listed_with_corruption = self._run("sessions", "list", environment=environment)
            self.assertIn(session_id, listed_with_corruption.stdout)
            self.assertIn("Warning:", listed_with_corruption.stderr)

            resumed = self._run("sessions", "resume", session_id, environment=environment)
            self.assertIn(f"Session: {session_id}", resumed.stdout)
            latest = self._run("sessions", "resume", environment=environment)
            self.assertIn(f"Session: {session_id}", latest.stdout)

            session_text = (sessions_dir / f"{session_id}.json").read_text(encoding="utf-8")
            self.assertNotIn("not-persisted-secret", session_text)
            self.assertNotIn("CODING_AGENT_", session_text)

            deleted = self._run("sessions", "delete", session_id, environment=environment)
            self.assertIn("Deleted session", deleted.stdout)

    def _run(
        self,
        *arguments: str,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "src.main", *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
