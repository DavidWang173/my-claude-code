from __future__ import annotations

import subprocess
import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from src.agent import AgentLoop, AgentRequest
from src.git_runtime import (
    GitAddTool,
    GitCommitTool,
    GitDiffCheckTool,
    GitRunTracker,
)
from src.models import ModelRequest, ModelResponse, ModelStreamChunk, ToolCall
from src.permissions import PermissionDecision, PermissionRequest
from src.providers import ProviderRegistry
from src.quality import suggest_test_commands
from src.sessions import JsonSessionStore
from src.shell_tools import ShellCommandPolicy
from src.tools import (
    CreateFileTool,
    ToolContext,
    ToolRegistry,
    workspace_tool_registry,
)


class GitRepositoryTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.container = Path(self._temporary.name).resolve()
        self.workspace = self.container / "workspace"
        self.workspace.mkdir()
        self._git("init", "--quiet")
        self._git("config", "user.name", "Agent Tests")
        self._git("config", "user.email", "agent@example.invalid")
        (self.workspace / "user.txt").write_text("committed\n", encoding="utf-8")
        (self.workspace / "agent.txt").write_text("before\n", encoding="utf-8")
        self._git("add", "user.txt", "agent.txt")
        self._git("commit", "--quiet", "-m", "initial")

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.workspace,
            check=True,
            capture_output=True,
            text=True,
        )


class GitRunTrackerTests(GitRepositoryTestCase):
    async def test_preexisting_changes_are_not_attributed_or_restored(self) -> None:
        (self.workspace / "user.txt").write_text("user work\n", encoding="utf-8")
        (self.workspace / "user-untracked.txt").write_text(
            "keep me\n", encoding="utf-8"
        )
        tracker = await GitRunTracker.capture(self.workspace)

        (self.workspace / "agent.txt").write_text("after\n", encoding="utf-8")
        (self.workspace / "created.txt").write_text("new\n", encoding="utf-8")
        tracker.mark_agent_paths(["agent.txt", "created.txt"])
        summary = await tracker.finish()

        self.assertEqual(
            set(summary.preexisting_files), {"user.txt", "user-untracked.txt"}
        )
        self.assertEqual(set(summary.changed_files), {"agent.txt", "created.txt"})
        self.assertEqual(set(summary.agent_only_files), {"agent.txt", "created.txt"})
        self.assertEqual(summary.overlapping_files, ())
        self.assertEqual(
            (self.workspace / "user.txt").read_text(encoding="utf-8"), "user work\n"
        )
        self.assertIn("-before", summary.diff)
        self.assertIn("+after", summary.diff)

    async def test_further_change_to_dirty_file_is_reported_as_overlap(self) -> None:
        (self.workspace / "user.txt").write_text("user baseline\n", encoding="utf-8")
        tracker = await GitRunTracker.capture(self.workspace)
        (self.workspace / "user.txt").write_text(
            "user baseline\nagent addition\n", encoding="utf-8"
        )
        tracker.mark_agent_paths(["user.txt"])

        summary = await tracker.finish()

        self.assertEqual(summary.changed_files, ("user.txt",))
        self.assertEqual(summary.overlapping_files, ("user.txt",))
        self.assertEqual(summary.agent_only_files, ())


class ControlledGitToolTests(GitRepositoryTestCase):
    def test_workspace_registry_exposes_controlled_git_tools(self) -> None:
        registry = workspace_tool_registry()

        self.assertIn("git_status", registry.names())
        self.assertIn("git_diff", registry.names())
        self.assertIn("git_diff_check", registry.names())
        self.assertIn("git_add", registry.names())
        self.assertIn("git_commit", registry.names())
        for name in ("git_diff_check", "git_add", "git_commit"):
            self.assertIs(
                registry.get(name).parameters["additionalProperties"], False
            )

    async def test_add_and_commit_only_task_local_files_after_approval(self) -> None:
        (self.workspace / "user.txt").write_text("user work\n", encoding="utf-8")
        tracker = await GitRunTracker.capture(self.workspace)
        (self.workspace / "agent.txt").write_text("agent work\n", encoding="utf-8")
        tracker.mark_agent_paths(["agent.txt"])
        context = ToolContext(
            "session",
            self.workspace,
            permission_granted=True,
            git_tracker=tracker,
        )

        denied_request = await GitAddTool().permission_request(
            {"paths": ["user.txt"]}, context
        )
        self.assertEqual(denied_request.effective_level.value, "deny")
        denied = await GitAddTool().execute({"paths": ["user.txt"]}, context)
        self.assertFalse(denied.success)
        self.assertIn("user changes", denied.error or "")

        add_request = await GitAddTool().permission_request(
            {"paths": ["agent.txt"]}, context
        )
        self.assertEqual(add_request.effective_level.value, "ask")
        self.assertEqual(add_request.command, "git add -- agent.txt")
        self.assertEqual(add_request.cwd, str(self.workspace))
        added = await GitAddTool().execute({"paths": ["agent.txt"]}, context)
        self.assertTrue(added.success, added.error)

        commit_request = await GitCommitTool().permission_request(
            {"message": "agent change"}, context
        )
        self.assertEqual(commit_request.effective_level.value, "ask")
        self.assertIn("Git hooks may run", commit_request.risk_reason or "")
        committed = await GitCommitTool().execute(
            {"message": "agent change"}, context
        )
        self.assertTrue(committed.success, committed.error)
        self.assertRegex(str(committed.metadata["commit"]), r"^[0-9a-f]{40}$")

        status = self._git("status", "--short").stdout
        self.assertIn(" M user.txt", status)
        self.assertNotIn("agent.txt", status)
        self.assertEqual(
            (self.workspace / "user.txt").read_text(encoding="utf-8"), "user work\n"
        )

    async def test_diff_check_reports_whitespace_failure_without_mutation(self) -> None:
        tracker = await GitRunTracker.capture(self.workspace)
        (self.workspace / "agent.txt").write_text("trailing  \n", encoding="utf-8")
        context = ToolContext("session", self.workspace, git_tracker=tracker)

        result = await GitDiffCheckTool().execute({}, context)

        self.assertTrue(result.success)
        self.assertFalse(result.metadata["passed"])
        self.assertNotEqual(result.metadata["exit_code"], 0)
        self.assertIn("trailing whitespace", result.content)

    def test_push_is_hard_denied_even_when_not_forced(self) -> None:
        policy = ShellCommandPolicy(("git push",))
        assessment = policy.assess_argv(
            ["git", "push", "origin", "main"], cwd=self.workspace
        )
        shell_assessment = policy.assess_shell(
            "git push --force origin main", cwd=self.workspace
        )

        self.assertEqual(assessment.level.value, "deny")
        self.assertEqual(shell_assessment.level.value, "deny")
        self.assertIn("never pushes", assessment.reason)


class GitAwareAgentIntegrationTests(GitRepositoryTestCase):
    def test_python_test_suggestions_do_not_execute_commands(self) -> None:
        (self.workspace / "tests").mkdir()

        suggestions = suggest_test_commands(self.workspace, ["src/example.py"])

        self.assertIn("python -m unittest discover -s tests -v", suggestions)
        self.assertIn("git_diff_check", suggestions)

    async def test_agent_report_separates_preexisting_worktree_changes(self) -> None:
        (self.workspace / "user.txt").write_text("user work\n", encoding="utf-8")
        provider = _FakeProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="create",
                            name="create_file",
                            arguments={"path": "created.txt", "content": "created\n"},
                        ),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="add",
                            name="git_add",
                            arguments={"paths": ["created.txt"]},
                        ),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="check",
                            name="git_diff_check",
                            arguments={"staged": True},
                        ),
                    )
                ),
                ModelResponse(text="done"),
            ]
        )
        sessions = JsonSessionStore(self.container / "sessions")
        loop = AgentLoop(
            ProviderRegistry((provider,)),
            ToolRegistry((CreateFileTool(), GitAddTool(), GitDiffCheckTool())),
            sessions,
            _AllowAll(),
        )

        result = await loop.run(
            AgentRequest(
                prompt="create one file",
                provider="fake",
                model="fake-model",
                workspace=self.workspace,
            )
        )

        assert result.report is not None
        self.assertEqual(result.report.git.agent_only_files, ("created.txt",))
        self.assertEqual(result.report.git.preexisting_files, ("user.txt",))
        self.assertEqual(result.report.git.overlapping_files, ())
        self.assertEqual(result.report.incomplete, ())
        self.assertIn("git_diff_check", result.report.suggested_tests)
        self.assertEqual(len(result.report.tests), 1)
        self.assertTrue(result.report.tests[0].passed)
        self.assertEqual(
            (self.workspace / "user.txt").read_text(encoding="utf-8"), "user work\n"
        )


class _AllowAll:
    def decide(self, request: PermissionRequest) -> PermissionDecision:
        self.last_request = request
        return PermissionDecision(True, "approved by test")


class _FakeProvider:
    name = "fake"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        del request
        if False:
            yield ModelStreamChunk()

    async def aclose(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
