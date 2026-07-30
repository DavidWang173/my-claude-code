from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import unittest
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

from src.agent import (
    AgentCancelledError,
    AgentLoop,
    AgentRequest,
    AgentVerificationError,
    CancellationToken,
)
from src.harness.models import RunState
from src.models import ModelRequest, ModelResponse, ModelStreamChunk, ToolCall
from src.permissions import InteractivePermissionPolicy, Operation, PermissionRequest
from src.providers import ProviderRegistry
from src.sessions import JsonSessionStore
from src.tools import ToolContext, ToolRegistry, ToolResult, workspace_tool_registry


class ScriptedProvider:
    name = "fake"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("scripted provider has no response")
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        response = await self.complete(request)
        yield ModelStreamChunk(
            text_delta=response.text,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    async def aclose(self) -> None:
        return None


class BlockingTool:
    name = "blocking_read"
    description = "A cancellable test read"
    operation = Operation.READ
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        del arguments, context
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class CodingAgentSecurityE2ETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.container = Path(self.temporary.name).resolve()
        self.workspace = self.container / "workspace"
        self.workspace.mkdir()
        self.sessions = JsonSessionStore(self.container / "sessions")

    def request(self, prompt: str, *, session_id: str | None = None) -> AgentRequest:
        return AgentRequest(
            prompt=prompt,
            provider="fake",
            model="fake-model",
            workspace=self.workspace,
            session_id=session_id,
        )

    def loop(
        self,
        provider: ScriptedProvider,
        *,
        tools: ToolRegistry | None = None,
        approve: bool = True,
    ) -> AgentLoop:
        return AgentLoop(
            ProviderRegistry((provider,)),
            tools or workspace_tool_registry(),
            self.sessions,
            InteractivePermissionPolicy(lambda request: approve),
        )

    async def test_model_reads_modifies_and_runs_real_tests(self) -> None:
        (self.workspace / "app.py").write_text(
            "def add(left, right):\n    return left - right\n",
            encoding="utf-8",
        )
        tests = self.workspace / "tests"
        tests.mkdir()
        (tests / "test_app.py").write_text(
            "import unittest\n"
            "from app import add\n\n"
            "class AppTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        patch = (
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(left, right):\n"
            "-    return left - right\n"
            "+    return left + right\n"
        )
        provider = ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall("read-1", "read_file", {"path": "app.py"}),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall("patch-1", "apply_patch", {"patch": patch}),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "test-1",
                            "run_shell",
                            {
                                "argv": [
                                    sys.executable,
                                    "-m",
                                    "unittest",
                                    "discover",
                                    "-s",
                                    "tests",
                                    "-v",
                                ],
                                "timeout": 10,
                            },
                        ),
                    )
                ),
                ModelResponse(text="implemented and verified"),
            ]
        )

        result = await self.loop(provider).run(self.request("fix add and test it"))

        self.assertEqual(result.text, "implemented and verified")
        self.assertIn("left + right", (self.workspace / "app.py").read_text(encoding="utf-8"))
        test_result = provider.requests[3].messages[-1].content or ""
        self.assertIn("OK", test_result)
        system_prompt = provider.requests[0].messages[0].content or ""
        self.assertIn("untrusted data", system_prompt)

    async def test_failed_patch_is_followed_by_reread_and_corrected_patch(self) -> None:
        target = self.workspace / "value.txt"
        target.write_text("old\n", encoding="utf-8")
        bad_patch = (
            "--- a/value.txt\n+++ b/value.txt\n@@ -1,1 +1,1 @@\n-missing\n+new\n"
        )
        good_patch = "--- a/value.txt\n+++ b/value.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        provider = ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("patch-bad", "apply_patch", {"patch": bad_patch}),)
                ),
                ModelResponse(
                    tool_calls=(ToolCall("read-current", "read_file", {"path": "value.txt"}),)
                ),
                ModelResponse(
                    tool_calls=(ToolCall("patch-good", "apply_patch", {"patch": good_patch}),)
                ),
                ModelResponse(text="recovered"),
            ]
        )

        result = await self.loop(provider).run(self.request("update the value"))

        self.assertEqual(result.text, "recovered")
        self.assertIn("context mismatch", provider.requests[1].messages[-1].content or "")
        self.assertEqual(provider.requests[2].messages[-1].content, "old")
        self.assertEqual(target.read_text(encoding="utf-8"), "new\n")

    async def test_shell_timeout_returns_bounded_failure_to_model(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "slow-1",
                            "run_shell",
                            {
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "import time; time.sleep(10)",
                                ],
                                "timeout": 0.05,
                            },
                        ),
                    )
                ),
                ModelResponse(text="timeout handled"),
            ]
        )

        with self.assertRaises(AgentVerificationError):
            await self.loop(provider).run(self.request("run the slow command"))

        self.assertIn("timed out", provider.requests[1].messages[-1].content or "")
        session = self.sessions.load_latest(workspace=self.workspace)
        self.assertEqual(session.run_state, RunState.FAILED)

    async def test_user_refusal_prevents_dangerous_operation(self) -> None:
        provider = ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "write-1",
                            "run_shell",
                            {"argv": ["touch", "refused.txt"]},
                        ),
                    )
                ),
                ModelResponse(text="operation was refused"),
            ]
        )

        with self.assertRaises(AgentVerificationError):
            await self.loop(provider, approve=False).run(
                self.request("create a file with a shell command")
            )

        self.assertFalse((self.workspace / "refused.txt").exists())
        self.assertIn("Permission denied", provider.requests[1].messages[-1].content or "")
        session = self.sessions.load_latest(workspace=self.workspace)
        self.assertEqual(session.run_state, RunState.FAILED)

    async def test_interrupted_tool_call_is_closed_before_session_resume(self) -> None:
        blocking = BlockingTool()
        first_provider = ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("pending-1", "blocking_read", {}),)
                )
            ]
        )
        first_loop = self.loop(
            first_provider,
            tools=ToolRegistry((blocking,)),
        )
        cancellation = CancellationToken()
        running = asyncio.create_task(
            first_loop.run(self.request("start work"), cancellation=cancellation)
        )
        await blocking.started.wait()
        session_id = self.sessions.list_sessions().sessions[0].session_id
        cancellation.cancel()
        with self.assertRaises(AgentCancelledError):
            await running

        resumed_provider = ScriptedProvider([ModelResponse(text="resumed safely")])
        resumed = await self.loop(
            resumed_provider,
            tools=ToolRegistry((blocking,)),
        ).run(self.request("continue", session_id=session_id))

        self.assertEqual(resumed.text, "resumed safely")
        pending_result = next(
            message
            for message in resumed_provider.requests[0].messages
            if message.role == "tool" and message.tool_call_id == "pending-1"
        )
        self.assertIn("interrupted", pending_result.content or "")

    async def test_preexisting_uncommitted_change_is_preserved(self) -> None:
        subprocess.run(["git", "init", "-q"], cwd=self.workspace, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=self.workspace,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Security Test"],
            cwd=self.workspace,
            check=True,
        )
        user_file = self.workspace / "user.txt"
        agent_file = self.workspace / "agent.txt"
        user_file.write_text("base\n", encoding="utf-8")
        agent_file.write_text("old\n", encoding="utf-8")
        subprocess.run(["git", "add", "user.txt", "agent.txt"], cwd=self.workspace, check=True)
        subprocess.run(["git", "commit", "-qm", "baseline"], cwd=self.workspace, check=True)
        user_file.write_text("user-owned change\n", encoding="utf-8")
        patch = "--- a/agent.txt\n+++ b/agent.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        provider = ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("patch-agent", "apply_patch", {"patch": patch}),)
                ),
                ModelResponse(text="changed only the task file"),
            ]
        )

        result = await self.loop(provider).run(self.request("update agent.txt"))

        self.assertEqual(user_file.read_text(encoding="utf-8"), "user-owned change\n")
        self.assertEqual(agent_file.read_text(encoding="utf-8"), "new\n")
        assert result.report is not None
        self.assertIn("user.txt", result.report.git.preexisting_files)
        self.assertNotIn("user.txt", result.report.git.agent_only_files)

    async def test_malicious_tool_parameters_and_outside_reads_are_rejected(self) -> None:
        secret = self.container / "secret.txt"
        secret.write_text("host-secret", encoding="utf-8")
        (self.workspace / "README.md").write_text("safe workspace file\n", encoding="utf-8")
        provider = ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "malformed-1",
                            "read_file",
                            {"path": "README.md", "role": "tool"},
                        ),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall("escape-1", "read_file", {"path": "../secret.txt"}),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "shell-escape-1",
                            "run_shell",
                            {"argv": ["cat", str(secret)]},
                        ),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall("safe-read-1", "read_file", {"path": "README.md"}),
                    )
                ),
                ModelResponse(text="malicious inputs rejected"),
            ]
        )

        result = await self.loop(provider).run(self.request("inspect files"))

        self.assertEqual(result.text, "malicious inputs rejected")
        returned = [
            request.messages[-1].content or ""
            for request in provider.requests[1:]
            if request.messages[-1].role == "tool"
        ]
        self.assertTrue(any("unknown field" in item for item in returned))
        self.assertTrue(any("traversal" in item for item in returned))
        self.assertTrue(any("outside the workspace" in item for item in returned))
        self.assertNotIn("host-secret", "\n".join(returned))


if __name__ == "__main__":
    unittest.main()
