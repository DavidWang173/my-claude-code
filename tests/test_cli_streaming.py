from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from src.agent import CancellationToken
from src.cli import CliDependencies, main
from src.config import AppConfig
from src.models import ModelRequest, ModelResponse, ModelStreamChunk, ToolCallDelta, Usage
from src.terminal_ui import InterruptAction, InterruptController


class FakeCliProvider:
    name = "fake"

    def __init__(self, streams: tuple[tuple[ModelStreamChunk, ...], ...]) -> None:
        self._streams = list(streams)
        self.requests: list[ModelRequest] = []
        self.closed = False

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("streaming CLI must not call complete")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        self.requests.append(request)
        if not self._streams:
            raise AssertionError("fake provider has no stream")
        for chunk in self._streams.pop(0):
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def text_stream(text: str = "done") -> tuple[ModelStreamChunk, ...]:
    midpoint = max(1, len(text) // 2)
    return (
        ModelStreamChunk(text_delta=text[:midpoint]),
        ModelStreamChunk(
            text_delta=text[midpoint:],
            usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            finish_reason="stop",
        ),
    )


def tool_stream(name: str, arguments: dict[str, object]) -> tuple[ModelStreamChunk, ...]:
    return (
        ModelStreamChunk(
            tool_call_deltas=(
                ToolCallDelta(
                    index=0,
                    id=f"{name}-1",
                    name=name,
                    arguments_delta=json.dumps(arguments),
                ),
            ),
            finish_reason="tool_calls",
        ),
    )


class StreamingCliIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        root = Path(self._temporary.name).resolve()
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.sessions = root / "sessions"
        self.environ = {
            "CODING_AGENT_PROVIDER": "fake",
            "CODING_AGENT_MODEL": "fake-model",
            "CODING_AGENT_SESSIONS_DIR": str(self.sessions),
        }

    def test_human_run_streams_without_color_when_redirected(self) -> None:
        provider = FakeCliProvider(
            (
                tool_stream("list_files", {"path": "."}),
                text_stream("hello"),
            )
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = main(
            ["run", "inspect", "project", "--workspace", str(self.workspace)],
            dependencies=self._dependencies(provider, stdout=stdout, stderr=stderr),
        )

        self.assertEqual(code, 0)
        self.assertTrue(provider.closed)
        rendered = stdout.getvalue()
        self.assertIn("model: fake-model", rendered)
        self.assertIn(f"workspace: {self.workspace}", rendered)
        self.assertIn("assistant> hello", rendered)
        self.assertRegex(
            rendered,
            r"tokens: prompt=\d+ completion=\d+ total=\d+",
        )
        self.assertNotIn("\x1b[", rendered)
        self.assertEqual(stderr.getvalue(), "")

    def test_json_mode_is_json_lines_and_contains_no_terminal_styles(self) -> None:
        provider = FakeCliProvider((text_stream("json answer"),))
        stdout = io.StringIO()

        code = main(
            [
                "run",
                "scripted task",
                "--workspace",
                str(self.workspace),
                "--json",
            ],
            dependencies=self._dependencies(provider, stdout=stdout),
        )

        self.assertEqual(code, 0)
        output = stdout.getvalue()
        records = [json.loads(line) for line in output.splitlines()]
        self.assertIn("session", {record["type"] for record in records})
        self.assertIn("text_delta", {record["type"] for record in records})
        self.assertIn("turn_summary", {record["type"] for record in records})
        self.assertNotIn("\x1b[", output)
        summary = next(record for record in records if record["type"] == "turn_summary")
        self.assertEqual(summary["usage"]["total_tokens"], 6)

    def test_shell_output_streams_as_json_events(self) -> None:
        (self.workspace / "test_sample.py").write_text(
            "import unittest\n\n"
            "class SampleTests(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        provider = FakeCliProvider(
            (
                tool_stream(
                    "run_shell",
                    {"argv": [sys.executable, "-m", "unittest"]},
                ),
                text_stream("tests complete"),
            )
        )
        stdout = io.StringIO()

        code = main(
            ["--json", "--workspace", str(self.workspace), "run", "run tests"],
            dependencies=self._dependencies(provider, stdout=stdout),
        )

        self.assertEqual(code, 0)
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        kinds = [record["type"] for record in records]
        self.assertIn("tool_call", kinds)
        self.assertIn("tool_output", kinds)
        self.assertIn("tool_result", kinds)
        summary = next(record for record in records if record["type"] == "turn_summary")
        self.assertTrue(summary["test_results"])

    def test_file_diff_is_shown_before_interactive_approval(self) -> None:
        provider = FakeCliProvider(
            (
                tool_stream(
                    "create_file",
                    {"path": "created.txt", "content": "created\n"},
                ),
                text_stream("file created"),
            )
        )
        stdin = TtyStringIO("y\n")
        stdout = TtyStringIO()

        code = main(
            ["--workspace", str(self.workspace), "run", "create file", "--no-color"],
            dependencies=self._dependencies(provider, stdin=stdin, stdout=stdout),
        )

        self.assertEqual(code, 0)
        rendered = stdout.getvalue()
        self.assertLess(rendered.index("diff preview:"), rendered.index("Approve?"))
        self.assertIn("--- /dev/null", rendered)
        self.assertIn("modified files: created.txt", rendered)
        self.assertEqual(
            (self.workspace / "created.txt").read_text(encoding="utf-8"), "created\n"
        )

    def test_chat_accepts_multiline_input(self) -> None:
        provider = FakeCliProvider((text_stream("understood"),))
        stdin = TtyStringIO("/multi\nfirst line\nsecond line\n.\n/exit\n")
        stdout = TtyStringIO()

        code = main(
            ["--workspace", str(self.workspace), "chat", "--no-color"],
            dependencies=self._dependencies(provider, stdin=stdin, stdout=stdout),
        )

        self.assertEqual(code, 0)
        user_messages = [
            message.content
            for message in provider.requests[0].messages
            if message.role == "user"
        ]
        self.assertEqual(user_messages[-1], "first line\nsecond line")
        self.assertIn("session:", stdout.getvalue())

    def test_sessions_config_and_diagnostics_have_json_interfaces(self) -> None:
        provider = FakeCliProvider(())
        for command in (["sessions"], ["config"], ["diagnostics"]):
            stdout = io.StringIO()
            code = main(
                ["--json", *command],
                dependencies=self._dependencies(provider, stdout=stdout),
            )
            self.assertEqual(code, 0)
            records = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertTrue(records)
            self.assertNotIn("\x1b[", stdout.getvalue())

    def _dependencies(
        self,
        provider: FakeCliProvider,
        *,
        stdin: io.StringIO | None = None,
        stdout: io.StringIO | None = None,
        stderr: io.StringIO | None = None,
    ) -> CliDependencies:
        return CliDependencies(
            provider_factory=lambda config: provider,
            stdin=stdin or io.StringIO(),
            stdout=stdout or io.StringIO(),
            stderr=stderr or io.StringIO(),
            environ=self.environ,
        )


class InterruptControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_interrupt_cancels_and_second_requests_exit(self) -> None:
        token = CancellationToken()
        controller = InterruptController()
        controller.bind(token)

        first = controller.handle_interrupt()
        second = controller.handle_interrupt()

        self.assertEqual(first, InterruptAction.CANCEL)
        self.assertTrue(token.cancelled)
        self.assertEqual(second, InterruptAction.EXIT)
        self.assertTrue(controller.exit_requested)


if __name__ == "__main__":
    unittest.main()
