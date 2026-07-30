from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.tools import (
    GitDiffTool,
    GitStatusTool,
    ListFilesTool,
    ReadFileTool,
    SearchTextTool,
    ToolContext,
    read_only_tool_registry,
)


class WorkspaceToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.container = Path(self._temporary.name).resolve()
        self.workspace = self.container / "workspace"
        self.workspace.mkdir()
        (self.workspace / "src").mkdir()
        (self.workspace / "README.md").write_text("heading\nneedle here\ntail\n", encoding="utf-8")
        (self.workspace / "src" / "app.py").write_text(
            "first\nsecond needle\nthird\nfourth\n", encoding="utf-8"
        )
        for ignored in (".git", "node_modules", "venv", "dist", "build"):
            directory = self.workspace / ignored
            directory.mkdir()
            (directory / "ignored.txt").write_text("needle", encoding="utf-8")
        nested_dependency = self.workspace / "src" / "node_modules"
        nested_dependency.mkdir()
        (nested_dependency / "ignored.txt").write_text("needle", encoding="utf-8")
        self.outside = self.container / "outside.txt"
        self.outside.write_text("outside secret", encoding="utf-8")
        self.context = ToolContext("test-session", self.workspace)

    async def test_registry_exposes_closed_model_schemas(self) -> None:
        registry = read_only_tool_registry()
        self.assertEqual(
            set(registry.names()),
            {
                "list_files",
                "read_file",
                "search_text",
                "git_status",
                "git_diff",
                "git_diff_check",
            },
        )
        for definition in registry.definitions():
            with self.subTest(tool=definition.name):
                self.assertEqual(definition.parameters["type"], "object")
                self.assertIs(definition.parameters["additionalProperties"], False)

    async def test_list_files_ignores_large_directories_and_is_structured(self) -> None:
        result = await ListFilesTool().execute({}, self.context)

        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertIn("README.md", result.content)
        self.assertIn("src/app.py", result.content)
        self.assertNotIn("ignored.txt", result.content)
        self.assertEqual(
            set(result.to_dict()), {"success", "content", "error", "metadata"}
        )

    async def test_absolute_path_inside_workspace_is_allowed(self) -> None:
        path = (self.workspace / "README.md").resolve()
        result = await ReadFileTool().execute({"path": str(path)}, self.context)

        self.assertTrue(result.success)
        self.assertIn("heading", result.content)

    async def test_path_traversal_absolute_escape_and_symlink_escape_are_rejected(self) -> None:
        link = self.workspace / "escape.txt"
        try:
            os.symlink(self.outside, link)
        except OSError as exc:
            self.skipTest(f"symbolic links unavailable: {exc}")

        attempts = (
            (ReadFileTool(), {"path": "../outside.txt"}, "traversal"),
            (ReadFileTool(), {"path": str(self.outside)}, "outside"),
            (ReadFileTool(), {"path": "escape.txt"}, "outside"),
            (ListFilesTool(), {"path": "../"}, "traversal"),
            (SearchTextTool(rg_executable=None), {"query": "secret", "path": "escape.txt"}, "outside"),
            (GitDiffTool(), {"path": str(self.outside)}, "outside"),
        )
        for tool, arguments, message in attempts:
            with self.subTest(tool=tool.name, arguments=arguments):
                result = await tool.execute(arguments, self.context)
                self.assertFalse(result.success)
                self.assertIn(message, result.error or "")

    async def test_read_file_supports_line_ranges_and_reports_truncation(self) -> None:
        result = await ReadFileTool().execute(
            {"path": "src/app.py", "start_line": 2, "line_count": 2},
            self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "second needle\nthird\n...[output truncated]")
        self.assertEqual(result.metadata["start_line"], 2)
        self.assertEqual(result.metadata["end_line"], 3)
        self.assertRegex(str(result.metadata["version"]), r"^[a-f0-9]{64}$")
        self.assertIs(result.metadata["truncated"], True)

    async def test_read_file_rejects_large_and_binary_files(self) -> None:
        (self.workspace / "large.txt").write_text("0123456789", encoding="utf-8")
        (self.workspace / "binary.dat").write_bytes(b"text\x00binary")

        large = await ReadFileTool(max_file_bytes=8).execute(
            {"path": "large.txt"}, self.context
        )
        binary = await ReadFileTool().execute({"path": "binary.dat"}, self.context)

        self.assertFalse(large.success)
        self.assertIn("size limit", large.error or "")
        self.assertEqual(large.metadata["file_size"], 10)
        self.assertFalse(binary.success)
        self.assertIn("binary", binary.error or "")

    async def test_list_output_limit_is_explicit(self) -> None:
        result = await ListFilesTool().execute({"max_entries": 1}, self.context)

        self.assertTrue(result.success)
        self.assertTrue(result.metadata["truncated"])
        self.assertIn("[output truncated]", result.content)

    async def test_search_uses_python_fallback_when_rg_is_unavailable(self) -> None:
        result = await SearchTextTool(rg_executable=None).execute(
            {"query": "needle"}, self.context
        )

        self.assertTrue(result.success)
        self.assertEqual(result.metadata["engine"], "python")
        self.assertEqual(result.metadata["matches"], 2)
        self.assertIn("README.md:2:1:needle here", result.content)
        self.assertIn("src/app.py:2:8:second needle", result.content)
        self.assertNotIn("ignored.txt", result.content)

    async def test_search_prefers_rg_when_available(self) -> None:
        if shutil.which("rg") is None:
            self.skipTest("rg is unavailable")
        result = await SearchTextTool().execute({"query": "needle"}, self.context)

        self.assertTrue(result.success)
        self.assertEqual(result.metadata["engine"], "rg")
        self.assertEqual(result.metadata["matches"], 2)

    async def test_git_tools_report_non_git_workspace(self) -> None:
        status = await GitStatusTool().execute({}, self.context)
        diff = await GitDiffTool().execute({}, self.context)

        self.assertFalse(status.success)
        self.assertIn("not a Git repository", status.error or "")
        self.assertFalse(diff.success)
        self.assertIn("not a Git repository", diff.error or "")


class GitWorkspaceToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is unavailable")
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.workspace = Path(self._temporary.name).resolve()
        self._git("init", "--quiet")
        tracked = self.workspace / "tracked.txt"
        tracked.write_text("before\n", encoding="utf-8")
        self._git("add", "tracked.txt")
        self._git(
            "-c",
            "user.name=Tool Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        )
        tracked.write_text("after\n", encoding="utf-8")
        self.context = ToolContext("test-session", self.workspace)

    def _git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=self.workspace,
            check=True,
            capture_output=True,
            text=True,
        )

    async def test_git_status_and_diff_are_read_only_and_bounded(self) -> None:
        status = await GitStatusTool().execute({}, self.context)
        diff = await GitDiffTool().execute({"path": "tracked.txt"}, self.context)

        self.assertTrue(status.success)
        self.assertFalse(status.metadata["clean"])
        self.assertIn("tracked.txt", status.content)
        self.assertTrue(diff.success)
        self.assertIn("-before", diff.content)
        self.assertIn("+after", diff.content)
        self.assertFalse(diff.metadata["staged"])


if __name__ == "__main__":
    unittest.main()
