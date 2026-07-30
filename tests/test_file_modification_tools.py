from __future__ import annotations

import difflib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.tools as tools_module
from src.permissions import (
    InteractivePermissionPolicy,
    Operation,
    PermissionRequest,
    ReadOnlyPermissionPolicy,
)
from src.tools import (
    ApplyPatchTool,
    CreateFileTool,
    ToolContext,
    workspace_tool_registry,
)


def unified_patch(path: str, old: str, new: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )


class FileModificationToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.container = Path(self._temporary.name).resolve()
        self.workspace = self.container / "workspace"
        self.workspace.mkdir()
        self.context = ToolContext("edit-session", self.workspace)
        self.outside = self.container / "outside.txt"
        self.outside.write_text("do not modify\n", encoding="utf-8")

    async def test_successful_patch_returns_diff_and_preserves_format(self) -> None:
        target = self.workspace / "example.txt"
        target.write_bytes(b"\xef\xbb\xbfalpha\r\nbeta\r\ngamma\r\n")
        patch_text = "\n".join(
            (
                "--- a/example.txt",
                "+++ b/example.txt",
                "@@ -1,3 +1,3 @@",
                " alpha",
                "-beta",
                "+BETA",
                " gamma",
            )
        )

        result = await ApplyPatchTool().execute({"patch": patch_text}, self.context)

        self.assertTrue(result.success, result.error)
        self.assertEqual(
            target.read_bytes(), b"\xef\xbb\xbfalpha\r\nBETA\r\ngamma\r\n"
        )
        self.assertIn("--- a/example.txt", result.content)
        self.assertIn("+++ b/example.txt", result.content)
        self.assertIn("-beta", result.content)
        self.assertIn("+BETA", result.content)
        self.assertEqual(result.metadata["files"], ["example.txt"])

    async def test_context_conflict_is_explicit_and_does_not_modify_file(self) -> None:
        target = self.workspace / "conflict.txt"
        original = "actual\ncontent\n"
        target.write_text(original, encoding="utf-8")
        patch_text = unified_patch("conflict.txt", "expected\ncontent\n", "changed\ncontent\n")

        result = await ApplyPatchTool().execute({"patch": patch_text}, self.context)

        self.assertFalse(result.success)
        self.assertIn("context mismatch", result.error or "")
        self.assertEqual(target.read_text(encoding="utf-8"), original)

    async def test_create_file_and_refuse_overwrite(self) -> None:
        tool = CreateFileTool()
        created = await tool.execute(
            {"path": "new.py", "content": "print('new')\n"}, self.context
        )
        refused = await tool.execute(
            {"path": "new.py", "content": "overwrite\n"}, self.context
        )

        self.assertTrue(created.success, created.error)
        self.assertEqual(
            (self.workspace / "new.py").read_text(encoding="utf-8"), "print('new')\n"
        )
        self.assertIn("--- /dev/null", created.content)
        self.assertFalse(refused.success)
        self.assertIn("refuses to overwrite", refused.error or "")
        self.assertEqual(
            (self.workspace / "new.py").read_text(encoding="utf-8"), "print('new')\n"
        )

    async def test_dry_run_previews_without_modifying_or_creating(self) -> None:
        target = self.workspace / "dry.txt"
        target.write_text("old\n", encoding="utf-8")
        patch_result = await ApplyPatchTool().execute(
            {"patch": unified_patch("dry.txt", "old\n", "new\n"), "dry_run": True},
            self.context,
        )
        create_result = await CreateFileTool().execute(
            {"path": "preview.txt", "content": "preview\n", "dry_run": True},
            self.context,
        )

        self.assertTrue(patch_result.success, patch_result.error)
        self.assertTrue(create_result.success, create_result.error)
        self.assertEqual(target.read_text(encoding="utf-8"), "old\n")
        self.assertFalse((self.workspace / "preview.txt").exists())
        self.assertIs(patch_result.metadata["dry_run"], True)
        self.assertIs(create_result.metadata["dry_run"], True)

    async def test_path_escape_and_symlink_escape_are_rejected(self) -> None:
        symlink = self.workspace / "escape.txt"
        try:
            os.symlink(self.outside, symlink)
        except OSError as exc:
            self.skipTest(f"symbolic links unavailable: {exc}")
        traversal = await CreateFileTool().execute(
            {"path": "../escaped.txt", "content": "bad"}, self.context
        )
        absolute = await CreateFileTool().execute(
            {"path": str(self.container / "absolute.txt"), "content": "bad"},
            self.context,
        )
        symlink_patch = await ApplyPatchTool().execute(
            {"patch": unified_patch("escape.txt", "do not modify\n", "modified\n")},
            self.context,
        )

        self.assertIn("traversal", traversal.error or "")
        self.assertIn("outside", absolute.error or "")
        self.assertIn("outside", symlink_patch.error or "")
        self.assertEqual(self.outside.read_text(encoding="utf-8"), "do not modify\n")

    async def test_protected_directories_and_secret_files_are_rejected(self) -> None:
        (self.workspace / ".git").mkdir()
        protected_directory = await CreateFileTool().execute(
            {"path": ".git/config", "content": "bad"}, self.context
        )
        protected_secret = await CreateFileTool().execute(
            {"path": ".env", "content": "TOKEN=bad\n"}, self.context
        )

        self.assertFalse(protected_directory.success)
        self.assertIn("protected directory", protected_directory.error or "")
        self.assertFalse(protected_secret.success)
        self.assertIn("secret file", protected_secret.error or "")

    async def test_multifile_context_failure_leaves_no_partial_changes(self) -> None:
        first = self.workspace / "first.txt"
        second = self.workspace / "second.txt"
        first.write_text("one\n", encoding="utf-8")
        second.write_text("two\n", encoding="utf-8")
        patch_text = "\n".join(
            (
                unified_patch("first.txt", "one\n", "ONE\n"),
                unified_patch("second.txt", "missing\n", "TWO\n"),
            )
        )

        result = await ApplyPatchTool().execute({"patch": patch_text}, self.context)

        self.assertFalse(result.success)
        self.assertIn("context mismatch", result.error or "")
        self.assertEqual(first.read_text(encoding="utf-8"), "one\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "two\n")

    async def test_write_failure_rolls_back_files_already_committed(self) -> None:
        first = self.workspace / "first.txt"
        second = self.workspace / "second.txt"
        first.write_text("one\n", encoding="utf-8")
        second.write_text("two\n", encoding="utf-8")
        patch_text = "\n".join(
            (
                unified_patch("first.txt", "one\n", "ONE\n"),
                unified_patch("second.txt", "two\n", "TWO\n"),
            )
        )
        real_replace = tools_module._atomic_replace_bytes

        def fail_second(
            path: Path,
            data: bytes,
            mode: int | None,
            **options: object,
        ) -> None:
            if path.name == "second.txt":
                raise OSError("simulated write failure")
            real_replace(path, data, mode, **options)

        with patch("src.tools._atomic_replace_bytes", side_effect=fail_second):
            result = await ApplyPatchTool().execute({"patch": patch_text}, self.context)

        self.assertFalse(result.success)
        self.assertIn("rolled back", result.error or "")
        self.assertEqual(first.read_text(encoding="utf-8"), "one\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "two\n")

    async def test_parent_symlink_swap_is_rejected_at_atomic_commit_boundary(self) -> None:
        parent = self.workspace / "safe"
        parent.mkdir()
        target = parent / "target.txt"
        target.write_text("inside\n", encoding="utf-8")
        moved = self.workspace / "moved"
        parent.rename(moved)
        outside_parent = self.container / "outside-dir"
        outside_parent.mkdir()
        outside_target = outside_parent / "target.txt"
        outside_target.write_text("outside\n", encoding="utf-8")
        os.symlink(outside_parent, parent)

        with self.assertRaisesRegex(
            tools_module.WorkspaceToolError,
            "symbolic link",
        ):
            tools_module._atomic_replace_bytes(
                target,
                b"escaped\n",
                0o644,
                root=self.workspace,
                expected=b"inside\n",
            )

        self.assertEqual(outside_target.read_text(encoding="utf-8"), "outside\n")
        self.assertEqual((moved / "target.txt").read_text(encoding="utf-8"), "inside\n")

    async def test_rollback_does_not_overwrite_a_concurrent_user_change(self) -> None:
        first = self.workspace / "first.txt"
        second = self.workspace / "second.txt"
        first.write_text("one\n", encoding="utf-8")
        second.write_text("two\n", encoding="utf-8")
        patch_text = "\n".join(
            (
                unified_patch("first.txt", "one\n", "ONE\n"),
                unified_patch("second.txt", "two\n", "TWO\n"),
            )
        )
        real_replace = tools_module._atomic_replace_bytes

        def fail_after_concurrent_change(
            path: Path,
            data: bytes,
            mode: int | None,
            **options: object,
        ) -> None:
            if path.name == "second.txt":
                first.write_text("concurrent user edit\n", encoding="utf-8")
                raise OSError("simulated second write failure")
            real_replace(path, data, mode, **options)

        with patch(
            "src.tools._atomic_replace_bytes",
            side_effect=fail_after_concurrent_change,
        ):
            result = await ApplyPatchTool().execute({"patch": patch_text}, self.context)

        self.assertFalse(result.success)
        self.assertIn("rollback was incomplete", result.error or "")
        self.assertEqual(first.read_text(encoding="utf-8"), "concurrent user edit\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "two\n")

    async def test_edit_limits_are_enforced(self) -> None:
        (self.workspace / "one.txt").write_text("one\n", encoding="utf-8")
        (self.workspace / "two.txt").write_text("two\n", encoding="utf-8")
        two_files = "\n".join(
            (
                unified_patch("one.txt", "one\n", "ONE\n"),
                unified_patch("two.txt", "two\n", "TWO\n"),
            )
        )
        too_many = await ApplyPatchTool(max_files=1).execute(
            {"patch": two_files}, self.context
        )
        too_large = await CreateFileTool(max_total_bytes=3).execute(
            {"path": "large.txt", "content": "four"}, self.context
        )

        self.assertIn("file limit", too_many.error or "")
        self.assertIn("byte limit", too_large.error or "")
        self.assertEqual((self.workspace / "one.txt").read_text(encoding="utf-8"), "one\n")
        self.assertFalse((self.workspace / "large.txt").exists())

    async def test_registry_and_permission_policies_expose_write_boundary(self) -> None:
        registry = workspace_tool_registry()
        self.assertEqual(registry.get("apply_patch").operation, Operation.WRITE)
        self.assertEqual(registry.get("create_file").operation, Operation.WRITE)
        request = PermissionRequest(Operation.WRITE, "apply_patch")
        self.assertFalse(ReadOnlyPermissionPolicy().decide(request).allowed)

        approvals: list[PermissionRequest] = []
        policy = InteractivePermissionPolicy(
            lambda pending: approvals.append(pending) or True
        )
        self.assertTrue(policy.decide(request).allowed)
        self.assertEqual(approvals, [request])


if __name__ == "__main__":
    unittest.main()
