from __future__ import annotations

import asyncio
import difflib
import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tools import ToolContext, ToolResult

from .permissions import Operation


_MAX_CAPTURE_BYTES = 2_000_000
_MAX_REPORT_BYTES = 500_000


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    exists: bool
    digest: str | None
    content: bytes | None


@dataclass(frozen=True)
class GitRunSummary:
    is_repository: bool
    baseline_head: str | None = None
    final_head: str | None = None
    baseline_status: str = ""
    baseline_diff: str = ""
    preexisting_files: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    agent_only_files: tuple[str, ...] = ()
    overlapping_files: tuple[str, ...] = ()
    new_files: tuple[str, ...] = ()
    removed_files: tuple[str, ...] = ()
    diff: str = ""
    diff_truncated: bool = False


async def _git(
    root: Path, *args: str, max_bytes: int = _MAX_CAPTURE_BYTES
) -> tuple[int, bytes, bytes, bool]:
    process = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    truncated = len(stdout) > max_bytes or len(stderr) > max_bytes
    return (
        process.returncode or 0,
        stdout[:max_bytes],
        stderr[:max_bytes],
        truncated,
    )


def _decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parse_porcelain_z(payload: bytes) -> set[str]:
    entries = payload.split(b"\0")
    paths: set[str] = set()
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        text = _decode(entry)
        if len(text) < 4:
            continue
        status = text[:2]
        path = text[3:]
        paths.add(path)
        if ("R" in status or "C" in status) and index < len(entries):
            renamed_from = _decode(entries[index])
            index += 1
            if renamed_from:
                paths.add(renamed_from)
    return paths


def _snapshot_worktree(root: Path, path: str) -> FileSnapshot:
    candidate = root / path
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return FileSnapshot(path=path, exists=False, digest=None, content=None)
    if not candidate.is_file():
        return FileSnapshot(path=path, exists=False, digest=None, content=None)
    try:
        content = candidate.read_bytes()
    except OSError:
        return FileSnapshot(path=path, exists=True, digest=None, content=None)
    retained = content if len(content) <= _MAX_CAPTURE_BYTES else None
    return FileSnapshot(path=path, exists=True, digest=_digest(content), content=retained)


def _snapshot_from_bytes(path: str, content: bytes | None) -> FileSnapshot:
    if content is None:
        return FileSnapshot(path=path, exists=False, digest=None, content=None)
    retained = content if len(content) <= _MAX_CAPTURE_BYTES else None
    return FileSnapshot(path=path, exists=True, digest=_digest(content), content=retained)


def _same(left: FileSnapshot, right: FileSnapshot) -> bool:
    return (
        left.exists == right.exists
        and left.digest == right.digest
        and left.digest is not None
    ) or (not left.exists and not right.exists)


class GitRunTracker:
    """Read-only task baseline used to attribute only this run's worktree delta."""

    def __init__(
        self,
        root: Path,
        *,
        is_repository: bool,
        baseline_head: str | None = None,
        baseline_status: str = "",
        baseline_diff: str = "",
        baseline_paths: set[str] | None = None,
        baseline_snapshots: dict[str, FileSnapshot] | None = None,
    ) -> None:
        self.root = root
        self.is_repository = is_repository
        self.baseline_head = baseline_head
        self.baseline_status = baseline_status
        self.baseline_diff = baseline_diff
        self.baseline_paths = baseline_paths or set()
        self.baseline_snapshots = baseline_snapshots or {}
        self._agent_paths: set[str] = set()

    @classmethod
    async def capture(cls, workspace: Path) -> GitRunTracker:
        root = workspace.resolve(strict=True)
        code, stdout, _, _ = await _git(root, "rev-parse", "--is-inside-work-tree")
        if code != 0 or _decode(stdout).strip() != "true":
            return cls(root, is_repository=False)

        _, head_bytes, _, _ = await _git(root, "rev-parse", "HEAD")
        head = _decode(head_bytes).strip() or None
        _, status_bytes, _, status_truncated = await _git(
            root, "status", "--short", "--branch", "--untracked-files=normal"
        )
        _, porcelain, _, _ = await _git(
            root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
        )
        _, unstaged, _, unstaged_truncated = await _git(
            root, "diff", "--no-ext-diff", "--binary"
        )
        _, staged, _, staged_truncated = await _git(
            root, "diff", "--cached", "--no-ext-diff", "--binary"
        )
        paths = _parse_porcelain_z(porcelain)
        snapshots = {path: _snapshot_worktree(root, path) for path in paths}
        status = _decode(status_bytes)
        if status_truncated:
            status += "\n[status truncated]"
        baseline_diff = _decode(unstaged)
        if staged:
            baseline_diff += "\n[staged]\n" + _decode(staged)
        if unstaged_truncated or staged_truncated:
            baseline_diff += "\n[baseline diff truncated]"
        return cls(
            root,
            is_repository=True,
            baseline_head=head,
            baseline_status=status,
            baseline_diff=baseline_diff,
            baseline_paths=paths,
            baseline_snapshots=snapshots,
        )

    def mark_agent_paths(self, paths: list[str] | tuple[str, ...]) -> None:
        for raw in paths:
            candidate = (self.root / raw).resolve(strict=False)
            try:
                relative = candidate.relative_to(self.root)
            except ValueError:
                continue
            self._agent_paths.add(relative.as_posix())

    @property
    def agent_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._agent_paths))

    def baseline_prompt(self) -> str:
        if not self.is_repository:
            return "Git baseline: this workspace is not a Git repository."
        if not self.baseline_paths:
            return "Git baseline: the worktree was clean when this task started."
        paths = ", ".join(sorted(self.baseline_paths)[:40])
        suffix = " …" if len(self.baseline_paths) > 40 else ""
        return (
            "Git baseline: the following paths already had user changes before this "
            f"task: {paths}{suffix}. Do not restore them, stage them, or attribute "
            "unchanged baseline content to this run."
        )

    async def _head_snapshot(self, path: str) -> FileSnapshot:
        if not self.baseline_head:
            return FileSnapshot(path=path, exists=False, digest=None, content=None)
        code, stdout, _, truncated = await _git(
            self.root,
            "show",
            f"{self.baseline_head}:{path}",
            max_bytes=_MAX_CAPTURE_BYTES + 1,
        )
        if code != 0:
            return FileSnapshot(path=path, exists=False, digest=None, content=None)
        if truncated:
            return FileSnapshot(
                path=path, exists=True, digest=_digest(stdout), content=None
            )
        return _snapshot_from_bytes(path, stdout)

    async def finish(self) -> GitRunSummary:
        if not self.is_repository:
            return GitRunSummary(is_repository=False)

        _, final_head_bytes, _, _ = await _git(self.root, "rev-parse", "HEAD")
        final_head = _decode(final_head_bytes).strip() or None
        _, porcelain, _, _ = await _git(
            self.root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
        )
        final_paths = _parse_porcelain_z(porcelain)
        candidates = set(self.baseline_paths) | final_paths | self._agent_paths
        if self.baseline_head and final_head and self.baseline_head != final_head:
            _, committed, _, _ = await _git(
                self.root,
                "diff",
                "--name-only",
                "-z",
                self.baseline_head,
                final_head,
            )
            candidates.update(
                _decode(part) for part in committed.split(b"\0") if part
            )

        baseline: dict[str, FileSnapshot] = {}
        final: dict[str, FileSnapshot] = {}
        for path in sorted(candidates):
            baseline[path] = self.baseline_snapshots.get(path) or await self._head_snapshot(
                path
            )
            final[path] = _snapshot_worktree(self.root, path)

        changed = sorted(path for path in candidates if not _same(baseline[path], final[path]))
        overlapping = sorted(set(changed) & self.baseline_paths)
        agent_only = sorted(
            path
            for path in changed
            if path in self._agent_paths and path not in self.baseline_paths
        )
        new_files = sorted(
            path for path in changed if not baseline[path].exists and final[path].exists
        )
        removed_files = sorted(
            path for path in changed if baseline[path].exists and not final[path].exists
        )
        diff, truncated = _build_incremental_diff(baseline, final, changed)
        return GitRunSummary(
            is_repository=True,
            baseline_head=self.baseline_head,
            final_head=final_head,
            baseline_status=self.baseline_status,
            baseline_diff=self.baseline_diff,
            preexisting_files=tuple(sorted(self.baseline_paths)),
            changed_files=tuple(changed),
            agent_only_files=tuple(agent_only),
            overlapping_files=tuple(overlapping),
            new_files=tuple(new_files),
            removed_files=tuple(removed_files),
            diff=diff,
            diff_truncated=truncated,
        )

    async def validate_stage_paths(self, paths: list[str]) -> tuple[bool, str]:
        if not self.is_repository:
            return False, "The workspace is not a Git repository."
        if not paths:
            return False, "At least one explicit file path is required."
        normalized: set[str] = set()
        for path in paths:
            requested = Path(path)
            if (
                not path
                or path in {".", ".."}
                or path.startswith(":")
                or requested.is_absolute()
                or ".." in requested.parts
                or any(character in path for character in "*?[]")
            ):
                return False, f"Unsafe or broad Git pathspec: {path!r}."
            candidate = (self.root / path).resolve(strict=False)
            try:
                relative = candidate.relative_to(self.root).as_posix()
            except ValueError:
                return False, f"Path is outside the workspace: {path!r}."
            if candidate.is_dir():
                return False, f"Directories are not accepted; name files explicitly: {path!r}."
            normalized.add(relative)
        baseline_overlap = sorted(normalized & self.baseline_paths)
        if baseline_overlap:
            return (
                False,
                "Refusing to stage paths that already contained user changes at task "
                f"start: {', '.join(baseline_overlap)}.",
            )
        unowned = sorted(normalized - self._agent_paths)
        if unowned:
            return (
                False,
                "Refusing to stage paths not recorded as modified by this agent run: "
                f"{', '.join(unowned)}.",
            )
        summary = await self.finish()
        unchanged = sorted(normalized - set(summary.changed_files))
        if unchanged:
            return False, f"No task-local changes remain for: {', '.join(unchanged)}."
        return True, ""

    async def validate_commit(self) -> tuple[bool, str, tuple[str, ...]]:
        if not self.is_repository:
            return False, "The workspace is not a Git repository.", ()
        code, staged_bytes, stderr, _ = await _git(
            self.root, "diff", "--cached", "--name-only", "-z"
        )
        if code != 0:
            return False, _decode(stderr).strip() or "Unable to inspect staged files.", ()
        staged = tuple(sorted(_decode(part) for part in staged_bytes.split(b"\0") if part))
        if not staged:
            return False, "There are no staged changes to commit.", ()
        baseline_overlap = sorted(set(staged) & self.baseline_paths)
        if baseline_overlap:
            return (
                False,
                "Refusing to commit staged paths that contained user changes at task "
                f"start: {', '.join(baseline_overlap)}.",
                staged,
            )
        unowned = sorted(set(staged) - self._agent_paths)
        if unowned:
            return (
                False,
                "Refusing to commit staged paths not owned by this agent run: "
                f"{', '.join(unowned)}.",
                staged,
            )
        return True, "", staged


def _build_incremental_diff(
    baseline: dict[str, FileSnapshot],
    final: dict[str, FileSnapshot],
    changed: list[str],
) -> tuple[str, bool]:
    chunks: list[str] = []
    size = 0
    truncated = False
    for path in changed:
        before = baseline[path]
        after = final[path]
        if before.content is None and before.exists or after.content is None and after.exists:
            chunk = f"Binary or oversized file changed: {path}\n"
        else:
            try:
                before_text = (before.content or b"").decode("utf-8")
                after_text = (after.content or b"").decode("utf-8")
            except UnicodeDecodeError:
                chunk = f"Binary file changed: {path}\n"
            else:
                chunk = "".join(
                    difflib.unified_diff(
                        before_text.splitlines(keepends=True),
                        after_text.splitlines(keepends=True),
                        fromfile=f"a/{path}",
                        tofile=f"b/{path}",
                    )
                )
        encoded = chunk.encode("utf-8")
        remaining = _MAX_REPORT_BYTES - size
        if len(encoded) > remaining:
            chunks.append(encoded[: max(0, remaining)].decode("utf-8", errors="ignore"))
            truncated = True
            break
        chunks.append(chunk)
        size += len(encoded)
    if truncated:
        chunks.append("\n[incremental diff truncated]\n")
    return "".join(chunks), truncated


class GitDiffCheckTool:
    name = "git_diff_check"
    description = "Check the current Git diff for whitespace errors without modifying it."
    operation = Operation.READ
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "staged": {
                "type": "boolean",
                "default": False,
                "description": "Check staged changes instead of unstaged changes.",
            }
        },
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        from .tools import ToolResult, validate_tool_arguments

        validate_tool_arguments(arguments, self.parameters)
        root = Path(context.working_directory).resolve(strict=True)
        repository_code, _, _, _ = await _git(
            root, "rev-parse", "--is-inside-work-tree"
        )
        if repository_code != 0:
            return ToolResult(success=False, error="workspace is not a Git repository")
        command = ["diff", "--check"]
        if arguments.get("staged", False):
            command.insert(1, "--cached")
        code, stdout, stderr, truncated = await _git(root, *command)
        content = _decode(stdout or stderr)
        return ToolResult(
            success=True,
            content=content or ("No whitespace errors." if code == 0 else "Diff check failed."),
            metadata={
                "passed": code == 0,
                "exit_code": code,
                "staged": bool(arguments.get("staged", False)),
                "truncated": truncated,
            },
        )


class GitAddTool:
    name = "git_add"
    description = (
        "Stage explicit task-local files after approval. Baseline user changes are refused."
    )
    operation = Operation.WRITE
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "uniqueItems": True,
            }
        },
        "required": ["paths"],
        "additionalProperties": False,
    }

    async def permission_request(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> Any:
        from .permissions import PermissionLevel, PermissionRequest
        from .tools import validate_tool_arguments

        validate_tool_arguments(arguments, self.parameters)
        tracker = context.git_tracker
        paths = list(arguments["paths"])
        if not isinstance(tracker, GitRunTracker):
            return PermissionRequest(
                operation=Operation.WRITE,
                target="git add",
                level=PermissionLevel.DENY,
                command=shlex.join(("git", "add", "--", *paths)),
                cwd=str(context.working_directory),
                risk_reason="No task Git baseline is available; safe attribution is impossible.",
            )
        allowed, error = await tracker.validate_stage_paths(paths)
        level = PermissionLevel.ASK if allowed else PermissionLevel.DENY
        summary = await tracker.finish()
        return PermissionRequest(
            operation=Operation.WRITE,
            target="git add: " + ", ".join(paths),
            level=level,
            command=shlex.join(("git", "add", "--", *paths)),
            cwd=str(context.working_directory),
            risk_reason=error
            or "Stages task-local changes in Git; this changes the repository index.",
            preview=summary.diff,
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        from .tools import ToolResult, validate_tool_arguments

        validate_tool_arguments(arguments, self.parameters)
        tracker = context.git_tracker
        paths = list(arguments["paths"])
        if not isinstance(tracker, GitRunTracker):
            return ToolResult(success=False, error="No task Git baseline is available.")
        allowed, error = await tracker.validate_stage_paths(paths)
        if not allowed:
            return ToolResult(success=False, error=error)
        if not context.permission_granted:
            return ToolResult(success=False, error="Approval is required before staging files.")
        code, stdout, stderr, truncated = await _git(
            Path(context.working_directory), "add", "--", *paths
        )
        return ToolResult(
            success=code == 0,
            content=_decode(stdout),
            error=None if code == 0 else _decode(stderr).strip(),
            metadata={"paths": paths, "exit_code": code, "truncated": truncated},
        )


class GitCommitTool:
    name = "git_commit"
    description = (
        "Commit already staged task-local files after approval. It never pushes."
    )
    operation = Operation.WRITE
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "minLength": 1, "maxLength": 500}
        },
        "required": ["message"],
        "additionalProperties": False,
    }

    async def permission_request(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> Any:
        from .permissions import PermissionLevel, PermissionRequest
        from .tools import validate_tool_arguments

        validate_tool_arguments(arguments, self.parameters)
        tracker = context.git_tracker
        message = arguments["message"].strip()
        if not isinstance(tracker, GitRunTracker):
            return PermissionRequest(
                operation=Operation.WRITE,
                target="git commit",
                level=PermissionLevel.DENY,
                command=shlex.join(("git", "commit", "--only", "-m", message)),
                cwd=str(context.working_directory),
                risk_reason="No task Git baseline is available; safe attribution is impossible.",
            )
        allowed, error, staged = await tracker.validate_commit()
        _, preview, _, _ = await _git(
            tracker.root, "diff", "--cached", "--no-ext-diff", "--binary"
        )
        return PermissionRequest(
            operation=Operation.WRITE,
            target="git commit",
            level=PermissionLevel.ASK if allowed else PermissionLevel.DENY,
            command=shlex.join(
                ("git", "commit", "--only", "-m", message, "--", *staged)
            ),
            cwd=str(context.working_directory),
            risk_reason=error
            or (
                "Creates a local Git commit for "
                f"{len(staged)} task-local file(s); Git hooks may run. No push is performed."
            ),
            preview=_decode(preview),
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        from .tools import ToolResult, validate_tool_arguments

        validate_tool_arguments(arguments, self.parameters)
        tracker = context.git_tracker
        message = arguments["message"].strip()
        if not isinstance(tracker, GitRunTracker):
            return ToolResult(success=False, error="No task Git baseline is available.")
        allowed, error, staged = await tracker.validate_commit()
        if not allowed:
            return ToolResult(success=False, error=error)
        if not context.permission_granted:
            return ToolResult(success=False, error="Approval is required before committing.")
        code, stdout, stderr, truncated = await _git(
            tracker.root, "commit", "--only", "-m", message, "--", *staged
        )
        if code != 0:
            return ToolResult(
                success=False,
                content=_decode(stdout),
                error=_decode(stderr).strip() or "git commit failed.",
                metadata={"paths": list(staged), "exit_code": code, "truncated": truncated},
            )
        _, head, _, _ = await _git(tracker.root, "rev-parse", "HEAD")
        return ToolResult(
            success=True,
            content=_decode(stdout),
            metadata={
                "paths": list(staged),
                "commit": _decode(head).strip(),
                "exit_code": code,
                "truncated": truncated,
            },
        )
