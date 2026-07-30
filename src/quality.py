from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .git_runtime import GitRunSummary
from .models import ToolCall
from .permissions import Operation
from .tools import ToolResult


@dataclass(frozen=True)
class TestResult:
    command: str
    passed: bool
    exit_code: int | None
    detail: str = ""


@dataclass(frozen=True)
class ToolExecutionRecord:
    name: str
    success: bool
    operation: Operation
    command: str | None = None
    exit_code: int | None = None
    evidence: str = ""


@dataclass(frozen=True)
class CompletionReport:
    completed: tuple[str, ...]
    incomplete: tuple[str, ...]
    tests: tuple[TestResult, ...]
    risks_and_follow_up: tuple[str, ...]
    suggested_tests: tuple[str, ...]
    git: GitRunSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "completed": list(self.completed),
            "incomplete": list(self.incomplete),
            "tests": [
                {
                    "command": result.command,
                    "passed": result.passed,
                    "exit_code": result.exit_code,
                    "detail": result.detail,
                }
                for result in self.tests
            ],
            "risks_and_follow_up": list(self.risks_and_follow_up),
            "suggested_tests": list(self.suggested_tests),
            "git": {
                "is_repository": self.git.is_repository,
                "baseline_head": self.git.baseline_head,
                "final_head": self.git.final_head,
                "preexisting_files": list(self.git.preexisting_files),
                "changed_files": list(self.git.changed_files),
                "agent_only_files": list(self.git.agent_only_files),
                "overlapping_files": list(self.git.overlapping_files),
                "new_files": list(self.git.new_files),
                "removed_files": list(self.git.removed_files),
                "diff": self.git.diff,
                "diff_truncated": self.git.diff_truncated,
            },
        }


class RunQualityTracker:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.expected_files: set[str] = set()
        self.tests: list[TestResult] = []
        self.diff_checks: list[TestResult] = []
        self.tool_failures: list[str] = []
        self.tool_executions: list[ToolExecutionRecord] = []
        self.read_evidence: list[str] = []
        self._suggested: set[str] = set()

    def observe(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        operation: Operation | None = None,
    ) -> None:
        metadata = result.metadata
        resolved_operation = operation or _operation_for_tool(call.name)
        raw_exit_code = metadata.get("exit_code")
        exit_code = raw_exit_code if isinstance(raw_exit_code, int) else None
        command = metadata.get("command")
        command_text = command if isinstance(command, str) else None
        self.tool_executions.append(
            ToolExecutionRecord(
                name=call.name,
                success=result.success,
                operation=resolved_operation,
                command=command_text,
                exit_code=exit_code,
                evidence=_bounded_evidence(result.content or result.error or ""),
            )
        )
        if result.success and resolved_operation is Operation.READ:
            target = metadata.get("path")
            if not isinstance(target, str):
                target = call.arguments.get("path")
            rendered_target = target if isinstance(target, str) else call.name
            self.read_evidence.append(f"{call.name}: {rendered_target}")
        if not result.success:
            self.tool_failures.append(f"{call.name}: {result.error or 'failed'}")
        if call.name in {"apply_patch", "create_file", "replace_range"}:
            if result.success and not bool(metadata.get("dry_run", False)):
                files = _string_list(metadata.get("files"))
                self.expected_files.update(files)
                self._suggested.update(suggest_test_commands(self.workspace, files))
        elif call.name == "run_shell":
            command = metadata.get("command")
            if isinstance(command, str) and _is_test_command(command):
                self.tests.append(
                    TestResult(
                        command=command,
                        passed=result.success and exit_code == 0,
                        exit_code=exit_code,
                        detail=(
                            result.error
                            or (result.content if exit_code not in {None, 0} else "")
                        ),
                    )
                )
        elif call.name == "git_diff_check":
            self.diff_checks.append(
                TestResult(
                    command="git diff --check"
                    + (" --cached" if metadata.get("staged") else ""),
                    passed=bool(metadata.get("passed", False)),
                    exit_code=exit_code,
                    detail=result.error or "",
                )
            )

    @property
    def suggested_tests(self) -> tuple[str, ...]:
        return tuple(sorted(self._suggested))

    def build_report(self, git: GitRunSummary) -> CompletionReport:
        completed: list[str] = []
        incomplete: list[str] = []
        risks: list[str] = []

        changed = set(git.changed_files)
        missing = (
            sorted(self.expected_files - changed)
            if git.is_repository
            else sorted(
                path
                for path in self.expected_files
                if not (self.workspace / path).exists()
            )
        )
        if self.expected_files and not missing:
            completed.append(
                "Expected files were modified: " + ", ".join(sorted(self.expected_files))
            )
        elif missing:
            incomplete.append(
                "Expected modifications are absent from the final worktree: "
                + ", ".join(missing)
            )

        if git.agent_only_files:
            completed.append(
                "Task-local Git changes: " + ", ".join(git.agent_only_files)
            )
        elif self.expected_files and not git.is_repository:
            completed.append(
                "Modified files (Git attribution unavailable): "
                + ", ".join(sorted(self.expected_files))
            )
        elif not self.expected_files:
            completed.append("No file modification was required or recorded.")

        all_checks = [*self.tests, *self.diff_checks]
        failures = [item for item in all_checks if not item.passed]
        if failures:
            incomplete.append(
                "Verification failed: " + ", ".join(item.command for item in failures)
            )
        elif all_checks:
            completed.append("Recorded verification completed successfully.")
        elif self.expected_files:
            applicable_tests = {
                command
                for command in self._suggested
                if command != "git_diff_check"
            }
            if applicable_tests or git.is_repository:
                incomplete.append("No appropriate test or diff check was recorded.")
            else:
                completed.append(
                    "No project test or Git diff check was applicable; "
                    "file-level verification is required."
                )

        unexplained_new = sorted(set(git.new_files) - self.expected_files)
        if unexplained_new:
            risks.append(
                "Unexplained new files appeared during the task: "
                + ", ".join(unexplained_new)
            )
        if git.overlapping_files:
            risks.append(
                "These files had user changes at task start and also changed during the "
                "task; they are not classified as agent-only: "
                + ", ".join(git.overlapping_files)
            )
        if self.tool_failures:
            risks.append("Tool failures occurred: " + "; ".join(self.tool_failures))
        if git.diff_truncated:
            risks.append("The task-local diff was truncated in the completion report.")
        if not git.is_repository:
            risks.append("The workspace is not a Git repository; Git attribution is unavailable.")
        if self._suggested and not all_checks:
            risks.append(
                "Suggested verification remains to be run through the Shell permission "
                "system: " + ", ".join(sorted(self._suggested))
            )

        return CompletionReport(
            completed=tuple(completed),
            incomplete=tuple(incomplete),
            tests=tuple(all_checks),
            risks_and_follow_up=tuple(risks),
            suggested_tests=self.suggested_tests,
            git=git,
        )


def suggest_test_commands(workspace: Path, files: list[str]) -> tuple[str, ...]:
    suffixes = {Path(path).suffix.casefold() for path in files}
    suggestions: set[str] = {"git_diff_check"}
    if ".py" in suffixes:
        if (workspace / "pytest.ini").exists() or (workspace / "conftest.py").exists():
            suggestions.add("python -m pytest")
        elif (workspace / "tests").is_dir():
            suggestions.add("python -m unittest discover -s tests -v")
    if suffixes & {".js", ".jsx", ".ts", ".tsx"} and (workspace / "package.json").exists():
        suggestions.add("npm test")
    if (workspace / "Cargo.toml").exists():
        suggestions.add("cargo test")
    if (workspace / "go.mod").exists():
        suggestions.add("go test ./...")
    return tuple(sorted(suggestions))


def _string_list(value: object) -> set[str]:
    if not isinstance(value, (list, tuple)):
        return set()
    return {item for item in value if isinstance(item, str)}


def _is_test_command(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    lowered = [token.casefold() for token in tokens]
    if not lowered:
        return False
    executable = Path(lowered[0]).name
    if executable in {"pytest", "tox", "nox"}:
        return True
    if executable.startswith("python") and any(
        token in {"pytest", "unittest"} for token in lowered
    ):
        return True
    if executable in {"npm", "pnpm", "yarn"} and any(
        token in {"test", "run"} for token in lowered[1:]
    ):
        return True
    return executable in {"cargo", "go"} and "test" in lowered[1:]


def _operation_for_tool(name: str) -> Operation:
    if name in {
        "list_files",
        "read_file",
        "search_text",
        "git_status",
        "git_diff",
        "git_diff_check",
        "inspect",
    }:
        return Operation.READ
    if name in {"apply_patch", "create_file", "replace_range", "git_add", "git_commit"}:
        return Operation.WRITE
    return Operation.EXECUTE


def _bounded_evidence(value: str, limit: int = 500) -> str:
    rendered = value.strip()
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit] + "…"
