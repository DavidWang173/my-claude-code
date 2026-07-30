"""Task-aware completion gate built on the existing quality report."""

from __future__ import annotations

import re
from pathlib import Path

from ..permissions import Operation
from ..quality import CompletionReport, RunQualityTracker, ToolExecutionRecord
from .models import (
    ExecutionPlan,
    FailureCategory,
    PlanStepStatus,
    TaskType,
    VerificationCheck,
    VerificationResult,
    VerificationStatus,
)

_PLACEHOLDER = re.compile(r"\[(?:todo|tbd)\]|<todo>|coming soon", re.I)
_MODIFICATION_CLAIM = re.compile(
    r"\b(?:i|we) (?:changed|modified|created|updated|fixed|implemented)\b|"
    r"(?:已经|已)(?:修改|创建|更新|修复|实现)",
    re.I,
)
_ENVIRONMENT_ERROR = re.compile(
    r"module not found|modulenotfounderror|command not found|no such file or directory|"
    r"missing dependency|cannot import|not installed",
    re.I,
)
_BUILD_COMMAND = re.compile(r"\b(build|compile|mypy|tsc)\b", re.I)
_CODE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
}


class VerificationGate:
    """Turn explicit evidence into the only authoritative completion decision."""

    def verify(
        self,
        *,
        task_type: TaskType,
        candidate: str,
        report: CompletionReport,
        quality: RunQualityTracker,
        plan: ExecutionPlan | None = None,
        no_tests_requested: bool = False,
        conflict_markers: tuple[str, ...] = (),
    ) -> VerificationResult:
        checks: list[VerificationCheck] = [
            self._candidate_check(candidate),
            self._report_check(report),
        ]
        if plan is not None:
            checks.append(self._plan_check(plan))

        if task_type is TaskType.INFORMATIONAL:
            checks.extend(self._informational_checks(candidate, quality))
        elif task_type is TaskType.INSPECTION:
            checks.extend(self._inspection_checks(quality))
        elif task_type is TaskType.MODIFICATION:
            checks.extend(
                self._modification_checks(
                    report,
                    quality,
                    no_tests_requested=no_tests_requested,
                    conflict_markers=conflict_markers,
                )
            )
        elif task_type is TaskType.EXECUTION:
            checks.extend(self._execution_checks(quality.tool_executions))

        failed = [check for check in checks if check.required and not check.passed]
        evidence = tuple(
            item
            for check in checks
            for item in check.evidence
            if check.executed
        )
        if not failed:
            return VerificationResult(
                passed=True,
                status=VerificationStatus.PASSED,
                checks=tuple(checks),
                evidence=evidence,
                failure_category=None,
                repairable=False,
                summary="All required verification checks passed.",
            )

        category, status, repairable = self._classify_failure(failed, report, quality)
        summary = "; ".join(
            check.failure_reason or f"{check.name} failed" for check in failed
        )
        return VerificationResult(
            passed=False,
            status=status,
            checks=tuple(checks),
            evidence=evidence,
            failure_category=category,
            repairable=repairable,
            summary=summary,
        )

    evaluate = verify

    def _candidate_check(self, candidate: str) -> VerificationCheck:
        present = bool(candidate.strip())
        no_placeholder = not _PLACEHOLDER.search(candidate)
        passed = present and no_placeholder
        reason = None
        if not present:
            reason = "The model did not produce a candidate response."
        elif not no_placeholder:
            reason = "The candidate contains an obvious unfinished placeholder."
        return VerificationCheck(
            name="candidate_complete",
            required=True,
            executed=True,
            passed=passed,
            evidence=("A non-empty candidate response was produced.",) if present else (),
            failure_reason=reason,
        )

    def _report_check(self, report: CompletionReport) -> VerificationCheck:
        return VerificationCheck(
            name="completion_report",
            required=True,
            executed=True,
            passed=not report.incomplete,
            evidence=tuple(report.completed),
            failure_reason=(
                "Completion report remains incomplete: " + "; ".join(report.incomplete)
                if report.incomplete
                else None
            ),
        )

    def _plan_check(self, plan: ExecutionPlan) -> VerificationCheck:
        incomplete = [
            step.id
            for step in plan.steps
            if step.status not in {PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED}
        ]
        return VerificationCheck(
            name="plan_steps",
            required=True,
            executed=True,
            passed=not incomplete,
            evidence=(f"Plan revision {plan.revision} evaluated.",),
            failure_reason=(
                "Required plan steps are incomplete: " + ", ".join(incomplete)
                if incomplete
                else None
            ),
        )

    def _informational_checks(
        self, candidate: str, quality: RunQualityTracker
    ) -> list[VerificationCheck]:
        claims_change = bool(_MODIFICATION_CLAIM.search(candidate))
        recorded_change = bool(quality.expected_files)
        return [
            VerificationCheck(
                name="no_false_modification_claim",
                required=True,
                executed=True,
                passed=not claims_change or recorded_change,
                evidence=(
                    ("Recorded file modifications support the response.",)
                    if recorded_change
                    else ()
                ),
                failure_reason=(
                    "The response claims file changes, but no successful modification was recorded."
                    if claims_change and not recorded_change
                    else None
                ),
            )
        ]

    def _inspection_checks(
        self, quality: RunQualityTracker
    ) -> list[VerificationCheck]:
        return [
            VerificationCheck(
                name="relevant_files_read",
                required=True,
                executed=bool(quality.read_evidence),
                passed=bool(quality.read_evidence),
                evidence=tuple(quality.read_evidence),
                failure_reason=(
                    None
                    if quality.read_evidence
                    else "No successful file or repository inspection was recorded."
                ),
            ),
            VerificationCheck(
                name="conclusions_have_evidence",
                required=True,
                executed=bool(quality.read_evidence),
                passed=bool(quality.read_evidence),
                evidence=tuple(quality.read_evidence),
                failure_reason=(
                    None
                    if quality.read_evidence
                    else "Inspection conclusions have no file or code-location evidence."
                ),
            ),
        ]

    def _modification_checks(
        self,
        report: CompletionReport,
        quality: RunQualityTracker,
        *,
        no_tests_requested: bool,
        conflict_markers: tuple[str, ...],
    ) -> list[VerificationCheck]:
        changed = bool(quality.expected_files) and not any(
            "Expected modifications are absent" in item for item in report.incomplete
        )
        unexplained = tuple(
            item
            for item in report.risks_and_follow_up
            if item.startswith("Unexplained new files")
        )
        diff_check = quality.diff_checks[-1] if quality.diff_checks else None
        if not report.git.is_repository:
            diff_executed = True
            diff_passed = not conflict_markers
            diff_evidence = (
                "Workspace is not a Git repository; used file-level conflict-marker validation.",
            )
            diff_reason = (
                "File-level alternative validation found conflict markers."
                if conflict_markers
                else None
            )
        else:
            diff_executed = diff_check is not None
            diff_passed = bool(diff_check and diff_check.passed)
            diff_evidence = (
                (f"{diff_check.command} exit={diff_check.exit_code}",)
                if diff_check is not None
                else ()
            )
            diff_reason = (
                "git diff --check was not executed."
                if diff_check is None
                else (
                    "git diff --check failed."
                    if not diff_check.passed
                    else None
                )
            )

        suggested_tests = {
            command
            for command in quality.suggested_tests
            if command != "git_diff_check"
        }
        code_changed = any(
            Path(path).suffix.casefold() in _CODE_SUFFIXES
            for path in quality.expected_files
        )
        tests_required = (bool(suggested_tests) or code_changed) and not no_tests_requested
        tests_executed = bool(quality.tests)
        tests_passed = tests_executed and all(result.passed for result in quality.tests)
        if not tests_required:
            tests_passed = True
        tests_reason = None
        if tests_required and not tests_executed:
            tests_reason = "Relevant tests were required but not executed."
        elif tests_required and not tests_passed:
            tests_reason = "One or more relevant tests failed."
        elif no_tests_requested:
            tests_reason = None

        return [
            VerificationCheck(
                name="expected_files_modified",
                required=True,
                executed=True,
                passed=changed,
                evidence=tuple(sorted(quality.expected_files)),
                failure_reason=(
                    None
                    if changed
                    else "The expected scoped file modification is absent."
                ),
            ),
            VerificationCheck(
                name="modification_scope",
                required=True,
                executed=True,
                passed=not unexplained,
                evidence=tuple(report.git.agent_only_files),
                failure_reason=(
                    "; ".join(unexplained) if unexplained else None
                ),
            ),
            VerificationCheck(
                name="git_diff_check",
                required=True,
                executed=diff_executed,
                passed=diff_passed,
                command="git diff --check" if report.git.is_repository else None,
                exit_code=diff_check.exit_code if diff_check is not None else None,
                evidence=diff_evidence,
                failure_reason=diff_reason,
            ),
            VerificationCheck(
                name="conflict_markers",
                required=True,
                executed=True,
                passed=not conflict_markers,
                evidence=tuple(conflict_markers),
                failure_reason=(
                    "Conflict markers remain in: " + ", ".join(conflict_markers)
                    if conflict_markers
                    else None
                ),
            ),
            VerificationCheck(
                name="relevant_tests",
                required=tests_required,
                executed=tests_executed or not tests_required,
                passed=tests_passed,
                command=", ".join(result.command for result in quality.tests) or None,
                evidence=tuple(
                    f"{result.command} exit={result.exit_code}"
                    for result in quality.tests
                )
                + (
                    ("User explicitly requested that tests not be run; alternative checks used.",)
                    if no_tests_requested
                    else ()
                ),
                failure_reason=tests_reason,
            ),
        ]

    def _execution_checks(
        self, records: list[ToolExecutionRecord]
    ) -> list[VerificationCheck]:
        executions = [
            record
            for record in records
            if record.operation in {Operation.EXECUTE, Operation.NETWORK}
            or record.command is not None
        ]
        successful = [
            record
            for record in executions
            if record.success
            and (record.exit_code == 0 if record.command is not None else True)
        ]
        exit_recorded = all(
            record.command is None or record.exit_code is not None for record in executions
        )
        errors_ignored = bool(successful)
        return [
            VerificationCheck(
                name="command_executed",
                required=True,
                executed=bool(executions),
                passed=bool(successful),
                command=successful[-1].command if successful else None,
                exit_code=successful[-1].exit_code if successful else None,
                evidence=tuple(record.evidence for record in successful if record.evidence),
                failure_reason=(
                    None
                    if successful
                    else "No successful goal-relevant command execution was recorded."
                ),
            ),
            VerificationCheck(
                name="exit_code_recorded",
                required=True,
                executed=bool(executions),
                passed=bool(executions) and exit_recorded,
                evidence=tuple(
                    f"{record.command} exit={record.exit_code}"
                    for record in executions
                    if record.command is not None
                ),
                failure_reason=(
                    None
                    if executions and exit_recorded
                    else "An executed command is missing its exit code."
                ),
            ),
            VerificationCheck(
                name="errors_not_ignored",
                required=True,
                executed=bool(executions),
                passed=bool(executions) and errors_ignored,
                failure_reason=(
                    None
                    if executions and errors_ignored
                    else "A command error remains unresolved."
                ),
            ),
        ]

    def _classify_failure(
        self,
        failed: list[VerificationCheck],
        report: CompletionReport,
        quality: RunQualityTracker,
    ) -> tuple[FailureCategory, VerificationStatus, bool]:
        details = " ".join(
            [
                *(check.failure_reason or "" for check in failed),
                *(result.detail for result in report.tests),
                *quality.tool_failures,
            ]
        )
        if _ENVIRONMENT_ERROR.search(details):
            return (
                FailureCategory.ENVIRONMENT_MISSING,
                VerificationStatus.BLOCKED,
                False,
            )
        if "timed out" in details.casefold() or "timeout" in details.casefold():
            return (
                FailureCategory.TIMEOUT,
                VerificationStatus.BLOCKED,
                False,
            )
        if "missing required field" in details.casefold() or "arguments." in details.casefold():
            return (
                FailureCategory.TOOL_ARGUMENT_ERROR,
                VerificationStatus.FAILED,
                True,
            )
        if report.tests and any(not result.passed for result in report.tests):
            failed_commands = " ".join(
                result.command for result in report.tests if not result.passed
            )
            category = (
                FailureCategory.BUILD_FAILURE
                if _BUILD_COMMAND.search(failed_commands)
                else FailureCategory.TEST_FAILURE
            )
            return category, VerificationStatus.FAILED, True
        if any(check.name == "relevant_tests" for check in failed):
            failed_commands = " ".join(result.command for result in report.tests)
            category = (
                FailureCategory.BUILD_FAILURE
                if _BUILD_COMMAND.search(failed_commands)
                else FailureCategory.TEST_FAILURE
            )
            return category, VerificationStatus.FAILED, True
        if any("Permission denied" in failure for failure in quality.tool_failures):
            return (
                FailureCategory.PERMISSION_DENIED,
                VerificationStatus.BLOCKED,
                False,
            )
        if quality.tool_failures:
            return (
                FailureCategory.TOOL_EXECUTION_ERROR,
                VerificationStatus.FAILED,
                True,
            )
        return (
            FailureCategory.REQUIREMENT_NOT_MET,
            VerificationStatus.FAILED,
            True,
        )
