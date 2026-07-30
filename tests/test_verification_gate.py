from __future__ import annotations

import unittest
from pathlib import Path

from src.git_runtime import GitRunSummary
from src.harness.models import FailureCategory, TaskType, VerificationStatus
from src.harness.verification import VerificationGate
from src.models import ToolCall
from src.permissions import Operation
from src.quality import CompletionReport, RunQualityTracker, TestResult as QualityTestResult
from src.tools import ToolResult


def report(
    *,
    incomplete: tuple[str, ...] = (),
    tests: tuple[QualityTestResult, ...] = (),
) -> CompletionReport:
    return CompletionReport(
        completed=("evidence collected",),
        incomplete=incomplete,
        tests=tests,
        risks_and_follow_up=(),
        suggested_tests=(),
        git=GitRunSummary(is_repository=False),
    )


class VerificationGateTests(unittest.TestCase):
    def test_report_incomplete_blocks_completion(self) -> None:
        result = VerificationGate().verify(
            task_type=TaskType.INFORMATIONAL,
            candidate="answer",
            report=report(incomplete=("work remains",)),
            quality=RunQualityTracker(Path.cwd()),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, FailureCategory.REQUIREMENT_NOT_MET)

    def test_modification_without_required_tests_does_not_pass(self) -> None:
        quality = RunQualityTracker(Path.cwd())
        quality.observe(
            ToolCall("write-1", "create_file", {"path": "module.py"}),
            ToolResult("created", metadata={"files": ["module.py"]}),
            operation=Operation.WRITE,
        )
        result = VerificationGate().verify(
            task_type=TaskType.MODIFICATION,
            candidate="implemented",
            report=report(),
            quality=quality,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failure_category, FailureCategory.TEST_FAILURE)

    def test_environment_missing_is_blocked_not_code_failure(self) -> None:
        failed_test = QualityTestResult(
            command="python -m unittest",
            passed=False,
            exit_code=1,
            detail="ModuleNotFoundError: missing dependency",
        )
        quality = RunQualityTracker(Path.cwd())
        quality.tests.append(failed_test)
        result = VerificationGate().verify(
            task_type=TaskType.EXECUTION,
            candidate="dependency missing",
            report=report(
                incomplete=("Verification failed",),
                tests=(failed_test,),
            ),
            quality=quality,
        )
        self.assertEqual(result.status, VerificationStatus.BLOCKED)
        self.assertEqual(result.failure_category, FailureCategory.ENVIRONMENT_MISSING)

    def test_inspection_requires_actual_read_evidence(self) -> None:
        quality = RunQualityTracker(Path.cwd())
        failed = VerificationGate().verify(
            task_type=TaskType.INSPECTION,
            candidate="looks correct",
            report=report(),
            quality=quality,
        )
        self.assertFalse(failed.passed)

        quality.observe(
            ToolCall("read-1", "read_file", {"path": "README.md"}),
            ToolResult("content", metadata={"path": "README.md"}),
            operation=Operation.READ,
        )
        passed = VerificationGate().verify(
            task_type=TaskType.INSPECTION,
            candidate="README.md documents the behavior.",
            report=report(),
            quality=quality,
        )
        self.assertTrue(passed.passed)


if __name__ == "__main__":
    unittest.main()
