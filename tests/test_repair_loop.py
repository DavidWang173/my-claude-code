from __future__ import annotations

import unittest

from src.harness.models import (
    FailureCategory,
    VerificationResult,
    VerificationStatus,
)
from src.harness.repair import RepairController, RepairPolicy


def failed_result(
    category: FailureCategory = FailureCategory.TEST_FAILURE,
    summary: str = "test failed",
) -> VerificationResult:
    return VerificationResult(
        passed=False,
        status=VerificationStatus.FAILED,
        checks=(),
        evidence=(),
        failure_category=category,
        repairable=True,
        summary=summary,
    )


class RepairControllerTests(unittest.TestCase):
    def test_first_failure_can_retry_but_same_failure_stops_mechanical_retry(self) -> None:
        controller = RepairController(RepairPolicy(max_repair_attempts=3, max_step_retries=3))
        first = controller.diagnose(
            failed_result(),
            affected_step="verify",
            repair_attempts=0,
            step_retries=0,
        )
        second = controller.diagnose(
            failed_result(),
            affected_step="verify",
            repair_attempts=1,
            step_retries=1,
        )
        self.assertTrue(first.retryable)
        self.assertFalse(second.retryable)
        self.assertIn("same failure", second.recommended_action)

    def test_global_and_step_budgets_prevent_infinite_loop(self) -> None:
        controller = RepairController(RepairPolicy(max_repair_attempts=2, max_step_retries=2))
        diagnosis = controller.diagnose(
            failed_result(summary="different"),
            affected_step="verify",
            repair_attempts=2,
            step_retries=0,
        )
        self.assertFalse(diagnosis.retryable)
        self.assertIn("budget", diagnosis.recommended_action)

    def test_permission_denial_requires_human_and_is_not_retried(self) -> None:
        diagnosis = RepairController().diagnose(
            failed_result(FailureCategory.PERMISSION_DENIED, "permission denied"),
            affected_step="implement",
            repair_attempts=0,
            step_retries=0,
        )
        self.assertTrue(diagnosis.requires_human)
        self.assertFalse(diagnosis.retryable)


if __name__ == "__main__":
    unittest.main()
