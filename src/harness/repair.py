"""Bounded diagnosis policy for failed verification attempts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    FailureCategory,
    RepairDiagnosis,
    VerificationResult,
)


@dataclass(frozen=True)
class RepairPolicy:
    max_repair_attempts: int = 2
    max_step_retries: int = 2

    def __post_init__(self) -> None:
        if self.max_repair_attempts < 0 or self.max_step_retries < 0:
            raise ValueError("repair budgets cannot be negative")


@dataclass
class RepairController:
    policy: RepairPolicy = field(default_factory=RepairPolicy)
    _last_signature: str | None = field(default=None, init=False)
    _same_error_count: int = field(default=0, init=False)

    def diagnose(
        self,
        result: VerificationResult,
        *,
        affected_step: str | None,
        repair_attempts: int,
        step_retries: int,
    ) -> RepairDiagnosis:
        category = result.failure_category or FailureCategory.UNKNOWN
        signature = f"{category.value}:{result.summary}"
        if signature == self._last_signature:
            self._same_error_count += 1
        else:
            self._last_signature = signature
            self._same_error_count = 1

        budget_available = (
            repair_attempts < self.policy.max_repair_attempts
            and step_retries < self.policy.max_step_retries
        )
        mechanical_loop = self._same_error_count >= 2
        requires_human = (
            category is FailureCategory.PERMISSION_DENIED
            or (
                category is FailureCategory.ENVIRONMENT_MISSING
                and not result.repairable
            )
        )
        retryable = (
            result.repairable
            and budget_available
            and not mechanical_loop
            and not requires_human
        )

        actions = {
            FailureCategory.TOOL_ARGUMENT_ERROR: "Correct the tool arguments and retry once.",
            FailureCategory.TOOL_EXECUTION_ERROR: "Inspect the tool error, reread affected state, and retry safely.",
            FailureCategory.TEST_FAILURE: "Diagnose the failing test, repair the implementation, and rerun verification.",
            FailureCategory.BUILD_FAILURE: "Diagnose the build output, repair the implementation, and rebuild.",
            FailureCategory.REQUIREMENT_NOT_MET: "Address the unmet acceptance criteria and collect evidence.",
            FailureCategory.CONTEXT_STALE: "Reread the affected files before making another change.",
            FailureCategory.PERMISSION_DENIED: "Use a permitted alternative or request explicit user action.",
            FailureCategory.ENVIRONMENT_MISSING: "Record the missing dependency and ask the user to provide or approve it.",
            FailureCategory.TIMEOUT: "Reduce the operation scope or ask the user for a larger time budget.",
            FailureCategory.UNKNOWN: "Reread the evidence and make one bounded corrective attempt.",
        }
        action = actions[category]
        if not budget_available:
            action = "Stop: the repair budget is exhausted."
        elif mechanical_loop:
            action = "Stop: the same failure repeated; do not retry mechanically."

        return RepairDiagnosis(
            failure_category=category,
            likely_cause=result.summary,
            affected_step=affected_step,
            recommended_action=action,
            requires_replan=category is FailureCategory.REQUIREMENT_NOT_MET,
            requires_reread=category
            in {
                FailureCategory.CONTEXT_STALE,
                FailureCategory.TOOL_EXECUTION_ERROR,
                FailureCategory.UNKNOWN,
            },
            requires_human=requires_human,
            retryable=retryable,
        )
