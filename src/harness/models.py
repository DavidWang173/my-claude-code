"""Typed, serialisable lifecycle models for the coding-agent harness.

The models deliberately contain summaries and evidence only.  They never
capture model reasoning or duplicate the conversation transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Mapping
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(UTC)


class RunState(str, Enum):
    PREPARED = "prepared"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


ALLOWED_TRANSITIONS: Mapping[RunState, frozenset[RunState]] = {
    RunState.PREPARED: frozenset(
        {RunState.PLANNING, RunState.EXECUTING, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.PLANNING: frozenset(
        {RunState.EXECUTING, RunState.FAILED, RunState.CANCELLED}
    ),
    RunState.EXECUTING: frozenset(
        {
            RunState.VERIFYING,
            RunState.WAITING_APPROVAL,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.VERIFYING: frozenset(
        {
            RunState.COMPLETED,
            RunState.REPAIRING,
            RunState.WAITING_APPROVAL,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.REPAIRING: frozenset(
        {
            RunState.EXECUTING,
            RunState.PLANNING,
            RunState.WAITING_APPROVAL,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.WAITING_APPROVAL: frozenset(
        {
            RunState.EXECUTING,
            RunState.VERIFYING,
            RunState.REPAIRING,
            RunState.FAILED,
            RunState.CANCELLED,
        }
    ),
    RunState.COMPLETED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
}


class InvalidRunStateTransition(RuntimeError):
    """Raised when code attempts a transition outside the lifecycle table."""


class TaskType(str, Enum):
    INFORMATIONAL = "informational"
    INSPECTION = "inspection"
    MODIFICATION = "modification"
    EXECUTION = "execution"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    id: str
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    dependencies: tuple[str, ...] = ()
    expected_output: str = ""
    verification_hint: str = ""
    retry_count: int = 0
    error_summary: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.description:
            raise ValueError("plan step id and description are required")
        if self.retry_count < 0:
            raise ValueError("plan step retry_count cannot be negative")
        if self.id in self.dependencies:
            raise ValueError("plan step cannot depend on itself")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "expected_output": self.expected_output,
            "verification_hint": self.verification_hint,
            "retry_count": self.retry_count,
            "error_summary": self.error_summary,
        }

    @classmethod
    def from_dict(cls, value: object) -> PlanStep:
        item = _mapping(value, "plan step")
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(entry, str) for entry in dependencies
        ):
            raise ValueError("plan step dependencies must be a string list")
        return cls(
            id=_required_text(item.get("id"), "plan step id"),
            description=_required_text(
                item.get("description"), "plan step description"
            ),
            status=PlanStepStatus(item.get("status", PlanStepStatus.PENDING.value)),
            dependencies=tuple(dependencies),
            expected_output=_optional_text(
                item.get("expected_output"), "expected_output"
            ),
            verification_hint=_optional_text(
                item.get("verification_hint"), "verification_hint"
            ),
            retry_count=_non_negative_int(item.get("retry_count", 0), "retry_count"),
            error_summary=_nullable_text(item.get("error_summary"), "error_summary"),
        )


@dataclass
class ExecutionPlan:
    id: str
    goal: str
    task_type: TaskType
    steps: list[PlanStep]
    acceptance_criteria: tuple[str, ...]
    status: PlanStatus = PlanStatus.ACTIVE
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    revision: int = 1

    def __post_init__(self) -> None:
        if not self.id or not self.goal:
            raise ValueError("plan id and goal are required")
        if not self.steps:
            raise ValueError("execution plan requires at least one step")
        if not self.acceptance_criteria:
            raise ValueError("execution plan requires acceptance criteria")
        if self.revision < 1:
            raise ValueError("plan revision must be positive")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("plan timestamps must include a timezone")
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("plan step ids must be unique")
        available: set[str] = set()
        for step in self.steps:
            if any(dependency not in available for dependency in step.dependencies):
                raise ValueError("plan dependencies must reference earlier steps")
            available.add(step.id)

    @classmethod
    def single_step(
        cls,
        *,
        goal: str,
        task_type: TaskType,
        acceptance_criteria: tuple[str, ...],
        description: str | None = None,
    ) -> ExecutionPlan:
        return cls(
            id=uuid4().hex,
            goal=goal,
            task_type=task_type,
            steps=[
                PlanStep(
                    id="step-1",
                    description=description or goal,
                    expected_output="A result satisfying the acceptance criteria.",
                    verification_hint="Verify every required acceptance criterion.",
                )
            ],
            acceptance_criteria=acceptance_criteria,
        )

    def touch(self) -> None:
        self.revision += 1
        self.updated_at = utc_now()

    def update_step(
        self,
        step_id: str,
        status: PlanStepStatus,
        *,
        error_summary: str | None = None,
        increment_retry: bool = False,
    ) -> PlanStep:
        step = self.step(step_id)
        step.status = status
        step.error_summary = error_summary
        if increment_retry:
            step.retry_count += 1
        self.touch()
        return step

    def step(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"unknown plan step: {step_id}")

    def next_incomplete_step(self) -> PlanStep | None:
        return next(
            (
                step
                for step in self.steps
                if step.status not in {PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED}
            ),
            None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "goal": self.goal,
            "task_type": self.task_type.value,
            "steps": [step.to_dict() for step in self.steps],
            "acceptance_criteria": list(self.acceptance_criteria),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: object) -> ExecutionPlan:
        item = _mapping(value, "execution plan")
        raw_steps = item.get("steps")
        raw_criteria = item.get("acceptance_criteria")
        if not isinstance(raw_steps, list):
            raise ValueError("execution plan steps must be a list")
        if not isinstance(raw_criteria, list) or not all(
            isinstance(entry, str) and entry for entry in raw_criteria
        ):
            raise ValueError("acceptance_criteria must be a non-empty string list")
        return cls(
            id=_required_text(item.get("id"), "plan id"),
            goal=_required_text(item.get("goal"), "plan goal"),
            task_type=TaskType(item.get("task_type")),
            steps=[PlanStep.from_dict(step) for step in raw_steps],
            acceptance_criteria=tuple(raw_criteria),
            status=PlanStatus(item.get("status", PlanStatus.ACTIVE.value)),
            created_at=_datetime(item.get("created_at"), "created_at"),
            updated_at=_datetime(item.get("updated_at"), "updated_at"),
            revision=_positive_int(item.get("revision", 1), "revision"),
        )


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class FailureCategory(str, Enum):
    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    TEST_FAILURE = "test_failure"
    BUILD_FAILURE = "build_failure"
    REQUIREMENT_NOT_MET = "requirement_not_met"
    CONTEXT_STALE = "context_stale"
    PERMISSION_DENIED = "permission_denied"
    ENVIRONMENT_MISSING = "environment_missing"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    required: bool
    executed: bool
    passed: bool
    command: str | None = None
    exit_code: int | None = None
    evidence: tuple[str, ...] = ()
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required": self.required,
            "executed": self.executed,
            "passed": self.passed,
            "command": self.command,
            "exit_code": self.exit_code,
            "evidence": list(self.evidence),
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> VerificationCheck:
        item = _mapping(value, "verification check")
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list) or not all(
            isinstance(entry, str) for entry in evidence
        ):
            raise ValueError("verification evidence must be a string list")
        return cls(
            name=_required_text(item.get("name"), "verification check name"),
            required=_boolean(item.get("required"), "required"),
            executed=_boolean(item.get("executed"), "executed"),
            passed=_boolean(item.get("passed"), "passed"),
            command=_nullable_text(item.get("command"), "command"),
            exit_code=_nullable_int(item.get("exit_code"), "exit_code"),
            evidence=tuple(evidence),
            failure_reason=_nullable_text(
                item.get("failure_reason"), "failure_reason"
            ),
        )


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    status: VerificationStatus
    checks: tuple[VerificationCheck, ...]
    evidence: tuple[str, ...]
    failure_category: FailureCategory | None
    repairable: bool
    summary: str

    def __post_init__(self) -> None:
        if self.passed != (self.status is VerificationStatus.PASSED):
            raise ValueError("verification passed flag and status disagree")
        if self.passed and self.failure_category is not None:
            raise ValueError("passed verification cannot have a failure category")

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "status": self.status.value,
            "checks": [check.to_dict() for check in self.checks],
            "evidence": list(self.evidence),
            "failure_category": (
                self.failure_category.value if self.failure_category else None
            ),
            "repairable": self.repairable,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, value: object) -> VerificationResult:
        item = _mapping(value, "verification result")
        raw_checks = item.get("checks", [])
        raw_evidence = item.get("evidence", [])
        if not isinstance(raw_checks, list):
            raise ValueError("verification checks must be a list")
        if not isinstance(raw_evidence, list) or not all(
            isinstance(entry, str) for entry in raw_evidence
        ):
            raise ValueError("verification evidence must be a string list")
        raw_category = item.get("failure_category")
        return cls(
            passed=_boolean(item.get("passed"), "passed"),
            status=VerificationStatus(item.get("status")),
            checks=tuple(VerificationCheck.from_dict(check) for check in raw_checks),
            evidence=tuple(raw_evidence),
            failure_category=(
                FailureCategory(raw_category) if raw_category is not None else None
            ),
            repairable=_boolean(item.get("repairable"), "repairable"),
            summary=_required_text(item.get("summary"), "verification summary"),
        )


@dataclass(frozen=True)
class RepairDiagnosis:
    failure_category: FailureCategory
    likely_cause: str
    affected_step: str | None
    recommended_action: str
    requires_replan: bool
    requires_reread: bool
    requires_human: bool
    retryable: bool


@dataclass
class RunCheckpoint:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    run_state: RunState = RunState.PREPARED
    task_type: TaskType = TaskType.INFORMATIONAL
    current_plan: ExecutionPlan | None = None
    current_step_id: str | None = None
    repair_attempts: int = 0
    last_verification: VerificationResult | None = None
    pending_approval: dict[str, object] | None = None
    checkpoint_version: int = 1
    completed_tool_call_ids: list[str] = field(default_factory=list)
    uncertain_tool_call_ids: list[str] = field(default_factory=list)
    decision_summary: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_state": self.run_state.value,
            "task_type": self.task_type.value,
            "current_plan": (
                self.current_plan.to_dict() if self.current_plan is not None else None
            ),
            "current_step_id": self.current_step_id,
            "repair_attempts": self.repair_attempts,
            "last_verification": (
                self.last_verification.to_dict()
                if self.last_verification is not None
                else None
            ),
            "pending_approval": self.pending_approval,
            "checkpoint_version": self.checkpoint_version,
            "completed_tool_call_ids": list(self.completed_tool_call_ids),
            "uncertain_tool_call_ids": list(self.uncertain_tool_call_ids),
            "decision_summary": self.decision_summary,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> RunCheckpoint:
        item = _mapping(value, "run checkpoint")
        raw_plan = item.get("current_plan")
        raw_verification = item.get("last_verification")
        pending = item.get("pending_approval")
        if pending is not None and not isinstance(pending, dict):
            raise ValueError("pending_approval must be an object or null")
        completed = _string_list(
            item.get("completed_tool_call_ids", []), "completed_tool_call_ids"
        )
        uncertain = _string_list(
            item.get("uncertain_tool_call_ids", []), "uncertain_tool_call_ids"
        )
        return cls(
            run_id=_required_text(item.get("run_id"), "run id"),
            run_state=RunState(item.get("run_state", RunState.PREPARED.value)),
            task_type=TaskType(
                item.get("task_type", TaskType.INFORMATIONAL.value)
            ),
            current_plan=(
                ExecutionPlan.from_dict(raw_plan) if raw_plan is not None else None
            ),
            current_step_id=_nullable_text(
                item.get("current_step_id"), "current_step_id"
            ),
            repair_attempts=_non_negative_int(
                item.get("repair_attempts", 0), "repair_attempts"
            ),
            last_verification=(
                VerificationResult.from_dict(raw_verification)
                if raw_verification is not None
                else None
            ),
            pending_approval=dict(pending) if pending is not None else None,
            checkpoint_version=_positive_int(
                item.get("checkpoint_version", 1), "checkpoint_version"
            ),
            completed_tool_call_ids=completed,
            uncertain_tool_call_ids=uncertain,
            decision_summary=_nullable_text(
                item.get("decision_summary"), "decision_summary"
            ),
            failure_reason=_nullable_text(
                item.get("failure_reason"), "failure_reason"
            ),
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _optional_text(value: object, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return value


def _nullable_text(value: object, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{label} must be text or null")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _non_negative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nullable_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer or null")
    return value


def _datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return result


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{label} must be a string list")
    return list(value)
