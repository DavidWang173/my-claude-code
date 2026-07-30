"""Single owner of Run lifecycle transitions and checkpoint persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING
from uuid import uuid4

from ..models import ToolCall
from ..permissions import Operation, PermissionRequest
from ..sessions import Session, SessionStore
from .events import EventStore, EventType, RunTracer
from .models import (
    ALLOWED_TRANSITIONS,
    ExecutionPlan,
    InvalidRunStateTransition,
    PlanStatus,
    PlanStepStatus,
    RunCheckpoint,
    RunState,
    TaskType,
    VerificationResult,
)
from .planning import Planner

if TYPE_CHECKING:
    from ..tools import ToolResult


class RunOrchestrator:
    """Own Run state; callers can observe ``state`` but cannot assign to it."""

    def __init__(
        self,
        session: Session,
        store: SessionStore,
        *,
        task_type: TaskType,
        planner: Planner | None = None,
        event_store: EventStore | None = None,
        resume: bool = True,
    ) -> None:
        self._session = session
        self._store = store
        self._planner = planner or Planner()
        restored = session.run_checkpoint() if resume else None
        if restored is not None and not restored.run_state.terminal:
            self._checkpoint = restored
            self._state = restored.run_state
        else:
            self._checkpoint = RunCheckpoint(
                run_id=uuid4().hex,
                run_state=RunState.PREPARED,
                task_type=task_type,
                uncertain_tool_call_ids=(
                    list(restored.uncertain_tool_call_ids)
                    if restored is not None
                    else []
                ),
                decision_summary=(
                    "A prior non-idempotent tool call has an unknown outcome; "
                    "human confirmation is required before retrying."
                    if restored is not None and restored.uncertain_tool_call_ids
                    else None
                ),
            )
            self._state = RunState.PREPARED
            self._persist()
        self._trace = RunTracer(
            event_store,
            run_id=self._checkpoint.run_id,
            session_id=session.id,
            run_state=self._state,
            provider=session.provider,
            model=session.model,
        )

    @property
    def state(self) -> RunState:
        return self._state

    @property
    def run_id(self) -> str:
        return self._checkpoint.run_id

    @property
    def trace(self) -> RunTracer:
        return self._trace

    @property
    def task_type(self) -> TaskType:
        return self._checkpoint.task_type

    @property
    def plan(self) -> ExecutionPlan | None:
        return self._checkpoint.current_plan

    @property
    def current_step_id(self) -> str | None:
        return self._checkpoint.current_step_id

    @property
    def repair_attempts(self) -> int:
        return self._checkpoint.repair_attempts

    @property
    def last_verification(self) -> VerificationResult | None:
        return self._checkpoint.last_verification

    @property
    def has_unsafe_recovery(self) -> bool:
        return bool(self._checkpoint.uncertain_tool_call_ids)

    def transition_to(
        self,
        new_state: RunState,
        *,
        verification: VerificationResult | None = None,
        reason: str | None = None,
    ) -> None:
        """Validate, apply, and durably checkpoint one state transition."""

        if new_state not in ALLOWED_TRANSITIONS[self._state]:
            raise InvalidRunStateTransition(
                f"illegal run state transition: {self._state.name} -> {new_state.name}"
            )
        if new_state is RunState.COMPLETED:
            result = verification or self._checkpoint.last_verification
            if result is None or not result.passed:
                raise InvalidRunStateTransition(
                    "VERIFYING -> COMPLETED requires a passed VerificationResult"
                )
        old_state = self._state
        self._state = new_state
        self._checkpoint.run_state = new_state
        self._trace.set_state(new_state)
        if verification is not None:
            self._checkpoint.last_verification = verification
        if reason:
            if new_state in {RunState.FAILED, RunState.CANCELLED}:
                self._checkpoint.failure_reason = reason
            else:
                self._checkpoint.decision_summary = reason
        if new_state is not RunState.WAITING_APPROVAL:
            self._checkpoint.pending_approval = None
        self._persist()
        self._trace.emit(
            EventType.RUN_STATE_CHANGED,
            input_summary={"from": old_state.value},
            output_summary={"to": new_state.value},
            success=True,
        )
        if old_state is RunState.REPAIRING and new_state is not RunState.REPAIRING:
            self._trace.finish(
                "repair",
                EventType.REPAIR_COMPLETED,
                plan_id=self.plan.id if self.plan else None,
                step_id=self.current_step_id,
                output_summary={"next_state": new_state.value},
                success=new_state not in {RunState.FAILED, RunState.CANCELLED},
            )
        if new_state is RunState.VERIFYING:
            self._trace.start(
                "verification",
                EventType.VERIFICATION_STARTED,
                plan_id=self.plan.id if self.plan else None,
                step_id=self.current_step_id,
                input_summary={"task_type": self.task_type.value},
            )
        elif new_state is RunState.REPAIRING:
            self._trace.start(
                "repair",
                EventType.REPAIR_STARTED,
                plan_id=self.plan.id if self.plan else None,
                step_id=self.current_step_id,
                input_summary={"attempt": self.repair_attempts},
            )
        terminal_event = {
            RunState.COMPLETED: EventType.RUN_COMPLETED,
            RunState.FAILED: EventType.RUN_FAILED,
            RunState.CANCELLED: EventType.RUN_CANCELLED,
        }.get(new_state)
        if terminal_event is not None:
            result = verification or self._checkpoint.last_verification
            self._trace.emit(
                terminal_event,
                duration_ms=self._trace.total_duration_ms(),
                plan_id=self.plan.id if self.plan else None,
                step_id=self.current_step_id,
                output_summary={
                    "verification_passed": result.passed if result else None,
                    "repair_attempts": self.repair_attempts,
                },
                success=new_state is RunState.COMPLETED,
                error_category=(
                    result.failure_category.value
                    if result is not None and result.failure_category is not None
                    else ("cancelled" if new_state is RunState.CANCELLED else None)
                ),
            )

    def prepare(self, goal: str) -> None:
        """Create a plan when policy requires one and enter execution."""

        if self._state is not RunState.PREPARED:
            return
        if self._planner.requires_plan(self.task_type, goal):
            self.transition_to(RunState.PLANNING)
            plan = self._planner.create_plan(goal=goal, task_type=self.task_type)
            self.set_plan(plan)
        self.transition_to(RunState.EXECUTING)

    def set_plan(self, plan: ExecutionPlan) -> None:
        if plan.task_type is not self.task_type:
            raise ValueError("plan task_type does not match the Run")
        previous_plan = self._checkpoint.current_plan
        self._checkpoint.current_plan = plan
        first = plan.next_incomplete_step()
        self._checkpoint.current_step_id = first.id if first else None
        if first is not None and first.status is PlanStepStatus.PENDING:
            first.status = PlanStepStatus.IN_PROGRESS
            plan.touch()
        self._persist()
        plan_event = self._trace.emit(
            EventType.PLAN_CREATED if previous_plan is None else EventType.PLAN_UPDATED,
            plan_id=plan.id,
            input_summary={
                "task_type": plan.task_type.value,
                "step_count": len(plan.steps),
                "acceptance_criteria_count": len(plan.acceptance_criteria),
            },
            output_summary={"revision": plan.revision, "status": plan.status.value},
            success=True,
        )
        if first is not None:
            self._trace.start(
                f"step:{first.id}",
                EventType.STEP_STARTED,
                parent_event_id=plan_event.event_id,
                plan_id=plan.id,
                step_id=first.id,
                input_summary={"retry_count": first.retry_count},
            )

    def update_step(
        self,
        step_id: str,
        status: PlanStepStatus,
        *,
        error_summary: str | None = None,
        increment_retry: bool = False,
    ) -> None:
        plan = self.plan
        if plan is None:
            raise ValueError("cannot update a step without an execution plan")
        previous_status = plan.step(step_id).status
        plan.update_step(
            step_id,
            status,
            error_summary=error_summary,
            increment_retry=increment_retry,
        )
        next_step = plan.next_incomplete_step()
        if next_step is not None:
            if next_step.status is PlanStepStatus.PENDING:
                next_step.status = PlanStepStatus.IN_PROGRESS
                plan.touch()
            self._checkpoint.current_step_id = next_step.id
        else:
            plan.status = PlanStatus.COMPLETED
            plan.touch()
            self._checkpoint.current_step_id = None
        self._persist()
        if status in {PlanStepStatus.COMPLETED, PlanStepStatus.FAILED}:
            self._trace.finish(
                f"step:{step_id}",
                (
                    EventType.STEP_COMPLETED
                    if status is PlanStepStatus.COMPLETED
                    else EventType.STEP_FAILED
                ),
                plan_id=plan.id,
                step_id=step_id,
                output_summary={
                    "from": previous_status.value,
                    "to": status.value,
                    "retry_count": plan.step(step_id).retry_count,
                },
                success=status is PlanStepStatus.COMPLETED,
                error_category=(
                    "step_failed" if status is PlanStepStatus.FAILED else None
                ),
            )
        plan_event = self._trace.emit(
            EventType.PLAN_UPDATED,
            plan_id=plan.id,
            step_id=step_id,
            output_summary={"revision": plan.revision, "status": plan.status.value},
            success=status is not PlanStepStatus.FAILED,
        )
        if (
            next_step is not None
            and next_step.id != step_id
            and next_step.status is PlanStepStatus.IN_PROGRESS
        ):
            self._trace.start(
                f"step:{next_step.id}",
                EventType.STEP_STARTED,
                parent_event_id=plan_event.event_id,
                plan_id=plan.id,
                step_id=next_step.id,
                input_summary={"retry_count": next_step.retry_count},
            )

    def mark_plan_candidate_complete(self) -> None:
        plan = self.plan
        if plan is None:
            return
        changed_steps = []
        for step in plan.steps:
            if step.status in {PlanStepStatus.PENDING, PlanStepStatus.IN_PROGRESS}:
                step.status = PlanStepStatus.COMPLETED
                changed_steps.append(step)
        plan.status = PlanStatus.COMPLETED
        plan.touch()
        self._checkpoint.current_step_id = None
        self._persist()
        for step in changed_steps:
            self._trace.finish(
                f"step:{step.id}",
                EventType.STEP_COMPLETED,
                plan_id=plan.id,
                step_id=step.id,
                output_summary={"status": step.status.value},
                success=True,
            )
        self._trace.emit(
            EventType.PLAN_UPDATED,
            plan_id=plan.id,
            output_summary={"revision": plan.revision, "status": plan.status.value},
            success=True,
        )

    def revise_plan(self, *, reason: str) -> None:
        plan = self.plan
        if plan is None:
            return
        plan.status = PlanStatus.ACTIVE
        plan.touch()
        step = plan.next_incomplete_step() or plan.steps[-1]
        step.status = PlanStepStatus.IN_PROGRESS
        step.error_summary = reason
        self._checkpoint.current_step_id = step.id
        self._checkpoint.decision_summary = reason
        self._persist()
        self._trace.emit(
            EventType.PLAN_UPDATED,
            plan_id=plan.id,
            step_id=step.id,
            input_summary={"revision_reason_category": "verification_failure"},
            output_summary={"revision": plan.revision, "status": plan.status.value},
            success=True,
        )
        self._trace.start(
            f"step:{step.id}",
            EventType.STEP_STARTED,
            plan_id=plan.id,
            step_id=step.id,
            input_summary={"retry_count": step.retry_count},
        )

    def upgrade_task_type(self, operation: Operation) -> None:
        upgraded = self.task_type
        if operation is Operation.WRITE:
            upgraded = TaskType.MODIFICATION
        elif operation in {Operation.EXECUTE, Operation.NETWORK} and upgraded in {
            TaskType.INFORMATIONAL,
            TaskType.INSPECTION,
        }:
            upgraded = TaskType.EXECUTION
        if upgraded is self.task_type:
            return
        self._checkpoint.task_type = upgraded
        if self.plan is None and self._planner.requires_plan(upgraded):
            self._checkpoint.current_plan = self._planner.fallback_plan(
                goal="Complete the requested modification.",
                task_type=upgraded,
            )
            first = self._checkpoint.current_plan.steps[0]
            first.status = PlanStepStatus.IN_PROGRESS
            self._checkpoint.current_step_id = first.id
        self._persist()

    def request_tool_call(self, call: ToolCall) -> None:
        self._trace.start(
            f"tool-request:{call.id}",
            EventType.TOOL_CALL_REQUESTED,
            parent_event_id=self._trace.last_model_event_id,
            plan_id=self.plan.id if self.plan else None,
            step_id=self.current_step_id,
            tool_call_id=call.id,
            input_summary={
                "tool_name": call.name,
                "argument_names": sorted(call.arguments),
                "argument_types": {
                    key: type(value).__name__ for key, value in call.arguments.items()
                },
            },
        )

    def begin_tool_call(
        self,
        call: ToolCall,
        *,
        operation: Operation,
    ) -> None:
        requested = self._trace.finish(
            f"tool-request:{call.id}",
            EventType.TOOL_CALL_STARTED,
            plan_id=self.plan.id if self.plan else None,
            step_id=self.current_step_id,
            tool_call_id=call.id,
            input_summary={"tool_name": call.name, "operation": operation.value},
        )
        self._trace.activate(f"tool:{call.id}", requested.event_id)
        if operation is not Operation.READ and call.id not in self._checkpoint.uncertain_tool_call_ids:
            self._checkpoint.uncertain_tool_call_ids.append(call.id)
            self._persist()

    def record_tool_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        operation: Operation,
    ) -> None:
        if call.id not in self._checkpoint.completed_tool_call_ids:
            self._checkpoint.completed_tool_call_ids.append(call.id)
        if call.id in self._checkpoint.uncertain_tool_call_ids:
            self._checkpoint.uncertain_tool_call_ids.remove(call.id)
        if result.success:
            self.upgrade_task_type(operation)
            self._advance_plan_for_tool(call, operation)
        else:
            self._persist()
        trace_key = (
            f"tool:{call.id}"
            if self._trace.has_active(f"tool:{call.id}")
            else f"tool-request:{call.id}"
        )
        self._trace.finish(
            trace_key,
            EventType.TOOL_CALL_COMPLETED if result.success else EventType.TOOL_CALL_FAILED,
            plan_id=self.plan.id if self.plan else None,
            step_id=self.current_step_id,
            tool_call_id=call.id,
            output_summary={
                "tool_name": call.name,
                "content_chars": len(result.content),
                "metadata_keys": sorted(str(key) for key in result.metadata),
            },
            success=result.success,
            error_category=None if result.success else "tool_execution_error",
        )

    def wait_for_approval(
        self,
        call: ToolCall,
        request: PermissionRequest,
    ) -> None:
        return_state = self._state
        if return_state not in {
            RunState.EXECUTING,
            RunState.VERIFYING,
            RunState.REPAIRING,
        }:
            raise InvalidRunStateTransition(
                f"cannot request approval while in {return_state.name}"
            )
        self.transition_to(RunState.WAITING_APPROVAL)
        self._trace.emit(
            EventType.TOOL_CALL_APPROVAL_REQUIRED,
            plan_id=self.plan.id if self.plan else None,
            step_id=self.current_step_id,
            tool_call_id=call.id,
            input_summary={
                "tool_name": call.name,
                "operation": request.operation.value,
            },
        )
        self._checkpoint.pending_approval = {
            "return_state": return_state.value,
            "tool_call_id": call.id,
            "tool_name": call.name,
            "operation": request.operation.value,
            "target": request.target,
            "arguments_fingerprint": _arguments_fingerprint(call.arguments),
        }
        self._persist()

    def resolve_approval(self, *, allowed: bool) -> None:
        pending = self._checkpoint.pending_approval
        if self._state is not RunState.WAITING_APPROVAL or pending is None:
            raise InvalidRunStateTransition("there is no pending approval to resolve")
        raw_return = pending.get("return_state")
        return_state = RunState(raw_return)
        if allowed:
            self.transition_to(return_state, reason="Permission request approved.")
        else:
            self.transition_to(
                RunState.REPAIRING,
                reason="Permission request denied; a permitted alternative is required.",
            )

    def record_verification(self, result: VerificationResult) -> None:
        if self._state is not RunState.VERIFYING:
            raise InvalidRunStateTransition(
                "verification evidence can only be recorded in VERIFYING"
            )
        self._checkpoint.last_verification = result
        self._persist()
        self._trace.finish(
            "verification",
            EventType.VERIFICATION_COMPLETED,
            plan_id=self.plan.id if self.plan else None,
            step_id=self.current_step_id,
            output_summary={
                "status": result.status.value,
                "check_count": len(result.checks),
                "failed_check_names": [
                    check.name for check in result.checks if check.required and not check.passed
                ],
                "evidence_count": len(result.evidence),
                "repairable": result.repairable,
            },
            success=result.passed,
            error_category=(
                result.failure_category.value if result.failure_category else None
            ),
        )

    def begin_repair(self, result: VerificationResult) -> None:
        self._checkpoint.repair_attempts += 1
        self.transition_to(
            RunState.REPAIRING,
            verification=result,
            reason=result.summary,
        )

    def _advance_plan_for_tool(self, call: ToolCall, operation: Operation) -> None:
        plan = self.plan
        if plan is None:
            self._persist()
            return
        previous = {step.id: step.status for step in plan.steps}
        if call.name == "create_file":
            for step in plan.steps:
                if step.id == "inspect" and step.status in {
                    PlanStepStatus.PENDING,
                    PlanStepStatus.IN_PROGRESS,
                }:
                    step.status = PlanStepStatus.SKIPPED
                    step.error_summary = "A new file did not require an existing-file read."
                    plan.touch()
        if call.name in {"git_diff_check", "run_shell"} and plan.steps:
            target = plan.steps[-1]
        elif operation is Operation.WRITE and len(plan.steps) >= 2:
            target = plan.steps[-2]
        else:
            target = plan.next_incomplete_step()
        if target is not None and target.status in {
            PlanStepStatus.PENDING,
            PlanStepStatus.IN_PROGRESS,
        }:
            target.status = PlanStepStatus.COMPLETED
            plan.touch()
        next_step = plan.next_incomplete_step()
        if next_step is not None:
            if next_step.status is PlanStepStatus.PENDING:
                next_step.status = PlanStepStatus.IN_PROGRESS
                plan.touch()
            self._checkpoint.current_step_id = next_step.id
        else:
            self._checkpoint.current_step_id = None
        self._persist()
        plan_event = self._trace.emit(
            EventType.PLAN_UPDATED,
            plan_id=plan.id,
            step_id=target.id if target is not None else None,
            output_summary={
                "revision": plan.revision,
                "status": plan.status.value,
                "source": "tool_result",
            },
            success=True,
        )
        for step in plan.steps:
            old_status = previous[step.id]
            if step.status is old_status:
                continue
            if step.status in {PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED}:
                self._trace.finish(
                    f"step:{step.id}",
                    EventType.STEP_COMPLETED,
                    plan_id=plan.id,
                    step_id=step.id,
                    output_summary={
                        "from": old_status.value,
                        "to": step.status.value,
                    },
                    success=True,
                )
            elif step.status is PlanStepStatus.IN_PROGRESS:
                self._trace.start(
                    f"step:{step.id}",
                    EventType.STEP_STARTED,
                    parent_event_id=plan_event.event_id,
                    plan_id=plan.id,
                    step_id=step.id,
                    input_summary={"retry_count": step.retry_count},
                )

    def _persist(self) -> None:
        self._checkpoint.checkpoint_version += 1
        self._session.apply_checkpoint(self._checkpoint)
        self._store.save(self._session)


def _arguments_fingerprint(arguments: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        encoded = repr(arguments)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
