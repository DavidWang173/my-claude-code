from __future__ import annotations

import unittest
from pathlib import Path

from src.git_runtime import GitRunSummary
from src.harness.models import (
    InvalidRunStateTransition,
    RunState,
    TaskType,
    VerificationResult,
    VerificationStatus,
)
from src.harness.orchestrator import RunOrchestrator
from src.sessions import Session


class MemoryStore:
    def __init__(self) -> None:
        self.saves = 0

    def save(self, session: Session) -> None:
        del session
        self.saves += 1


def passed_verification() -> VerificationResult:
    return VerificationResult(
        passed=True,
        status=VerificationStatus.PASSED,
        checks=(),
        evidence=("verified",),
        failure_category=None,
        repairable=False,
        summary="passed",
    )


class RunStateTests(unittest.TestCase):
    def test_legal_transitions_are_checkpointed_and_completion_requires_gate(self) -> None:
        session = Session.create(workspace=Path.cwd())
        store = MemoryStore()
        run = RunOrchestrator(
            session,
            store,  # type: ignore[arg-type]
            task_type=TaskType.INFORMATIONAL,
        )
        initial_saves = store.saves

        run.transition_to(RunState.EXECUTING)
        run.transition_to(RunState.VERIFYING)
        with self.assertRaisesRegex(
            InvalidRunStateTransition, "passed VerificationResult"
        ):
            run.transition_to(RunState.COMPLETED)
        run.transition_to(
            RunState.COMPLETED,
            verification=passed_verification(),
        )

        self.assertEqual(run.state, RunState.COMPLETED)
        self.assertEqual(session.run_state, RunState.COMPLETED)
        self.assertGreaterEqual(store.saves, initial_saves + 3)

    def test_illegal_transition_has_explicit_error_and_state_is_read_only(self) -> None:
        run = RunOrchestrator(
            Session.create(),
            MemoryStore(),  # type: ignore[arg-type]
            task_type=TaskType.INFORMATIONAL,
        )
        with self.assertRaisesRegex(
            InvalidRunStateTransition, "PREPARED -> COMPLETED"
        ):
            run.transition_to(
                RunState.COMPLETED,
                verification=passed_verification(),
            )
        with self.assertRaises(AttributeError):
            run.state = RunState.FAILED  # type: ignore[misc]

    def test_cancel_and_failure_are_terminal(self) -> None:
        for terminal in (RunState.CANCELLED, RunState.FAILED):
            with self.subTest(terminal=terminal):
                run = RunOrchestrator(
                    Session.create(),
                    MemoryStore(),  # type: ignore[arg-type]
                    task_type=TaskType.INFORMATIONAL,
                )
                run.transition_to(terminal, reason="bounded stop")
                with self.assertRaises(InvalidRunStateTransition):
                    run.transition_to(RunState.EXECUTING)


if __name__ == "__main__":
    unittest.main()
