from __future__ import annotations

import unittest

from src.harness.models import PlanStepStatus, RunState, TaskType
from src.harness.planning import Planner
from src.sessions import SCHEMA_VERSION, Session, session_from_dict, session_to_dict


class RunRecoveryTests(unittest.TestCase):
    def test_current_plan_step_survives_session_round_trip(self) -> None:
        session = Session.create()
        plan = Planner().create_plan(
            goal="Implement feature and verify it",
            task_type=TaskType.MODIFICATION,
        )
        plan.steps[0].status = PlanStepStatus.COMPLETED
        plan.steps[1].status = PlanStepStatus.IN_PROGRESS
        session.run_id = "a" * 32
        session.run_state = RunState.EXECUTING
        session.task_type = TaskType.MODIFICATION
        session.current_plan = plan
        session.current_step_id = "implement"
        session.checkpoint_version = 4

        restored = session_from_dict(session_to_dict(session))

        self.assertEqual(restored.current_step_id, "implement")
        assert restored.current_plan is not None
        self.assertEqual(
            restored.current_plan.step("inspect").status,
            PlanStepStatus.COMPLETED,
        )
        self.assertEqual(restored.current_plan.revision, plan.revision)

    def test_schema_two_loads_with_safe_checkpoint_defaults(self) -> None:
        session = Session.create()
        payload = session_to_dict(session)
        payload["schema_version"] = 2
        for key in (
            "run_id",
            "run_state",
            "task_type",
            "current_plan",
            "current_step_id",
            "repair_attempts",
            "last_verification",
            "pending_approval",
            "checkpoint_version",
            "completed_tool_call_ids",
            "uncertain_tool_call_ids",
            "run_decision_summary",
            "run_failure_reason",
        ):
            payload.pop(key)

        restored = session_from_dict(payload)

        self.assertEqual(restored.schema_version, SCHEMA_VERSION)
        self.assertIsNone(restored.run_state)
        self.assertEqual(restored.repair_attempts, 0)
        self.assertEqual(restored.completed_tool_call_ids, [])

    def test_checkpoint_schema_has_no_private_chain_of_thought_field(self) -> None:
        session = Session.create()
        payload = session_to_dict(session)
        rendered = repr(payload).casefold()
        self.assertNotIn("chain_of_thought", rendered)
        self.assertNotIn("reasoning_content", rendered)


if __name__ == "__main__":
    unittest.main()
