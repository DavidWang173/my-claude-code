from __future__ import annotations

import unittest

from src.harness.models import PlanStepStatus, TaskType
from src.harness.planning import Planner


class PlannerTests(unittest.TestCase):
    def test_simple_informational_task_skips_planner(self) -> None:
        self.assertFalse(
            Planner().requires_plan(TaskType.INFORMATIONAL, "What does this mean?")
        )

    def test_complex_modification_gets_valid_ordered_plan(self) -> None:
        planner = Planner()
        self.assertTrue(
            planner.requires_plan(
                TaskType.MODIFICATION,
                "Implement the feature and run tests",
            )
        )
        plan = planner.create_plan(
            goal="Implement the feature and run tests",
            task_type=TaskType.MODIFICATION,
        )
        self.assertEqual([step.id for step in plan.steps], ["inspect", "implement", "verify"])
        self.assertEqual(plan.steps[1].dependencies, ("inspect",))

    def test_invalid_planner_output_falls_back_to_single_step(self) -> None:
        plan = Planner().create_plan(
            goal="Implement safely",
            task_type=TaskType.MODIFICATION,
            planner_output={"steps": "not-a-list"},
        )
        self.assertEqual(len(plan.steps), 1)
        self.assertTrue(plan.acceptance_criteria)

    def test_plan_update_increments_revision_and_stores_no_private_reasoning(self) -> None:
        plan = Planner().fallback_plan(
            goal="Change one file",
            task_type=TaskType.MODIFICATION,
        )
        revision = plan.revision
        plan.update_step("step-1", PlanStepStatus.COMPLETED)
        payload = plan.to_dict()

        self.assertGreater(plan.revision, revision)
        rendered_keys = repr(payload).casefold()
        self.assertNotIn("chain_of_thought", rendered_keys)
        self.assertNotIn("reasoning_content", rendered_keys)


if __name__ == "__main__":
    unittest.main()
