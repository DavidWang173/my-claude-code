"""Small, schema-validated planning policy.

The first phase intentionally uses an ordered plan.  ``dependencies`` remain
available for future scheduling but can only point to earlier steps.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from uuid import uuid4

from .models import ExecutionPlan, PlanStep, TaskType

_MODIFICATION = re.compile(
    r"\b(implement|add|create|change|modify|edit|update|fix|repair|refactor|"
    r"remove|delete|rename|write|patch|migrate|build)\b|"
    r"(实现|新增|添加|创建|修改|修复|重构|删除|重命名|迁移|编写)",
    re.IGNORECASE,
)
_EXECUTION = re.compile(
    r"\b(run|execute|launch|start|stop|install|deploy|publish|test|build)\b|"
    r"(运行|执行|启动|停止|安装|部署|发布|测试|构建)",
    re.IGNORECASE,
)
_INSPECTION = re.compile(
    r"\b(inspect|review|audit|analy[sz]e|investigate|diagnose|read|find|search|"
    r"trace)\b|"
    r"(检查|审查|分析|调查|诊断|读取|查找|搜索|代码解释)",
    re.IGNORECASE,
)
_READ_ONLY = re.compile(r"\b(read[- ]?only|do not (?:change|modify|edit))\b|只读|不要修改", re.I)


def classify_task(prompt: str) -> TaskType:
    """Conservatively classify the requested deliverable, not model prose."""

    if _READ_ONLY.search(prompt):
        return TaskType.INSPECTION
    if re.search(r"\battempt (?:a )?(?:write|change|edit)\b", prompt, re.I):
        return TaskType.INFORMATIONAL
    if _MODIFICATION.search(prompt):
        return TaskType.MODIFICATION
    if _EXECUTION.search(prompt):
        return TaskType.EXECUTION
    if _INSPECTION.search(prompt):
        return TaskType.INSPECTION
    return TaskType.INFORMATIONAL


class Planner:
    """Create and validate a lightweight ordered execution plan."""

    def requires_plan(self, task_type: TaskType, prompt: str = "") -> bool:
        if task_type is not TaskType.MODIFICATION:
            return False
        lowered = prompt.casefold()
        complexity_markers = (
            " and ",
            "\n-",
            "\n1.",
            "multiple",
            "across",
            "phase",
            "refactor",
            "migrate",
            "test",
            "verify",
            "完整",
            "多个",
            "阶段",
            "重构",
            "迁移",
            "测试",
            "验证",
            "以及",
        )
        return any(marker in lowered for marker in complexity_markers)

    def create_plan(
        self,
        *,
        goal: str,
        task_type: TaskType,
        planner_output: Mapping[str, object] | None = None,
    ) -> ExecutionPlan:
        """Return a valid plan, falling back to a safe single-step plan.

        A provider-backed planner may pass decoded JSON through
        ``planner_output``.  Invalid output is never accepted as a plan.
        """

        if planner_output is not None:
            try:
                payload = dict(planner_output)
                payload.setdefault("id", uuid4().hex)
                payload.setdefault("goal", goal)
                payload.setdefault("task_type", task_type.value)
                return ExecutionPlan.from_dict(payload)
            except (KeyError, TypeError, ValueError):
                return self.fallback_plan(goal=goal, task_type=task_type)
        return self._default_plan(goal=goal, task_type=task_type)

    def fallback_plan(self, *, goal: str, task_type: TaskType) -> ExecutionPlan:
        return ExecutionPlan.single_step(
            goal=goal,
            task_type=task_type,
            acceptance_criteria=self.acceptance_criteria(task_type),
            description="Complete the requested task and collect explicit verification evidence.",
        )

    def acceptance_criteria(self, task_type: TaskType) -> tuple[str, ...]:
        if task_type is TaskType.MODIFICATION:
            return (
                "The requested files contain the intended scoped changes.",
                "The final diff has no whitespace errors or conflict markers.",
                "Relevant tests or a documented alternative verification pass.",
            )
        if task_type is TaskType.EXECUTION:
            return (
                "The requested command was actually executed.",
                "Its exit code and goal-relevant output were recorded.",
            )
        if task_type is TaskType.INSPECTION:
            return (
                "Relevant files were actually read.",
                "Conclusions cite concrete file or code evidence.",
            )
        return (
            "The response answers the user's request without unsupported claims.",
        )

    def _default_plan(self, *, goal: str, task_type: TaskType) -> ExecutionPlan:
        if task_type is not TaskType.MODIFICATION:
            return self.fallback_plan(goal=goal, task_type=task_type)
        return ExecutionPlan(
            id=uuid4().hex,
            goal=goal,
            task_type=task_type,
            acceptance_criteria=self.acceptance_criteria(task_type),
            steps=[
                PlanStep(
                    id="inspect",
                    description="Inspect the relevant implementation and tests.",
                    expected_output="A bounded summary of the existing behavior and edit scope.",
                    verification_hint="Record concrete file reads.",
                ),
                PlanStep(
                    id="implement",
                    description="Apply the smallest scoped implementation change.",
                    dependencies=("inspect",),
                    expected_output="Only task-relevant files are modified.",
                    verification_hint="Review the task-local diff and changed paths.",
                ),
                PlanStep(
                    id="verify",
                    description="Run required checks and resolve any failures.",
                    dependencies=("implement",),
                    expected_output="Diff checks and relevant tests or alternatives pass.",
                    verification_hint="Record commands, exit codes, and evidence.",
                ),
            ],
        )
