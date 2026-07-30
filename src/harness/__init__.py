"""Public lifecycle API for the phase-one coding-agent harness."""

from .events import (
    EventStore,
    EventType,
    JsonlEventStore,
    RunEvent,
    RunTracer,
    TRACE_SCHEMA_VERSION,
)
from .models import (
    ALLOWED_TRANSITIONS,
    ExecutionPlan,
    FailureCategory,
    InvalidRunStateTransition,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
    RepairDiagnosis,
    RunCheckpoint,
    RunState,
    TaskType,
    VerificationCheck,
    VerificationResult,
    VerificationStatus,
)
__all__ = [
    "ALLOWED_TRANSITIONS",
    "ExecutionPlan",
    "EventStore",
    "EventType",
    "FailureCategory",
    "InvalidRunStateTransition",
    "JsonlEventStore",
    "PlanStatus",
    "PlanStep",
    "PlanStepStatus",
    "ProjectIndexer",
    "RetrievalCandidate",
    "RetrievalQuery",
    "ContextRanker",
    "ContextRetrievalService",
    "InstructionResolver",
    "RepairDiagnosis",
    "RunCheckpoint",
    "RunEvent",
    "RunOrchestrator",
    "RunState",
    "RunTracer",
    "TRACE_SCHEMA_VERSION",
    "TaskType",
    "VerificationCheck",
    "VerificationResult",
    "VerificationStatus",
    "Planner",
    "RepairController",
    "RepairPolicy",
    "VerificationGate",
    "classify_task",
]


def __getattr__(name: str) -> object:
    """Lazily expose components that depend on the existing Session module."""

    if name == "RunOrchestrator":
        from .orchestrator import RunOrchestrator

        return RunOrchestrator
    if name in {"Planner", "classify_task"}:
        from .planning import Planner, classify_task

        return {"Planner": Planner, "classify_task": classify_task}[name]
    if name in {"RepairController", "RepairPolicy"}:
        from .repair import RepairController, RepairPolicy

        return {
            "RepairController": RepairController,
            "RepairPolicy": RepairPolicy,
        }[name]
    if name == "VerificationGate":
        from .verification import VerificationGate

        return VerificationGate
    if name in {
        "ContextRanker",
        "ContextRetrievalService",
        "InstructionResolver",
        "ProjectIndexer",
        "RetrievalCandidate",
        "RetrievalQuery",
    }:
        from .project_knowledge import (
            ContextRanker,
            ContextRetrievalService,
            InstructionResolver,
            ProjectIndexer,
            RetrievalCandidate,
            RetrievalQuery,
        )

        return {
            "ContextRanker": ContextRanker,
            "ContextRetrievalService": ContextRetrievalService,
            "InstructionResolver": InstructionResolver,
            "ProjectIndexer": ProjectIndexer,
            "RetrievalCandidate": RetrievalCandidate,
            "RetrievalQuery": RetrievalQuery,
        }[name]
    raise AttributeError(name)
