"""Permission decisions for potentially sensitive agent operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Operation(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"


class PermissionLevel(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRequest:
    operation: Operation
    target: str
    level: PermissionLevel | None = None
    command: str | None = None
    cwd: str | None = None
    risk_reason: str | None = None
    preview: str | None = None

    @property
    def effective_level(self) -> PermissionLevel:
        if self.level is not None:
            return self.level
        return PermissionLevel.ALLOW if self.operation is Operation.READ else PermissionLevel.ASK

    def describe(self) -> str:
        details = [f"level: {self.effective_level.value}", f"target: {self.target}"]
        if self.command is not None:
            details.append(f"command: {self.command}")
        if self.cwd is not None:
            details.append(f"cwd: {self.cwd}")
        if self.risk_reason is not None:
            details.append(f"risk: {self.risk_reason}")
        if self.preview is not None:
            details.append(f"preview:\n{self.preview}")
        return "\n".join(details)


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str


class PermissionPolicy(Protocol):
    def decide(self, request: PermissionRequest) -> PermissionDecision: ...


class ReadOnlyPermissionPolicy:
    """Conservative policy that permits inspection and rejects mutations."""

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        level = request.effective_level
        if level is PermissionLevel.ALLOW:
            return PermissionDecision(True, "operation is classified as allow")
        if level is PermissionLevel.DENY:
            return PermissionDecision(False, request.risk_reason or "operation is denied")
        return PermissionDecision(False, f"{request.operation.value} operations require approval")


class NonInteractivePermissionPolicy(ReadOnlyPermissionPolicy):
    """Allow only pre-classified safe operations; ASK defaults to refusal."""


class InteractivePermissionPolicy:
    """Ask an injected UI callback before potentially mutating operations.

    The callback boundary keeps terminal, desktop, and other approval UIs out of
    the agent runtime. Read operations remain non-interactive by default.
    """

    def __init__(
        self,
        approve: Callable[[PermissionRequest], bool | PermissionDecision],
    ) -> None:
        self._approve = approve

    def decide(self, request: PermissionRequest) -> PermissionDecision:
        level = request.effective_level
        if level is PermissionLevel.ALLOW:
            return PermissionDecision(True, "operation is classified as allow")
        if level is PermissionLevel.DENY:
            return PermissionDecision(False, request.risk_reason or "operation is denied")
        answer = self._approve(request)
        if isinstance(answer, PermissionDecision):
            return answer
        if answer:
            return PermissionDecision(True, "approved interactively")
        return PermissionDecision(False, "denied interactively")
