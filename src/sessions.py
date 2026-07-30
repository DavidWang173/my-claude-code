"""Versioned local sessions stored outside the project workspace."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from .context import ContextState, context_state_from_dict, context_state_to_dict
from .harness.models import (
    ExecutionPlan,
    RunCheckpoint,
    RunState,
    TaskType,
    VerificationResult,
)
from .models import (
    AssistantMessage,
    Message,
    MessageRole,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)

SCHEMA_VERSION = 4
_READABLE_SCHEMA_VERSIONS = frozenset({1, 2, 3, SCHEMA_VERSION})
_SESSION_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
DEFAULT_MAX_SESSION_BYTES = 32_000_000
SessionMessage = Message


def default_session_directory(environ: Mapping[str, str] | None = None) -> Path:
    """Return an OS-appropriate user data directory, never the repository."""

    if environ is None:
        environ = os.environ
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "coding-agent" / "sessions"


@dataclass
class Session:
    schema_version: int
    session_id: str
    workspace: Path
    created_at: datetime
    updated_at: datetime
    provider: str
    model: str
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    context: ContextState = field(default_factory=ContextState)
    run_id: str | None = None
    run_state: RunState | None = None
    task_type: TaskType | None = None
    current_plan: ExecutionPlan | None = None
    current_step_id: str | None = None
    repair_attempts: int = 0
    last_verification: VerificationResult | None = None
    pending_approval: dict[str, object] | None = None
    checkpoint_version: int = 0
    completed_tool_call_ids: list[str] = field(default_factory=list)
    uncertain_tool_call_ids: list[str] = field(default_factory=list)
    run_decision_summary: str | None = None
    run_failure_reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        workspace: Path | None = None,
        provider: str = "unconfigured",
        model: str = "unconfigured",
    ) -> Session:
        now = datetime.now(UTC)
        return cls(
            schema_version=SCHEMA_VERSION,
            session_id=uuid4().hex,
            workspace=(workspace or Path.cwd()).expanduser().resolve(),
            created_at=now,
            updated_at=now,
            provider=provider,
            model=model,
        )

    @property
    def id(self) -> str:
        """Compatibility alias for the task-1 session skeleton."""

        return self.session_id

    def add_message(self, message: Message | str, content: str | None = None) -> None:
        """Append a validated message and refresh the update timestamp.

        The ``role, content`` form is retained for compatibility. New code
        should pass one of the four concrete message classes.
        """

        if isinstance(message, str):
            message = Message(role=cast(MessageRole, message), content=content)
        _validate_tool_relationships([*self.messages, message])
        self.messages.append(message)
        self.updated_at = datetime.now(UTC)

    def add_usage(self, usage: Usage) -> None:
        self.usage = Usage(
            prompt_tokens=self.usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens + usage.completion_tokens,
            total_tokens=self.usage.total_tokens + usage.total_tokens,
        )
        self.updated_at = datetime.now(UTC)

    def apply_checkpoint(self, checkpoint: RunCheckpoint) -> None:
        """Copy one validated lifecycle checkpoint into the existing Session."""

        self.run_id = checkpoint.run_id
        self.run_state = checkpoint.run_state
        self.task_type = checkpoint.task_type
        self.current_plan = checkpoint.current_plan
        self.current_step_id = checkpoint.current_step_id
        self.repair_attempts = checkpoint.repair_attempts
        self.last_verification = checkpoint.last_verification
        self.pending_approval = (
            dict(checkpoint.pending_approval)
            if checkpoint.pending_approval is not None
            else None
        )
        self.checkpoint_version = checkpoint.checkpoint_version
        self.completed_tool_call_ids = list(checkpoint.completed_tool_call_ids)
        self.uncertain_tool_call_ids = list(checkpoint.uncertain_tool_call_ids)
        self.run_decision_summary = checkpoint.decision_summary
        self.run_failure_reason = checkpoint.failure_reason
        self.updated_at = datetime.now(UTC)

    def run_checkpoint(self) -> RunCheckpoint | None:
        if self.run_state is None:
            return None
        return RunCheckpoint(
            run_id=self.run_id or uuid4().hex,
            run_state=self.run_state,
            task_type=self.task_type or TaskType.INFORMATIONAL,
            current_plan=self.current_plan,
            current_step_id=self.current_step_id,
            repair_attempts=self.repair_attempts,
            last_verification=self.last_verification,
            pending_approval=(
                dict(self.pending_approval)
                if self.pending_approval is not None
                else None
            ),
            checkpoint_version=max(1, self.checkpoint_version),
            completed_tool_call_ids=list(self.completed_tool_call_ids),
            uncertain_tool_call_ids=list(self.uncertain_tool_call_ids),
            decision_summary=self.run_decision_summary,
            failure_reason=self.run_failure_reason,
        )


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    workspace: Path
    created_at: datetime
    updated_at: datetime
    provider: str
    model: str
    message_count: int
    usage: Usage

    @classmethod
    def from_session(cls, session: Session) -> SessionSummary:
        return cls(
            session_id=session.session_id,
            workspace=session.workspace,
            created_at=session.created_at,
            updated_at=session.updated_at,
            provider=session.provider,
            model=session.model,
            message_count=len(session.messages),
            usage=session.usage,
        )


@dataclass(frozen=True)
class SessionFileError:
    path: Path
    message: str


@dataclass(frozen=True)
class SessionListResult:
    sessions: tuple[SessionSummary, ...]
    errors: tuple[SessionFileError, ...] = ()


class SessionStore(Protocol):
    def create(self, *, workspace: Path, provider: str, model: str) -> Session: ...

    def save(self, session: Session) -> None: ...

    def load(self, session_id: str) -> Session: ...

    def load_latest(self, *, workspace: Path | None = None) -> Session: ...

    def list_sessions(self) -> SessionListResult: ...

    def delete(self, session_id: str) -> None: ...


class SessionError(RuntimeError):
    pass


class SessionNotFoundError(SessionError):
    pass


class SessionCorruptedError(SessionError):
    pass


class SessionStorageError(SessionError):
    pass


class JsonSessionStore:
    """JSON session store using durable same-directory atomic replacement."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        secrets: tuple[str | None, ...] = (),
        max_session_bytes: int = DEFAULT_MAX_SESSION_BYTES,
    ) -> None:
        requested_root = (root or default_session_directory()).expanduser()
        if requested_root.is_symlink():
            raise ValueError("session directory must not be a symbolic link")
        self._root = requested_root.resolve()
        if _inside_git_worktree(self._root):
            raise ValueError("session directory must be outside a Git workspace")
        if max_session_bytes <= 0:
            raise ValueError("max_session_bytes must be positive")
        self._secrets = tuple(secret for secret in secrets if secret)
        self._max_session_bytes = max_session_bytes

    @property
    def root(self) -> Path:
        return self._root

    def create(self, *, workspace: Path, provider: str, model: str) -> Session:
        session = Session.create(workspace=workspace, provider=provider, model=model)
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        if session.schema_version != SCHEMA_VERSION:
            raise SessionStorageError(
                f"cannot save unsupported session schema {session.schema_version}"
            )
        _validate_session(session)
        path = self._path_for(session.session_id)
        self._ensure_root()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._root,
                prefix=f".{session.session_id}-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                payload = _redact_secrets(session_to_dict(session), self._secrets)
                encoded_bytes = 0
                encoder = json.JSONEncoder(ensure_ascii=False, indent=2)
                for chunk in encoder.iterencode(payload):
                    encoded_bytes += len(chunk.encode("utf-8"))
                    if encoded_bytes > self._max_session_bytes:
                        raise SessionStorageError(
                            "session exceeds the configured storage size limit"
                        )
                    stream.write(chunk)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, path)
        except SessionStorageError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        except (OSError, TypeError, ValueError):
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise SessionStorageError(f"unable to save session {session.session_id}") from None

    def load(self, session_id: str) -> Session:
        self._secure_existing_root()
        path = self._path_for(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"session not found: {session_id}")
        return self._load_path(path)

    def load_latest(self, *, workspace: Path | None = None) -> Session:
        expected_workspace = workspace.expanduser().resolve() if workspace is not None else None
        result = self.list_sessions()
        summaries = (
            summary
            for summary in result.sessions
            if expected_workspace is None or summary.workspace == expected_workspace
        )
        try:
            latest = next(summaries)
        except StopIteration:
            scope = f" for workspace {expected_workspace}" if expected_workspace else ""
            raise SessionNotFoundError(f"no resumable sessions found{scope}") from None
        return self.load(latest.session_id)

    def list_sessions(self) -> SessionListResult:
        if not self._root.exists():
            return SessionListResult(())
        self._secure_existing_root()
        summaries: list[SessionSummary] = []
        errors: list[SessionFileError] = []
        for path in self._root.glob("*.json"):
            try:
                summaries.append(SessionSummary.from_session(self._load_path(path)))
            except SessionCorruptedError as exc:
                errors.append(SessionFileError(path=path, message=str(exc)))
        summaries.sort(key=lambda item: item.updated_at, reverse=True)
        errors.sort(key=lambda item: item.path.name)
        return SessionListResult(tuple(summaries), tuple(errors))

    def delete(self, session_id: str) -> None:
        self._secure_existing_root()
        path = self._path_for(session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            raise SessionNotFoundError(f"session not found: {session_id}") from None
        except OSError:
            raise SessionStorageError(f"unable to delete session {session_id}") from None

    def _load_path(self, path: Path) -> Session:
        descriptor: int | None = None
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("session path is not a regular file")
            if file_stat.st_size > self._max_session_bytes:
                raise ValueError("session file exceeds the configured size limit")
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = None
                payload = json.load(stream)
            session = session_from_dict(payload)
            if path.stem != session.session_id:
                raise ValueError("session filename does not match session_id")
            return session
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise SessionCorruptedError(f"session file is corrupted: {path.name}") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _path_for(self, session_id: str) -> Path:
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("invalid session id")
        return self._root / f"{session_id}.json"

    def _ensure_root(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self._root, 0o700)
        except OSError:
            raise SessionStorageError("unable to create the user session directory") from None

    def _secure_existing_root(self) -> None:
        try:
            if not self._root.is_dir():
                raise SessionStorageError("session storage path is not a directory")
            if os.name == "posix":
                os.chmod(self._root, 0o700)
        except OSError:
            raise SessionStorageError("unable to secure the user session directory") from None


def message_to_dict(message: Message) -> dict[str, object]:
    payload: dict[str, object] = {"role": message.role, "content": message.content}
    if message.role == "assistant" and message.tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
            for call in message.tool_calls
        ]
    elif message.role == "tool":
        payload["tool_call_id"] = message.tool_call_id
    return payload


def message_from_dict(payload: object) -> Message:
    item = _mapping(payload, "message")
    role = item.get("role")
    content = item.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("message content must be text or null")
    if role == "system":
        return SystemMessage(_required_content(content, role))
    if role == "user":
        return UserMessage(_required_content(content, role))
    if role == "assistant":
        raw_calls = item.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError("assistant tool_calls must be a list")
        calls = tuple(_tool_call_from_dict(call) for call in raw_calls)
        return AssistantMessage(content, tool_calls=calls)
    if role == "tool":
        tool_call_id = item.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            raise ValueError("tool result requires tool_call_id")
        return ToolMessage(_required_content(content, role), tool_call_id=tool_call_id)
    raise ValueError("message role is invalid")


def session_to_dict(session: Session) -> dict[str, object]:
    return {
        "schema_version": session.schema_version,
        "session_id": session.session_id,
        "workspace": str(session.workspace),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "provider": session.provider,
        "model": session.model,
        "messages": [message_to_dict(message) for message in session.messages],
        "usage": {
            "prompt_tokens": session.usage.prompt_tokens,
            "completion_tokens": session.usage.completion_tokens,
            "total_tokens": session.usage.total_tokens,
        },
        "context": context_state_to_dict(session.context),
        "run_state": session.run_state.value if session.run_state else None,
        "task_type": session.task_type.value if session.task_type else None,
        "current_plan": (
            session.current_plan.to_dict() if session.current_plan is not None else None
        ),
        "current_step_id": session.current_step_id,
        "repair_attempts": session.repair_attempts,
        "last_verification": (
            session.last_verification.to_dict()
            if session.last_verification is not None
            else None
        ),
        "pending_approval": session.pending_approval,
        "checkpoint_version": session.checkpoint_version,
        "run_id": session.run_id,
        "completed_tool_call_ids": list(session.completed_tool_call_ids),
        "uncertain_tool_call_ids": list(session.uncertain_tool_call_ids),
        "run_decision_summary": session.run_decision_summary,
        "run_failure_reason": session.run_failure_reason,
    }


def _redact_secrets(value: object, secrets: tuple[str, ...]) -> object:
    if not secrets:
        return value
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact_secrets(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [_redact_secrets(item, secrets) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _redact_secrets(item, secrets)
            for key, item in value.items()
        }
    return value


def session_from_dict(payload: object) -> Session:
    item = _mapping(payload, "session")
    schema_version = item.get("schema_version")
    if schema_version not in _READABLE_SCHEMA_VERSIONS:
        raise SessionCorruptedError(
            f"unsupported session schema: {schema_version}; "
            f"expected one of {sorted(_READABLE_SCHEMA_VERSIONS)}"
        )
    session_id = item.get("session_id")
    workspace_value = item.get("workspace")
    provider = item.get("provider")
    model = item.get("model")
    if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("session_id is invalid")
    if not isinstance(workspace_value, str) or not Path(workspace_value).is_absolute():
        raise ValueError("session workspace must be an absolute path")
    if not isinstance(provider, str) or not provider:
        raise ValueError("session provider is invalid")
    if not isinstance(model, str) or not model:
        raise ValueError("session model is invalid")

    raw_messages = item.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("session messages must be a list")
    messages = [message_from_dict(message) for message in raw_messages]
    usage_item = _mapping(item.get("usage"), "usage")
    usage = Usage(
        prompt_tokens=_token_count(usage_item.get("prompt_tokens"), "prompt_tokens"),
        completion_tokens=_token_count(
            usage_item.get("completion_tokens"), "completion_tokens"
        ),
        total_tokens=_token_count(usage_item.get("total_tokens"), "total_tokens"),
    )
    raw_plan = item.get("current_plan")
    raw_verification = item.get("last_verification")
    raw_pending = item.get("pending_approval")
    if raw_pending is not None and not isinstance(raw_pending, dict):
        raise ValueError("pending_approval must be an object or null")
    completed_tool_call_ids = _string_list(
        item.get("completed_tool_call_ids", []), "completed_tool_call_ids"
    )
    uncertain_tool_call_ids = _string_list(
        item.get("uncertain_tool_call_ids", []), "uncertain_tool_call_ids"
    )
    raw_run_state = item.get("run_state")
    raw_task_type = item.get("task_type")
    session = Session(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        workspace=Path(workspace_value),
        created_at=_datetime(item.get("created_at"), "created_at"),
        updated_at=_datetime(item.get("updated_at"), "updated_at"),
        provider=provider,
        model=model,
        messages=messages,
        usage=usage,
        context=(
            context_state_from_dict(item.get("context"))
            if schema_version >= 2
            else ContextState()
        ),
        run_id=_nullable_text(item.get("run_id"), "run_id"),
        run_state=RunState(raw_run_state) if raw_run_state is not None else None,
        task_type=TaskType(raw_task_type) if raw_task_type is not None else None,
        current_plan=(
            ExecutionPlan.from_dict(raw_plan) if raw_plan is not None else None
        ),
        current_step_id=_nullable_text(
            item.get("current_step_id"), "current_step_id"
        ),
        repair_attempts=_token_count(
            item.get("repair_attempts", 0), "repair_attempts"
        ),
        last_verification=(
            VerificationResult.from_dict(raw_verification)
            if raw_verification is not None
            else None
        ),
        pending_approval=dict(raw_pending) if raw_pending is not None else None,
        checkpoint_version=_token_count(
            item.get("checkpoint_version", 0), "checkpoint_version"
        ),
        completed_tool_call_ids=completed_tool_call_ids,
        uncertain_tool_call_ids=uncertain_tool_call_ids,
        run_decision_summary=_nullable_text(
            item.get("run_decision_summary"), "run_decision_summary"
        ),
        run_failure_reason=_nullable_text(
            item.get("run_failure_reason"), "run_failure_reason"
        ),
    )
    _validate_session(session)
    return session


def _validate_session(session: Session) -> None:
    if session.schema_version != SCHEMA_VERSION:
        raise ValueError("session schema_version is unsupported")
    if not _SESSION_ID_PATTERN.fullmatch(session.session_id):
        raise ValueError("session_id is invalid")
    if not session.workspace.is_absolute():
        raise ValueError("session workspace must be absolute")
    if not session.provider or not session.model:
        raise ValueError("session provider and model are required")
    if session.created_at.tzinfo is None or session.updated_at.tzinfo is None:
        raise ValueError("session timestamps must include a timezone")
    if session.updated_at < session.created_at:
        raise ValueError("session updated_at precedes created_at")
    _validate_tool_relationships(session.messages)
    if session.repair_attempts < 0 or session.checkpoint_version < 0:
        raise ValueError("session checkpoint counters cannot be negative")
    if session.run_state is None and any(
        (
            session.run_id,
            session.task_type,
            session.current_plan,
            session.current_step_id,
            session.last_verification,
            session.pending_approval,
        )
    ):
        raise ValueError("run checkpoint data requires run_state")


def _validate_tool_relationships(messages: list[Message]) -> None:
    requested: set[str] = set()
    completed: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                if call.id in requested:
                    raise ValueError(f"duplicate tool call id: {call.id}")
                requested.add(call.id)
        elif message.role == "tool":
            call_id = cast(str, message.tool_call_id)
            if call_id not in requested:
                raise ValueError(f"tool result references unknown call: {call_id}")
            if call_id in completed:
                raise ValueError(f"duplicate tool result for call: {call_id}")
            completed.add(call_id)


def _tool_call_from_dict(payload: object) -> ToolCall:
    item = _mapping(payload, "tool call")
    call_id = item.get("id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise ValueError("tool call id and name must be text")
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must be an object")
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_content(content: str | None, role: object) -> str:
    if content is None:
        raise ValueError(f"{role} message content is required")
    return content


def _token_count(value: object, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nullable_text(value: object, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{label} must be text or null")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{label} must be a string list")
    return list(value)


def _datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _inside_git_worktree(path: Path) -> bool:
    return any((candidate / ".git").exists() for candidate in (path, *path.parents))
