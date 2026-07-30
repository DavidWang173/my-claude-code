"""Structured, local observability for one harness Run.

The trace contains lifecycle facts and bounded summaries only. It deliberately
does not persist prompts, model output, tool output, or model reasoning.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .models import RunState

logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|"
    r"password|passwd|secret|cookie|session[_-]?cookie)",
    re.IGNORECASE,
)
_PRIVATE_REASONING_KEY = re.compile(
    r"(?:chain[_-]?of[_-]?thought|reasoning(?:[_-]?content)?|private[_-]?reasoning|"
    r"internal[_-]?thoughts?|scratchpad|cot)",
    re.IGNORECASE,
)
_SECRET_TEXT = (
    re.compile(r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9._~+/\-=]{8,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|"
        r"passwd|secret|cookie)\b\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)
_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/[A-Za-z0-9_.@+~ -]+){2,}|"
    r"(?<![A-Za-z0-9_.-])[A-Za-z]:\\(?:[^\\\s]+\\)+[^\\\s]*"
)
_REDACTED = "[REDACTED]"
_REDACTED_PATH = "[REDACTED_PATH]"


class EventType(str, Enum):
    RUN_STARTED = "RUN_STARTED"
    RUN_STATE_CHANGED = "RUN_STATE_CHANGED"
    PLAN_CREATED = "PLAN_CREATED"
    PLAN_UPDATED = "PLAN_UPDATED"
    STEP_STARTED = "STEP_STARTED"
    STEP_COMPLETED = "STEP_COMPLETED"
    STEP_FAILED = "STEP_FAILED"
    MODEL_CALL_STARTED = "MODEL_CALL_STARTED"
    MODEL_CALL_COMPLETED = "MODEL_CALL_COMPLETED"
    TOOL_CALL_REQUESTED = "TOOL_CALL_REQUESTED"
    TOOL_CALL_APPROVAL_REQUIRED = "TOOL_CALL_APPROVAL_REQUIRED"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_COMPLETED = "TOOL_CALL_COMPLETED"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    VERIFICATION_STARTED = "VERIFICATION_STARTED"
    VERIFICATION_COMPLETED = "VERIFICATION_COMPLETED"
    REPAIR_STARTED = "REPAIR_STARTED"
    REPAIR_COMPLETED = "REPAIR_COMPLETED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"
    RUN_CANCELLED = "RUN_CANCELLED"


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class RunEvent:
    event_id: str
    timestamp: datetime
    run_id: str
    session_id: str
    parent_event_id: str | None
    event_type: EventType
    run_state: RunState
    plan_id: str | None = None
    step_id: str | None = None
    tool_call_id: str | None = None
    provider: str | None = None
    model: str | None = None
    duration_ms: float | None = None
    input_summary: Mapping[str, JsonValue] | None = None
    output_summary: Mapping[str, JsonValue] | None = None
    success: bool | None = None
    error_category: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.event_id or not _RUN_ID.fullmatch(self.run_id):
            raise ValueError("event_id and a safe run_id are required")
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.timestamp.tzinfo is None:
            raise ValueError("event timestamp must include a timezone")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("event duration cannot be negative")
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported trace schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "parent_event_id": self.parent_event_id,
            "event_type": self.event_type.value,
            "run_state": self.run_state.value,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "tool_call_id": self.tool_call_id,
            "provider": self.provider,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "input_summary": dict(self.input_summary) if self.input_summary else None,
            "output_summary": dict(self.output_summary) if self.output_summary else None,
            "success": self.success,
            "error_category": self.error_category,
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> RunEvent:
        if not isinstance(value, Mapping):
            raise ValueError("trace event must be an object")
        input_summary = _optional_mapping(value.get("input_summary"), "input_summary")
        output_summary = _optional_mapping(value.get("output_summary"), "output_summary")
        metadata = _optional_mapping(value.get("metadata"), "metadata") or {}
        timestamp = datetime.fromisoformat(_required_string(value.get("timestamp"), "timestamp"))
        duration = value.get("duration_ms")
        if duration is not None and (
            not isinstance(duration, (int, float)) or isinstance(duration, bool)
        ):
            raise ValueError("duration_ms must be numeric")
        success = value.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValueError("success must be boolean")
        return cls(
            event_id=_required_string(value.get("event_id"), "event_id"),
            timestamp=timestamp,
            run_id=_required_string(value.get("run_id"), "run_id"),
            session_id=_required_string(value.get("session_id"), "session_id"),
            parent_event_id=_optional_string(value.get("parent_event_id"), "parent_event_id"),
            event_type=EventType(value.get("event_type")),
            run_state=RunState(value.get("run_state")),
            plan_id=_optional_string(value.get("plan_id"), "plan_id"),
            step_id=_optional_string(value.get("step_id"), "step_id"),
            tool_call_id=_optional_string(value.get("tool_call_id"), "tool_call_id"),
            provider=_optional_string(value.get("provider"), "provider"),
            model=_optional_string(value.get("model"), "model"),
            duration_ms=float(duration) if duration is not None else None,
            input_summary=input_summary,
            output_summary=output_summary,
            success=success,
            error_category=_optional_string(value.get("error_category"), "error_category"),
            metadata=metadata,
            schema_version=value.get("schema_version", TRACE_SCHEMA_VERSION),  # type: ignore[arg-type]
        )


class EventStore(Protocol):
    def append(self, event: RunEvent) -> bool: ...

    def read(self, run_id: str) -> tuple[RunEvent, ...]: ...


class NullEventStore:
    """No-op store used by embedders that have not configured local traces."""

    def append(self, event: RunEvent) -> bool:
        del event
        return True

    def read(self, run_id: str) -> tuple[RunEvent, ...]:
        del run_id
        return ()


class JsonlEventStore:
    """Append-only, one-file-per-Run JSONL store with corrupt-line recovery."""

    def __init__(
        self,
        root: Path,
        *,
        secrets: tuple[str | None, ...] = (),
        max_file_bytes: int = 64_000_000,
    ) -> None:
        requested = root.expanduser()
        if requested.is_symlink():
            raise ValueError("trace directory must not be a symbolic link")
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        self._root = requested.resolve()
        self._secrets = tuple(item for item in secrets if item)
        self._max_file_bytes = max_file_bytes

    @property
    def root(self) -> Path:
        return self._root

    def append(self, event: RunEvent) -> bool:
        """Best-effort append. Storage failures are warnings, never Run failures."""

        descriptor: int | None = None
        try:
            self._ensure_root()
            path = self._path(event.run_id)
            flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            current = os.fstat(descriptor)
            if not stat.S_ISREG(current.st_mode):
                raise OSError("trace target is not a regular file")
            if current.st_size > self._max_file_bytes:
                raise OSError("trace file exceeds its size limit")
            payload = sanitize_structured(event.to_dict(), secrets=self._secrets)
            encoded = (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
            if current.st_size + len(encoded) > self._max_file_bytes:
                raise OSError("trace file exceeds its size limit")
            if current.st_size and os.pread(descriptor, 1, current.st_size - 1) != b"\n":
                os.write(descriptor, b"\n")
            os.write(descriptor, encoded)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o600)
            return True
        except (OSError, TypeError, ValueError):
            logger.warning(
                "unable to append local Run trace event run_id=%s event_type=%s",
                event.run_id,
                event.event_type.value,
                exc_info=True,
            )
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def read(self, run_id: str) -> tuple[RunEvent, ...]:
        path = self._path(run_id)
        if not path.exists():
            return ()
        events: list[RunEvent] = []
        try:
            if path.stat().st_size > self._max_file_bytes:
                logger.warning("local Run trace exceeds size limit run_id=%s", run_id)
                return ()
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        payload = json.loads(line)
                        event = RunEvent.from_dict(payload)
                        if event.run_id != run_id:
                            raise ValueError("event run_id does not match trace file")
                    except (json.JSONDecodeError, TypeError, ValueError):
                        logger.warning(
                            "skipping corrupt local Run trace line run_id=%s line=%d",
                            run_id,
                            line_number,
                        )
                        continue
                    events.append(event)
        except (OSError, UnicodeError):
            logger.warning("unable to read local Run trace run_id=%s", run_id, exc_info=True)
            return ()
        return tuple(events)

    def query(
        self,
        run_id: str,
        *,
        event_type: EventType | None = None,
        success: bool | None = None,
        limit: int | None = None,
    ) -> tuple[RunEvent, ...]:
        if limit is not None and limit < 0:
            raise ValueError("trace query limit cannot be negative")
        events = (
            event
            for event in self.read(run_id)
            if (event_type is None or event.event_type is event_type)
            and (success is None or event.success is success)
        )
        selected = tuple(events)
        return selected if limit is None else selected[-limit:]

    def _path(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("invalid run_id")
        return self._root / f"{run_id}.jsonl"

    def _ensure_root(self) -> None:
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self._root.is_symlink() or not self._root.is_dir():
            raise OSError("trace directory is not a regular directory")
        if os.name == "posix":
            os.chmod(self._root, 0o700)


class RunTracer:
    """Small lifecycle facade that owns parent links and elapsed timings."""

    def __init__(
        self,
        store: EventStore | None,
        *,
        run_id: str,
        session_id: str,
        run_state: RunState,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self._store = store or NullEventStore()
        self.run_id = run_id
        self.session_id = session_id
        self.provider = provider
        self.model = model
        self._state = run_state
        self._active: dict[str, tuple[str, float]] = {}
        try:
            existing = self._store.read(run_id)
        except Exception:
            logger.warning("unable to inspect existing local Run trace run_id=%s", run_id)
            existing = ()
        started = next(
            (event for event in existing if event.event_type is EventType.RUN_STARTED),
            None,
        )
        if started is None:
            started = self.emit(
                EventType.RUN_STARTED,
                parent_event_id=None,
                input_summary={"resumed": False},
                success=True,
            )
        self.root_event_id = started.event_id
        self.started_at = started.timestamp
        self.last_model_event_id: str | None = next(
            (
                event.event_id
                for event in reversed(existing)
                if event.event_type is EventType.MODEL_CALL_COMPLETED
            ),
            None,
        )

    def set_state(self, state: RunState) -> None:
        self._state = state

    def emit(
        self,
        event_type: EventType,
        *,
        parent_event_id: str | None = None,
        plan_id: str | None = None,
        step_id: str | None = None,
        tool_call_id: str | None = None,
        duration_ms: float | None = None,
        input_summary: Mapping[str, object] | None = None,
        output_summary: Mapping[str, object] | None = None,
        success: bool | None = None,
        error_category: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> RunEvent:
        event = RunEvent(
            event_id=uuid4().hex,
            timestamp=datetime.now(UTC),
            run_id=self.run_id,
            session_id=self.session_id,
            parent_event_id=(
                parent_event_id
                if parent_event_id is not None or event_type is EventType.RUN_STARTED
                else getattr(self, "root_event_id", None)
            ),
            event_type=event_type,
            run_state=self._state,
            plan_id=plan_id,
            step_id=step_id,
            tool_call_id=tool_call_id,
            provider=self.provider,
            model=self.model,
            duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
            input_summary=_as_summary(input_summary),
            output_summary=_as_summary(output_summary),
            success=success,
            error_category=error_category,
            metadata=_as_summary(metadata) or {},
        )
        try:
            self._store.append(event)
        except Exception:
            logger.warning(
                "unable to append local Run trace event run_id=%s event_type=%s",
                self.run_id,
                event_type.value,
            )
        if event_type is EventType.MODEL_CALL_COMPLETED:
            self.last_model_event_id = event.event_id
        return event

    def start(self, key: str, event_type: EventType, **fields: object) -> RunEvent:
        event = self.emit(event_type, **fields)  # type: ignore[arg-type]
        self._active[key] = (event.event_id, time.monotonic())
        return event

    def finish(self, key: str, event_type: EventType, **fields: object) -> RunEvent:
        active = self._active.pop(key, None)
        parent_id = active[0] if active is not None else None
        duration_ms = (
            (time.monotonic() - active[1]) * 1000 if active is not None else None
        )
        fields.setdefault("parent_event_id", parent_id)
        fields.setdefault("duration_ms", duration_ms)
        return self.emit(event_type, **fields)  # type: ignore[arg-type]

    def activate(self, key: str, parent_event_id: str) -> None:
        """Start timing work whose start event was emitted by another pair."""

        self._active[key] = (parent_event_id, time.monotonic())

    def has_active(self, key: str) -> bool:
        return key in self._active

    def total_duration_ms(self) -> float:
        return max(
            0.0,
            (datetime.now(UTC) - self.started_at.astimezone(UTC)).total_seconds() * 1000,
        )


def sanitize_structured(
    value: object,
    *,
    secrets: tuple[str, ...] = (),
    _key: str | None = None,
) -> JsonValue:
    """Return bounded JSON data with secrets, private reasoning, and paths removed."""

    if _key is not None and _PRIVATE_REASONING_KEY.search(_key):
        return _REDACTED
    if _key is not None and _SENSITIVE_KEY.search(_key):
        return _REDACTED
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, Enum):
        return sanitize_structured(value.value, secrets=secrets, _key=_key)
    if isinstance(value, Path):
        return _REDACTED_PATH
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, _REDACTED)
        for pattern in _SECRET_TEXT:
            redacted = pattern.sub(_REDACTED, redacted)
        redacted = _ABSOLUTE_PATH.sub(_REDACTED_PATH, redacted)
        return redacted[:4096]
    if isinstance(value, Mapping):
        output: dict[str, JsonValue] = {}
        for raw_key, item in list(value.items())[:128]:
            key = str(raw_key)[:128]
            if _PRIVATE_REASONING_KEY.search(key):
                continue
            output[key] = sanitize_structured(item, secrets=secrets, _key=key)
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_structured(item, secrets=secrets, _key=_key) for item in list(value)[:128]]
    return f"<{type(value).__name__}>"


def _as_summary(value: Mapping[str, object] | None) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    sanitized = sanitize_structured(value)
    if not isinstance(sanitized, dict):
        raise ValueError("event summary must be an object")
    return sanitized


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _optional_mapping(value: object, name: str) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return dict(value)  # type: ignore[return-value]
