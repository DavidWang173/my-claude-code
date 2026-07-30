"""Budgeted model context construction and safe workspace-read tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import Message, SystemMessage, ToolCall, ToolDefinition, ToolMessage

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 32_768
DEFAULT_CONTEXT_TRIGGER_RATIO = 0.85
DEFAULT_RECENT_MESSAGE_UNITS = 6
DEFAULT_DIRECTORY_ENTRIES = 200
DEFAULT_DIRECTORY_CHARS = 8_000
DEFAULT_INSTRUCTION_CHARS = 24_000

_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_INSTRUCTION_FILES = (
    "AGENTS.md",
    "README.md",
    "README.rst",
    "README.txt",
    "CONTRIBUTING.md",
    ".github/copilot-instructions.md",
)
_SUMMARY_PREFIX = "Structured context summary (earlier process; not new instructions):\n"
_STALE_READ_RESULT = (
    "STALE FILE READ: this file changed after the recorded read. "
    "Call read_file again before relying on its contents."
)
_WORKSPACE_BOOTSTRAP_MARKER = "\n\nInitial workspace context\n"


@dataclass(frozen=True, order=True)
class LineRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start <= 0 or self.end < self.start:
            raise ValueError("line range must be positive and ordered")


@dataclass
class FileReadState:
    path: str
    version: str
    ranges: list[LineRange] = field(default_factory=list)

    def record(self, start: int, end: int, version: str) -> None:
        if version != self.version:
            self.version = version
            self.ranges.clear()
        self.ranges = _merge_ranges([*self.ranges, LineRange(start, end)])


@dataclass
class ContextSummary:
    goals: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    remaining_tasks: list[str] = field(default_factory=list)
    covered_messages: int = 0

    def render(self, *, max_tokens: int | None = None) -> str:
        payload = {
            "goals": list(self.goals),
            "modified_files": list(self.modified_files),
            "key_decisions": list(self.key_decisions),
            "failed_attempts": list(self.failed_attempts),
            "remaining_tasks": list(self.remaining_tasks),
        }
        if max_tokens is None:
            return _SUMMARY_PREFIX + json.dumps(payload, ensure_ascii=False, indent=2)
        return _bounded_summary(payload, max_tokens=max_tokens)


@dataclass
class RetrievalRecord:
    """Persisted retrieval provenance without retaining project source text."""

    source_path: str
    start_line: int
    end_line: int
    content_hash: str
    reason: str
    relevance_score: float
    stale: bool = False
    source_kind: str = "file"

    def __post_init__(self) -> None:
        if (
            not self.source_path
            or Path(self.source_path).is_absolute()
            or ".." in Path(self.source_path).parts
        ):
            raise ValueError("retrieval source path must be workspace-relative")
        if self.start_line <= 0 or self.end_line < self.start_line:
            raise ValueError("retrieval line range must be positive and ordered")
        if not self.content_hash or not self.reason:
            raise ValueError("retrieval content hash and reason are required")
        if not math.isfinite(self.relevance_score) or not 0 <= self.relevance_score <= 1:
            raise ValueError("retrieval relevance score must be between zero and one")
        if self.source_kind not in {"file", "instruction", "git", "runtime"}:
            raise ValueError("retrieval source kind is invalid")


@dataclass
class ContextState:
    read_files: dict[str, FileReadState] = field(default_factory=dict)
    read_call_paths: dict[str, str] = field(default_factory=dict)
    stale_read_calls: set[str] = field(default_factory=set)
    retrieval_results: list[RetrievalRecord] = field(default_factory=list)
    workspace_bootstrap_sent: bool = False
    summary: ContextSummary = field(default_factory=ContextSummary)
    compression_count: int = 0
    last_prompt_tokens: int = 0
    last_usage_estimated: bool = False


@dataclass(frozen=True)
class ContextSelection:
    messages: tuple[Message, ...]
    estimated_tokens: int
    compressed: bool
    omitted_messages: int = 0


class ToolResultLike(Protocol):
    @property
    def success(self) -> bool: ...

    @property
    def metadata(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class _MessageUnit:
    indices: tuple[int, ...]
    messages: tuple[Message, ...]
    pending_tool_result: bool = False


class ContextManager:
    """Build provider context without discarding the session's full local history."""

    def __init__(
        self,
        *,
        max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        trigger_ratio: float = DEFAULT_CONTEXT_TRIGGER_RATIO,
        recent_units: int = DEFAULT_RECENT_MESSAGE_UNITS,
        directory_entries: int = DEFAULT_DIRECTORY_ENTRIES,
        directory_chars: int = DEFAULT_DIRECTORY_CHARS,
        instruction_chars: int = DEFAULT_INSTRUCTION_CHARS,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max context tokens must be positive")
        if not 0 < trigger_ratio <= 1:
            raise ValueError("context trigger ratio must be between zero and one")
        if recent_units < 0:
            raise ValueError("recent context units must not be negative")
        if min(directory_entries, directory_chars, instruction_chars) <= 0:
            raise ValueError("workspace context limits must be positive")
        self.max_tokens = max_tokens
        self._trigger_ratio = trigger_ratio
        self._recent_units = recent_units
        self._directory_entries = directory_entries
        self._directory_chars = directory_chars
        self._instruction_chars = instruction_chars

    @property
    def retrieval_token_budget(self) -> int:
        """Reserve a bounded share for ephemeral, source-backed retrieval."""

        return max(64, min(4_096, self.max_tokens // 3))

    def initial_system_prompt(self, workspace: Path, base_prompt: str) -> str:
        """Add names-only directory context and explicit instruction files."""

        root = workspace.expanduser().resolve()
        directory = _compact_directory(
            root,
            max_entries=self._directory_entries,
            max_chars=self._directory_chars,
        )
        instructions = _read_instruction_files(root, max_chars=self._instruction_chars)
        return (
            f"{base_prompt.rstrip()}\n\n"
            "Initial workspace context\n"
            "-------------------------\n"
            "The directory section contains names only. Read source files with tools before "
            "using or changing their contents.\n\n"
            f"Compact directory:\n{directory}\n\n"
            f"Explicit project instructions:\n{instructions}\n"
        )

    def select(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
        state: ContextState,
        *,
        session_id: str,
        dynamic_context: str | None = None,
    ) -> ContextSelection:
        """Return a budgeted provider view while keeping tool groups atomic."""

        full = _mask_stale_read_results(messages, state.stale_read_calls)
        if state.workspace_bootstrap_sent and full and full[0].role == "system":
            full = (
                SystemMessage(_without_workspace_bootstrap(full[0].content or "")),
                *full[1:],
            )
        state.workspace_bootstrap_sent = True
        protected_prefix = 1 if full and full[0].role == "system" else 0
        if dynamic_context:
            if protected_prefix:
                combined = SystemMessage(
                    f"{(full[0].content or '').rstrip()}\n\n{dynamic_context}"
                )
                full = (combined, *full[1:])
            else:
                full = (SystemMessage(dynamic_context), *full)
                protected_prefix = 1
        if not full:
            return ContextSelection((), estimate_request_tokens((), tools), False)
        full_tokens = estimate_request_tokens(full, tools)
        trigger = max(1, math.floor(self.max_tokens * self._trigger_ratio))
        if full_tokens < trigger:
            return ContextSelection(full, full_tokens, False)

        protected = tuple(full[:protected_prefix])
        body_start = protected_prefix
        units = _message_units(full, body_start)
        active_users = _active_user_indices(full)
        mandatory = {
            index
            for index, unit in enumerate(units)
            if unit.pending_tool_result or any(item in active_users for item in unit.indices)
        }
        selected = set(mandatory)
        for index in range(len(units) - 1, -1, -1):
            if len(selected - mandatory) >= self._recent_units:
                break
            selected.add(index)

        tools_tokens = estimate_tools_tokens(tools)
        summary_reserve = min(1_200, max(160, self.max_tokens // 5))
        base_messages = protected
        while True:
            chosen = tuple(
                message
                for index, unit in enumerate(units)
                if index in selected
                for message in unit.messages
            )
            estimated_without_summary = (
                estimate_messages_tokens((*base_messages, *chosen)) + tools_tokens
            )
            optional = sorted(selected - mandatory)
            if estimated_without_summary + summary_reserve <= self.max_tokens or not optional:
                break
            selected.remove(optional[0])

        omitted_units = [
            unit for index, unit in enumerate(units) if index not in selected
        ]
        omitted = tuple(message for unit in omitted_units for message in unit.messages)
        active_tasks = [
            full[index].content or ""
            for index in sorted(active_users)
            if full[index].content
        ]
        _update_summary(state.summary, omitted, full, active_tasks)
        state.compression_count += 1

        chosen_messages = tuple(
            message
            for index, unit in enumerate(units)
            if index in selected
            for message in unit.messages
        )
        fixed_tokens = (
            estimate_messages_tokens((*base_messages, *chosen_messages))
            + tools_tokens
        )
        summary_message = SystemMessage(
            state.summary.render(
                max_tokens=max(32, self.max_tokens - fixed_tokens - 8)
            )
        )
        selected_messages = (
            (*base_messages, summary_message, *chosen_messages)
            if omitted
            else (*base_messages, *chosen_messages)
        )
        estimated = estimate_request_tokens(selected_messages, tools)
        logger.info(
            "context compressed session=%s messages_before=%d messages_after=%d "
            "omitted=%d estimated_tokens=%d budget=%d count=%d",
            session_id,
            len(full),
            len(selected_messages),
            len(omitted),
            estimated,
            self.max_tokens,
            state.compression_count,
        )
        return ContextSelection(
            tuple(selected_messages),
            estimated,
            True,
            omitted_messages=len(omitted),
        )

    def observe_tool_result(
        self,
        state: ContextState,
        call: ToolCall,
        result: ToolResultLike,
    ) -> None:
        """Track reads and conservatively invalidate versions after write attempts."""

        metadata = result.metadata
        if call.name in {"apply_patch", "create_file", "replace_range"}:
            if metadata.get("dry_run", call.arguments.get("dry_run", False)):
                return
            raw_files = metadata.get("files")
            modified = (
                {path for path in raw_files if isinstance(path, str)}
                if isinstance(raw_files, (list, tuple))
                else set(_modified_paths(call))
            )
            for path in modified:
                state.read_files.pop(path, None)
            state.stale_read_calls.update(
                call_id
                for call_id, path in state.read_call_paths.items()
                if path in modified
            )
            return
        if call.name == "run_shell":
            # A process may write before failing; argv classification cannot prove otherwise.
            state.read_files.clear()
            state.stale_read_calls.update(state.read_call_paths)
            return
        if not result.success:
            return
        if call.name == "read_file":
            path = metadata.get("path")
            start = metadata.get("start_line")
            end = metadata.get("end_line")
            version = metadata.get("version")
            if (
                isinstance(path, str)
                and isinstance(start, int)
                and isinstance(end, int)
                and isinstance(version, str)
                and start > 0
                and end >= start
            ):
                existing = state.read_files.get(path)
                if existing is None:
                    existing = FileReadState(path=path, version=version)
                    state.read_files[path] = existing
                existing.record(start, end, version)
                state.read_call_paths[call.id] = path
                state.stale_read_calls.discard(call.id)
            return


def estimate_request_tokens(
    messages: Sequence[Message],
    tools: Sequence[ToolDefinition] = (),
) -> int:
    return estimate_messages_tokens(messages) + estimate_tools_tokens(tools)


def estimate_messages_tokens(messages: Sequence[Message]) -> int:
    total = 2
    for message in messages:
        total += 4 + _estimate_text(message.role)
        if message.content:
            total += _estimate_text(message.content)
        if message.tool_call_id:
            total += _estimate_text(message.tool_call_id)
        for call in message.tool_calls:
            total += 6 + _estimate_text(call.id) + _estimate_text(call.name)
            total += _estimate_json(call.arguments)
    return total


def estimate_tools_tokens(tools: Sequence[ToolDefinition]) -> int:
    return sum(
        8
        + _estimate_text(tool.name)
        + _estimate_text(tool.description)
        + _estimate_json(tool.parameters)
        for tool in tools
    )


def estimate_completion_tokens(text: str, calls: Sequence[ToolCall]) -> int:
    return _estimate_text(text) + sum(
        6 + _estimate_text(call.name) + _estimate_json(call.arguments)
        for call in calls
    )


def context_state_to_dict(state: ContextState) -> dict[str, object]:
    return {
        "read_files": [
            {
                "path": item.path,
                "version": item.version,
                "ranges": [[line_range.start, line_range.end] for line_range in item.ranges],
            }
            for item in sorted(state.read_files.values(), key=lambda value: value.path)
        ],
        "read_call_paths": dict(sorted(state.read_call_paths.items())),
        "stale_read_calls": sorted(state.stale_read_calls),
        "workspace_bootstrap_sent": state.workspace_bootstrap_sent,
        "retrieval_results": [
            {
                "source_path": item.source_path,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "content_hash": item.content_hash,
                "reason": item.reason,
                "relevance_score": item.relevance_score,
                "stale": item.stale,
                "source_kind": item.source_kind,
            }
            for item in state.retrieval_results
        ],
        "summary": {
            "goals": list(state.summary.goals),
            "modified_files": list(state.summary.modified_files),
            "key_decisions": list(state.summary.key_decisions),
            "failed_attempts": list(state.summary.failed_attempts),
            "remaining_tasks": list(state.summary.remaining_tasks),
            "covered_messages": state.summary.covered_messages,
        },
        "compression_count": state.compression_count,
        "last_prompt_tokens": state.last_prompt_tokens,
        "last_usage_estimated": state.last_usage_estimated,
    }


def context_state_from_dict(value: object) -> ContextState:
    if value is None:
        return ContextState()
    if not isinstance(value, Mapping):
        raise ValueError("session context must be an object")
    raw_files = value.get("read_files", [])
    if not isinstance(raw_files, list):
        raise ValueError("session context read_files must be a list")
    read_files: dict[str, FileReadState] = {}
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            raise ValueError("session context file entry must be an object")
        path = raw_file.get("path")
        version = raw_file.get("version")
        raw_ranges = raw_file.get("ranges", [])
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(version, str)
            or not version
            or not isinstance(raw_ranges, list)
        ):
            raise ValueError("session context file entry is invalid")
        ranges: list[LineRange] = []
        for raw_range in raw_ranges:
            if (
                not isinstance(raw_range, list)
                or len(raw_range) != 2
                or not all(isinstance(item, int) for item in raw_range)
            ):
                raise ValueError("session context line range is invalid")
            ranges.append(LineRange(raw_range[0], raw_range[1]))
        read_files[path] = FileReadState(path, version, _merge_ranges(ranges))

    raw_call_paths = value.get("read_call_paths", {})
    if not isinstance(raw_call_paths, Mapping):
        raise ValueError("session context read_call_paths must be an object")
    read_call_paths: dict[str, str] = {}
    for call_id, path in raw_call_paths.items():
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError("session context read call entry is invalid")
        read_call_paths[call_id] = path
    stale_read_calls = set(
        _string_list(value.get("stale_read_calls", []), "stale_read_calls")
    )
    if not stale_read_calls.issubset(read_call_paths):
        raise ValueError("session context stale read call is unknown")
    workspace_bootstrap_sent = _boolean(
        value.get("workspace_bootstrap_sent", False),
        "workspace_bootstrap_sent",
    )

    raw_retrieval_results = value.get("retrieval_results", [])
    if not isinstance(raw_retrieval_results, list):
        raise ValueError("session context retrieval_results must be a list")
    retrieval_results: list[RetrievalRecord] = []
    for raw_result in raw_retrieval_results:
        if not isinstance(raw_result, Mapping):
            raise ValueError("session context retrieval result must be an object")
        source_path = raw_result.get("source_path")
        start_line = raw_result.get("start_line")
        end_line = raw_result.get("end_line")
        content_hash = raw_result.get("content_hash")
        reason = raw_result.get("reason")
        relevance_score = raw_result.get("relevance_score")
        stale = raw_result.get("stale", False)
        source_kind = raw_result.get("source_kind", "file")
        if (
            not isinstance(source_path, str)
            or not isinstance(start_line, int)
            or not isinstance(end_line, int)
            or not isinstance(content_hash, str)
            or not isinstance(reason, str)
            or not isinstance(relevance_score, (int, float))
            or isinstance(relevance_score, bool)
            or not isinstance(stale, bool)
            or not isinstance(source_kind, str)
        ):
            raise ValueError("session context retrieval result is invalid")
        retrieval_results.append(
            RetrievalRecord(
                source_path=source_path,
                start_line=start_line,
                end_line=end_line,
                content_hash=content_hash,
                reason=reason,
                relevance_score=float(relevance_score),
                stale=stale,
                source_kind=source_kind,
            )
        )

    raw_summary = value.get("summary", {})
    if not isinstance(raw_summary, Mapping):
        raise ValueError("session context summary must be an object")
    summary = ContextSummary(
        goals=_string_list(raw_summary.get("goals", []), "goals"),
        modified_files=_string_list(
            raw_summary.get("modified_files", []), "modified_files"
        ),
        key_decisions=_string_list(
            raw_summary.get("key_decisions", []), "key_decisions"
        ),
        failed_attempts=_string_list(
            raw_summary.get("failed_attempts", []), "failed_attempts"
        ),
        remaining_tasks=_string_list(
            raw_summary.get("remaining_tasks", []), "remaining_tasks"
        ),
        covered_messages=_non_negative_int(
            raw_summary.get("covered_messages", 0), "covered_messages"
        ),
    )
    return ContextState(
        read_files=read_files,
        read_call_paths=read_call_paths,
        stale_read_calls=stale_read_calls,
        retrieval_results=retrieval_results,
        workspace_bootstrap_sent=workspace_bootstrap_sent,
        summary=summary,
        compression_count=_non_negative_int(
            value.get("compression_count", 0), "compression_count"
        ),
        last_prompt_tokens=_non_negative_int(
            value.get("last_prompt_tokens", 0), "last_prompt_tokens"
        ),
        last_usage_estimated=_boolean(
            value.get("last_usage_estimated", False), "last_usage_estimated"
        ),
    )


def file_version(data: bytes) -> str:
    """Return a stable content version without retaining file contents."""

    return hashlib.sha256(data).hexdigest()


def _without_workspace_bootstrap(value: str) -> str:
    base, marker, _ = value.partition(_WORKSPACE_BOOTSTRAP_MARKER)
    if not marker:
        return value
    return (
        base.rstrip()
        + "\n\nWorkspace context is retrieved dynamically for the current task and plan step."
    )


def _compact_directory(root: Path, *, max_entries: int, max_chars: int) -> str:
    if not root.is_dir():
        return "[workspace directory is unavailable]"
    entries: list[str] = []
    truncated = False
    try:
        for current, directories, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            relative_dir = current_path.relative_to(root)
            depth = len(relative_dir.parts)
            directories[:] = [
                name
                for name in sorted(directories)
                if name not in _IGNORED_DIRECTORIES
                and not (current_path / name).is_symlink()
                and depth < 2
            ]
            names = [*(f"{name}/" for name in directories), *sorted(files)]
            for name in names:
                relative = relative_dir / name
                rendered = relative.as_posix()
                if len(entries) >= max_entries:
                    truncated = True
                    break
                candidate = "\n".join([*entries, rendered])
                if len(candidate) > max_chars:
                    truncated = True
                    break
                entries.append(rendered)
            if truncated:
                break
    except OSError:
        return "[workspace directory could not be inspected]"
    if truncated:
        entries.append("...[additional entries omitted]")
    return "\n".join(entries) if entries else "[workspace contains no visible files]"


def _read_instruction_files(root: Path, *, max_chars: int) -> str:
    sections: list[str] = []
    remaining = max_chars
    for relative_name in _INSTRUCTION_FILES:
        candidate = root / relative_name
        try:
            resolved = candidate.resolve(strict=True)
            if (
                not resolved.is_relative_to(root)
                or not resolved.is_file()
                or resolved.stat().st_size > max_chars * 4
            ):
                continue
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeError, RuntimeError):
            continue
        if remaining <= 0:
            break
        visible = content[:remaining]
        marker = "\n...[instruction file truncated]" if len(content) > len(visible) else ""
        section = f"### {relative_name}\n{visible}{marker}"
        sections.append(section)
        remaining -= len(section)
    return "\n\n".join(sections) if sections else "[no explicit instruction files found]"


def _message_units(messages: Sequence[Message], start: int) -> list[_MessageUnit]:
    units: list[_MessageUnit] = []
    consumed_results: set[int] = set()
    index = start
    while index < len(messages):
        if index in consumed_results:
            index += 1
            continue
        message = messages[index]
        if message.role == "assistant" and message.tool_calls:
            call_ids = {call.id for call in message.tool_calls}
            group_indices = [index]
            group_messages = [message]
            completed: set[str] = set()
            for cursor in range(index + 1, len(messages)):
                candidate = messages[cursor]
                if candidate.role == "tool" and candidate.tool_call_id in call_ids:
                    group_indices.append(cursor)
                    group_messages.append(candidate)
                    completed.add(candidate.tool_call_id or "")
                    consumed_results.add(cursor)
            units.append(
                _MessageUnit(
                    tuple(group_indices),
                    tuple(group_messages),
                    pending_tool_result=completed != call_ids,
                )
            )
            index += 1
            continue
        units.append(_MessageUnit((index,), (message,)))
        index += 1
    return units


def _mask_stale_read_results(
    messages: Sequence[Message],
    stale_call_ids: set[str],
) -> tuple[Message, ...]:
    if not stale_call_ids:
        return tuple(messages)
    return tuple(
        ToolMessage(_STALE_READ_RESULT, tool_call_id=message.tool_call_id or "")
        if message.role == "tool" and message.tool_call_id in stale_call_ids
        else message
        for message in messages
    )


def _active_user_indices(messages: Sequence[Message]) -> set[int]:
    last_completion = -1
    for index, message in enumerate(messages):
        if message.role == "assistant" and not message.tool_calls:
            last_completion = index
    return {
        index
        for index, message in enumerate(messages)
        if index > last_completion and message.role == "user"
    }


def _update_summary(
    summary: ContextSummary,
    omitted: Sequence[Message],
    full: Sequence[Message],
    active_tasks: Sequence[str],
) -> None:
    call_by_id = {
        call.id: call
        for message in full
        if message.role == "assistant"
        for call in message.tool_calls
    }
    result_by_id = {
        message.tool_call_id: message
        for message in full
        if message.role == "tool" and message.tool_call_id is not None
    }
    for message in omitted:
        if message.role == "user" and message.content:
            _append_unique(summary.goals, _clip(message.content, 600), maximum=12)
        elif message.role == "assistant":
            if message.content:
                _append_unique(
                    summary.key_decisions,
                    _clip(message.content, 500),
                    maximum=16,
                )
            for call in message.tool_calls:
                tool_result = result_by_id.get(call.id)
                if tool_result is None or (tool_result.content or "").startswith("ERROR:"):
                    continue
                for path in _modified_paths(call):
                    _append_unique(summary.modified_files, path, maximum=100)
        elif message.role == "tool" and message.content and message.content.startswith("ERROR:"):
            call = call_by_id.get(message.tool_call_id or "")
            name = call.name if call is not None else "tool"
            _append_unique(
                summary.failed_attempts,
                f"{name} failed; inspect the retained session/tool result before retrying",
                maximum=16,
            )
    summary.remaining_tasks = [_clip(task, 1_500) for task in active_tasks[-4:]]
    summary.covered_messages = max(summary.covered_messages, len(omitted))


def _modified_paths(call: ToolCall) -> tuple[str, ...]:
    if call.arguments.get("dry_run", False):
        return ()
    if call.name == "create_file":
        path = call.arguments.get("path")
        return (path,) if isinstance(path, str) else ()
    if call.name == "apply_patch":
        patch = call.arguments.get("patch")
        if not isinstance(patch, str):
            return ()
        paths = []
        for line in patch.splitlines():
            if line.startswith("+++ "):
                path = line[4:].strip()
                if path != "/dev/null":
                    paths.append(path.removeprefix("b/"))
        return tuple(paths)
    return ()


def _merge_ranges(ranges: Sequence[LineRange]) -> list[LineRange]:
    merged: list[LineRange] = []
    for item in sorted(ranges):
        if not merged or item.start > merged[-1].end + 1:
            merged.append(item)
            continue
        previous = merged[-1]
        merged[-1] = LineRange(previous.start, max(previous.end, item.end))
    return merged


def _estimate_text(value: str) -> int:
    return max(1, math.ceil(len(value.encode("utf-8")) / 4))


def _estimate_json(value: object) -> int:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    return _estimate_text(rendered)


def _bounded_summary(
    payload: dict[str, list[str]],
    *,
    max_tokens: int,
) -> str:
    """Render all summary fields while bounding retained detail."""

    working = {
        "goals": [_clip(value, 300) for value in payload["goals"][-4:]],
        "modified_files": [_clip(value, 160) for value in payload["modified_files"][-20:]],
        "key_decisions": [_clip(value, 300) for value in payload["key_decisions"][-5:]],
        "failed_attempts": [_clip(value, 220) for value in payload["failed_attempts"][-5:]],
        "remaining_tasks": [_clip(value, 400) for value in payload["remaining_tasks"][-4:]],
    }

    def render() -> str:
        return _SUMMARY_PREFIX + json.dumps(working, ensure_ascii=False, separators=(",", ":"))

    rendered = render()
    removal_order = (
        "goals",
        "key_decisions",
        "failed_attempts",
        "modified_files",
        "remaining_tasks",
    )
    while _estimate_text(rendered) > max_tokens:
        reduced = False
        for key in removal_order:
            if len(working[key]) > 1:
                del working[key][0]
                reduced = True
                break
        if reduced:
            rendered = render()
            continue
        for maximum in (160, 80, 40):
            clipped = {
                key: [_clip(value, maximum) for value in values]
                for key, values in working.items()
            }
            if clipped != working:
                working = clipped
                reduced = True
                rendered = render()
                if _estimate_text(rendered) <= max_tokens:
                    return rendered
        if _estimate_text(rendered) > max_tokens:
            for key in removal_order:
                if working[key]:
                    working[key].clear()
                    reduced = True
                    rendered = render()
                    break
        if not reduced:
            break
    return rendered


def _append_unique(values: list[str], value: str, *, maximum: int) -> None:
    if value and value not in values:
        values.append(value)
    if len(values) > maximum:
        del values[: len(values) - maximum]


def _clip(value: str, maximum: int) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= maximum else compact[:maximum] + "...[truncated]"


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"session context {label} must be a string list")
    return list(value)


def _non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"session context {label} must be a non-negative integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"session context {label} must be boolean")
    return value
