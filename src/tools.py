"""Tool contracts, registries, and bounded workspace operations."""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .context import file_version
from .models import ToolDefinition
from .permissions import Operation, PermissionLevel, PermissionRequest
from .sandbox import SandboxRuntime


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    working_directory: Path
    permission_granted: bool = False
    approved_request: PermissionRequest | None = None
    output_handler: Callable[[str, str], Awaitable[None]] | None = None
    git_tracker: object | None = None


@dataclass(frozen=True, init=False)
class ToolResult:
    success: bool
    content: str
    error: str | None
    metadata: Mapping[str, object]

    def __init__(
        self,
        content: str = "",
        is_error: bool = False,
        *,
        success: bool | None = None,
        error: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        resolved_success = not is_error if success is None else success
        if is_error and resolved_success:
            raise ValueError("ToolResult cannot be both successful and an error")
        if not resolved_success and error is None:
            error = content or "tool execution failed"
        object.__setattr__(self, "success", resolved_success)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def is_error(self) -> bool:
        return not self.success

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, object]: ...

    @property
    def operation(self) -> Operation: ...

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult: ...


class ToolNotFoundError(LookupError):
    pass


class ToolValidationError(ValueError):
    pass


class ToolRegistry:
    """Holds the tools available to one agent instance."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"tool is not registered: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        )


def validate_tool_arguments(
    arguments: Mapping[str, object],
    schema: Mapping[str, object],
) -> None:
    """Validate arguments against the strict JSON-schema subset tools expose.

    Object properties are closed by default. The supported constraints cover
    agent tool inputs without becoming a general JSON Schema implementation.
    """

    encoded_bytes = 0
    try:
        encoder = json.JSONEncoder(allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        for chunk in encoder.iterencode(arguments):
            encoded_bytes += len(chunk.encode("utf-8"))
            if encoded_bytes > DEFAULT_MAX_TOOL_ARGUMENT_BYTES:
                raise ToolValidationError(
                    "arguments exceed the maximum serialized size of "
                    f"{DEFAULT_MAX_TOOL_ARGUMENT_BYTES} bytes"
                )
    except ToolValidationError:
        raise
    except (TypeError, ValueError):
        raise ToolValidationError("arguments must contain only valid JSON values") from None
    _validate_value(arguments, schema, path="arguments")


def _validate_value(value: object, schema: Mapping[str, object], *, path: str) -> None:
    expected_type = schema.get("type")
    if not isinstance(expected_type, str):
        raise ToolValidationError(f"{path} schema must declare a type")
    if not _matches_type(value, expected_type):
        raise ToolValidationError(f"{path} must be {expected_type}")

    enum = schema.get("enum")
    if enum is not None:
        if not isinstance(enum, list):
            raise ToolValidationError(f"{path} schema enum must be a list")
        if value not in enum:
            raise ToolValidationError(f"{path} is not an allowed value")

    if expected_type == "object":
        _validate_object(value, schema, path=path)
    elif expected_type == "array":
        _validate_array(value, schema, path=path)
    elif expected_type == "string":
        minimum = schema.get("minLength")
        if minimum is not None and (not isinstance(minimum, int) or len(value) < minimum):
            raise ToolValidationError(f"{path} is shorter than minLength")
        maximum = schema.get("maxLength")
        if maximum is not None and (not isinstance(maximum, int) or len(value) > maximum):
            raise ToolValidationError(f"{path} is longer than maxLength")
    elif expected_type in {"integer", "number"}:
        minimum = schema.get("minimum")
        if minimum is not None and (
            not isinstance(minimum, (int, float)) or value < minimum  # type: ignore[operator]
        ):
            raise ToolValidationError(f"{path} is below minimum")
        maximum = schema.get("maximum")
        if maximum is not None and (
            not isinstance(maximum, (int, float)) or value > maximum  # type: ignore[operator]
        ):
            raise ToolValidationError(f"{path} is above maximum")


def _validate_object(value: object, schema: Mapping[str, object], *, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ToolValidationError(f"{path} must be object")
    if not all(isinstance(key, str) for key in value):
        raise ToolValidationError(f"{path} keys must be strings")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ToolValidationError(f"{path} schema properties must be an object")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise ToolValidationError(f"{path} schema required must be a string list")
    missing = [key for key in required if key not in value]
    if missing:
        raise ToolValidationError(f"{path} is missing required field: {missing[0]}")

    additional = schema.get("additionalProperties", False)
    if not isinstance(additional, (bool, Mapping)):
        raise ToolValidationError(
            f"{path} schema additionalProperties must be boolean or an object"
        )
    for key, item in value.items():
        property_schema = properties.get(key)
        if property_schema is None:
            if additional is False:
                raise ToolValidationError(f"{path} contains unknown field: {key}")
            if isinstance(additional, Mapping):
                _validate_value(item, additional, path=f"{path}.{key}")
            continue
        if not isinstance(property_schema, Mapping):
            raise ToolValidationError(f"{path}.{key} schema must be an object")
        _validate_value(item, property_schema, path=f"{path}.{key}")


def _validate_array(value: object, schema: Mapping[str, object], *, path: str) -> None:
    if not isinstance(value, list):
        raise ToolValidationError(f"{path} must be array")
    minimum = schema.get("minItems")
    if minimum is not None and (not isinstance(minimum, int) or len(value) < minimum):
        raise ToolValidationError(f"{path} has fewer items than minItems")
    maximum = schema.get("maxItems")
    if maximum is not None and (not isinstance(maximum, int) or len(value) > maximum):
        raise ToolValidationError(f"{path} has more items than maxItems")
    item_schema = schema.get("items")
    if not isinstance(item_schema, Mapping):
        raise ToolValidationError(f"{path} schema must declare items")
    unique_items = schema.get("uniqueItems", False)
    if not isinstance(unique_items, bool):
        raise ToolValidationError(f"{path} schema uniqueItems must be boolean")
    if unique_items:
        rendered = [
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in value
        ]
        if len(rendered) != len(set(rendered)):
            raise ToolValidationError(f"{path} must contain unique items")
    for index, item in enumerate(value):
        _validate_value(item, item_schema, path=f"{path}[{index}]")


def _matches_type(value: object, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    raise ToolValidationError(f"unsupported schema type: {expected_type}")


DEFAULT_IGNORED_DIRECTORIES = frozenset(
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
DEFAULT_MAX_OUTPUT_CHARS = 100_000
DEFAULT_MAX_FILE_BYTES = 1_000_000
DEFAULT_MAX_TOOL_ARGUMENT_BYTES = 2_000_000
DEFAULT_MAX_PATH_CHARS = 4096
_TRUNCATION_MARKER = "\n...[output truncated]"
_AUTO_RG = object()


class WorkspaceToolError(ValueError):
    pass


class ListFilesTool:
    name = "list_files"
    operation = Operation.READ
    description = (
        "List files under a workspace-relative directory while ignoring common build and "
        "dependency directories."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": DEFAULT_MAX_PATH_CHARS,
            },
            "max_entries": {"type": "integer", "minimum": 1, "maximum": 5000},
        },
        "additionalProperties": False,
    }
    def __init__(self, *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> None:
        self._max_output_chars = max_output_chars

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        try:
            validate_tool_arguments(arguments, self.parameters)
            root = _workspace_root(context)
            target = _resolve_workspace_path(root, _string_argument(arguments, "path", "."))
            if not target.is_dir() and not target.is_file():
                raise WorkspaceToolError("path is not a file or directory")
            max_entries = _integer_argument(arguments, "max_entries", 1000)
            entries: list[str] = []
            truncated = False
            for file_path in _iter_workspace_files(root, target):
                if len(entries) >= max_entries:
                    truncated = True
                    break
                entries.append(file_path.relative_to(root).as_posix())
            content, output_truncated, original_characters = _bounded_output(
                "\n".join(entries),
                max_chars=self._max_output_chars,
                truncated=truncated,
            )
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "path": _display_path(root, target),
                    "entries": len(entries),
                    "truncated": output_truncated,
                    "original_characters": original_characters,
                    "ignored_directories": sorted(DEFAULT_IGNORED_DIRECTORIES),
                },
            )
        except (ToolValidationError, WorkspaceToolError, OSError, ValueError) as exc:
            return _failure(_safe_workspace_error(exc))


class ReadFileTool:
    name = "read_file"
    operation = Operation.READ
    description = (
        "Read a bounded UTF-8 text range from a workspace file. Binary and oversized files "
        "are rejected."
    )

    def __init__(
        self,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_lines: int = 2000,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self._max_file_bytes = max_file_bytes
        self._max_lines = max_lines
        self._max_output_chars = max_output_chars
        self.parameters: Mapping[str, object] = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": DEFAULT_MAX_PATH_CHARS,
                },
                "start_line": {"type": "integer", "minimum": 1},
                "line_count": {"type": "integer", "minimum": 1, "maximum": max_lines},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_file_bytes,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        try:
            validate_tool_arguments(arguments, self.parameters)
            root = _workspace_root(context)
            target = _resolve_workspace_path(root, _string_argument(arguments, "path"))
            if not target.is_file():
                raise WorkspaceToolError("path is not a regular file")
            file_size = target.stat().st_size
            byte_limit = _integer_argument(arguments, "max_bytes", self._max_file_bytes)
            if file_size > byte_limit:
                return _failure(
                    f"file exceeds size limit of {byte_limit} bytes",
                    path=_display_path(root, target),
                    file_size=file_size,
                    max_bytes=byte_limit,
                )
            data = target.read_bytes()
            text = _decode_text(data)
            lines = text.splitlines()
            start_line = _integer_argument(arguments, "start_line", 1)
            line_count = _integer_argument(arguments, "line_count", min(200, self._max_lines))
            start_index = min(start_line - 1, len(lines))
            end_index = min(start_index + line_count, len(lines))
            selected = "\n".join(lines[start_index:end_index])
            line_truncated = end_index < len(lines)
            content, truncated, original_characters = _bounded_output(
                selected,
                max_chars=self._max_output_chars,
                truncated=line_truncated,
            )
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "path": _display_path(root, target),
                    "file_size": file_size,
                    "total_lines": len(lines),
                    "start_line": start_line,
                    "end_line": end_index,
                    "version": file_version(data),
                    "truncated": truncated,
                    "original_characters": original_characters,
                },
            )
        except UnicodeError:
            return _failure("binary or non-UTF-8 file cannot be read as text")
        except (ToolValidationError, WorkspaceToolError, OSError, ValueError) as exc:
            return _failure(_safe_workspace_error(exc))


class SearchTextTool:
    name = "search_text"
    operation = Operation.READ
    description = (
        "Search literal text in workspace files, preferring ripgrep and falling back to a "
        "bounded Python scanner."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 8192},
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": DEFAULT_MAX_PATH_CHARS,
            },
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        rg_executable: str | None | object = _AUTO_RG,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self._rg_executable = (
            shutil.which("rg") if rg_executable is _AUTO_RG else rg_executable
        )
        self._max_file_bytes = max_file_bytes
        self._max_output_chars = max_output_chars

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        try:
            validate_tool_arguments(arguments, self.parameters)
            root = _workspace_root(context)
            target = _resolve_workspace_path(root, _string_argument(arguments, "path", "."))
            query = _string_argument(arguments, "query")
            max_results = _integer_argument(arguments, "max_results", 200)
            if self._rg_executable is not None:
                try:
                    return await self._search_with_rg(
                        root, target, query=query, max_results=max_results
                    )
                except FileNotFoundError:
                    pass
            return self._search_with_python(root, target, query=query, max_results=max_results)
        except (ToolValidationError, WorkspaceToolError, OSError, ValueError) as exc:
            return _failure(_safe_workspace_error(exc))

    async def _search_with_rg(
        self,
        root: Path,
        target: Path,
        *,
        query: str,
        max_results: int,
    ) -> ToolResult:
        assert isinstance(self._rg_executable, str)
        relative_target = _display_path(root, target)
        command = [
            self._rg_executable,
            "--fixed-strings",
            "--line-number",
            "--column",
            "--no-heading",
            "--color=never",
            "--hidden",
        ]
        for ignored in sorted(DEFAULT_IGNORED_DIRECTORIES):
            command.extend(("--glob", f"!**/{ignored}/**"))
        command.extend(("--", query, relative_target))
        return_code, stdout, stderr = await _run_process(command, cwd=root)
        if return_code not in {0, 1}:
            return _failure(
                "ripgrep search failed",
                engine="rg",
                exit_code=return_code,
                detail=_short_error(stderr),
            )
        lines = stdout.splitlines()
        limited = lines[:max_results]
        content, truncated, original_characters = _bounded_output(
            "\n".join(limited),
            max_chars=self._max_output_chars,
            truncated=len(lines) > max_results,
        )
        return ToolResult(
            success=True,
            content=content,
            metadata={
                "engine": "rg",
                "path": relative_target,
                "matches": len(limited),
                "truncated": truncated,
                "original_characters": original_characters,
            },
        )

    def _search_with_python(
        self,
        root: Path,
        target: Path,
        *,
        query: str,
        max_results: int,
    ) -> ToolResult:
        matches: list[str] = []
        truncated = False
        skipped_binary = 0
        skipped_large = 0
        for file_path in _iter_workspace_files(root, target):
            try:
                if file_path.stat().st_size > self._max_file_bytes:
                    skipped_large += 1
                    continue
                text = _decode_text(file_path.read_bytes())
            except (OSError, UnicodeError):
                skipped_binary += 1
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                column = line.find(query)
                if column < 0:
                    continue
                if len(matches) >= max_results:
                    truncated = True
                    break
                display_line = line if len(line) <= 500 else f"{line[:500]}...[line truncated]"
                matches.append(
                    f"{file_path.relative_to(root).as_posix()}:{line_number}:{column + 1}:"
                    f"{display_line}"
                )
            if truncated:
                break
        content, output_truncated, original_characters = _bounded_output(
            "\n".join(matches),
            max_chars=self._max_output_chars,
            truncated=truncated,
        )
        return ToolResult(
            success=True,
            content=content,
            metadata={
                "engine": "python",
                "path": _display_path(root, target),
                "matches": len(matches),
                "truncated": output_truncated,
                "original_characters": original_characters,
                "skipped_binary": skipped_binary,
                "skipped_large": skipped_large,
            },
        )


class GitStatusTool:
    name = "git_status"
    operation = Operation.READ
    description = "Show read-only Git branch and working-tree status limited to the workspace."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> None:
        self._max_output_chars = max_output_chars

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        try:
            validate_tool_arguments(arguments, self.parameters)
            root = _workspace_root(context)
            git_error = await _git_workspace_error(root)
            if git_error is not None:
                return _failure(git_error)
            return_code, stdout, stderr = await _run_process(
                [
                    "git",
                    "-c",
                    "core.pager=cat",
                    "status",
                    "--short",
                    "--branch",
                    "--untracked-files=normal",
                    "--",
                    ".",
                ],
                cwd=root,
            )
            if return_code != 0:
                return _failure(
                    "git status failed",
                    exit_code=return_code,
                    detail=_short_error(stderr),
                )
            rendered = stdout or "Working tree clean."
            content, truncated, original_characters = _bounded_output(
                rendered,
                max_chars=self._max_output_chars,
            )
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "truncated": truncated,
                    "original_characters": original_characters,
                    "clean": not any(
                        line and not line.startswith("##")
                        for line in stdout.splitlines()
                    ),
                },
            )
        except FileNotFoundError:
            return _failure("git executable was not found")
        except (ToolValidationError, WorkspaceToolError, OSError, ValueError) as exc:
            return _failure(_safe_workspace_error(exc))


class GitDiffTool:
    name = "git_diff"
    operation = Operation.READ
    description = "Show a read-only Git diff, optionally staged or restricted to one path."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "staged": {"type": "boolean"},
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": DEFAULT_MAX_PATH_CHARS,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, *, max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> None:
        self._max_output_chars = max_output_chars

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        try:
            validate_tool_arguments(arguments, self.parameters)
            root = _workspace_root(context)
            path_argument = arguments.get("path")
            if path_argument is None:
                relative_target = "."
            else:
                target = _resolve_workspace_path(root, _string_argument(arguments, "path"))
                relative_target = _display_path(root, target)
            git_error = await _git_workspace_error(root)
            if git_error is not None:
                return _failure(git_error)
            command = [
                "git",
                "-c",
                "core.pager=cat",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
            ]
            if arguments.get("staged", False):
                command.append("--cached")
            command.extend(("--", relative_target))
            return_code, stdout, stderr = await _run_process(command, cwd=root)
            if return_code != 0:
                return _failure(
                    "git diff failed",
                    exit_code=return_code,
                    detail=_short_error(stderr),
                )
            rendered = stdout or "No differences."
            content, truncated, original_characters = _bounded_output(
                rendered,
                max_chars=self._max_output_chars,
            )
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "path": relative_target,
                    "staged": bool(arguments.get("staged", False)),
                    "truncated": truncated,
                    "original_characters": original_characters,
                },
            )
        except FileNotFoundError:
            return _failure("git executable was not found")
        except (ToolValidationError, WorkspaceToolError, OSError, ValueError) as exc:
            return _failure(_safe_workspace_error(exc))


DEFAULT_MAX_EDIT_FILES = 20
DEFAULT_MAX_EDIT_BYTES = 2_000_000
_PROTECTED_WRITE_DIRECTORIES = frozenset(
    {".aws", ".config", ".docker", ".git", ".gnupg", ".kube", ".ssh"}
)
_PROTECTED_WRITE_FILES = frozenset(
    {
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_ecdsa",
        "id_rsa",
        "known_hosts",
    }
)
_PROTECTED_WRITE_SUFFIXES = frozenset(
    {".der", ".key", ".p12", ".pem", ".pfx"}
)
_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)


@dataclass(frozen=True)
class _PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _FilePatch:
    path: str
    hunks: tuple[_PatchHunk, ...]


@dataclass(frozen=True)
class _FileChange:
    path: Path
    display_path: str
    original: bytes | None
    updated: bytes
    mode: int | None


class ApplyPatchTool:
    name = "apply_patch"
    operation = Operation.WRITE
    description = (
        "Apply a strict unified diff to existing workspace text files transactionally. "
        "Hunk locations and context must match exactly; use dry_run to preview."
    )
    def __init__(
        self,
        *,
        max_files: int = DEFAULT_MAX_EDIT_FILES,
        max_total_bytes: int = DEFAULT_MAX_EDIT_BYTES,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        if max_files <= 0 or max_total_bytes <= 0 or max_output_chars <= 0:
            raise ValueError("edit limits must be positive")
        self._max_files = max_files
        self._max_total_bytes = max_total_bytes
        self._max_output_chars = max_output_chars
        self.parameters: Mapping[str, object] = {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "minLength": 1,
                },
                "dry_run": {"type": "boolean"},
            },
            "required": ["patch"],
            "additionalProperties": False,
        }

    def permission_request(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> PermissionRequest:
        validate_tool_arguments(arguments, self.parameters)
        root = _workspace_root(context)
        patch_text = _string_argument(arguments, "patch")
        if len(patch_text.encode("utf-8")) > self._max_total_bytes:
            raise WorkspaceToolError(
                f"patch exceeds total byte limit of {self._max_total_bytes}"
            )
        file_patches = _parse_unified_patch(patch_text)
        if len(file_patches) > self._max_files:
            raise WorkspaceToolError(f"patch exceeds file limit of {self._max_files}")
        paths: list[str] = []
        for file_patch in file_patches:
            target = _resolve_workspace_path(root, file_patch.path)
            _validate_writable_path(root, target)
            paths.append(_display_path(root, target))
        dry_run = _boolean_argument(arguments, "dry_run", False)
        return PermissionRequest(
            operation=Operation.WRITE,
            target=f"apply_patch: {', '.join(paths)}",
            level=PermissionLevel.ALLOW if dry_run else PermissionLevel.ASK,
            cwd=str(root),
            risk_reason=(
                "dry-run only; no files will be changed"
                if dry_run
                else f"patch will modify {len(paths)} workspace file(s)"
            ),
            preview=patch_text,
        )

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        try:
            validate_tool_arguments(arguments, self.parameters)
            root = _workspace_root(context)
            patch_text = _string_argument(arguments, "patch")
            if len(patch_text.encode("utf-8")) > self._max_total_bytes:
                raise WorkspaceToolError(
                    f"patch exceeds total byte limit of {self._max_total_bytes}"
                )
            file_patches = _parse_unified_patch(patch_text)
            if len(file_patches) > self._max_files:
                raise WorkspaceToolError(
                    f"patch exceeds file limit of {self._max_files}"
                )

            changes: list[_FileChange] = []
            rendered_diffs: list[str] = []
            total_bytes = 0
            for file_patch in file_patches:
                target = _resolve_workspace_path(root, file_patch.path)
                _validate_writable_path(root, target)
                if not target.is_file():
                    raise WorkspaceToolError(
                        f"patch target is not a regular file: {file_patch.path}"
                    )
                original = target.read_bytes()
                text, bom, newline, final_newline = _decode_editable_text(original)
                updated_text = _apply_file_patch(text, file_patch, newline, final_newline)
                updated = bom + updated_text.encode("utf-8")
                if updated == original:
                    raise WorkspaceToolError(
                        f"patch makes no changes to {file_patch.path}"
                    )
                total_bytes += len(original) + len(updated)
                if total_bytes > self._max_total_bytes:
                    raise WorkspaceToolError(
                        f"edit exceeds total byte limit of {self._max_total_bytes}"
                    )
                display_path = _display_path(root, target)
                changes.append(
                    _FileChange(
                        path=target,
                        display_path=display_path,
                        original=original,
                        updated=updated,
                        mode=stat.S_IMODE(target.stat().st_mode),
                    )
                )
                rendered_diffs.append(
                    _unified_diff(text, updated_text, display_path)
                )

            dry_run = _boolean_argument(arguments, "dry_run", False)
            if not dry_run:
                _commit_changes(changes, root=root)
            content, truncated, original_characters = _bounded_output(
                "\n".join(rendered_diffs),
                max_chars=self._max_output_chars,
            )
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "dry_run": dry_run,
                    "files": [change.display_path for change in changes],
                    "file_count": len(changes),
                    "total_bytes": total_bytes,
                    "truncated": truncated,
                    "original_characters": original_characters,
                },
            )
        except UnicodeError:
            return _failure("patch targets must be UTF-8 text files")
        except (ToolValidationError, WorkspaceToolError, OSError, ValueError) as exc:
            return _failure(_safe_workspace_error(exc))


class CreateFileTool:
    name = "create_file"
    operation = Operation.WRITE
    description = (
        "Create one new UTF-8 workspace file without overwriting an existing path. "
        "Use dry_run to preview the unified diff."
    )
    def __init__(
        self,
        *,
        max_total_bytes: int = DEFAULT_MAX_EDIT_BYTES,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        if max_total_bytes <= 0 or max_output_chars <= 0:
            raise ValueError("edit limits must be positive")
        self._max_total_bytes = max_total_bytes
        self._max_output_chars = max_output_chars
        self.parameters: Mapping[str, object] = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": DEFAULT_MAX_PATH_CHARS,
                },
                "content": {"type": "string"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    def permission_request(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> PermissionRequest:
        validate_tool_arguments(arguments, self.parameters)
        root = _workspace_root(context)
        target = _resolve_new_workspace_path(root, _string_argument(arguments, "path"))
        _validate_writable_path(root, target)
        content_value = _string_argument(arguments, "content")
        if len(content_value.encode("utf-8")) > self._max_total_bytes:
            raise WorkspaceToolError(
                f"file exceeds total byte limit of {self._max_total_bytes}"
            )
        display_path = _display_path(root, target)
        dry_run = _boolean_argument(arguments, "dry_run", False)
        return PermissionRequest(
            operation=Operation.WRITE,
            target=f"create_file: {display_path}",
            level=PermissionLevel.ALLOW if dry_run else PermissionLevel.ASK,
            cwd=str(root),
            risk_reason=(
                "dry-run only; no file will be created"
                if dry_run
                else "a new workspace file will be created"
            ),
            preview=_unified_diff("", content_value, display_path, created=True),
        )

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        try:
            validate_tool_arguments(arguments, self.parameters)
            root = _workspace_root(context)
            requested_path = _string_argument(arguments, "path")
            target = _resolve_new_workspace_path(root, requested_path)
            _validate_writable_path(root, target)
            content_value = _string_argument(arguments, "content")
            updated = content_value.encode("utf-8")
            if len(updated) > self._max_total_bytes:
                raise WorkspaceToolError(
                    f"file exceeds total byte limit of {self._max_total_bytes}"
                )
            display_path = _display_path(root, target)
            change = _FileChange(target, display_path, None, updated, None)
            dry_run = _boolean_argument(arguments, "dry_run", False)
            if not dry_run:
                _commit_changes([change], root=root)
            rendered = _unified_diff("", content_value, display_path, created=True)
            content, truncated, original_characters = _bounded_output(
                rendered,
                max_chars=self._max_output_chars,
            )
            return ToolResult(
                success=True,
                content=content,
                metadata={
                    "dry_run": dry_run,
                    "files": [display_path],
                    "file_count": 1,
                    "total_bytes": len(updated),
                    "truncated": truncated,
                    "original_characters": original_characters,
                },
            )
        except (ToolValidationError, WorkspaceToolError, OSError, ValueError) as exc:
            return _failure(_safe_workspace_error(exc))


def file_modification_tools() -> tuple[Tool, ...]:
    return (ApplyPatchTool(), CreateFileTool())


def workspace_tools(
    *,
    shell_allowlist: Iterable[str] = (),
    sandbox_runtime_factory: Callable[[Path], SandboxRuntime] | None = None,
) -> tuple[Tool, ...]:
    from .git_runtime import GitAddTool, GitCommitTool
    from .shell_tools import ShellCommandPolicy, ShellTool

    return (
        *read_only_workspace_tools(),
        *file_modification_tools(),
        GitAddTool(),
        GitCommitTool(),
        ShellTool(
            policy=ShellCommandPolicy(shell_allowlist),
            runtime_factory=sandbox_runtime_factory,
        ),
    )


def workspace_tool_registry(
    *,
    shell_allowlist: Iterable[str] = (),
    sandbox_runtime_factory: Callable[[Path], SandboxRuntime] | None = None,
) -> ToolRegistry:
    return ToolRegistry(
        workspace_tools(
            shell_allowlist=shell_allowlist,
            sandbox_runtime_factory=sandbox_runtime_factory,
        )
    )


def read_only_workspace_tools() -> tuple[Tool, ...]:
    from .git_runtime import GitDiffCheckTool

    return (
        ListFilesTool(),
        ReadFileTool(),
        SearchTextTool(),
        GitStatusTool(),
        GitDiffTool(),
        GitDiffCheckTool(),
    )


def read_only_tool_registry() -> ToolRegistry:
    return ToolRegistry(read_only_workspace_tools())


def _workspace_root(context: ToolContext) -> Path:
    try:
        root = context.working_directory.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise WorkspaceToolError("workspace does not exist or cannot be resolved") from None
    if not root.is_dir() or not root.is_absolute():
        raise WorkspaceToolError("workspace must be an absolute directory")
    return root


def _resolve_workspace_path(root: Path, raw_path: str) -> Path:
    requested = Path(raw_path).expanduser()
    if ".." in requested.parts:
        raise WorkspaceToolError("path traversal is not allowed")
    candidate = requested if requested.is_absolute() else root / requested
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        raise WorkspaceToolError("path does not exist or cannot be resolved") from None
    if not resolved.is_relative_to(root):
        raise WorkspaceToolError("path resolves outside the workspace")
    return resolved


def _resolve_new_workspace_path(root: Path, raw_path: str) -> Path:
    requested = Path(raw_path).expanduser()
    if ".." in requested.parts:
        raise WorkspaceToolError("path traversal is not allowed")
    candidate = requested if requested.is_absolute() else root / requested
    try:
        parent = candidate.parent.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError):
        raise WorkspaceToolError("parent directory does not exist or cannot be resolved") from None
    if not parent.is_relative_to(root):
        raise WorkspaceToolError("path resolves outside the workspace")
    target = parent / candidate.name
    if target.exists() or target.is_symlink():
        raise WorkspaceToolError("create_file refuses to overwrite an existing path")
    return target


def _validate_writable_path(root: Path, target: Path) -> None:
    relative = target.relative_to(root)
    lowered_parts = tuple(part.casefold() for part in relative.parts)
    protected_directory = next(
        (part for part in lowered_parts[:-1] if part in _PROTECTED_WRITE_DIRECTORIES),
        None,
    )
    if protected_directory is not None:
        raise WorkspaceToolError(
            f"modification of protected directory is not allowed: {protected_directory}"
        )
    name = target.name.casefold()
    is_environment_file = name == ".env" or (
        name.startswith(".env.")
        and not name.endswith((".example", ".sample", ".template"))
    )
    is_secret_named = name in _PROTECTED_WRITE_FILES or name.startswith("secrets.")
    if is_environment_file or is_secret_named or target.suffix.casefold() in _PROTECTED_WRITE_SUFFIXES:
        raise WorkspaceToolError("modification of a potential secret file is not allowed")


def _parse_unified_patch(patch_text: str) -> tuple[_FilePatch, ...]:
    lines = patch_text.splitlines()
    if not lines:
        raise WorkspaceToolError("patch is empty")
    parsed: list[_FilePatch] = []
    seen_paths: set[str] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith(("diff --git ", "index ")):
            index += 1
            continue
        if not line.startswith("--- "):
            raise WorkspaceToolError(
                f"invalid unified diff: expected '---' header at line {index + 1}"
            )
        old_path = _parse_patch_path(line[4:])
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise WorkspaceToolError("invalid unified diff: missing '+++' header")
        new_path = _parse_patch_path(lines[index][4:])
        index += 1
        if old_path == "/dev/null" or new_path == "/dev/null":
            raise WorkspaceToolError(
                "apply_patch only modifies existing files; use create_file for new files"
            )
        if old_path != new_path:
            raise WorkspaceToolError("renames are not supported by apply_patch")
        if old_path in seen_paths:
            raise WorkspaceToolError(f"patch contains duplicate file section: {old_path}")
        seen_paths.add(old_path)

        hunks: list[_PatchHunk] = []
        while index < len(lines) and not lines[index].startswith("--- "):
            if not lines[index]:
                index += 1
                continue
            match = _HUNK_HEADER.match(lines[index])
            if match is None:
                if lines[index].startswith(("diff --git ", "index ")):
                    index += 1
                    continue
                raise WorkspaceToolError(
                    f"invalid unified diff hunk header at line {index + 1}"
                )
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            index += 1
            hunk_lines: list[tuple[str, str]] = []
            old_seen = 0
            new_seen = 0
            while old_seen < old_count or new_seen < new_count:
                if index >= len(lines):
                    raise WorkspaceToolError("invalid unified diff: incomplete hunk")
                hunk_line = lines[index]
                if hunk_line == "\\ No newline at end of file":
                    if not hunk_lines:
                        raise WorkspaceToolError(
                            f"invalid newline marker at line {index + 1}"
                        )
                    index += 1
                    continue
                if not hunk_line or hunk_line[0] not in {" ", "+", "-"}:
                    raise WorkspaceToolError(
                        f"invalid unified diff hunk line at line {index + 1}"
                    )
                marker, value = hunk_line[0], hunk_line[1:]
                if marker != "+":
                    old_seen += 1
                if marker != "-":
                    new_seen += 1
                if old_seen > old_count or new_seen > new_count:
                    raise WorkspaceToolError("unified diff hunk line counts do not match header")
                hunk_lines.append((marker, value))
                index += 1
            if index < len(lines) and lines[index] == "\\ No newline at end of file":
                index += 1
            hunks.append(
                _PatchHunk(
                    old_start,
                    old_count,
                    new_start,
                    new_count,
                    tuple(hunk_lines),
                )
            )
        if not hunks:
            raise WorkspaceToolError(f"patch contains no hunks for {old_path}")
        parsed.append(_FilePatch(old_path, tuple(hunks)))
    if not parsed:
        raise WorkspaceToolError("patch contains no file changes")
    return tuple(parsed)


def _parse_patch_path(raw_path: str) -> str:
    path = raw_path.split("\t", 1)[0]
    if path.startswith(('"', "'")):
        raise WorkspaceToolError("quoted patch paths are not supported")
    if path.startswith(("a/", "b/")):
        path = path[2:]
    if not path:
        raise WorkspaceToolError("patch path is empty")
    return path


def _decode_editable_text(data: bytes) -> tuple[str, bytes, str, bool]:
    bom = b"\xef\xbb\xbf" if data.startswith(b"\xef\xbb\xbf") else b""
    payload = data[len(bom) :]
    text = _decode_text(payload)
    if "\r\n" in text:
        newline = "\r\n"
    elif "\r" in text and "\n" not in text:
        newline = "\r"
    else:
        newline = "\n"
    return text, bom, newline, text.endswith(("\n", "\r"))


def _apply_file_patch(
    original_text: str,
    file_patch: _FilePatch,
    newline: str,
    final_newline: bool,
) -> str:
    original_lines = original_text.splitlines()
    updated_lines: list[str] = []
    cursor = 0
    for hunk in file_patch.hunks:
        old_index = 0 if hunk.old_start == 0 else hunk.old_start - 1
        if old_index < cursor or old_index > len(original_lines):
            raise WorkspaceToolError(
                f"patch hunk has an invalid or overlapping location in {file_patch.path}"
            )
        expected = [value for marker, value in hunk.lines if marker != "+"]
        replacement = [value for marker, value in hunk.lines if marker != "-"]
        actual = original_lines[old_index : old_index + len(expected)]
        if actual != expected:
            raise WorkspaceToolError(
                f"patch context mismatch in {file_patch.path} at old line {hunk.old_start}"
            )
        updated_lines.extend(original_lines[cursor:old_index])
        updated_lines.extend(replacement)
        cursor = old_index + len(expected)
    updated_lines.extend(original_lines[cursor:])
    rendered = newline.join(updated_lines)
    if final_newline and updated_lines:
        rendered += newline
    return rendered


def _unified_diff(
    original: str,
    updated: str,
    display_path: str,
    *,
    created: bool = False,
) -> str:
    from_path = "/dev/null" if created else f"a/{display_path}"
    rendered = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=from_path,
            tofile=f"b/{display_path}",
            lineterm="",
        )
    )
    if created and not rendered:
        return f"--- /dev/null\n+++ b/{display_path}\n@@ -0,0 +0,0 @@"
    return rendered


def _commit_changes(changes: Iterable[_FileChange], *, root: Path) -> None:
    committed: list[_FileChange] = []
    try:
        for change in changes:
            if change.original is None:
                _assert_safe_commit_target(root, change.path, existing=False)
                _atomic_create_bytes(change.path, change.updated, root=root)
            else:
                _assert_safe_commit_target(root, change.path, existing=True)
                _atomic_replace_bytes(
                    change.path,
                    change.updated,
                    change.mode,
                    root=root,
                    expected=change.original,
                )
            committed.append(change)
    except Exception as exc:
        rollback_errors: list[str] = []
        for change in reversed(committed):
            try:
                if change.original is None:
                    _remove_created_bytes_if_unchanged(
                        root,
                        change.path,
                        change.updated,
                    )
                else:
                    _assert_safe_commit_target(root, change.path, existing=True)
                    _atomic_replace_bytes(
                        change.path,
                        change.original,
                        change.mode,
                        root=root,
                        expected=change.updated,
                    )
            except (OSError, WorkspaceToolError):
                rollback_errors.append(change.display_path)
        if rollback_errors:
            raise WorkspaceToolError(
                "write failed and rollback was incomplete for: " + ", ".join(rollback_errors)
            ) from exc
        if isinstance(exc, WorkspaceToolError):
            raise
        raise WorkspaceToolError("write failed; completed changes were rolled back") from exc


def _atomic_replace_bytes(
    path: Path,
    data: bytes,
    mode: int | None,
    *,
    root: Path | None = None,
    expected: bytes | None = None,
) -> None:
    if root is not None and os.name == "posix":
        _secure_atomic_replace_bytes(root, path, data, mode, expected=expected)
        return
    _assert_canonical_parent(path)
    if path.is_symlink():
        raise WorkspaceToolError("refusing to replace a symbolic link")
    if expected is not None:
        try:
            current = path.read_bytes()
        except OSError:
            raise WorkspaceToolError("file changed or disappeared before commit") from None
        if current != expected:
            raise WorkspaceToolError("file changed before commit")
    temporary = _write_temporary_bytes(path.parent, data, mode)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_bytes(
    path: Path,
    data: bytes,
    *,
    root: Path | None = None,
) -> None:
    if root is not None and os.name == "posix":
        _secure_atomic_create_bytes(root, path, data)
        return
    _assert_canonical_parent(path)
    if path.exists() or path.is_symlink():
        raise WorkspaceToolError(
            f"create_file refuses to overwrite an existing path: {path.name}"
        )
    temporary = _write_temporary_bytes(path.parent, data, 0o644)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise WorkspaceToolError(
                f"create_file refuses to overwrite an existing path: {path.name}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _secure_atomic_replace_bytes(
    root: Path,
    path: Path,
    data: bytes,
    mode: int | None,
    *,
    expected: bytes | None,
) -> None:
    parent_descriptor = _open_secure_parent(root, path)
    temporary_name: str | None = None
    try:
        current = _read_name_without_links(parent_descriptor, path.name)
        if expected is not None and current != expected:
            raise WorkspaceToolError("file changed before commit")
        temporary_name = _write_temporary_at(parent_descriptor, data, mode)
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)


def _secure_atomic_create_bytes(root: Path, path: Path, data: bytes) -> None:
    parent_descriptor = _open_secure_parent(root, path)
    temporary_name: str | None = None
    try:
        temporary_name = _write_temporary_at(parent_descriptor, data, 0o644)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise WorkspaceToolError(
                f"create_file refuses to overwrite an existing path: {path.name}"
            ) from None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)


def _remove_created_bytes_if_unchanged(
    root: Path,
    path: Path,
    expected: bytes,
) -> None:
    if os.name != "posix":
        _assert_safe_commit_target(root, path, existing=True)
        if path.read_bytes() != expected:
            raise WorkspaceToolError("concurrent change prevents rollback")
        path.unlink()
        return
    parent_descriptor = _open_secure_parent(root, path)
    try:
        if _read_name_without_links(parent_descriptor, path.name) != expected:
            raise WorkspaceToolError("concurrent change prevents rollback")
        os.unlink(path.name, dir_fd=parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _open_secure_parent(root: Path, path: Path) -> int:
    try:
        relative_parent = path.parent.relative_to(root)
        expected_root = os.stat(root, follow_symlinks=False)
    except (OSError, ValueError):
        raise WorkspaceToolError("write target escaped or changed before commit") from None
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError:
        raise WorkspaceToolError("workspace changed before commit") from None
    try:
        opened_root = os.fstat(descriptor)
        if (
            opened_root.st_dev != expected_root.st_dev
            or opened_root.st_ino != expected_root.st_ino
        ):
            raise WorkspaceToolError("workspace changed before commit")
        for part in relative_parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError:
        os.close(descriptor)
        raise WorkspaceToolError(
            "write target parent changed or became a symbolic link before commit"
        ) from None
    except Exception:
        os.close(descriptor)
        raise


def _read_name_without_links(parent_descriptor: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError:
        raise WorkspaceToolError("file changed or disappeared before commit") from None
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise WorkspaceToolError("write target is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _write_temporary_at(
    parent_descriptor: int,
    data: bytes,
    mode: int | None,
) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for _ in range(100):
        name = f".coding-agent-edit-{secrets.token_hex(12)}"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
            break
        except FileExistsError:
            continue
    else:
        raise WorkspaceToolError("unable to allocate a temporary edit file")
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
        if mode is not None:
            os.fchmod(descriptor, mode)
        return name
    except Exception:
        try:
            os.unlink(name, dir_fd=parent_descriptor)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)


def _assert_safe_commit_target(root: Path, path: Path, *, existing: bool) -> None:
    """Revalidate resolved paths at the commit boundary.

    Approval and patch preparation may take time. Re-resolving immediately
    before each filesystem mutation prevents a swapped symlink or renamed
    parent from redirecting the prepared write.
    """

    _assert_canonical_parent(path)
    try:
        path.parent.relative_to(root)
    except ValueError:
        raise WorkspaceToolError("write target parent escaped the workspace") from None
    if existing:
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            raise WorkspaceToolError("write target changed before commit") from None
        if resolved != path or path.is_symlink() or not path.is_file():
            raise WorkspaceToolError("write target changed or became a symbolic link")
    elif path.exists() or path.is_symlink():
        raise WorkspaceToolError("create_file refuses to overwrite an existing path")


def _assert_canonical_parent(path: Path) -> None:
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        raise WorkspaceToolError("write target parent changed before commit") from None
    if resolved_parent != path.parent:
        raise WorkspaceToolError(
            "write target parent changed or became a symbolic link before commit"
        )


def _write_temporary_bytes(parent: Path, data: bytes, mode: int | None) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=".coding-agent-edit-", dir=parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _iter_workspace_files(root: Path, target: Path) -> Iterable[Path]:
    if target.is_file():
        relative = target.relative_to(root)
        if not any(part in DEFAULT_IGNORED_DIRECTORIES for part in relative.parts):
            yield target
        return
    for current, directories, files in os.walk(target, followlinks=False):
        current_path = Path(current)
        safe_directories: list[str] = []
        for name in sorted(directories):
            if name in DEFAULT_IGNORED_DIRECTORIES:
                continue
            candidate = current_path / name
            if candidate.is_symlink():
                continue
            try:
                if candidate.resolve(strict=True).is_relative_to(root):
                    safe_directories.append(name)
            except (OSError, RuntimeError):
                continue
        directories[:] = safe_directories
        for name in sorted(files):
            candidate = current_path / name
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if not resolved.is_file() or not resolved.is_relative_to(root):
                continue
            relative = resolved.relative_to(root)
            if any(part in DEFAULT_IGNORED_DIRECTORIES for part in relative.parts):
                continue
            yield resolved


def _decode_text(data: bytes) -> str:
    sample = data[:8192]
    if b"\x00" in sample:
        raise UnicodeError("binary data contains NUL bytes")
    text = data.decode("utf-8")
    if sample:
        control_bytes = sum(byte < 9 or 13 < byte < 32 for byte in sample)
        if control_bytes / len(sample) > 0.05:
            raise UnicodeError("binary data contains control bytes")
    return text


def _bounded_output(
    content: str,
    *,
    max_chars: int,
    truncated: bool = False,
) -> tuple[str, bool, int]:
    original_characters = len(content)
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True
    if truncated:
        content = f"{content}{_TRUNCATION_MARKER}"
    return content, truncated, original_characters


async def _git_workspace_error(root: Path) -> str | None:
    return_code, stdout, _ = await _run_process(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
    )
    if return_code != 0 or stdout.strip() != "true":
        return "workspace is not a Git repository"
    return None


async def _run_process(command: list[str], *, cwd: Path) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    return (
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _failure(error: str, **metadata: object) -> ToolResult:
    return ToolResult(success=False, error=error, metadata=metadata)


def _safe_workspace_error(error: Exception) -> str:
    if isinstance(error, (WorkspaceToolError, ToolValidationError)):
        return str(error)
    return "workspace operation failed"


def _display_path(root: Path, target: Path) -> str:
    relative = target.relative_to(root)
    return "." if not relative.parts else relative.as_posix()


def _string_argument(
    arguments: Mapping[str, object],
    name: str,
    default: str | None = None,
) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str):
        raise ToolValidationError(f"arguments.{name} must be string")
    return value


def _integer_argument(arguments: Mapping[str, object], name: str, default: int) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(f"arguments.{name} must be integer")
    return value


def _boolean_argument(arguments: Mapping[str, object], name: str, default: bool) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ToolValidationError(f"arguments.{name} must be boolean")
    return value


def _short_error(stderr: str, *, max_chars: int = 500) -> str:
    stripped = stderr.strip()
    return stripped[:max_chars] if stripped else "no diagnostic output"
