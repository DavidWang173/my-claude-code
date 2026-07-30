"""Terminal input and event rendering for the coding-agent CLI.

This module deliberately contains no model loop, tool execution, or session
persistence logic. It renders Agent events and collects user decisions/input.
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import FrameType
from collections.abc import Callable
from typing import IO, Protocol

from .agent import AgentEvent, AgentEventKind, AgentResult, CancellationToken
from .models import ToolCall, Usage
from .permissions import PermissionRequest
from .tools import ToolResult

_ANSI = {
    "bold": "\x1b[1m",
    "cyan": "\x1b[36m",
    "dim": "\x1b[2m",
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "yellow": "\x1b[33m",
    "reset": "\x1b[0m",
}


@dataclass
class TurnSummary:
    modified_files: set[str] = field(default_factory=set)
    test_results: list[str] = field(default_factory=list)

    def observe(self, event: AgentEvent) -> None:
        if event.kind is not AgentEventKind.TOOL_RESULT or event.tool_result is None:
            return
        result = event.tool_result
        metadata = result.metadata
        if result.success and not metadata.get("dry_run", False):
            files = metadata.get("files", ())
            if isinstance(files, (list, tuple)):
                self.modified_files.update(str(path) for path in files)
        if event.tool_call is not None and event.tool_call.name == "run_shell":
            reason = metadata.get("risk_reason")
            if isinstance(reason, str) and "test" in reason.casefold():
                command = str(metadata.get("command", "test command"))
                exit_code = metadata.get("exit_code")
                self.test_results.append(f"{command} -> exit {exit_code}")


class EventRenderer(Protocol):
    def show_context(self, *, model: str, workspace: Path, session_id: str) -> None: ...

    def render_event(self, event: AgentEvent) -> None: ...

    def render_completion(self, result: AgentResult, summary: TurnSummary) -> None: ...

    def render_error(self, message: str, *, error_type: str = "error") -> None: ...

    def render_cancelled(self) -> None: ...

    def emit_record(self, record: object) -> None: ...


class HumanRenderer:
    def __init__(
        self,
        output: IO[str],
        error: IO[str],
        input_stream: IO[str],
        *,
        color: bool | None = None,
        shell_view_chars: int = 8000,
        diff_view_chars: int = 30_000,
    ) -> None:
        tty = bool(getattr(output, "isatty", lambda: False)())
        self._color = tty and "NO_COLOR" not in os.environ if color is None else color
        self._output = output
        self._error = error
        self._input = input_stream
        self._shell_view_chars = shell_view_chars
        self._diff_view_chars = diff_view_chars
        self._assistant_open = False
        self._shell_chars: dict[str, int] = {}
        self._shell_truncated: set[str] = set()

    def show_context(self, *, model: str, workspace: Path, session_id: str) -> None:
        self._line(self._styled("Coding Agent", "bold", "cyan"))
        self._line(f"model: {model}")
        self._line(f"workspace: {workspace}")
        self._line(f"session: {session_id}")

    def show_chat_hint(self) -> None:
        self._line(
            self._styled(
                "Enter submits; end a line with \\ to continue, or use /multi then .; /exit quits.",
                "dim",
            )
        )

    def render_event(self, event: AgentEvent) -> None:
        if event.kind is AgentEventKind.TEXT_DELTA:
            if not self._assistant_open:
                self._write(self._styled("assistant> ", "green"))
                self._assistant_open = True
            self._write(event.text)
            return
        if event.kind is AgentEventKind.TOOL_CALL and event.tool_call is not None:
            self._close_assistant_line()
            self._line(
                f"{self._styled('tool>', 'cyan')} {event.tool_call.name} "
                f"{_brief_arguments(event.tool_call)}"
            )
            return
        if event.kind is AgentEventKind.TOOL_OUTPUT:
            self._close_assistant_line()
            self._render_tool_output(event)
            return
        if event.kind is AgentEventKind.TOOL_RESULT and event.tool_result is not None:
            self._close_assistant_line()
            self._render_tool_result(event.tool_call, event.tool_result)
            return
        if event.kind is AgentEventKind.COMPLETED:
            self._close_assistant_line()

    def render_completion(self, result: AgentResult, summary: TurnSummary) -> None:
        self._close_assistant_line()
        self._line(self._styled("Turn complete", "bold", "green"))
        verification = result.verification
        if verification is not None:
            self._line(
                self._styled(
                    f"Verification: {verification.status.value.upper()} — "
                    f"{verification.summary}",
                    "green" if verification.passed else "red",
                )
            )
        report = result.report
        if report is None:
            files = ", ".join(sorted(summary.modified_files)) or "none"
            tests = "; ".join(summary.test_results) or "not reported"
            self._line(f"completed: modified files: {files}")
            self._line(f"incomplete: none")
            self._line(f"test results: {tests}")
            self._line("risks and follow-up: none reported")
        else:
            self._render_report_section("Completed", report.completed, "none")
            self._render_report_section("Incomplete", report.incomplete, "none")
            test_lines = tuple(
                f"{'PASS' if item.passed else 'FAIL'} {item.command}"
                f" (exit={item.exit_code})"
                for item in report.tests
            )
            self._render_report_section("Test results", test_lines, "not run")
            self._render_report_section(
                "Risks and follow-up", report.risks_and_follow_up, "none"
            )
            agent_files = ", ".join(report.git.agent_only_files) or "none"
            legacy_files = agent_files
            if legacy_files == "none" and summary.modified_files:
                legacy_files = ", ".join(sorted(summary.modified_files))
            self._line(f"modified files: {legacy_files}")
            preexisting = ", ".join(report.git.preexisting_files) or "none"
            overlap = ", ".join(report.git.overlapping_files) or "none"
            self._line(f"task-local files: {agent_files}")
            self._line(f"pre-existing user changes: {preexisting}")
            self._line(f"overlapping paths (not agent-only): {overlap}")
        self._line(
            "tokens: "
            f"prompt={result.usage.prompt_tokens} "
            f"completion={result.usage.completion_tokens} "
            f"total={result.usage.total_tokens}"
        )

    def _render_report_section(
        self, title: str, values: tuple[str, ...], empty: str
    ) -> None:
        self._line(self._styled(f"{title}:", "bold"))
        if not values:
            self._line(f"  - {empty}")
            return
        for value in values:
            self._line(f"  - {value}")

    def render_error(self, message: str, *, error_type: str = "error") -> None:
        self._close_assistant_line()
        label = "Warning" if error_type == "warning" else error_type
        self._error.write(self._styled(f"{label}: {message}\n", "red"))
        self._error.flush()

    def render_cancelled(self) -> None:
        self._close_assistant_line()
        self._line(self._styled("Current operation cancelled. Press Ctrl+C again to exit.", "yellow"))

    def render_interrupt_exit(self) -> None:
        self._close_assistant_line()
        self._line(self._styled("Exiting.", "yellow"))

    def approve(self, request: PermissionRequest) -> bool:
        self._close_assistant_line()
        self._line(self._styled("Approval required", "bold", "yellow"))
        if request.command is not None:
            self._line(f"command: {request.command}")
        self._line(f"target: {request.target}")
        if request.cwd is not None:
            self._line(f"cwd: {request.cwd}")
        if request.risk_reason is not None:
            self._line(f"risk: {request.risk_reason}")
        if request.preview is not None:
            preview, truncated = _truncate(request.preview, self._diff_view_chars)
            self._line(self._styled("diff preview:", "cyan"))
            self._line(preview)
            if truncated:
                self._line(self._styled("...[diff preview truncated]", "yellow"))
        self._write("Approve? [y/N] ")
        answer = self._input.readline()
        return answer.strip().casefold() in {"y", "yes"}

    def emit_record(self, record: object) -> None:
        if isinstance(record, str):
            self._line(record)
        else:
            self._line(json.dumps(record, ensure_ascii=False, indent=2, default=str))

    def _render_tool_output(self, event: AgentEvent) -> None:
        call_id = event.tool_call.id if event.tool_call is not None else "unknown"
        used = self._shell_chars.get(call_id, 0)
        available = self._shell_view_chars - used
        if available <= 0:
            if call_id not in self._shell_truncated:
                self._line(self._styled("...[shell output hidden after view limit]", "yellow"))
                self._shell_truncated.add(call_id)
            return
        visible = event.text[:available]
        self._shell_chars[call_id] = used + len(visible)
        prefix = f"[{event.stream or 'tool'}] "
        self._write(self._styled(prefix, "dim"))
        self._write(visible)
        if visible and not visible.endswith("\n"):
            self._write("\n")
        if len(event.text) > len(visible) and call_id not in self._shell_truncated:
            self._line(self._styled("...[shell output hidden after view limit]", "yellow"))
            self._shell_truncated.add(call_id)

    def _render_tool_result(self, call: ToolCall | None, result: ToolResult) -> None:
        name = call.name if call is not None else "tool"
        if result.success:
            exit_code = result.metadata.get("exit_code")
            detail = f" exit={exit_code}" if exit_code is not None else ""
            self._line(self._styled(f"✓ {name}{detail}", "green"))
        else:
            self._line(self._styled(f"✗ {name}: {result.error or 'failed'}", "red"))

    def _styled(self, value: str, *styles: str) -> str:
        if not self._color:
            return value
        return "".join(_ANSI[style] for style in styles) + value + _ANSI["reset"]

    def _close_assistant_line(self) -> None:
        if self._assistant_open:
            self._write("\n")
            self._assistant_open = False

    def _write(self, value: str) -> None:
        self._output.write(value)
        self._output.flush()

    def _line(self, value: str = "") -> None:
        self._write(f"{value}\n")


class JsonRenderer:
    """JSON Lines renderer with no ANSI or human prompts."""

    def __init__(self, output: IO[str]) -> None:
        self._output = output

    def show_context(self, *, model: str, workspace: Path, session_id: str) -> None:
        self._emit(
            {
                "type": "session",
                "model": model,
                "workspace": str(workspace),
                "session_id": session_id,
            }
        )

    def render_event(self, event: AgentEvent) -> None:
        payload: dict[str, object] = {"type": event.kind.value}
        if event.text:
            payload["text"] = event.text
        if event.stream is not None:
            payload["stream"] = event.stream
        if event.tool_call is not None:
            payload["tool_call"] = {
                "id": event.tool_call.id,
                "name": event.tool_call.name,
                "arguments": dict(event.tool_call.arguments),
            }
        if event.tool_result is not None:
            payload["tool_result"] = event.tool_result.to_dict()
        self._emit(payload)

    def render_completion(self, result: AgentResult, summary: TurnSummary) -> None:
        report = result.report
        modified_files = (
            list(report.git.agent_only_files or tuple(sorted(summary.modified_files)))
            if report is not None
            else sorted(summary.modified_files)
        )
        test_results = (
            [
                {
                    "command": item.command,
                    "passed": item.passed,
                    "exit_code": item.exit_code,
                    "detail": item.detail,
                }
                for item in report.tests
            ]
            if report is not None
            else summary.test_results
        )
        self._emit(
            {
                "type": "turn_summary",
                "session_id": result.session_id,
                "modified_files": modified_files,
                "test_results": test_results,
                "report": report.to_dict() if report is not None else None,
                "verification": (
                    result.verification.to_dict()
                    if result.verification is not None
                    else None
                ),
                "status": result.status.value,
                "usage": _usage_dict(result.usage),
                "turns": result.turns,
                "tool_calls": result.tool_calls,
            }
        )

    def render_error(self, message: str, *, error_type: str = "error") -> None:
        self._emit({"type": error_type, "message": message})

    def render_cancelled(self) -> None:
        self._emit({"type": "cancelled", "message": "current operation cancelled"})

    def emit_record(self, record: object) -> None:
        self._emit({"type": "record", "data": record})

    def _emit(self, payload: object) -> None:
        self._output.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self._output.flush()


class InterruptAction(str, Enum):
    CANCEL = "cancel"
    EXIT = "exit"


class InterruptController:
    """First interrupt cancels the active token; a second requests process exit."""

    def __init__(self) -> None:
        self.exit_requested = False
        self._token: CancellationToken | None = None

    def bind(self, token: CancellationToken) -> None:
        self._token = token

    def unbind(self) -> None:
        self._token = None

    def handle_interrupt(self) -> InterruptAction:
        if self._token is not None and not self._token.cancelled:
            self._token.cancel()
            return InterruptAction.CANCEL
        self.exit_requested = True
        if self._token is not None:
            self._token.cancel()
        return InterruptAction.EXIT


class SignalBinding:
    def __init__(
        self,
        controller: InterruptController,
        on_action: Callable[[InterruptAction], None],
    ) -> None:
        self._controller = controller
        self._on_action = on_action
        self._previous: object | None = None

    def __enter__(self) -> SignalBinding:
        self._previous = signal.getsignal(signal.SIGINT)

        def handler(signum: int, frame: FrameType | None) -> None:
            del signum, frame
            self._on_action(self._controller.handle_interrupt())

        signal.signal(signal.SIGINT, handler)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)  # type: ignore[arg-type]


def read_prompt(input_stream: IO[str], renderer: HumanRenderer) -> str | None:
    if not bool(getattr(input_stream, "isatty", lambda: False)()):
        value = input_stream.read()
        return value.strip() or None
    renderer._write("you> ")
    first = input_stream.readline()
    if not first:
        return None
    value = first.rstrip("\r\n")
    if value.strip() in {"/exit", "/quit"}:
        return None
    if value.strip() == "/multi":
        lines: list[str] = []
        while True:
            renderer._write("...  ")
            line = input_stream.readline()
            if not line:
                break
            line = line.rstrip("\r\n")
            if line == ".":
                break
            lines.append(line)
        return "\n".join(lines).strip() or None
    lines = []
    while value.endswith("\\"):
        lines.append(value[:-1])
        renderer._write("...  ")
        continuation = input_stream.readline()
        if not continuation:
            break
        value = continuation.rstrip("\r\n")
    lines.append(value)
    return "\n".join(lines).strip() or None


def _brief_arguments(call: ToolCall, *, max_chars: int = 240) -> str:
    if call.name == "apply_patch":
        patch = call.arguments.get("patch")
        if isinstance(patch, str):
            paths = [line[4:].removeprefix("b/") for line in patch.splitlines() if line.startswith("+++ ")]
            return json.dumps({"files": paths, "dry_run": call.arguments.get("dry_run", False)})
    if call.name == "create_file":
        return json.dumps(
            {
                "path": call.arguments.get("path"),
                "dry_run": call.arguments.get("dry_run", False),
            },
            ensure_ascii=False,
        )
    rendered = json.dumps(dict(call.arguments), ensure_ascii=False, default=str)
    return rendered if len(rendered) <= max_chars else rendered[:max_chars] + "...[truncated]"


def _usage_dict(usage: Usage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _truncate(value: str, maximum: int) -> tuple[str, bool]:
    return (value, False) if len(value) <= maximum else (value[:maximum], True)
