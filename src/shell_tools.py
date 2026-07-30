"""Controlled subprocess execution with explicit risk classification.

This module is a policy layer, not an operating-system sandbox. It keeps the
working directory fixed, avoids a shell by default, and makes risky commands
visible to the permission system before a process starts.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .permissions import Operation, PermissionLevel, PermissionRequest
from .sandbox import (
    LocalSandboxRuntime,
    SandboxArtifact,
    SandboxCommand,
    SandboxError,
    SandboxExecutionResult,
    SandboxPolicy,
    SandboxRuntime,
)
from .tools import (
    ToolContext,
    ToolResult,
    ToolValidationError,
    WorkspaceToolError,
    _workspace_root,
    validate_tool_arguments,
)

DEFAULT_SHELL_TIMEOUT = 120.0
MAX_SHELL_TIMEOUT = 600.0
DEFAULT_MAX_SHELL_OUTPUT_BYTES = 1_000_000
_OUTPUT_TRUNCATION_MARKER = "\n...[command output truncated]"

_SAFE_EXECUTABLES = frozenset(
    {"cat", "echo", "grep", "head", "ls", "pwd", "rg", "sort", "tail", "uniq", "wc"}
)
_WRITE_EXECUTABLES = frozenset(
    {"chmod", "chown", "cp", "install", "ln", "mkdir", "mv", "rm", "tee", "touch"}
)
_NETWORK_EXECUTABLES = frozenset(
    {"curl", "ftp", "nc", "netcat", "scp", "sftp", "ssh", "telnet", "wget"}
)
_PACKAGE_EXECUTABLES = frozenset(
    {"brew", "cargo", "gem", "go", "npm", "npx", "pip", "pip3", "pnpm", "poetry", "uv", "yarn"}
)
_SHELL_EXECUTABLES = frozenset(
    {"bash", "dash", "fish", "ksh", "sh", "zsh"}
)
_INTERPRETER_EXECUTABLES = frozenset(
    {"node", "perl", "php", "python", "python3", "ruby"}
)
_KEY_DIRECTORY_NAMES = frozenset({".aws", ".gnupg", ".kube", ".ssh"})
_CREDENTIAL_FILE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
_CREDENTIAL_FILE_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_GIT_READ_SUBCOMMANDS = frozenset(
    {"blame", "describe", "diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}
)
_GIT_WRITE_SUBCOMMANDS = frozenset(
    {
        "add",
        "am",
        "apply",
        "bisect",
        "branch",
        "checkout",
        "cherry-pick",
        "clean",
        "clone",
        "commit",
        "fetch",
        "init",
        "merge",
        "mv",
        "pull",
        "push",
        "rebase",
        "reset",
        "restore",
        "revert",
        "rm",
        "stash",
        "switch",
        "tag",
    }
)


class ShellStreamKind(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    TRUNCATED = "truncated"
    COMPLETED = "completed"


@dataclass(frozen=True)
class ShellStreamEvent:
    kind: ShellStreamKind
    data: str = ""
    result: ToolResult | None = None


@dataclass(frozen=True)
class CommandAssessment:
    level: PermissionLevel
    operation: Operation
    reason: str
    command: str
    cwd: Path
    shell_mode: bool

    def permission_request(self) -> PermissionRequest:
        return PermissionRequest(
            operation=self.operation,
            target="run_shell",
            level=self.level,
            command=self.command,
            cwd=str(self.cwd),
            risk_reason=self.reason,
        )


@dataclass(frozen=True)
class _PreparedCommand:
    argv: tuple[str, ...] | None
    shell_command: str | None
    timeout: float
    assessment: CommandAssessment


class ShellCommandPolicy:
    """Classify commands using conservative, reviewable rules.

    Allowlist entries are argv prefixes parsed with ``shlex.split``. They cannot
    override hard denials, shell-mode approval, protected-path checks, or
    explicit working-directory escape.
    """

    def __init__(self, allowlist: Iterable[str | Sequence[str]] = ()) -> None:
        parsed: list[tuple[str, ...]] = []
        for entry in allowlist:
            tokens = tuple(shlex.split(entry)) if isinstance(entry, str) else tuple(entry)
            if not tokens or not all(isinstance(token, str) and token for token in tokens):
                raise ValueError("shell allowlist entries must contain non-empty argv tokens")
            parsed.append(tokens)
        self._allowlist = tuple(parsed)

    def assess_argv(self, argv: Sequence[str], *, cwd: Path) -> CommandAssessment:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ToolValidationError("arguments.argv must contain non-empty strings")
        tokens = tuple(argv)
        command = shlex.join(tokens)
        executable = Path(tokens[0]).name.casefold()

        denial = self._hard_denial(tokens, command=command, cwd=cwd, shell_mode=False)
        if denial is not None:
            return CommandAssessment(
                PermissionLevel.DENY, Operation.EXECUTE, denial, command, cwd, False
            )

        level, operation, reason = self._base_argv_assessment(tokens, executable)
        if (
            level is PermissionLevel.ASK
            and self._matches_allowlist(tokens)
            and not self._allowlist_is_unsafe(tokens, executable)
        ):
            level = PermissionLevel.ALLOW
            reason = "command matches a configured argv allowlist prefix"
        return CommandAssessment(level, operation, reason, command, cwd, False)

    def assess_shell(self, command: str, *, cwd: Path) -> CommandAssessment:
        if not command.strip():
            raise ToolValidationError("arguments.shell_command must not be empty")
        try:
            tokens = tuple(shlex.split(command))
        except ValueError:
            tokens = ()
        denial = self._hard_denial(tokens, command=command, cwd=cwd, shell_mode=True)
        if denial is not None:
            return CommandAssessment(
                PermissionLevel.DENY, Operation.EXECUTE, denial, command, cwd, True
            )
        operation = (
            Operation.NETWORK
            if tokens and Path(tokens[0]).name.casefold() in _NETWORK_EXECUTABLES
            else Operation.EXECUTE
        )
        return CommandAssessment(
            PermissionLevel.ASK,
            operation,
            "shell syntax mode can expand variables, redirect files, and start pipelines",
            command,
            cwd,
            True,
        )

    def _hard_denial(
        self,
        tokens: tuple[str, ...],
        *,
        command: str,
        cwd: Path,
        shell_mode: bool,
    ) -> str | None:
        lowered = tuple(token.casefold() for token in tokens)
        executable = Path(lowered[0]).name if lowered else ""
        if any(Path(token).name == "sudo" for token in lowered) or re.search(
            r"(^|[;&|]\s*)sudo(?:\s|$)", command
        ):
            return "sudo is not permitted"
        if executable == "rm" and _has_recursive_force_flags(lowered[1:]):
            return "recursive forced deletion (rm -rf) is denied"
        if re.search(r"\brm\s+(?:-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\b", command):
            return "recursive forced deletion (rm -rf) is denied"
        if executable == "git":
            git_args = _git_arguments(lowered[1:])
            if git_args and git_args[0] == "push":
                return "git push is disabled by default; this agent never pushes"
            if len(git_args) >= 2 and git_args[0] == "reset" and "--hard" in git_args[1:]:
                return "git reset --hard is denied"
            if git_args and git_args[0] == "clean" and _has_force_flag(git_args[1:]):
                return "forced git clean is denied"
        if re.search(r"\bgit(?:\s+-[^\s]+)*\s+push\b", command, re.IGNORECASE):
            return "git push is disabled by default; this agent never pushes"
        if re.search(r"\bgit\s+reset\b[^\n;&|]*--hard\b", command, re.IGNORECASE):
            return "git reset --hard is denied"
        if shell_mode and re.search(
            r"\b(?:curl|wget)\b[^\n]*\|\s*(?:/[\w.-]+/)*(?:ba|z|k|da)?sh\b",
            command,
            re.IGNORECASE,
        ):
            return "downloading content directly into a shell is denied"
        if executable in {"mkfs", "reboot", "shutdown"}:
            return f"destructive system command is denied: {executable}"
        if _references_key_directory(tokens, command):
            return "reading common credential files or private-key directories is denied"
        outside_read = _outside_explicit_path(tokens, cwd=cwd)
        if outside_read is not None:
            return f"accessing a path outside the workspace is denied: {outside_read}"
        outside_write = _outside_write_target(tokens, cwd=cwd)
        if outside_write is not None:
            return f"writing outside the workspace is denied: {outside_write}"
        if shell_mode:
            redirect = _outside_shell_redirect(command, cwd=cwd)
            if redirect is not None:
                return f"shell redirection outside the workspace is denied: {redirect}"
        return None

    def _base_argv_assessment(
        self,
        tokens: tuple[str, ...],
        executable: str,
    ) -> tuple[PermissionLevel, Operation, str]:
        args = tuple(token.casefold() for token in tokens[1:])
        if executable == "git":
            unsafe_option = _unsafe_git_option(args)
            if unsafe_option is not None:
                return PermissionLevel.ASK, Operation.EXECUTE, unsafe_option
            git_args = _git_arguments(args)
            subcommand = git_args[0] if git_args else ""
            if subcommand in _GIT_READ_SUBCOMMANDS:
                return PermissionLevel.ALLOW, Operation.READ, "recognized read-only Git command"
            if subcommand in _GIT_WRITE_SUBCOMMANDS:
                operation = Operation.NETWORK if subcommand in {"clone", "fetch", "pull", "push"} else Operation.WRITE
                return PermissionLevel.ASK, operation, f"Git {subcommand} can change state"
            return PermissionLevel.ASK, Operation.EXECUTE, "unrecognized Git command"
        if executable in _NETWORK_EXECUTABLES:
            return PermissionLevel.ASK, Operation.NETWORK, "command can access the network"
        if executable in _WRITE_EXECUTABLES:
            return PermissionLevel.ASK, Operation.WRITE, "command can modify files"
        if _is_python_executable(executable) and _is_python_package_install(args):
            return PermissionLevel.ASK, Operation.NETWORK, "dependency installation can write files and access the network"
        if executable in {"cargo", "go"} and args and args[0] == "test":
            return PermissionLevel.ALLOW, Operation.EXECUTE, "recognized test runner"
        if executable in _PACKAGE_EXECUTABLES and _is_package_install(executable, args):
            return PermissionLevel.ASK, Operation.NETWORK, "dependency installation can write files and access the network"
        if executable in {"pytest", "py.test"}:
            return PermissionLevel.ALLOW, Operation.EXECUTE, "recognized test runner"
        if _is_python_executable(executable) and _is_python_test_command(args):
            return PermissionLevel.ALLOW, Operation.EXECUTE, "recognized Python test command"
        if executable in {"npm", "pnpm", "yarn"} and _is_javascript_test_command(args):
            return PermissionLevel.ALLOW, Operation.EXECUTE, "recognized package test command"
        if executable == "find" and any(
            arg in {
                "-delete",
                "-exec",
                "-execdir",
                "-ok",
                "-fprint",
                "-fprintf",
                "-fls",
            }
            for arg in args
        ):
            return PermissionLevel.ASK, Operation.WRITE, "find action can execute commands or delete files"
        if executable == "sed":
            return PermissionLevel.ASK, Operation.EXECUTE, "sed expressions may write files"
        if executable == "rg" and any(
            arg == "--pre" or arg.startswith("--pre=") for arg in args
        ):
            return PermissionLevel.ASK, Operation.EXECUTE, "ripgrep --pre can execute a command"
        if executable == "sort" and any(
            arg in {"-o", "--output"} or arg.startswith("--output=") for arg in args
        ):
            return PermissionLevel.ASK, Operation.WRITE, "sort output option can modify files"
        if executable in _SAFE_EXECUTABLES or executable == "find":
            return PermissionLevel.ALLOW, Operation.READ, "recognized read-only command"
        return PermissionLevel.ASK, Operation.EXECUTE, "command is not in the built-in safe command set"

    def _matches_allowlist(self, tokens: tuple[str, ...]) -> bool:
        return any(tokens[: len(prefix)] == prefix for prefix in self._allowlist)

    @staticmethod
    def _allowlist_is_unsafe(tokens: tuple[str, ...], executable: str) -> bool:
        lowered = tuple(token.casefold() for token in tokens[1:])
        if executable in _SHELL_EXECUTABLES:
            return True
        if (executable in _INTERPRETER_EXECUTABLES or _is_python_executable(executable)) and any(
            token in {"-c", "-e", "--eval"} for token in lowered
        ):
            return True
        return executable in {"env", "xargs"}


class ShellTool:
    name = "run_shell"
    operation = Operation.EXECUTE
    description = (
        "Run a bounded command in the workspace. Prefer argv; shell_command is a separate "
        "riskier mode that always requires approval. Returns stdout, stderr, and exit code."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 8192},
                "minItems": 1,
                "maxItems": 256,
            },
            "shell_command": {"type": "string", "minLength": 1, "maxLength": 32768},
            "cwd": {"type": "string", "minLength": 1, "maxLength": 4096},
            "timeout": {"type": "number", "minimum": 0.01, "maximum": MAX_SHELL_TIMEOUT},
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        policy: ShellCommandPolicy | None = None,
        timeout: float = DEFAULT_SHELL_TIMEOUT,
        max_output_bytes: int = DEFAULT_MAX_SHELL_OUTPUT_BYTES,
        runtime: SandboxRuntime | None = None,
        runtime_factory: Callable[[Path], SandboxRuntime] | None = None,
    ) -> None:
        if timeout <= 0 or timeout > MAX_SHELL_TIMEOUT:
            raise ValueError(f"timeout must be between 0 and {MAX_SHELL_TIMEOUT:g} seconds")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self._policy = policy or ShellCommandPolicy()
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        if runtime is not None and runtime_factory is not None:
            raise ValueError("provide runtime or runtime_factory, not both")
        self._runtime = runtime
        self._runtime_factory = runtime_factory or (
            lambda workspace: LocalSandboxRuntime(
                workspace,
                policy=SandboxPolicy.local_default(timeout=MAX_SHELL_TIMEOUT),
            )
        )

    def permission_request(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> PermissionRequest:
        return self._prepare(arguments, context).assessment.permission_request()

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> ToolResult:
        completed: ToolResult | None = None
        async for event in self.execute_stream(arguments, context):
            if event.kind is ShellStreamKind.COMPLETED:
                completed = event.result
            elif context.output_handler is not None:
                await context.output_handler(event.kind.value, event.data)
        if completed is None:
            return ToolResult(success=False, error="command ended without a result")
        return completed

    async def execute_stream(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> AsyncIterator[ShellStreamEvent]:
        try:
            prepared = self._prepare(arguments, context)
        except (ToolValidationError, WorkspaceToolError, OSError, ValueError) as exc:
            yield ShellStreamEvent(
                ShellStreamKind.COMPLETED,
                result=ToolResult(success=False, error=str(exc)),
            )
            return
        assessment = prepared.assessment
        if assessment.level is PermissionLevel.DENY:
            yield ShellStreamEvent(
                ShellStreamKind.COMPLETED,
                result=_policy_failure(assessment, "command denied"),
            )
            return
        if assessment.level is PermissionLevel.ASK and not context.permission_granted:
            yield ShellStreamEvent(
                ShellStreamKind.COMPLETED,
                result=_policy_failure(
                    assessment,
                    "command requires approval; non-interactive execution defaults to deny",
                ),
            )
            return

        runtime = self._runtime or self._runtime_factory(assessment.cwd)
        if runtime.workspace != assessment.cwd:
            yield ShellStreamEvent(
                ShellStreamKind.COMPLETED,
                result=ToolResult(
                    success=False,
                    error="sandbox workspace does not match the tool workspace",
                    metadata=_command_metadata(assessment, exit_code=None),
                ),
            )
            return

        queue: asyncio.Queue[tuple[str | None, bytes]] = asyncio.Queue()

        async def handle_output(stream_name: str, chunk: bytes) -> None:
            await queue.put((stream_name, chunk))

        async def run_in_sandbox() -> tuple[
            SandboxExecutionResult, tuple[SandboxArtifact, ...]
        ]:
            await runtime.prepare()
            try:
                execution = await runtime.execute(
                    SandboxCommand(
                        argv=prepared.argv,
                        shell_command=prepared.shell_command,
                        cwd=assessment.cwd,
                        timeout=prepared.timeout,
                        max_output_bytes=self._max_output_bytes,
                    ),
                    output_handler=handle_output,
                )
                artifacts = await runtime.collect_artifacts()
                return execution, artifacts
            finally:
                await runtime.cleanup()

        async def managed_execution() -> tuple[
            SandboxExecutionResult, tuple[SandboxArtifact, ...]
        ]:
            try:
                return await run_in_sandbox()
            finally:
                await queue.put((None, b""))

        task = asyncio.create_task(managed_execution())
        try:
            while True:
                stream_name, chunk = await queue.get()
                if stream_name is None:
                    break
                yield ShellStreamEvent(
                    ShellStreamKind(stream_name),
                    data=chunk.decode("utf-8", errors="replace"),
                )
            execution, artifacts = await task
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        except (SandboxError, OSError, ValueError) as exc:
            yield ShellStreamEvent(
                ShellStreamKind.COMPLETED,
                result=ToolResult(
                    success=False,
                    error=str(exc),
                    metadata={
                        **_command_metadata(assessment, exit_code=None),
                        "sandbox_runtime": type(runtime).__name__,
                        "security_level": runtime.security_level.value,
                    },
                ),
            )
            return

        if execution.truncated:
            yield ShellStreamEvent(
                ShellStreamKind.TRUNCATED,
                data=_OUTPUT_TRUNCATION_MARKER,
            )
        rendered = _render_output(execution, truncated=execution.truncated)
        metadata = _command_metadata(
            assessment,
            exit_code=execution.exit_code,
            timed_out=execution.timed_out,
            truncated=execution.truncated,
            output_bytes=execution.output_bytes,
        )
        metadata.update(
            {
                "sandbox_runtime": type(runtime).__name__,
                "security_level": runtime.security_level.value,
                "artifacts": [
                    {"path": artifact.path, "size": artifact.size}
                    for artifact in artifacts
                ],
                "files": [artifact.path for artifact in artifacts],
            }
        )
        if execution.timed_out:
            result = ToolResult(
                success=False,
                content=rendered,
                error=f"command timed out after {prepared.timeout:g} seconds",
                metadata=metadata,
            )
        else:
            result = ToolResult(success=True, content=rendered, metadata=metadata)
        yield ShellStreamEvent(ShellStreamKind.COMPLETED, result=result)

    def _prepare(
        self,
        arguments: Mapping[str, object],
        context: ToolContext,
    ) -> _PreparedCommand:
        validate_tool_arguments(arguments, self.parameters)
        root = _workspace_root(context)
        requested_cwd = arguments.get("cwd")
        if requested_cwd is not None:
            if not isinstance(requested_cwd, str):
                raise ToolValidationError("arguments.cwd must be string")
            candidate = Path(requested_cwd).expanduser()
            candidate = candidate if candidate.is_absolute() else root / candidate
            try:
                resolved_cwd = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                raise WorkspaceToolError("shell cwd does not exist or cannot be resolved") from None
            if resolved_cwd != root:
                raise WorkspaceToolError("shell cwd must be exactly the current workspace")

        argv_value = arguments.get("argv")
        shell_value = arguments.get("shell_command")
        if (argv_value is None) == (shell_value is None):
            raise ToolValidationError("provide exactly one of argv or shell_command")
        timeout_value = arguments.get("timeout", self._timeout)
        if not isinstance(timeout_value, (int, float)) or isinstance(timeout_value, bool):
            raise ToolValidationError("arguments.timeout must be number")
        timeout = float(timeout_value)
        if argv_value is not None:
            if not isinstance(argv_value, list):
                raise ToolValidationError("arguments.argv must be array")
            argv = tuple(argv_value)
            assessment = self._policy.assess_argv(argv, cwd=root)
            return _PreparedCommand(argv, None, timeout, assessment)
        if not isinstance(shell_value, str):
            raise ToolValidationError("arguments.shell_command must be string")
        assessment = self._policy.assess_shell(shell_value, cwd=root)
        return _PreparedCommand(None, shell_value, timeout, assessment)

def _policy_failure(assessment: CommandAssessment, prefix: str) -> ToolResult:
    return ToolResult(
        success=False,
        error=f"{prefix}: {assessment.reason}",
        metadata=_command_metadata(assessment, exit_code=None),
    )


def _command_metadata(
    assessment: CommandAssessment,
    *,
    exit_code: int | None,
    timed_out: bool = False,
    truncated: bool = False,
    output_bytes: int = 0,
) -> dict[str, object]:
    return {
        "command": assessment.command,
        "cwd": str(assessment.cwd),
        "shell_mode": assessment.shell_mode,
        "permission_level": assessment.level.value,
        "risk_reason": assessment.reason,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "truncated": truncated,
        "output_bytes": output_bytes,
    }


def _render_output(execution: SandboxExecutionResult, *, truncated: bool) -> str:
    sections: list[str] = []
    for name, data in (("stdout", execution.stdout), ("stderr", execution.stderr)):
        if data:
            sections.append(f"[{name}]\n{data.decode('utf-8', errors='replace')}")
    if not sections:
        sections.append("Command produced no output.")
    rendered = "\n".join(sections)
    if truncated:
        rendered += _OUTPUT_TRUNCATION_MARKER
    return rendered


def _git_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    remaining: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token in {"-c", "-C", "--git-dir", "--work-tree"}:
            index += 2
            continue
        if token.startswith(("--git-dir=", "--work-tree=")):
            index += 1
            continue
        remaining.extend(arguments[index:])
        break
    return tuple(remaining)


def _has_recursive_force_flags(arguments: Sequence[str]) -> bool:
    recursive = any(token in {"-r", "-R", "--recursive"} or (token.startswith("-") and "r" in token.casefold()[1:]) for token in arguments)
    forced = _has_force_flag(arguments)
    return recursive and forced


def _has_force_flag(arguments: Sequence[str]) -> bool:
    return any(token in {"-f", "--force"} or (token.startswith("-") and "f" in token.casefold()[1:]) for token in arguments)


def _references_key_directory(tokens: Sequence[str], command: str) -> bool:
    if re.search(r"(?:^|[/\\])\.(?:ssh|aws|gnupg|kube)(?:[/\\]|$)", command, re.IGNORECASE):
        return True
    for token in tokens:
        expanded = token.replace("$HOME", str(Path.home())).replace("${HOME}", str(Path.home()))
        try:
            path = Path(expanded).expanduser()
            parts = path.parts
        except RuntimeError:
            continue
        if any(part.casefold() in _KEY_DIRECTORY_NAMES for part in parts):
            return True
        name = path.name.casefold()
        is_environment_file = name == ".env" or (
            name.startswith(".env.")
            and not name.endswith((".example", ".sample", ".template"))
        )
        if (
            is_environment_file
            or name in _CREDENTIAL_FILE_NAMES
            or path.suffix.casefold() in _CREDENTIAL_FILE_SUFFIXES
        ):
            return True
    return False


def _outside_write_target(tokens: Sequence[str], *, cwd: Path) -> str | None:
    if not tokens:
        return None
    executable = Path(tokens[0]).name.casefold()
    lowered = tuple(token.casefold() for token in tokens[1:])
    if executable == "git":
        for index, token in enumerate(tokens[1:]):
            if token == "-C" and index + 2 <= len(tokens[1:]):
                target = tokens[index + 2]
                if _path_is_outside(target, cwd):
                    return target
        return None
    if executable not in _WRITE_EXECUTABLES:
        return None
    operands = [token for token in tokens[1:] if not token.startswith("-")]
    if executable in {"cp", "install", "ln", "mv"}:
        candidates = operands[-1:]
    elif executable in {"chmod", "chown"}:
        candidates = operands[1:]
    else:
        candidates = operands
    for candidate in candidates:
        if _path_is_outside(candidate, cwd):
            return candidate
    if executable == "rm" and any(token == "--" for token in lowered):
        return next((item for item in candidates if _path_is_outside(item, cwd)), None)
    return None


def _outside_explicit_path(tokens: Sequence[str], *, cwd: Path) -> str | None:
    """Reject explicit filesystem operands that resolve beyond the workspace.

    This is intentionally conservative. The command policy is an approval
    boundary, not a reason for an allow-classified command to read arbitrary
    host files merely because its process cwd is inside the workspace.
    """

    for token in tokens[1:]:
        candidate_text = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
        if not candidate_text or candidate_text == "-" or (
            candidate_text.startswith("-") and "=" not in token
        ):
            continue
        try:
            candidate_path = Path(candidate_text).expanduser()
        except RuntimeError:
            return candidate_text
        has_parent_traversal = ".." in candidate_path.parts
        workspace_candidate = (
            candidate_path if candidate_path.is_absolute() else cwd / candidate_path
        )
        explicitly_path_like = (
            candidate_path.is_absolute()
            or candidate_text.startswith("~")
            or has_parent_traversal
            or workspace_candidate.exists()
            or workspace_candidate.is_symlink()
        )
        if explicitly_path_like and _path_is_outside(candidate_text, cwd):
            return candidate_text
    return None


def _unsafe_git_option(arguments: Sequence[str]) -> str | None:
    """Identify Git options that can execute helpers or alter command scope."""

    lowered = tuple(argument.casefold() for argument in arguments)
    if any(
        token in {"-c", "--config-env", "--exec-path"}
        or token.startswith(("-c=", "--config-env=", "--exec-path="))
        for token in lowered
    ):
        return "Git configuration and executable-path overrides require approval"
    git_args = _git_arguments(lowered)
    if any(token in {"--ext-diff", "--textconv"} for token in git_args[1:]):
        return "Git external diff and text-conversion helpers can execute commands"
    return None


def _outside_shell_redirect(command: str, *, cwd: Path) -> str | None:
    pattern = re.compile(r"(?:>>?|\btee(?:\s+-[a-zA-Z]+)*\s+)(?:\s*)(['\"]?)([^\s;&|]+)\1")
    for match in pattern.finditer(command):
        target = match.group(2)
        if _path_is_outside(target, cwd):
            return target
    return None


def _path_is_outside(raw_path: str, workspace: Path) -> bool:
    if raw_path in {"-", "."}:
        return False
    path = Path(raw_path).expanduser()
    candidate = path if path.is_absolute() else workspace / path
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return True
    return not resolved.is_relative_to(workspace)


def _is_package_install(executable: str, arguments: Sequence[str]) -> bool:
    if executable in {"pip", "pip3"}:
        return bool(arguments and arguments[0] == "install")
    if executable == "uv":
        return any(argument in {"add", "pip", "sync"} for argument in arguments[:2])
    if executable in {"npm", "pnpm", "yarn"}:
        return bool(arguments and arguments[0] in {"add", "ci", "i", "install"})
    return True


def _is_python_test_command(arguments: Sequence[str]) -> bool:
    return len(arguments) >= 2 and arguments[0] == "-m" and arguments[1] in {"pytest", "unittest"}


def _is_python_package_install(arguments: Sequence[str]) -> bool:
    return len(arguments) >= 3 and arguments[:2] == ("-m", "pip") and arguments[2] == "install"


def _is_python_executable(executable: str) -> bool:
    return executable in {"python", "python3"} or re.fullmatch(
        r"python3(?:\.\d+)+", executable
    ) is not None


def _is_javascript_test_command(arguments: Sequence[str]) -> bool:
    return bool(arguments and (arguments[0] == "test" or arguments[:2] == ("run", "test")))
