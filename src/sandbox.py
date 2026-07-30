"""Process isolation boundary for local and container-backed execution.

Permission approval and command classification deliberately live above this
module.  A sandbox constrains an already-approved operation; it never decides
whether an operation is allowed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import stat
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol
from uuid import uuid4


class SecurityLevel(str, Enum):
    APPLICATION_ONLY = "APPLICATION_ONLY"
    CONTAINER = "CONTAINER"


class WorkspaceMountMode(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class NetworkMode(str, Enum):
    DISABLED = "none"
    BRIDGE = "bridge"


class SecretPolicy(str, Enum):
    DENY = "deny"
    ALLOWLIST = "allowlist"


_DEFAULT_ENVIRONMENT_ALLOWLIST = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "SYSTEMROOT",
    "TERM",
    "TMPDIR",
)
_SENSITIVE_ENVIRONMENT_FRAGMENTS = (
    "API_KEY",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)
_DEFAULT_MEMORY_LIMIT = 512 * 1024 * 1024
_DEFAULT_ARTIFACT_LIMIT = 100 * 1024 * 1024


@dataclass(frozen=True)
class SandboxPolicy:
    workspace_mount_mode: WorkspaceMountMode = WorkspaceMountMode.READ_WRITE
    allowed_paths: tuple[Path, ...] = ()
    network_mode: NetworkMode = NetworkMode.DISABLED
    cpu_limit: float = 1.0
    memory_limit: int = _DEFAULT_MEMORY_LIMIT
    process_limit: int = 128
    timeout: float = 120.0
    environment_allowlist: tuple[str, ...] = _DEFAULT_ENVIRONMENT_ALLOWLIST
    secret_policy: SecretPolicy = SecretPolicy.DENY

    def __post_init__(self) -> None:
        if self.cpu_limit <= 0:
            raise ValueError("cpu_limit must be positive")
        if self.memory_limit <= 0:
            raise ValueError("memory_limit must be positive")
        if self.process_limit <= 0:
            raise ValueError("process_limit must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if not all(isinstance(path, Path) for path in self.allowed_paths):
            raise ValueError("allowed_paths must contain Path values")
        if not all(
            name
            and name.replace("_", "").isalnum()
            and name.upper() == name
            for name in self.environment_allowlist
        ):
            raise ValueError(
                "environment_allowlist must contain uppercase environment names"
            )

    @classmethod
    def local_default(cls, *, timeout: float = 120.0) -> SandboxPolicy:
        return cls(timeout=timeout)

    @classmethod
    def container_default(
        cls,
        *,
        workspace_writable: bool = False,
        timeout: float = 120.0,
    ) -> SandboxPolicy:
        return cls(
            workspace_mount_mode=(
                WorkspaceMountMode.READ_WRITE
                if workspace_writable
                else WorkspaceMountMode.READ_ONLY
            ),
            network_mode=NetworkMode.DISABLED,
            cpu_limit=1.0,
            memory_limit=_DEFAULT_MEMORY_LIMIT,
            process_limit=128,
            timeout=timeout,
            environment_allowlist=_DEFAULT_ENVIRONMENT_ALLOWLIST,
            secret_policy=SecretPolicy.DENY,
        )


@dataclass(frozen=True)
class SandboxCommand:
    argv: tuple[str, ...] | None = None
    shell_command: str | None = None
    cwd: Path | None = None
    timeout: float | None = None
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if (self.argv is None) == (self.shell_command is None):
            raise ValueError("provide exactly one of argv or shell_command")
        if self.argv is not None and (
            not self.argv or not all(isinstance(item, str) and item for item in self.argv)
        ):
            raise ValueError("argv must contain non-empty strings")
        if self.shell_command is not None and not self.shell_command.strip():
            raise ValueError("shell_command must not be empty")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")


@dataclass(frozen=True)
class SandboxExecutionResult:
    stdout: bytes
    stderr: bytes
    exit_code: int | None
    timed_out: bool = False
    truncated: bool = False
    output_bytes: int = 0


@dataclass(frozen=True)
class SandboxArtifact:
    path: str
    size: int


@dataclass(frozen=True)
class SandboxHealth:
    healthy: bool
    security_level: SecurityLevel
    detail: str


SandboxOutputHandler = Callable[[str, bytes], Awaitable[None]]


class SandboxRuntime(Protocol):
    @property
    def workspace(self) -> Path: ...

    @property
    def policy(self) -> SandboxPolicy: ...

    @property
    def security_level(self) -> SecurityLevel: ...

    async def prepare(self) -> None: ...

    async def execute(
        self,
        command: SandboxCommand,
        *,
        output_handler: SandboxOutputHandler | None = None,
    ) -> SandboxExecutionResult: ...

    async def read_file(self, path: str | Path) -> bytes: ...

    async def write_file(self, path: str | Path, data: bytes) -> None: ...

    async def collect_artifacts(self) -> tuple[SandboxArtifact, ...]: ...

    async def cleanup(self) -> None: ...

    async def health_check(self) -> SandboxHealth: ...


class SandboxError(RuntimeError):
    pass


class SandboxUnavailableError(SandboxError):
    pass


class SandboxPolicyError(SandboxError):
    pass


class LocalSandboxRuntime:
    """Compatibility runtime that preserves the current host execution model.

    This runtime provides application-level checks only.  It does not claim
    kernel, container, VM, network, CPU, memory, or process isolation.
    """

    security_level = SecurityLevel.APPLICATION_ONLY

    def __init__(
        self,
        workspace: Path,
        *,
        policy: SandboxPolicy | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._workspace = _resolved_workspace(workspace)
        self._policy = policy or SandboxPolicy.local_default()
        self._environ = dict(os.environ if environ is None else environ)
        self._prepared = False
        _validate_policy_paths(self._workspace, self._policy)

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    async def prepare(self) -> None:
        self._prepared = True

    async def execute(
        self,
        command: SandboxCommand,
        *,
        output_handler: SandboxOutputHandler | None = None,
    ) -> SandboxExecutionResult:
        if not self._prepared:
            raise SandboxError("sandbox runtime has not been prepared")
        cwd = _resolve_runtime_cwd(
            self._workspace,
            command.cwd or self._workspace,
            self._policy,
        )
        environment = _filtered_environment(
            self._environ,
            self._policy,
            home=str(self._workspace),
        )
        timeout = min(command.timeout or self._policy.timeout, self._policy.timeout)
        return await _execute_process(
            argv=command.argv,
            shell_command=command.shell_command,
            cwd=cwd,
            environment=environment,
            timeout=timeout,
            max_output_bytes=command.max_output_bytes,
            output_handler=output_handler,
        )

    async def read_file(self, path: str | Path) -> bytes:
        target = _resolve_runtime_path(
            self._workspace, path, self._policy, must_exist=True
        )
        if not target.is_file():
            raise SandboxPolicyError("sandbox read target is not a regular file")
        return target.read_bytes()

    async def write_file(self, path: str | Path, data: bytes) -> None:
        if self._policy.workspace_mount_mode is WorkspaceMountMode.READ_ONLY:
            raise SandboxPolicyError("sandbox workspace is read-only")
        requested = Path(path).expanduser()
        candidate = requested if requested.is_absolute() else self._workspace / requested
        try:
            raw_parent = candidate.parent.absolute()
            canonical_parent = candidate.parent.resolve(strict=True)
        except (OSError, RuntimeError):
            raise SandboxPolicyError("sandbox write parent does not exist") from None
        if canonical_parent != raw_parent or candidate.is_symlink():
            raise SandboxPolicyError(
                "sandbox write target contains or is a symbolic link"
            )
        target = _resolve_runtime_path(
            self._workspace, path, self._policy, must_exist=False
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor: int | None = None
        try:
            if os.name == "posix":
                parent_descriptor = _open_workspace_parent(self._workspace, target)
                descriptor = os.open(
                    target.name,
                    flags,
                    0o644,
                    dir_fd=parent_descriptor,
                )
            else:
                descriptor = os.open(target, flags, 0o644)
        except OSError:
            raise SandboxPolicyError("sandbox write target could not be opened safely") from None
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)
        try:
            view = memoryview(data)
            written = 0
            while written < len(view):
                written += os.write(descriptor, view[written:])
        finally:
            os.close(descriptor)

    async def collect_artifacts(self) -> tuple[SandboxArtifact, ...]:
        return _collect_artifact_metadata(self._workspace / "artifacts")

    async def cleanup(self) -> None:
        self._prepared = False

    async def health_check(self) -> SandboxHealth:
        return SandboxHealth(
            healthy=True,
            security_level=self.security_level,
            detail=(
                "Local execution is available with application-level checks only; "
                "network and resource limits are not OS-enforced."
            ),
        )


class ContainerSandboxRuntime:
    """Experimental Docker CLI backed runtime.

    Docker is invoked behind the SandboxRuntime boundary.  Callers never use a
    Docker API directly, and absence of Docker is reported rather than falling
    back to local execution.
    """

    security_level = SecurityLevel.CONTAINER

    def __init__(
        self,
        workspace: Path,
        *,
        policy: SandboxPolicy | None = None,
        image: str = "python:3.11-alpine",
        docker_executable: str = "docker",
        host_environ: Mapping[str, str] | None = None,
    ) -> None:
        if not image.strip():
            raise ValueError("container image must not be empty")
        if not docker_executable.strip():
            raise ValueError("docker executable must not be empty")
        self._workspace = _resolved_workspace(workspace)
        self._policy = policy or SandboxPolicy.container_default()
        self._image = image
        self._docker_executable = docker_executable
        self._host_environ = dict(os.environ if host_environ is None else host_environ)
        self._container_name: str | None = None
        self._artifact_directory: tempfile.TemporaryDirectory[str] | None = None
        _validate_policy_paths(self._workspace, self._policy)

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    async def health_check(self) -> SandboxHealth:
        executable = shutil.which(
            self._docker_executable,
            path=self._host_environ.get("PATH"),
        )
        if executable is None:
            return SandboxHealth(
                healthy=False,
                security_level=self.security_level,
                detail=(
                    f"Docker executable {self._docker_executable!r} is not available; "
                    "container execution cannot start."
                ),
            )
        result = await _execute_process(
            argv=(executable, "version", "--format", "{{.Server.Version}}"),
            shell_command=None,
            cwd=self._workspace,
            environment=_docker_host_environment(self._host_environ),
            timeout=min(self._policy.timeout, 10.0),
            max_output_bytes=16_384,
        )
        if result.timed_out or result.exit_code != 0:
            return SandboxHealth(
                healthy=False,
                security_level=self.security_level,
                detail="Docker is installed but its daemon is unavailable.",
            )
        return SandboxHealth(
            healthy=True,
            security_level=self.security_level,
            detail="Docker daemon is available.",
        )

    async def prepare(self) -> None:
        if self._container_name is not None:
            return
        health = await self.health_check()
        if not health.healthy:
            raise SandboxUnavailableError(health.detail)
        image_check = await self._docker_control(
            "image",
            "inspect",
            self._image,
            timeout=min(self._policy.timeout, 10.0),
        )
        if image_check.exit_code != 0:
            raise SandboxUnavailableError(
                f"Container image {self._image!r} is not available locally; "
                "pull or build it explicitly before selecting container mode."
            )

        artifacts = tempfile.TemporaryDirectory(prefix="coding-agent-artifacts-")
        artifact_path = Path(artifacts.name).resolve()
        os.chmod(artifact_path, 0o777)
        container_name = f"coding-agent-{uuid4().hex}"
        create = await self._docker_control(
            *self._build_create_arguments(container_name, artifact_path),
            timeout=min(self._policy.timeout, 30.0),
        )
        if create.exit_code != 0:
            artifacts.cleanup()
            raise SandboxUnavailableError(
                "Docker could not create the requested sandbox container."
            )
        self._container_name = container_name
        self._artifact_directory = artifacts
        start = await self._docker_control(
            "start",
            container_name,
            timeout=min(self._policy.timeout, 30.0),
        )
        if start.exit_code != 0:
            await self.cleanup()
            raise SandboxUnavailableError("Docker could not start the sandbox container.")

    def _build_create_arguments(
        self,
        container_name: str,
        artifact_path: Path,
    ) -> tuple[str, ...]:
        access = (
            ",readonly"
            if self._policy.workspace_mount_mode is WorkspaceMountMode.READ_ONLY
            else ""
        )
        mounts: list[str] = []
        if self._policy.allowed_paths:
            for allowed_path in _allowed_roots(self._workspace, self._policy):
                relative = allowed_path.relative_to(self._workspace)
                destination = (
                    "/workspace"
                    if not relative.parts
                    else f"/workspace/{relative.as_posix()}"
                )
                mounts.extend(
                    (
                        "--mount",
                        f"type=bind,src={allowed_path},dst={destination}{access}",
                    )
                )
        else:
            mounts.extend(
                (
                    "--mount",
                    f"type=bind,src={self._workspace},dst=/workspace{access}",
                )
            )
        return (
            "create",
            "--name",
            container_name,
            "--workdir",
            "/workspace",
            "--user",
            "65532:65532",
            "--network",
            self._policy.network_mode.value,
            "--cpus",
            f"{self._policy.cpu_limit:g}",
            "--memory",
            str(self._policy.memory_limit),
            "--pids-limit",
            str(self._policy.process_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=67108864",
            *(
                ("--tmpfs", "/workspace:rw,nosuid,nodev,size=67108864")
                if self._policy.allowed_paths
                else ()
            ),
            *mounts,
            "--mount",
            f"type=bind,src={artifact_path},dst=/artifacts",
            "--env",
            "HOME=/tmp",
            self._image,
            "sh",
            "-c",
            "while :; do sleep 3600; done",
        )

    async def execute(
        self,
        command: SandboxCommand,
        *,
        output_handler: SandboxOutputHandler | None = None,
    ) -> SandboxExecutionResult:
        if self._container_name is None:
            raise SandboxError("sandbox runtime has not been prepared")
        _resolve_runtime_cwd(
            self._workspace,
            command.cwd or self._workspace,
            self._policy,
        )
        environment = _filtered_environment(
            self._host_environ,
            self._policy,
            home="/tmp",
        )
        arguments = [
            self._docker_executable,
            "exec",
            "--workdir",
            "/workspace",
        ]
        for name, value in sorted(environment.items()):
            arguments.extend(("--env", f"{name}={value}"))
        arguments.append(self._container_name)
        if command.argv is not None:
            arguments.extend(command.argv)
        else:
            assert command.shell_command is not None
            arguments.extend(("sh", "-lc", command.shell_command))
        timeout = min(command.timeout or self._policy.timeout, self._policy.timeout)
        result = await _execute_process(
            argv=tuple(arguments),
            shell_command=None,
            cwd=self._workspace,
            environment=_docker_host_environment(self._host_environ),
            timeout=timeout,
            max_output_bytes=command.max_output_bytes,
            output_handler=output_handler,
        )
        if result.timed_out:
            await self.cleanup()
        else:
            paused = await self._docker_control(
                "pause",
                self._container_name,
                timeout=min(self._policy.timeout, 10.0),
            )
            if paused.exit_code != 0:
                await self.cleanup()
                raise SandboxError(
                    "Docker could not freeze the sandbox before artifact collection."
                )
        return result

    async def read_file(self, path: str | Path) -> bytes:
        target = _resolve_runtime_path(
            self._workspace, path, self._policy, must_exist=True
        )
        if not target.is_file():
            raise SandboxPolicyError("sandbox read target is not a regular file")
        return target.read_bytes()

    async def write_file(self, path: str | Path, data: bytes) -> None:
        if self._policy.workspace_mount_mode is WorkspaceMountMode.READ_ONLY:
            raise SandboxPolicyError("sandbox workspace is read-only")
        # Host-side compatibility writes use the same path boundary as Local.
        compatibility = LocalSandboxRuntime(
            self._workspace,
            policy=self._policy,
            environ={},
        )
        await compatibility.write_file(path, data)

    async def collect_artifacts(self) -> tuple[SandboxArtifact, ...]:
        if self._artifact_directory is None:
            return ()
        source = Path(self._artifact_directory.name)
        artifacts = _collect_artifact_metadata(
            source,
            max_total_bytes=_DEFAULT_ARTIFACT_LIMIT,
        )
        if not artifacts:
            return ()
        container_name = self._container_name
        if container_name is None:
            raise SandboxError("sandbox container is not prepared")
        destination = _create_artifact_destination(
            self._workspace,
            container_name,
        )
        exported: list[SandboxArtifact] = []
        for artifact in artifacts:
            source_file = source / artifact.path
            target = destination / artifact.path
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_new_file(source_file, target)
            exported.append(
                SandboxArtifact(
                    path=target.relative_to(self._workspace).as_posix(),
                    size=artifact.size,
                )
            )
        return tuple(exported)

    async def cleanup(self) -> None:
        container_name = self._container_name
        self._container_name = None
        if container_name is not None:
            await self._docker_control(
                "rm",
                "--force",
                container_name,
                timeout=min(self._policy.timeout, 30.0),
            )
        artifacts = self._artifact_directory
        self._artifact_directory = None
        if artifacts is not None:
            artifacts.cleanup()

    async def _docker_control(
        self,
        *arguments: str,
        timeout: float,
    ) -> SandboxExecutionResult:
        return await _execute_process(
            argv=(self._docker_executable, *arguments),
            shell_command=None,
            cwd=self._workspace,
            environment=_docker_host_environment(self._host_environ),
            timeout=timeout,
            max_output_bytes=64_000,
        )


def _resolved_workspace(workspace: Path) -> Path:
    try:
        resolved = workspace.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise SandboxPolicyError("sandbox workspace does not exist") from None
    if not resolved.is_dir():
        raise SandboxPolicyError("sandbox workspace is not a directory")
    return resolved


def _validate_policy_paths(workspace: Path, policy: SandboxPolicy) -> None:
    for allowed in policy.allowed_paths:
        try:
            resolved = allowed.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise SandboxPolicyError(
                f"allowed sandbox path does not exist: {allowed}"
            ) from None
        if not resolved.is_relative_to(workspace):
            raise SandboxPolicyError(
                f"allowed sandbox path is outside the workspace: {allowed}"
            )


def _allowed_roots(workspace: Path, policy: SandboxPolicy) -> tuple[Path, ...]:
    if not policy.allowed_paths:
        return (workspace,)
    return tuple(path.expanduser().resolve(strict=True) for path in policy.allowed_paths)


def _resolve_runtime_path(
    workspace: Path,
    raw_path: str | Path,
    policy: SandboxPolicy,
    *,
    must_exist: bool,
) -> Path:
    requested = Path(raw_path).expanduser()
    candidate = requested if requested.is_absolute() else workspace / requested
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError):
        raise SandboxPolicyError("sandbox path does not exist or cannot be resolved") from None
    if not any(resolved.is_relative_to(root) for root in _allowed_roots(workspace, policy)):
        raise SandboxPolicyError("sandbox path is outside allowed_paths")
    return resolved


def _resolve_runtime_cwd(
    workspace: Path,
    raw_path: Path,
    policy: SandboxPolicy,
) -> Path:
    resolved = _resolve_runtime_path(workspace, raw_path, policy, must_exist=True)
    if not resolved.is_dir():
        raise SandboxPolicyError("sandbox cwd is not a directory")
    return resolved


def _filtered_environment(
    source: Mapping[str, str],
    policy: SandboxPolicy,
    *,
    home: str,
) -> dict[str, str]:
    environment = {"HOME": home}
    for name in policy.environment_allowlist:
        if (
            policy.secret_policy is SecretPolicy.DENY
            and _is_sensitive_environment_name(name)
        ):
            continue
        value = source.get(name)
        if value is not None:
            environment[name] = value
    environment.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return environment


def _is_sensitive_environment_name(name: str) -> bool:
    upper = name.upper()
    return any(fragment in upper for fragment in _SENSITIVE_ENVIRONMENT_FRAGMENTS)


def _docker_host_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Environment for the Docker client, never the container payload."""

    allowed = {
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "DOCKER_TLS_VERIFY",
        "HOME",
        "PATH",
        "SYSTEMROOT",
        "TMPDIR",
    }
    return {name: value for name, value in source.items() if name in allowed}


def _open_workspace_parent(workspace: Path, target: Path) -> int:
    try:
        relative_parent = target.parent.relative_to(workspace)
    except ValueError:
        raise SandboxPolicyError("sandbox write target escaped the workspace") from None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(workspace, flags)
    try:
        for part in relative_parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


async def _execute_process(
    *,
    argv: Sequence[str] | None,
    shell_command: str | None,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
    max_output_bytes: int,
    output_handler: SandboxOutputHandler | None = None,
) -> SandboxExecutionResult:
    common: dict[str, object] = {
        "cwd": cwd,
        "env": dict(environment),
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "start_new_session": True,
    }
    try:
        if argv is not None:
            process = await asyncio.create_subprocess_exec(*argv, **common)  # type: ignore[arg-type]
        else:
            assert shell_command is not None
            process = await asyncio.create_subprocess_shell(  # type: ignore[arg-type]
                shell_command,
                **common,
            )
    except OSError:
        raise SandboxError("command could not be started") from None

    queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue(maxsize=32)
    readers = (
        asyncio.create_task(_read_stream("stdout", process.stdout, queue)),
        asyncio.create_task(_read_stream("stderr", process.stderr, queue)),
    )
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    captured_bytes = 0
    truncated = False
    active_streams = 2
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    timed_out = False
    try:
        while active_streams:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            try:
                stream_name, chunk = await asyncio.wait_for(queue.get(), remaining)
            except asyncio.TimeoutError:
                raise TimeoutError from None
            if chunk is None:
                active_streams -= 1
                continue
            available = max_output_bytes - captured_bytes
            kept = chunk[: max(available, 0)]
            if kept:
                captured[stream_name].extend(kept)
                captured_bytes += len(kept)
                if output_handler is not None:
                    await output_handler(stream_name, kept)
            if len(chunk) > len(kept):
                truncated = True
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError
        try:
            await asyncio.wait_for(process.wait(), remaining)
        except asyncio.TimeoutError:
            raise TimeoutError from None
    except TimeoutError:
        timed_out = True
        await _terminate_process(process)
    except asyncio.CancelledError:
        await _terminate_process(process)
        raise
    finally:
        if process.returncode is None:
            await _terminate_process(process)
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)

    return SandboxExecutionResult(
        stdout=bytes(captured["stdout"]),
        stderr=bytes(captured["stderr"]),
        exit_code=process.returncode,
        timed_out=timed_out,
        truncated=truncated,
        output_bytes=captured_bytes,
    )


async def _read_stream(
    name: str,
    stream: asyncio.StreamReader | None,
    queue: asyncio.Queue[tuple[str, bytes | None]],
) -> None:
    try:
        if stream is not None:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                await queue.put((name, chunk))
    finally:
        await queue.put((name, None))


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), 0.5)
        return
    except asyncio.TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


def _collect_artifact_metadata(
    root: Path,
    *,
    max_total_bytes: int = _DEFAULT_ARTIFACT_LIMIT,
) -> tuple[SandboxArtifact, ...]:
    if not root.exists():
        return ()
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise SandboxPolicyError("artifact directory cannot be resolved") from None
    artifacts: list[SandboxArtifact] = []
    total_bytes = 0
    for current, directories, files in os.walk(resolved_root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if not (current_path / name).is_symlink()
        ]
        for name in sorted(files):
            candidate = current_path / name
            try:
                file_stat = candidate.stat(follow_symlinks=False)
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if (
                candidate.is_symlink()
                or not stat.S_ISREG(file_stat.st_mode)
                or not resolved.is_relative_to(resolved_root)
            ):
                continue
            total_bytes += file_stat.st_size
            if total_bytes > max_total_bytes:
                raise SandboxPolicyError("sandbox artifacts exceed the collection limit")
            artifacts.append(
                SandboxArtifact(
                    path=resolved.relative_to(resolved_root).as_posix(),
                    size=file_stat.st_size,
                )
            )
    return tuple(artifacts)


def _create_artifact_destination(workspace: Path, run_name: str) -> Path:
    root = workspace / "artifacts"
    if root.is_symlink():
        raise SandboxPolicyError("workspace artifacts path is a symbolic link")
    try:
        root.mkdir(mode=0o700, exist_ok=True)
        resolved_root = root.resolve(strict=True)
    except OSError:
        raise SandboxPolicyError("workspace artifacts directory cannot be created") from None
    if resolved_root != root:
        raise SandboxPolicyError("workspace artifacts path escaped the workspace")
    destination = root / run_name
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        raise SandboxPolicyError("sandbox artifact destination already exists") from None
    except OSError:
        raise SandboxPolicyError("sandbox artifact destination cannot be created") from None
    return destination


def _copy_new_file(source: Path, target: Path) -> None:
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, read_flags)
        source_stat = os.fstat(source_descriptor)
    except OSError:
        raise SandboxPolicyError("sandbox artifact source changed before copy") from None
    if not stat.S_ISREG(source_stat.st_mode):
        os.close(source_descriptor)
        raise SandboxPolicyError("sandbox artifact source is not a regular file")

    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    write_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        target_descriptor = os.open(target, write_flags, 0o600)
    except OSError:
        os.close(source_descriptor)
        raise SandboxPolicyError("sandbox artifact target could not be created safely") from None
    try:
        while True:
            chunk = os.read(source_descriptor, 64 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            written = 0
            while written < len(view):
                written += os.write(target_descriptor, view[written:])
    finally:
        os.close(source_descriptor)
        os.close(target_descriptor)
