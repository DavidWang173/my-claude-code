from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from src.sandbox import (
    ContainerSandboxRuntime,
    LocalSandboxRuntime,
    NetworkMode,
    SandboxArtifact,
    SandboxCommand,
    SandboxExecutionResult,
    SandboxHealth,
    SandboxPolicy,
    SandboxPolicyError,
    SecretPolicy,
    SecurityLevel,
    WorkspaceMountMode,
)
from src.shell_tools import ShellCommandPolicy, ShellTool
from src.tools import ToolContext


class _RecordingRuntime:
    security_level = SecurityLevel.CONTAINER

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.policy = SandboxPolicy.container_default()
        self.prepared = 0
        self.executed: list[SandboxCommand] = []
        self.cleaned = 0

    async def prepare(self) -> None:
        self.prepared += 1

    async def execute(
        self,
        command: SandboxCommand,
        *,
        output_handler=None,
    ) -> SandboxExecutionResult:
        self.executed.append(command)
        if output_handler is not None:
            await output_handler("stdout", b"isolated\n")
        return SandboxExecutionResult(
            stdout=b"isolated\n",
            stderr=b"",
            exit_code=0,
            output_bytes=9,
        )

    async def read_file(self, path: str | Path) -> bytes:
        raise AssertionError("not used")

    async def write_file(self, path: str | Path, data: bytes) -> None:
        raise AssertionError("not used")

    async def collect_artifacts(self) -> tuple[SandboxArtifact, ...]:
        return ()

    async def cleanup(self) -> None:
        self.cleaned += 1

    async def health_check(self) -> SandboxHealth:
        return SandboxHealth(True, self.security_level, "test runtime")


class _CleanupRecordingContainerRuntime(ContainerSandboxRuntime):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace)
        self.control_calls: list[tuple[str, ...]] = []

    async def _docker_control(
        self,
        *arguments: str,
        timeout: float,
    ) -> SandboxExecutionResult:
        self.control_calls.append(tuple(arguments))
        return SandboxExecutionResult(b"", b"", 0)


class SandboxSecurityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.container = Path(self._temporary.name).resolve()
        self.workspace = self.container / "workspace"
        self.workspace.mkdir()

    async def test_local_runtime_is_explicitly_application_only(self) -> None:
        runtime = LocalSandboxRuntime(self.workspace)
        health = await runtime.health_check()

        self.assertTrue(health.healthy)
        self.assertEqual(runtime.security_level, SecurityLevel.APPLICATION_ONLY)
        self.assertIn("application-level", health.detail)
        self.assertIn("not OS-enforced", health.detail)

    async def test_environment_allowlist_excludes_secrets(self) -> None:
        policy = SandboxPolicy(
            environment_allowlist=(
                "CODING_AGENT_API_KEY",
                "PATH",
                "VISIBLE_VALUE",
            ),
            secret_policy=SecretPolicy.DENY,
        )
        runtime = LocalSandboxRuntime(
            self.workspace,
            policy=policy,
            environ={
                "CODING_AGENT_API_KEY": "must-not-cross-boundary",
                "PATH": "/usr/bin:/bin",
                "UNLISTED_VALUE": "hidden",
                "VISIBLE_VALUE": "visible",
            },
        )
        await runtime.prepare()
        result = await runtime.execute(
            SandboxCommand(
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import os; "
                        "print(os.getenv('CODING_AGENT_API_KEY')); "
                        "print(os.getenv('UNLISTED_VALUE')); "
                        "print(os.getenv('VISIBLE_VALUE')); "
                        "print(os.environ['HOME'])"
                    ),
                )
            )
        )
        await runtime.cleanup()

        lines = result.stdout.decode().splitlines()
        self.assertEqual(lines[:3], ["None", "None", "visible"])
        self.assertEqual(lines[3], str(self.workspace))
        self.assertNotIn(b"must-not-cross-boundary", result.stdout)

    async def test_read_write_paths_and_symlinks_stay_in_allowed_paths(self) -> None:
        allowed = self.workspace / "allowed"
        allowed.mkdir()
        permitted_file = allowed / "ok.txt"
        permitted_file.write_text("ok", encoding="utf-8")
        denied_file = self.workspace / "denied.txt"
        denied_file.write_text("denied", encoding="utf-8")
        outside_file = self.container / "outside.txt"
        outside_file.write_text("outside", encoding="utf-8")
        (allowed / "escape").symlink_to(outside_file)
        (allowed / "write-link").symlink_to(permitted_file)
        runtime = LocalSandboxRuntime(
            self.workspace,
            policy=SandboxPolicy(allowed_paths=(allowed,)),
        )

        self.assertEqual(await runtime.read_file("allowed/ok.txt"), b"ok")
        with self.assertRaises(SandboxPolicyError):
            await runtime.read_file("denied.txt")
        with self.assertRaises(SandboxPolicyError):
            await runtime.read_file("allowed/escape")
        with self.assertRaises(SandboxPolicyError):
            await runtime.write_file(outside_file, b"changed")
        with self.assertRaises(SandboxPolicyError):
            await runtime.write_file("allowed/write-link", b"changed")
        await runtime.write_file("allowed/new.txt", b"new")
        self.assertEqual(outside_file.read_bytes(), b"outside")
        self.assertEqual(permitted_file.read_bytes(), b"ok")
        self.assertEqual((allowed / "new.txt").read_bytes(), b"new")

    async def test_read_only_policy_blocks_runtime_writes(self) -> None:
        runtime = LocalSandboxRuntime(
            self.workspace,
            policy=SandboxPolicy(
                workspace_mount_mode=WorkspaceMountMode.READ_ONLY,
            ),
        )

        with self.assertRaises(SandboxPolicyError):
            await runtime.write_file("new.txt", b"content")
        self.assertFalse((self.workspace / "new.txt").exists())

    async def test_timeout_is_enforced_and_process_is_terminated(self) -> None:
        runtime = LocalSandboxRuntime(
            self.workspace,
            policy=SandboxPolicy(timeout=0.05),
        )
        await runtime.prepare()
        result = await runtime.execute(
            SandboxCommand(
                argv=(sys.executable, "-c", "import time; time.sleep(10)"),
                timeout=5,
            )
        )
        await runtime.cleanup()

        self.assertTrue(result.timed_out)
        self.assertIsNotNone(result.exit_code)

    async def test_shell_and_test_commands_use_runtime_after_permission_boundary(self) -> None:
        runtime = _RecordingRuntime(self.workspace)
        tool = ShellTool(runtime=runtime)
        context = ToolContext("sandbox-test", self.workspace)

        test_result = await tool.execute(
            {"argv": [sys.executable, "-m", "unittest"]},
            context,
        )
        denied_write = await tool.execute(
            {"argv": ["touch", "not-created.txt"]},
            context,
        )

        self.assertTrue(test_result.success, test_result.error)
        self.assertIn("isolated", test_result.content)
        self.assertEqual(test_result.metadata["security_level"], "CONTAINER")
        self.assertEqual(runtime.prepared, 1)
        self.assertEqual(len(runtime.executed), 1)
        self.assertEqual(runtime.cleaned, 1)
        self.assertFalse(denied_write.success)
        self.assertFalse((self.workspace / "not-created.txt").exists())

    async def test_missing_docker_fails_closed_without_local_fallback(self) -> None:
        runtime = ContainerSandboxRuntime(
            self.workspace,
            docker_executable="coding-agent-test-docker-does-not-exist",
        )
        result = await ShellTool(
            runtime=runtime,
            policy=ShellCommandPolicy(("/usr/bin/true",)),
        ).execute(
            {"argv": ["/usr/bin/true"]},
            ToolContext("container-test", self.workspace),
        )

        self.assertFalse(result.success)
        self.assertIn("Docker executable", result.error or "")
        self.assertIn("not available", result.error or "")
        self.assertEqual(result.metadata["security_level"], "CONTAINER")

    async def test_container_defaults_encode_network_identity_and_resource_limits(
        self,
    ) -> None:
        policy = SandboxPolicy.container_default(workspace_writable=False)
        runtime = ContainerSandboxRuntime(self.workspace, policy=policy)
        artifacts = self.container / "artifact-staging"
        artifacts.mkdir()
        arguments = runtime._build_create_arguments("test-container", artifacts)
        rendered = " ".join(arguments)

        self.assertIn("--network none", rendered)
        self.assertIn("--user 65532:65532", rendered)
        self.assertIn("--cpus 1", rendered)
        self.assertIn(f"--memory {512 * 1024 * 1024}", rendered)
        self.assertIn("--pids-limit 128", rendered)
        self.assertIn("--cap-drop ALL", rendered)
        self.assertIn("--security-opt no-new-privileges", rendered)
        self.assertIn("--read-only", arguments)
        self.assertIn("dst=/workspace,readonly", rendered)
        self.assertNotIn("docker.sock", rendered)
        self.assertNotIn("API_KEY", rendered)

    async def test_container_artifacts_are_recovered_before_cleanup(self) -> None:
        runtime = ContainerSandboxRuntime(self.workspace)
        staging = tempfile.TemporaryDirectory()
        self.addCleanup(staging.cleanup)
        runtime._artifact_directory = staging
        runtime._container_name = "test-artifact-export"
        report = Path(staging.name) / "reports" / "result.txt"
        report.parent.mkdir()
        report.write_text("verified", encoding="utf-8")

        artifacts = await runtime.collect_artifacts()

        self.assertEqual(
            artifacts,
            (
                SandboxArtifact(
                    path="artifacts/test-artifact-export/reports/result.txt",
                    size=8,
                ),
            ),
        )
        recovered = self.workspace / artifacts[0].path
        self.assertEqual(recovered.read_text(encoding="utf-8"), "verified")
        runtime._artifact_directory = None
        runtime._container_name = None

    async def test_container_cleanup_force_removes_container_and_staging(self) -> None:
        runtime = _CleanupRecordingContainerRuntime(self.workspace)
        staging = tempfile.TemporaryDirectory()
        staging_path = Path(staging.name)
        runtime._artifact_directory = staging
        runtime._container_name = "test-cleanup"

        await runtime.cleanup()

        self.assertEqual(
            runtime.control_calls,
            [("rm", "--force", "test-cleanup")],
        )
        self.assertFalse(staging_path.exists())
        self.assertIsNone(runtime._container_name)
        self.assertIsNone(runtime._artifact_directory)

    async def test_container_allowed_paths_are_mounted_without_full_workspace(
        self,
    ) -> None:
        source = self.workspace / "source"
        source.mkdir()
        runtime = ContainerSandboxRuntime(
            self.workspace,
            policy=SandboxPolicy(
                workspace_mount_mode=WorkspaceMountMode.READ_ONLY,
                allowed_paths=(source,),
                network_mode=NetworkMode.DISABLED,
                cpu_limit=1.0,
                memory_limit=512 * 1024 * 1024,
                process_limit=128,
                timeout=120.0,
                environment_allowlist=("PATH",),
                secret_policy=SecretPolicy.DENY,
            ),
        )
        artifacts = self.container / "artifact-staging"
        artifacts.mkdir()
        arguments = runtime._build_create_arguments("test-container", artifacts)
        rendered = " ".join(arguments)

        self.assertIn(f"src={source},dst=/workspace/source,readonly", rendered)
        self.assertNotIn(f"src={self.workspace},dst=/workspace,", rendered)
        self.assertIn("/workspace:rw,nosuid,nodev", rendered)

    def test_invalid_resource_boundaries_are_rejected(self) -> None:
        for kwargs in (
            {"cpu_limit": 0},
            {"memory_limit": 0},
            {"process_limit": 0},
            {"timeout": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SandboxPolicy(**kwargs)


if __name__ == "__main__":
    unittest.main()
