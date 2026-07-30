from __future__ import annotations

import io
import logging
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from src.config import load_config
from src.logging_config import SecretRedactingFilter
from src.permissions import Operation, PermissionRequest, ReadOnlyPermissionPolicy
from src.providers import ProviderNotFoundError, ProviderRegistry
from src.sessions import JsonSessionStore, Session
from src.tools import ToolNotFoundError, ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_pyproject_requires_python_311_and_defines_command(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["scripts"]["agent"], "src.main:main")
        self.assertEqual(project["scripts"]["coding-agent"], "src.main:main")

    def test_expected_module_boundaries_import(self) -> None:
        for module in (
            "src.cli",
            "src.config",
            "src.context",
            "src.providers",
            "src.openai_provider",
            "src.agent",
            "src.tools",
            "src.shell_tools",
            "src.sandbox",
            "src.sessions",
            "src.permissions",
            "src.terminal_ui",
            "src.harness",
            "src.harness.models",
            "src.harness.orchestrator",
            "src.harness.planning",
            "src.harness.project_knowledge",
            "src.harness.verification",
            "src.harness.repair",
        ):
            with self.subTest(module=module):
                __import__(module)

    def test_cli_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("usage: agent", result.stdout)
        self.assertIn("diagnostics", result.stdout)

    def test_diagnostics_summary_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "diagnostics", "summary"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Coding Agent Diagnostics Summary", result.stdout)

    def test_config_repr_and_logging_do_not_expose_api_key(self) -> None:
        secret = "secret-test-key"
        config = load_config({"CODING_AGENT_API_KEY": secret})
        self.assertNotIn(secret, repr(config))

        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.addFilter(SecretRedactingFilter((config.api_key,)))
        logger = logging.getLogger("redaction-test")
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.info("credential=%s", secret)
        self.assertNotIn(secret, output.getvalue())
        self.assertIn("[REDACTED]", output.getvalue())

    def test_empty_registries_report_missing_entries(self) -> None:
        with self.assertRaises(ProviderNotFoundError):
            ProviderRegistry().get("missing")
        with self.assertRaises(ToolNotFoundError):
            ToolRegistry().get("missing")

    def test_default_permissions_are_read_only(self) -> None:
        policy = ReadOnlyPermissionPolicy()
        self.assertTrue(policy.decide(PermissionRequest(Operation.READ, "README.md")).allowed)
        self.assertFalse(policy.decide(PermissionRequest(Operation.WRITE, "README.md")).allowed)

    def test_json_session_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JsonSessionStore(Path(directory))
            session = Session.create()
            session.add_message("user", "hello")
            store.save(session)
            restored = store.load(session.id)
        self.assertEqual(restored.id, session.id)
        self.assertEqual(restored.messages[0].content, "hello")


if __name__ == "__main__":
    unittest.main()
