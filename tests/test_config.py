from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import ConfigError, load_config
from src.openai_provider import OpenAICompatibleConfig


class LayeredConfigTests(unittest.TestCase):
    def test_cli_overrides_environment_file_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.toml"
            config_file.write_text(
                '\n'.join(
                    (
                        'api_key = "file-key"',
                        'base_url = "https://file.example/v1"',
                        'model = "file-model"',
                        'timeout = 15',
                        'max_tokens = 1000',
                        'max_context_tokens = 12000',
                        'shell_allowlist = ["ruff check", "pytest"]',
                    )
                ),
                encoding="utf-8",
            )
            config = load_config(
                {
                    "CODING_AGENT_API_KEY": "environment-key",
                    "CODING_AGENT_TIMEOUT": "25",
                    "CODING_AGENT_MAX_CONTEXT_TOKENS": "16000",
                },
                cli_values={
                    "api_key": "cli-key",
                    "model": "cli-model",
                    "max_context_tokens": 20000,
                },
                config_file=config_file,
            )

        self.assertEqual(config.api_key, "cli-key")
        self.assertEqual(config.model, "cli-model")
        self.assertEqual(config.timeout, 25.0)
        self.assertEqual(config.base_url, "https://file.example/v1")
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.max_context_tokens, 20000)
        self.assertEqual(config.provider, "openai-compatible")
        self.assertEqual(config.shell_allowlist, ("ruff check", "pytest"))
        provider_config = OpenAICompatibleConfig.from_app_config(config)
        self.assertEqual(provider_config.model, "cli-model")
        self.assertNotIn("cli-key", repr(provider_config))

    def test_invalid_api_key_type_does_not_expose_value(self) -> None:
        secret = "should-never-appear"
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.toml"
            config_file.write_text(f'api_key = {{ value = "{secret}" }}', encoding="utf-8")
            with self.assertRaises(ConfigError) as raised:
                load_config({}, config_file=config_file)
        self.assertNotIn(secret, str(raised.exception))

    def test_shell_allowlist_uses_standard_configuration_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.toml"
            config_file.write_text(
                'shell_allowlist = ["file command"]', encoding="utf-8"
            )
            environment_config = load_config(
                {"CODING_AGENT_SHELL_ALLOWLIST": "env one, env two"},
                config_file=config_file,
            )
            cli_config = load_config(
                {"CODING_AGENT_SHELL_ALLOWLIST": "env command"},
                cli_values={"shell_allowlist": ["cli command"]},
                config_file=config_file,
            )

        self.assertEqual(environment_config.shell_allowlist, ("env one", "env two"))
        self.assertEqual(cli_config.shell_allowlist, ("cli command",))


if __name__ == "__main__":
    unittest.main()
