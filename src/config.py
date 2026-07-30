"""Layered application configuration without global state."""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .sessions import default_session_directory

DEFAULT_CONFIG_FILE = Path.home() / ".config" / "coding-agent" / "config.toml"

_DEFAULTS: Mapping[str, object] = {
    "provider": "openai-compatible",
    "api_key": None,
    "base_url": "https://api.openai.com/v1",
    "model": None,
    "timeout": 60.0,
    "max_tokens": 4096,
    "max_context_tokens": 32768,
    "sessions_dir": str(default_session_directory()),
    "log_level": "INFO",
    "shell_allowlist": (),
}

_ENVIRONMENT_KEYS = {
    "provider": "CODING_AGENT_PROVIDER",
    "api_key": "CODING_AGENT_API_KEY",
    "base_url": "CODING_AGENT_BASE_URL",
    "model": "CODING_AGENT_MODEL",
    "timeout": "CODING_AGENT_TIMEOUT",
    "max_tokens": "CODING_AGENT_MAX_TOKENS",
    "max_context_tokens": "CODING_AGENT_MAX_CONTEXT_TOKENS",
    "sessions_dir": "CODING_AGENT_SESSIONS_DIR",
    "log_level": "CODING_AGENT_LOG_LEVEL",
    "shell_allowlist": "CODING_AGENT_SHELL_ALLOWLIST",
}


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime settings.

    ``api_key`` is excluded from dataclass representations. Callers must also
    avoid serialising this object into logs or session data.
    """

    provider: str = "openai-compatible"
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "https://api.openai.com/v1"
    model: str | None = None
    timeout: float = 60.0
    max_tokens: int = 4096
    max_context_tokens: int = 32768
    sessions_dir: Path = field(default_factory=default_session_directory)
    log_level: str = "INFO"
    shell_allowlist: tuple[str, ...] = ()


class ConfigError(ValueError):
    """Raised when configuration cannot be read or validated."""


def load_config(
    environ: Mapping[str, str] | None = None,
    *,
    cli_values: Mapping[str, object] | None = None,
    config_file: Path | None = None,
) -> AppConfig:
    """Load settings with CLI > environment > user file > defaults priority."""

    if environ is None:
        import os

        environ = os.environ
    cli_values = cli_values or {}

    selected_file = config_file or DEFAULT_CONFIG_FILE
    file_values = _read_config_file(selected_file, required=config_file is not None)

    merged: dict[str, object] = {}
    for key, default in _DEFAULTS.items():
        cli_value = cli_values.get(key)
        environment_value = environ.get(_ENVIRONMENT_KEYS[key])
        file_value = file_values.get(key)
        merged[key] = _first_defined(cli_value, environment_value, file_value, default)

    log_level = str(merged["log_level"]).upper()
    if log_level not in logging.getLevelNamesMapping():
        raise ConfigError("invalid value for log_level")

    provider = _required_text(merged["provider"], "provider")
    base_url = _required_text(merged["base_url"], "base_url").rstrip("/")
    model = _optional_text(merged["model"])
    api_key = _optional_text(merged["api_key"])
    timeout = _positive_float(merged["timeout"], "timeout")
    max_tokens = _positive_int(merged["max_tokens"], "max_tokens")
    max_context_tokens = _positive_int(
        merged["max_context_tokens"], "max_context_tokens"
    )
    sessions_dir = Path(_required_text(merged["sessions_dir"], "sessions_dir"))
    shell_allowlist = _command_allowlist(merged["shell_allowlist"])

    return AppConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        max_context_tokens=max_context_tokens,
        sessions_dir=sessions_dir,
        log_level=log_level,
        shell_allowlist=shell_allowlist,
    )


def _read_config_file(path: Path, *, required: bool) -> Mapping[str, object]:
    try:
        with path.expanduser().open("rb") as stream:
            payload = tomllib.load(stream)
    except FileNotFoundError:
        if required:
            raise ConfigError(f"configuration file not found: {path}") from None
        return {}
    except (OSError, tomllib.TOMLDecodeError):
        raise ConfigError(f"unable to read configuration file: {path}") from None
    return payload


def _first_defined(*values: object) -> object:
    return next((value for value in values if value is not None), None)


def _required_text(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError("text configuration value has an invalid type")
    return value.strip() or None


def _positive_float(value: object, key: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ConfigError(f"{key} must be a positive number") from None
    if parsed <= 0:
        raise ConfigError(f"{key} must be a positive number")
    return parsed


def _positive_int(value: object, key: str) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ConfigError(f"{key} must be a positive integer") from None
    if parsed <= 0:
        raise ConfigError(f"{key} must be a positive integer")
    return parsed


def _command_allowlist(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        entries = value.split(",")
    elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        entries = list(value)
    else:
        raise ConfigError("shell_allowlist must be a string list")
    return tuple(entry.strip() for entry in entries if entry.strip())
