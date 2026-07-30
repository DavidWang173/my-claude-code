"""Read-only diagnostics kept separate from the agent execution path."""

from __future__ import annotations

from .models import PortingBacklog, PortingModule
from .port_manifest import PortManifest, build_port_manifest

ARCHITECTURE_BOUNDARIES = (
    "cli",
    "config",
    "providers",
    "agent",
    "context",
    "tools",
    "sessions",
    "permissions",
)


def build_diagnostic_backlog() -> PortingBacklog:
    """Build compatibility metadata used by the legacy summary renderer."""

    modules = [
        PortingModule(
            name,
            f"Provide the {name} architecture boundary",
            f"src/{name}.py",
            "implemented",
        )
        for name in ARCHITECTURE_BOUNDARIES
    ]
    return PortingBacklog(title="Tool surface", modules=modules)


def render_summary(manifest: PortManifest | None = None) -> str:
    """Render current architecture diagnostics without invoking the agent."""

    current_manifest = manifest or build_port_manifest()
    lines = [
        "# Coding Agent Diagnostics Summary",
        "",
        current_manifest.to_markdown(),
        "",
        "Architecture boundaries:",
        *(f"- `{name}`" for name in ARCHITECTURE_BOUNDARIES),
        "",
        "Runtime status: the OpenAI-compatible provider and bounded agent loop are "
        "available with read-only inspection, transactional file edits, and controlled "
        "permission-gated command execution through a streaming terminal and JSON Lines CLI. "
        "Task-scoped Git baselines separate pre-existing user changes from agent-only work, "
        "and completion reports include verification and risk checks.",
    ]
    return "\n".join(lines)
