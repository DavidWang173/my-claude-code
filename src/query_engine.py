from __future__ import annotations

from dataclasses import dataclass

from .commands import build_command_backlog
from .diagnostics import build_diagnostic_backlog
from .port_manifest import PortManifest, build_port_manifest


@dataclass
class QueryEnginePort:
    manifest: PortManifest

    @classmethod
    def from_workspace(cls) -> 'QueryEnginePort':
        return cls(manifest=build_port_manifest())

    def render_summary(self) -> str:
        command_backlog = build_command_backlog()
        tool_backlog = build_diagnostic_backlog()
        sections = [
            '# Python Porting Workspace Summary',
            '',
            '> Legacy compatibility diagnostics; this report is not part of the agent loop.',
            '',
            self.manifest.to_markdown(),
            '',
            f'{command_backlog.title}:',
            *command_backlog.summary_lines(),
            '',
            f'{tool_backlog.title}:',
            *tool_backlog.summary_lines(),
        ]
        return '\n'.join(sections)
