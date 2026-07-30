from __future__ import annotations

from .models import PortingBacklog, PortingModule

PORTED_COMMANDS = (
    PortingModule('main', 'Expose a Python CLI for manifest and backlog reporting', 'src/main.py', 'implemented'),
    PortingModule('summary', 'Render a Markdown overview of the current porting workspace', 'src/query_engine.py', 'implemented'),
    PortingModule('subsystems', 'List the current Python modules participating in the rewrite', 'src/port_manifest.py', 'implemented'),
)


def build_command_backlog() -> PortingBacklog:
    return PortingBacklog(title='Command surface', modules=list(PORTED_COMMANDS))
