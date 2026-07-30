from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .models import Subsystem

DEFAULT_SRC_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PortManifest:
    src_root: Path
    total_python_files: int
    top_level_modules: tuple[Subsystem, ...]

    def to_markdown(self) -> str:
        lines = [
            f'Source root: `{self.src_root}`',
            f'Total Python files: **{self.total_python_files}**',
            '',
            'Top-level Python modules:',
        ]
        for module in self.top_level_modules:
            noun = 'file' if module.file_count == 1 else 'files'
            lines.append(f'- `{module.name}` ({module.file_count} {noun}) — {module.notes}')
        return '\n'.join(lines)


def build_port_manifest(src_root: Path | None = None) -> PortManifest:
    root = src_root or DEFAULT_SRC_ROOT
    files = sorted(path for path in root.rglob('*.py') if path.is_file())
    counter = Counter(
        path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else path.name
        for path in files
    )
    notes = {
        '__init__.py': 'package export surface',
        'main.py': 'CLI entrypoint',
        'cli.py': 'command-line parsing and dispatch',
        'config.py': 'configuration loading',
        'context.py': 'budgeted model context, summaries, and file-read version tracking',
        'providers.py': 'model provider contracts and registry',
        'openai_provider.py': 'OpenAI-compatible model adapter',
        'agent.py': 'bounded agent loop and streaming runtime',
        'tools.py': 'tool protocol, validation, registry, and bounded workspace tools',
        'git_runtime.py': 'task Git baselines, incremental attribution, and reviewed Git writes',
        'quality.py': 'test suggestions, verification tracking, and completion quality reports',
        'shell_tools.py': 'controlled subprocess execution and command risk classification',
        'terminal_ui.py': 'terminal and JSON event rendering with interactive input',
        'sessions.py': 'session persistence',
        'permissions.py': 'permission policy contracts',
        'logging_config.py': 'logging configuration and secret redaction',
        'diagnostics.py': 'read-only architecture diagnostics',
        'port_manifest.py': 'workspace manifest generation',
        'query_engine.py': 'legacy diagnostics compatibility',
        'commands.py': 'legacy command diagnostics metadata',
        'models.py': 'provider-neutral messages, tool calls, responses, usage, and diagnostics models',
        'harness': 'Run lifecycle, planning, verification, repair, and checkpoint models',
    }
    modules = tuple(
        Subsystem(name=name, path=f'src/{name}', file_count=count, notes=notes.get(name, 'Python port support module'))
        for name, count in counter.most_common()
    )
    return PortManifest(src_root=root, total_python_files=len(files), top_level_modules=modules)
