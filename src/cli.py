"""Command-line application composition and dispatch."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO

from .agent import (
    AgentCancelledError,
    AgentError,
    AgentEventKind,
    AgentLimits,
    AgentLoop,
    AgentRequest,
    AgentResult,
    CancellationToken,
)
from .config import AppConfig, ConfigError, load_config
from .context import ContextManager
from .diagnostics import render_summary
from .harness.events import EventType, JsonlEventStore
from .logging_config import configure_logging
from .models import Usage
from .permissions import InteractivePermissionPolicy, NonInteractivePermissionPolicy
from .port_manifest import PortManifest, build_port_manifest
from .providers import ModelProvider, ProviderError, ProviderRegistry
from .query_engine import QueryEnginePort
from .sessions import JsonSessionStore, Session, SessionError, SessionStore
from .terminal_ui import (
    EventRenderer,
    HumanRenderer,
    InterruptAction,
    InterruptController,
    JsonRenderer,
    SignalBinding,
    TurnSummary,
    read_prompt,
)
from .tools import workspace_tool_registry

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_DIAGNOSTICS = ("summary", "manifest", "subsystems")


@dataclass
class CliDependencies:
    """Injectable process edges used by CLI integration tests and alternate UIs."""

    provider_factory: Callable[[AppConfig], ModelProvider] | None = None
    session_store_factory: Callable[[Path], SessionStore] | None = None
    stdin: IO[str] | None = None
    stdout: IO[str] | None = None
    stderr: IO[str] | None = None
    environ: Mapping[str, str] | None = None

    def streams(self) -> tuple[IO[str], IO[str], IO[str]]:
        return (
            self.stdin or sys.stdin,
            self.stdout or sys.stdout,
            self.stderr or sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Independent streaming terminal coding agent",
    )
    parser.add_argument("--config-file", type=Path, help="path to a user TOML configuration file")
    parser.add_argument("--log-level", choices=_LOG_LEVELS, help="override the configured log level")
    parser.add_argument("--provider", help="model provider adapter")
    parser.add_argument("--api-key", help="provider API key; environment variables are safer")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--model", help="model identifier")
    parser.add_argument("--timeout", type=float, help="provider request timeout in seconds")
    parser.add_argument("--max-tokens", type=int, help="maximum completion tokens")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="maximum model context budget before structured compression",
    )
    parser.add_argument(
        "--shell-allow",
        action="append",
        help="allow an exact argv command prefix (repeatable; cannot override hard denials)",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="agent workspace")
    parser.add_argument("--json", action="store_true", help="emit JSON Lines without terminal styling")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI terminal colors")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="start an interactive multi-turn session")
    _add_runtime_output_options(chat_parser)

    run_parser = subparsers.add_parser("run", help="run one task and exit")
    run_parser.add_argument("prompt", nargs="+", help="task for the agent")
    _add_runtime_output_options(run_parser)

    resume_parser = subparsers.add_parser("resume", help="resume a session by ID or latest")
    resume_parser.add_argument("session_id", nargs="?")
    resume_parser.add_argument("--prompt", help="run one task after resuming instead of opening chat")
    _add_runtime_output_options(resume_parser)

    sessions_parser = subparsers.add_parser("sessions", help="list or manage local sessions")
    _add_output_options(sessions_parser)
    session_commands = sessions_parser.add_subparsers(dest="session_command", required=False)
    new_session = session_commands.add_parser("new", help="create an empty local session")
    new_session.add_argument("--workspace", type=Path, default=Path.cwd())
    session_commands.add_parser("list", help="list valid local sessions")
    legacy_resume = session_commands.add_parser("resume", help="show a resumable session")
    legacy_resume.add_argument("session_id", nargs="?")
    delete_session = session_commands.add_parser("delete", help="delete a session by ID")
    delete_session.add_argument("session_id")

    trace_parser = subparsers.add_parser("trace", help="inspect a local Run event trace")
    trace_parser.add_argument("run_id", help="Run identifier from a session checkpoint")
    trace_parser.add_argument("--event-type", choices=tuple(item.value for item in EventType))
    trace_parser.add_argument("--failed", action="store_true", help="show failed events only")
    trace_parser.add_argument("--limit", type=_non_negative_int)
    _add_output_options(trace_parser)

    config_parser = subparsers.add_parser("config", help="show effective non-secret configuration")
    _add_output_options(config_parser)

    diagnostics_parser = subparsers.add_parser(
        "diagnostics", help="inspect the project without starting the agent"
    )
    diagnostics_parser.add_argument("diagnostic", nargs="?", choices=_DIAGNOSTICS, default="summary")
    diagnostics_parser.add_argument("--limit", type=_non_negative_int, default=16)
    _add_output_options(diagnostics_parser)

    # Temporary compatibility aliases remain diagnostics-only.
    subparsers.add_parser("summary", help="legacy alias for diagnostics summary")
    subparsers.add_parser("manifest", help="legacy alias for diagnostics manifest")
    legacy_subsystems = subparsers.add_parser("subsystems", help="legacy diagnostics alias")
    legacy_subsystems.add_argument("--limit", type=_non_negative_int, default=16)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    dependencies: CliDependencies | None = None,
) -> int:
    deps = dependencies or CliDependencies()
    stdin, stdout, stderr = deps.streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(
            deps.environ,
            cli_values={
                "log_level": args.log_level,
                "provider": args.provider,
                "api_key": args.api_key,
                "base_url": args.base_url,
                "model": args.model,
                "timeout": args.timeout,
                "max_tokens": args.max_tokens,
                "max_context_tokens": args.max_context_tokens,
                "shell_allowlist": args.shell_allow,
            },
            config_file=args.config_file,
        )
    except ConfigError as exc:
        parser.error(str(exc))

    configure_logging(config.log_level, secrets=(config.api_key,))
    renderer: EventRenderer
    if args.json:
        renderer = JsonRenderer(stdout)
    else:
        color = False if args.no_color else None
        renderer = HumanRenderer(stdout, stderr, stdin, color=color)

    if args.command == "config":
        renderer.emit_record(_safe_config_record(config))
        return 0
    if args.command == "diagnostics":
        return _run_diagnostics(args.diagnostic, args.limit, renderer, json_mode=args.json)
    if args.command in {"summary", "manifest", "subsystems"}:
        return _run_legacy_diagnostics(args.command, getattr(args, "limit", 16), stdout)
    if args.command == "sessions":
        try:
            return _run_sessions(args, config, deps, renderer, json_mode=args.json)
        except (SessionError, ValueError) as exc:
            renderer.render_error(str(exc), error_type="session_error")
            return 1
    if args.command == "trace":
        try:
            return _run_trace(args, config, renderer, json_mode=args.json)
        except ValueError as exc:
            renderer.render_error(str(exc), error_type="trace_error")
            return 1

    controller = InterruptController()

    def handle_interrupt(action: InterruptAction) -> None:
        if action is InterruptAction.CANCEL:
            renderer.render_cancelled()
        else:
            if isinstance(renderer, HumanRenderer):
                renderer.render_interrupt_exit()
            raise KeyboardInterrupt

    try:
        with SignalBinding(controller, handle_interrupt):
            return asyncio.run(
                _run_agent_command(args, config, deps, renderer, controller, stdin)
            )
    except KeyboardInterrupt:
        return 130
    except (AgentError, ProviderError, SessionError, ConfigError, ValueError) as exc:
        renderer.render_error(str(exc), error_type=type(exc).__name__)
        return 1


async def _run_agent_command(
    args: argparse.Namespace,
    config: AppConfig,
    deps: CliDependencies,
    renderer: EventRenderer,
    controller: InterruptController,
    stdin: IO[str],
) -> int:
    workspace = args.workspace.expanduser().resolve()
    store = (
        deps.session_store_factory(config.sessions_dir)
        if deps.session_store_factory is not None
        else JsonSessionStore(config.sessions_dir, secrets=(config.api_key,))
    )

    resumed: Session | None = None
    if args.command == "resume":
        resumed = (
            store.load(args.session_id)
            if args.session_id
            else store.load_latest(workspace=workspace)
        )
        workspace = resumed.workspace
        config = replace(config, provider=resumed.provider, model=resumed.model)

    provider_factory = deps.provider_factory or _default_provider_factory
    provider = provider_factory(config)
    if provider.name != config.provider:
        raise ConfigError(
            f"configured provider {config.provider!r} does not match factory provider {provider.name!r}"
        )
    if config.model is None:
        raise ConfigError("model is required for chat, run, and resume")

    session = resumed or store.create(
        workspace=workspace,
        provider=config.provider,
        model=config.model,
    )
    renderer.show_context(model=config.model, workspace=workspace, session_id=session.id)
    interactive = (
        not args.json
        and bool(getattr(stdin, "isatty", lambda: False)())
        and isinstance(renderer, HumanRenderer)
    )
    permissions = (
        InteractivePermissionPolicy(renderer.approve)
        if interactive and isinstance(renderer, HumanRenderer)
        else NonInteractivePermissionPolicy()
    )
    loop = AgentLoop(
        ProviderRegistry((provider,)),
        workspace_tool_registry(shell_allowlist=config.shell_allowlist),
        store,
        permissions,
        limits=AgentLimits(timeout_seconds=max(config.timeout * 5, 60.0)),
        context=ContextManager(max_tokens=config.max_context_tokens),
        event_store=JsonlEventStore(
            config.sessions_dir.parent / "traces",
            secrets=(config.api_key,),
        ),
    )

    try:
        if args.command == "run":
            prompt = " ".join(args.prompt).strip()
            completed = await _run_turn(
                loop,
                prompt,
                session,
                config,
                renderer,
                controller,
            )
            return 0 if completed else 130
        if args.command == "resume" and args.prompt:
            completed = await _run_turn(
                loop,
                args.prompt,
                session,
                config,
                renderer,
                controller,
            )
            return 0 if completed else 130
        return await _chat_loop(
            loop,
            session,
            config,
            renderer,
            controller,
            stdin,
            interactive=interactive,
        )
    finally:
        await provider.aclose()


async def _run_turn(
    loop: AgentLoop,
    prompt: str,
    session: Session,
    config: AppConfig,
    renderer: EventRenderer,
    controller: InterruptController,
) -> bool:
    token = CancellationToken()
    controller.bind(token)
    summary = TurnSummary()
    completed: AgentResult | None = None
    starting_usage = session.usage
    try:
        async for event in loop.run_stream(
            AgentRequest(
                prompt=prompt,
                provider=config.provider,
                model=config.model or session.model,
                workspace=session.workspace,
                session_id=session.id,
            ),
            cancellation=token,
        ):
            summary.observe(event)
            renderer.render_event(event)
            if event.kind is AgentEventKind.COMPLETED:
                completed = event.result
    except AgentCancelledError:
        return False
    finally:
        controller.unbind()
    if completed is None:
        raise AgentError("agent ended without a completed result")
    turn_usage = Usage(
        prompt_tokens=completed.usage.prompt_tokens - starting_usage.prompt_tokens,
        completion_tokens=(
            completed.usage.completion_tokens - starting_usage.completion_tokens
        ),
        total_tokens=completed.usage.total_tokens - starting_usage.total_tokens,
    )
    renderer.render_completion(replace(completed, usage=turn_usage), summary)
    return True


async def _chat_loop(
    loop: AgentLoop,
    session: Session,
    config: AppConfig,
    renderer: EventRenderer,
    controller: InterruptController,
    stdin: IO[str],
    *,
    interactive: bool,
) -> int:
    if isinstance(renderer, HumanRenderer) and interactive:
        renderer.show_chat_hint()
    consumed_redirect = False
    while not controller.exit_requested:
        if isinstance(renderer, HumanRenderer):
            try:
                prompt = read_prompt(stdin, renderer)
            except KeyboardInterrupt:
                return 130
        else:
            if consumed_redirect:
                break
            prompt = stdin.read().strip() or None
            consumed_redirect = True
        if prompt is None:
            break
        completed = await _run_turn(loop, prompt, session, config, renderer, controller)
        if not completed and not interactive:
            return 130
    return 130 if controller.exit_requested else 0


def _run_sessions(
    args: argparse.Namespace,
    config: AppConfig,
    deps: CliDependencies,
    renderer: EventRenderer,
    *,
    json_mode: bool,
) -> int:
    store = (
        deps.session_store_factory(config.sessions_dir)
        if deps.session_store_factory is not None
        else JsonSessionStore(config.sessions_dir, secrets=(config.api_key,))
    )
    command = args.session_command or "list"
    if command == "new":
        session = store.create(
            workspace=args.workspace,
            provider=config.provider,
            model=config.model or "unconfigured",
        )
        _emit_record(renderer, {"session_id": session.id} if json_mode else session.id)
        return 0
    if command == "list":
        result = store.list_sessions()
        if json_mode:
            _emit_record(
                renderer,
                {
                    "sessions": [
                        {
                            "session_id": item.session_id,
                            "updated_at": item.updated_at.isoformat(),
                            "provider": item.provider,
                            "model": item.model,
                            "message_count": item.message_count,
                            "workspace": str(item.workspace),
                            "total_tokens": item.usage.total_tokens,
                        }
                        for item in result.sessions
                    ],
                    "errors": [error.message for error in result.errors],
                },
            )
        else:
            for item in result.sessions:
                _emit_record(
                    renderer,
                    f"{item.session_id}\t{item.updated_at.isoformat()}\t"
                    f"{item.provider}/{item.model}\t{item.message_count}\t{item.workspace}",
                )
            for error in result.errors:
                renderer.render_error(error.message, error_type="warning")
        return 0
    if command == "resume":
        session = store.load(args.session_id) if args.session_id else store.load_latest(workspace=Path.cwd())
        _emit_record(renderer, _session_record(session) if json_mode else _session_text(session))
        return 0
    if command == "delete":
        store.delete(args.session_id)
        _emit_record(
            renderer,
            {"deleted_session_id": args.session_id}
            if json_mode
            else f"Deleted session {args.session_id}",
        )
        return 0
    raise ValueError(f"unknown session command: {command}")


def _run_trace(
    args: argparse.Namespace,
    config: AppConfig,
    renderer: EventRenderer,
    *,
    json_mode: bool,
) -> int:
    store = JsonlEventStore(
        config.sessions_dir.parent / "traces",
        secrets=(config.api_key,),
    )
    events = store.query(
        args.run_id,
        event_type=EventType(args.event_type) if args.event_type else None,
        success=False if args.failed else None,
        limit=args.limit,
    )
    if json_mode:
        for event in events:
            _emit_record(renderer, event.to_dict())
        return 0
    if not events:
        _emit_record(renderer, f"No trace events found for Run {args.run_id}")
        return 0
    for event in events:
        duration = (
            f"\t{event.duration_ms:.3f}ms"
            if event.duration_ms is not None
            else ""
        )
        outcome = (
            f"\t{'ok' if event.success else 'failed'}"
            if event.success is not None
            else ""
        )
        links = "\t".join(
            value
            for value in (event.step_id, event.tool_call_id)
            if value is not None
        )
        suffix = f"\t{links}" if links else ""
        _emit_record(
            renderer,
            f"{event.timestamp.isoformat()}\t{event.event_type.value}"
            f"{outcome}{duration}{suffix}",
        )
    return 0


def _run_diagnostics(
    command: str,
    limit: int,
    renderer: EventRenderer,
    *,
    json_mode: bool,
) -> int:
    manifest = build_port_manifest()
    if json_mode:
        record = _manifest_record(manifest)
        record["diagnostic"] = command
        if command == "subsystems":
            record["modules"] = record["modules"][:limit]  # type: ignore[index]
        _emit_record(renderer, record)
        return 0
    if command == "summary":
        _emit_record(renderer, render_summary(manifest))
    elif command == "manifest":
        _emit_record(renderer, manifest.to_markdown())
    elif command == "subsystems":
        for subsystem in manifest.top_level_modules[:limit]:
            _emit_record(renderer, f"{subsystem.name}\t{subsystem.file_count}\t{subsystem.notes}")
    else:
        raise ValueError(f"unknown diagnostics command: {command}")
    return 0


def _run_legacy_diagnostics(command: str, limit: int, stdout: IO[str]) -> int:
    if command == "summary":
        stdout.write(QueryEnginePort.from_workspace().render_summary() + "\n")
        return 0
    manifest = build_port_manifest()
    if command == "manifest":
        stdout.write(manifest.to_markdown() + "\n")
    else:
        for subsystem in manifest.top_level_modules[:limit]:
            stdout.write(f"{subsystem.name}\t{subsystem.file_count}\t{subsystem.notes}\n")
    return 0


def _default_provider_factory(config: AppConfig) -> ModelProvider:
    from .openai_provider import OpenAICompatibleConfig, OpenAICompatibleProvider

    if config.provider != "openai-compatible":
        raise ConfigError(f"no CLI provider factory is configured for {config.provider!r}")
    return OpenAICompatibleProvider(OpenAICompatibleConfig.from_app_config(config))


def _safe_config_record(config: AppConfig) -> dict[str, object]:
    return {
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "timeout": config.timeout,
        "max_tokens": config.max_tokens,
        "max_context_tokens": config.max_context_tokens,
        "sessions_dir": str(config.sessions_dir),
        "log_level": config.log_level,
        "shell_allowlist": list(config.shell_allowlist),
        "api_key_configured": config.api_key is not None,
    }


def _session_record(session: Session) -> dict[str, object]:
    return {
        "session_id": session.id,
        "workspace": str(session.workspace),
        "provider": session.provider,
        "model": session.model,
        "messages": len(session.messages),
        "total_tokens": session.usage.total_tokens,
    }


def _session_text(session: Session) -> str:
    return (
        f"Session: {session.id}\nWorkspace: {session.workspace}\n"
        f"Provider: {session.provider}\nModel: {session.model}\n"
        f"Messages: {len(session.messages)}\nTokens: {session.usage.total_tokens}"
    )


def _manifest_record(manifest: PortManifest) -> dict[str, object]:
    return {
        "source_root": str(manifest.src_root),
        "total_python_files": manifest.total_python_files,
        "modules": [
            {
                "name": module.name,
                "path": module.path,
                "file_count": module.file_count,
                "notes": module.notes,
            }
            for module in manifest.top_level_modules
        ],
    }


def _emit_record(renderer: EventRenderer, record: object) -> None:
    renderer.emit_record(record)


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)


def _add_runtime_output_options(parser: argparse.ArgumentParser) -> None:
    _add_output_options(parser)
    parser.add_argument("--workspace", type=Path, default=argparse.SUPPRESS)
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=argparse.SUPPRESS,
        help="override the context compression budget",
    )


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed
