# Independent Python Coding Agent

This repository contains the engineering foundation for an independent coding
agent written in Python. It is under active development: the current release
defines architecture, configuration, persistence, permissions, command-line
boundaries, read-only workspace inspection, permission-gated transactional file
editing, controlled command execution, and a streaming terminal interface. An
OpenAI-compatible provider and bounded core agent loop are available;
unrestricted file overwrites and unreviewed risky commands are intentionally not
built in.

The project is not affiliated with Anthropic and does not claim compatibility
with Claude Code.

## Requirements

- Python 3.11 or newer
- `httpx` (installed automatically) for asynchronous HTTP and streaming

## Install and run

Create a virtual environment, then install the project in editable mode:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Inspect the command surface and start a task:

```bash
.venv/bin/agent --help
.venv/bin/agent run "inspect the project and run its tests"
```

`coding-agent` remains an installation alias, but new examples use `agent`.

The daily command surface is:

```bash
agent chat
agent run "任务"
agent resume [session-id]
agent sessions
agent config
agent diagnostics
```

`chat` accepts normal one-line input, a trailing `\` for continuation, or
`/multi` followed by lines and a final `.`. The first Ctrl+C cancels the active
model/tool operation; a second requests exit. Redirected stdin/stdout works
without color. Add `--json` before or after a runtime command for JSON Lines
events suitable for scripts.

## Diagnostics

Diagnostics are read-only and do not enter the core agent flow:

```bash
agent diagnostics
agent diagnostics manifest
agent diagnostics subsystems --limit 16
```

The source checkout also supports:

```bash
python3 -m src.main diagnostics summary
```

For existing users of the original workspace, `summary`, `manifest`, and
`subsystems` remain temporary compatibility aliases.

## Architecture

The package remains named `src` temporarily so existing imports and
`python -m src.main` continue to work. New integrations should use the
`agent` console command.

```text
src/
├── cli.py              # argument parsing and command dispatch
├── config.py           # environment-based configuration
├── providers.py        # provider protocol, domain errors, and registry
├── openai_provider.py  # OpenAI-compatible HTTP and streaming adapter
├── agent.py            # bounded async/sync agent loop and streaming events
├── context.py          # budgeted context selection, summaries, and file-read versions
├── tools.py            # tool protocol, schemas, registry, and bounded workspace tools
├── git_runtime.py      # task Git baselines, incremental attribution, and reviewed Git writes
├── quality.py          # verification tracking, test suggestions, and completion checks
├── shell_tools.py      # subprocess execution, streaming, and risk classification
├── terminal_ui.py      # human terminal and style-free JSON Lines rendering
├── sessions.py         # JSON session persistence
├── permissions.py      # read-only and callback-driven interactive permission policies
├── diagnostics.py      # manifest and summary reporting
├── logging_config.py   # logging setup and secret redaction
└── main.py             # executable module and console-script target
```

Dependencies are passed to agent instances explicitly. There are no global
provider, tool, session, or permission singletons.

## Configuration

Configuration is loaded from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODING_AGENT_PROVIDER` | `openai-compatible` | Provider adapter selection |
| `CODING_AGENT_MODEL` | unset | Model identifier |
| `CODING_AGENT_API_KEY` | unset | Provider credential |
| `CODING_AGENT_BASE_URL` | `https://api.openai.com/v1` | Compatible API root |
| `CODING_AGENT_TIMEOUT` | `60` | Request timeout in seconds |
| `CODING_AGENT_MAX_TOKENS` | `4096` | Completion token limit |
| `CODING_AGENT_MAX_CONTEXT_TOKENS` | `32768` | Context budget before compression |
| `CODING_AGENT_SESSIONS_DIR` | OS user data directory | Local session storage |
| `CODING_AGENT_LOG_LEVEL` | `INFO` | Application log level |
| `CODING_AGENT_SHELL_ALLOWLIST` | unset | Comma-separated argv command prefixes |

Values are resolved in this order: CLI arguments, environment variables, the
user configuration file, then defaults. The default user file is
`~/.config/coding-agent/config.toml`; it uses the same lower-case names:

```toml
provider = "openai-compatible"
base_url = "https://example.invalid/v1"
model = "your-model-name"
timeout = 60
max_tokens = 4096
max_context_tokens = 32768
shell_allowlist = ["ruff check", "pytest"]
```

Prefer `CODING_AGENT_API_KEY` for credentials. The project never writes API
keys to configuration files or sessions, excludes them from configuration
representations, redacts them from application logs, and omits upstream response
bodies from domain exceptions. `--api-key` exists for explicit CLI precedence,
but environment variables avoid shell-history and process-list exposure.

## Provider API

`ModelProvider` is independent of the CLI and agent loop. It exposes one-shot
and asynchronous streaming completion methods over provider-neutral `Message`,
`ToolCall`, `ModelResponse`, `ModelStreamChunk`, and `Usage` models. The first
adapter targets OpenAI-compatible Chat Completions APIs and supports text,
tool calls, token usage, stream cancellation, and safe domain errors for timeout,
authentication, rate limiting, server failures, connection failures, and invalid
responses. Future Anthropic or local adapters can implement the same protocol
without changing callers.

## Agent runtime

`AgentLoop` receives provider, tool, session, and permission dependencies
explicitly. For each task it constructs a workspace-aware system prompt, calls
the selected provider, validates and executes tool calls in order, persists each
assistant/tool step, and continues until the model returns final text.

The runtime offers `await run(...)`, `run_sync(...)`, and an asynchronous
`run_stream(...)` event iterator. `AgentLimits` bounds model turns, total tool
calls, and wall-clock duration; `CancellationToken` interrupts in-flight model
or tool awaits. Identical repeated calls are blocked, while validation failures,
permission denials, and tool exceptions become tool results so the model can
correct its approach. Tool input objects are closed by default and validated
against the strict JSON-schema subset declared by each tool.

## Context management

The first model request contains the user task, absolute workspace path, a
bounded names-only directory overview, and explicit project instruction files
such as `AGENTS.md`, `README.md`, and `CONTRIBUTING.md`. It does not preload
ordinary source-file contents; the model must inspect those through
`read_file` and the other workspace tools.

Successful reads record the workspace-relative path, exact line range, and a
SHA-256 content version in the session's context metadata. Overlapping ranges
are merged. Non-dry-run patch/create attempts invalidate affected entries, and
Shell tool runs conservatively invalidate all tracked reads because partial or
indirect filesystem effects cannot be inferred reliably from argv classification.
Invalidated historical `read_file` results remain intact in local session
history, but the Provider-facing view replaces their content with a stale-read
marker so the model must read the current version again.

Before each provider call, `ContextManager` estimates the request size,
including tool schemas. Near `max_context_tokens`, it retains the system
prompt, exact active user constraints, recent conversation units, and any
assistant tool call together with all of its tool results. Older process is
represented by a structured summary containing goals, modified files, key
decisions, failed attempts, and remaining work. Full history remains in local
session storage for recovery and audit; only the provider-facing view is
compressed. Compression logs contain counts and session IDs, never message,
file, or tool-output contents.

When a provider supplies token usage, those values are accumulated unchanged.
If it omits usage entirely, the runtime falls back to a UTF-8-size estimate for
that model turn.

## Read-only workspace tools

The built-in registry exposes `list_files`, `read_file`, `search_text`,
`git_status`, `git_diff`, and `git_diff_check`. Every tool returns `success`,
`content`, `error`, and `metadata`, and publishes a closed JSON Schema for model
tool calling.

File paths are resolved before use and must remain inside the active workspace;
path traversal, absolute-path escape, and symlink escape are rejected. Common
large directories such as `.git`, `node_modules`, `venv`, `dist`, and `build`
are skipped. Text reads are bounded and reject binary files. Search prefers
`rg` and uses a bounded Python scanner when ripgrep is unavailable. Git tools
invoke only fixed read-only status, diff, and whitespace-check operations.

## Git awareness and completion quality

At the beginning of every agent task, the runtime records the current commit,
porcelain status, staged and unstaged baseline diff, and byte snapshots of
already-dirty paths. At completion it compares the final files with that exact
starting worktree state. Unchanged user edits are excluded from the task delta;
paths changed on top of pre-existing edits are listed separately as overlapping
and are never described as agent-only. No cleanup, restoration, staging,
commit, or push is performed as part of baseline tracking.

The full workspace registry also provides `git_add` and `git_commit`. Both are
`ask` operations and show the exact command, absolute cwd, risk reason, and diff
preview. `git_add` requires explicit file paths and refuses broad pathspecs,
paths dirty at task start, and paths not recorded as modified by the current
agent run. `git_commit` refuses staged baseline/user files and only creates a
local commit after approval; it never pushes. Git hooks may run during an
approved commit.

After a file edit, the runtime suggests focused verification based on changed
file types and project markers (for example Python, JavaScript, Rust, or Go
tests), plus `git_diff_check`. Suggestions are not executed automatically:
commands still pass through `run_shell` and its permission policy. Before a
normal completion, the core runtime checks expected file changes, recorded test
or diff-check results, failures, and unexplained new files. Human and JSON
completion reports distinguish completed work, incomplete work, test results,
and risks/follow-up, and expose pre-existing, agent-only, and overlapping Git
paths separately.

## File modification tools

`workspace_tool_registry()` adds `apply_patch` and `create_file` to the read-only
set, along with controlled `run_shell` execution. `apply_patch` accepts a strict
unified diff for existing UTF-8 text files;
all declared hunks and context must match their exact locations. `create_file`
uses exclusive creation and never overwrites an existing path. Both support
`dry_run` and return the resulting unified diff.

Edits are confined to the resolved workspace and reject path traversal,
symlink escape, `.git` and common user configuration directories, environment
credential files, and common private-key formats. Multi-file patches are fully
prepared before commit, use bounded file/byte counts, detect concurrent changes,
write atomically, and restore saved original bytes if a later write fails.

Each tool declares its permission operation. `ReadOnlyPermissionPolicy` rejects
the editing tools, while `InteractivePermissionPolicy` delegates write approval
to an injected UI callback so the Agent remains independent of terminal or GUI
frameworks.

## Terminal interface

The CLI is an application-composition layer over `AgentLoop`; it does not
implement a second model or tool loop. Provider and session factories are
injectable for tests and alternate frontends. Human mode streams model text,
shows concise tool arguments, bounds visible Shell output, presents file diffs
before approval, and ends each turn with modified files, recognized test-command
results, and per-turn token usage. It displays the active model, absolute
workspace, and session ID before execution.

JSON mode emits one JSON object per line for session context, text deltas, tool
calls, stdout/stderr chunks, tool results, completion, errors, and the final turn
summary. It never uses ANSI styling or interactive approval; consequently,
commands classified as `ask` are refused in JSON/non-interactive mode unless a
safe configured policy has already classified them as `allow`.

## Controlled command execution

`run_shell` uses an argv array by default and always starts commands with the
resolved workspace as its exact `cwd`. A separate `shell_command` input exists
for pipelines, redirects, variable expansion, and other shell syntax; this mode
is explicitly marked as higher risk and always requires approval. The tool
streams stdout and stderr independently through `AgentEventKind.TOOL_OUTPUT`,
then returns bounded output, exit code, timeout state, truncation state, command,
cwd, and classification metadata. Non-zero exit codes are normal command results
so the model can inspect and react to them. Timeouts and cancellation terminate
the process group.

Commands are classified as:

- `allow`: recognized read-only inspection and common test commands;
- `ask`: file writes, dependency installation, Git mutations, network access,
  unknown commands, and all explicit shell-syntax commands;
- `deny`: patterns such as `rm -rf`, `git reset --hard`, forced Git clean,
  any `git push`, `sudo`, download-to-shell pipelines, common credential directories, explicit
  workspace-external writes, and workspace-external cwd requests.

Every `ask` request carries the exact rendered command, absolute cwd, and risk
reason. `InteractivePermissionPolicy` passes this complete request to its UI
callback. `NonInteractivePermissionPolicy` rejects `ask` by default. Configure
argv prefix allowlisting through `shell_allowlist`, the
`CODING_AGENT_SHELL_ALLOWLIST` environment variable, or repeatable
`--shell-allow`; hard denials, shell mode, dynamic interpreter code, protected
paths, and cwd restrictions cannot be downgraded by this allowlist. Application
composition should pass `config.shell_allowlist` to
`workspace_tool_registry(shell_allowlist=...)`.

### Sandbox runtime

Every `run_shell` command, including recognized test and build commands, starts
through the `SandboxRuntime` boundary after command classification and
permission approval. The interface provides `prepare`, `execute`, `read_file`,
`write_file`, `collect_artifacts`, `cleanup`, and `health_check`.

`LocalSandboxRuntime` is the compatibility default. It preserves the existing
cwd, path, symlink, dangerous-command, minimal-environment, streaming, output,
and timeout behavior and explicitly reports
`security_level=APPLICATION_ONLY`. It is not kernel isolation and cannot
OS-enforce its network, CPU, memory, or process policy fields.

`ContainerSandboxRuntime` is an experimental Docker CLI implementation. It
fails closed when Docker, its daemon, or the selected local image is unavailable
and never silently falls back to Local. Its default policy uses a read-only
workspace, no network, UID/GID `65532`, a read-only root filesystem, dropped
capabilities, `no-new-privileges`, CPU/memory/PID limits, and an allowlisted
environment without API keys. It never mounts the Docker socket. Callers must
explicitly choose whether a task needs a writable workspace:

```python
from src.sandbox import ContainerSandboxRuntime, SandboxPolicy
from src.tools import workspace_tool_registry

registry = workspace_tool_registry(
    sandbox_runtime_factory=lambda workspace: ContainerSandboxRuntime(
        workspace,
        policy=SandboxPolicy.container_default(workspace_writable=False),
        image="python:3.11-alpine",
    )
)
```

Container images are not pulled automatically. Output written to `/artifacts`
is copied to a unique `artifacts/<run-id>/` directory in the workspace before
the container and its staging directory are removed.

PermissionManager remains outside the sandbox. A sandbox constrains an
already-approved operation; it does not approve commands and cannot turn a
denied command into an allowed one.

Filesystem tools remain outside the container for compatibility. Their current
host-side workspace confinement, symlink rejection, atomic writes, concurrent
change checks, Git attribution, previews, and approval flow are stronger and
more reviewable than routing those operations through a short-lived container.
The runtime read/write methods establish a future adapter boundary, but are not
used to bypass these controls. Containerized Shell commands still see only the
workspace paths selected by `allowed_paths`; an empty value means the whole
workspace.

## Local sessions

Sessions use versioned JSON and are stored under the operating system's user
data directory by default. On macOS this is
`~/Library/Application Support/coding-agent/sessions`; on Linux it follows
`$XDG_DATA_HOME` or `~/.local/share`. A configured session directory inside any
Git workspace is rejected.

```bash
agent sessions
agent resume                                 # latest for the current workspace
agent resume SESSION_ID

# Temporary management compatibility commands:
agent sessions new --workspace /absolute/path/to/project
agent sessions delete SESSION_ID
```

Each session records `schema_version`, its ID, absolute workspace, timestamps,
provider/model, typed message history, cumulative token usage, and an optional
Run checkpoint. Version 3 checkpoints contain the explicit Run state, task
type, current structured plan and step, repair count, last verification result,
pending approval summary, and confirmed/uncertain tool-call IDs. Version 1 and
2 files still load with safe empty checkpoint defaults. Writes use a
same-directory temporary file, filesystem flush, and atomic replacement.
On POSIX systems the directory is forced to mode `0700` and each session file
to `0600`; symlinked session files are never followed. The configured API key is
redacted if it appears in persisted message or tool-result text. If a process is
interrupted after a tool request was saved, resume closes that pending request
with an explicit interrupted result before sending new model input; it never
silently replays the tool.
Corrupted files are reported individually and skipped while valid sessions
remain available. The fixed metadata schema has no field for API keys or
environment snapshots. Bounded tool results, including Shell output needed by
the model, are stored as ordinary tool messages rather than session metadata.

## Harness lifecycle

Every CLI turn is managed by the lifecycle in `src.harness`. Simple
informational work goes directly from preparation to execution; complex
modifications receive a schema-validated ordered plan. Model text is only a
candidate result: all task types enter verification, and modification or
execution work can reach completion only after the task-aware gate accepts
recorded evidence. Repair attempts and per-step retries are both bounded at two
by default. Approval prompts are persisted as `WAITING_APPROVAL`, while
cancellation, timeout, exhausted limits, and unsafe recovery end in explicit
terminal states.

The full threat/control/test mapping is maintained in
[`SECURITY.md`](SECURITY.md).

## Tests

Run the complete test suite with the standard library:

```bash
python3 -m unittest discover -s tests -v
```

The suite covers legacy diagnostics, module boundaries, layered configuration,
CLI behavior, provider text/tool/stream responses, stream interruption, domain
errors, message round trips, tool-result linkage, atomic session persistence,
corruption isolation, single/multiple/failing tool loops, streaming, limits,
cancellation, repeated-call protection, registries, permissions, and credential
redaction, plus workspace path containment, bounded text reads, binary rejection,
search fallback, output truncation, Git/non-Git behavior, exact patch conflicts,
transaction rollback, dry runs, protected paths, exclusive file creation, and
permission-gated writes, plus command classification, approval details,
stdout/stderr streaming, exit codes, timeout, cancellation, and output limits.
Context tests cover bounded small/large workspace overviews, exact active
constraints, atomic tool-call/result retention, structured compression,
read-range/version tracking, invalidation, persisted schema migration, and
provider-reported versus estimated usage.
Provider tests also cover bounded HTTP/SSE response sizes, forged tool-result
rejection, and bounded HTTP 429 retry with `Retry-After`.
CLI integration tests use an injected fake provider and also cover human/JSON
streaming, multiline input, diff approval, Ctrl+C state, redirected output, and
no-color rendering.
Git integration tests create repositories with pre-existing uncommitted user
changes and verify task-only attribution, overlap reporting, reviewed add/commit,
diff checks, push denial, and preservation of the original worktree changes.
Security end-to-end tests cover real read/patch/test execution, failed-patch
recovery, Shell timeout, refusal, interrupted-session resume, preservation of
existing dirty changes, and malicious tool-argument rejection.
