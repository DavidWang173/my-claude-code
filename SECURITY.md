# Security model and verification

This agent treats the model, workspace contents, tool arguments, command output,
provider responses, and persisted sessions as separate trust boundaries. The
core controls below are enforced in code and covered by automated tests.

| Risk | Enforced control | Test evidence |
| --- | --- | --- |
| Workspace path traversal | Reject `..`, resolve paths, require the final target to remain below the resolved workspace, and reject Shell operands that explicitly resolve outside it. | `test_workspace_tools`, `test_file_modification_tools`, `test_security_e2e` |
| Symbolic-link escape | Reject resolved targets outside the workspace, do not follow directory links while walking, revalidate canonical parents at commit, reject link-swapped write targets, and open session files with `O_NOFOLLOW` where available. | `test_path_escape_and_symlink_escape_are_rejected`, `test_parent_symlink_swap_is_rejected_at_atomic_commit_boundary`, `test_session_symlink_is_never_followed` |
| Shell injection | Use argv execution without a shell by default. Shell syntax is a distinct `ASK` operation. Parameter-level execution/write features such as `rg --pre`, Git external diff/config overrides, `sort -o`, and `sed` require approval. | `test_argv_metacharacters_are_literal_without_shell`, `test_read_only_classification_rejects_parameter_level_bypasses` |
| Prompt/tool parameter injection | Mark all workspace/tool material as untrusted data in the system prompt. Tool schemas are closed, JSON-only, size-bounded, and enforce nested types and unique arrays. | `test_invalid_arguments_become_tool_error_without_execution`, `test_malicious_tool_parameters_and_outside_reads_are_rejected` |
| API key leakage | Exclude credentials from config representations and subprocess environments, redact configured secrets from logs and persisted session strings, and omit upstream bodies from errors. | `test_config_repr_and_logging_do_not_expose_api_key`, `test_session_permissions_and_known_secret_redaction`, `test_domain_errors_are_clear_and_do_not_leak_api_key` |
| Session file permissions | Store outside Git worktrees, force directories to `0700` and files to `0600` on POSIX, use same-directory atomic replacement, bound file size, and refuse symlinked session files. | `test_session_permissions_and_known_secret_redaction`, `test_atomic_write_preserves_previous_file_when_replace_fails`, `test_session_symlink_is_never_followed` |
| Sensitive logging | Application logs record counts, IDs, status codes, and retry timing, not prompts, file content, tool output, response bodies, or Authorization headers; configured secrets pass through a redacting filter. | `test_provider_logs_do_not_contain_api_key`, `test_config_repr_and_logging_do_not_expose_api_key` |
| Infinite agent loop | Bound model turns, tool calls, stream chunks, and wall time; block identical tool-call fingerprints within a run and reject reused call IDs. | `test_max_turns_stops_the_loop`, `test_max_tool_calls_stops_before_execution`, `test_repeated_identical_call_is_blocked` |
| Oversized output / memory | Bound provider response bytes and SSE event bytes, agent text/tool-argument/tool-result sizes, workspace results, Shell capture, context, and session files. | `test_oversized_response_and_forged_tool_result_are_rejected`, `test_model_and_tool_outputs_are_bounded_before_persistence`, `test_output_limit_is_explicit` |
| Half-applied patch | Prepare and validate every hunk before writing, compare current bytes at commit, replace atomically, roll back completed writes, and refuse to overwrite a concurrent user edit during rollback. | `test_multifile_context_failure_leaves_no_partial_changes`, `test_write_failure_rolls_back_files_already_committed`, `test_rollback_does_not_overwrite_a_concurrent_user_change` |
| Overwriting existing Git work | Capture an exact task baseline, preserve and separately report pre-existing paths, mark overlapping changes, and refuse to stage or commit baseline paths. | `test_preexisting_uncommitted_change_is_preserved`, `test_preexisting_changes_are_not_attributed_or_restored`, `test_add_and_commit_only_task_local_files_after_approval` |
| Non-interactive approval bypass | `ASK` defaults to refusal outside an interactive UI. Hard denials cannot be downgraded by approval or command allowlists. The exact command, cwd, risk, and preview are bound to approval and rechecked before execution. | `test_ask_defaults_to_deny_without_approval`, `test_allowlist_cannot_override_hard_boundaries`, `test_user_refusal_prevents_dangerous_operation` |
| Sandbox downgrade | Permission decisions remain outside `SandboxRuntime`. Container mode reports missing Docker/daemon/image and never falls back to Local; Local is explicitly labeled `APPLICATION_ONLY`. | `test_shell_and_test_commands_use_runtime_after_permission_boundary`, `test_missing_docker_fails_closed_without_local_fallback`, `test_local_runtime_is_explicitly_application_only` |
| Container network and host exposure | Container defaults use `--network none`, a non-root UID/GID, read-only root, dropped capabilities, `no-new-privileges`, selected workspace mounts, and no Docker socket mount. | `test_container_defaults_encode_network_identity_and_resource_limits`, `test_container_allowed_paths_are_mounted_without_full_workspace` |
| Container secret/environment leakage | Container payload variables come only from `environment_allowlist`; secret-like names are dropped under the default policy and provider API keys are never automatically forwarded. | `test_environment_allowlist_excludes_secrets`, `test_container_defaults_encode_network_identity_and_resource_limits` |
| Container resource exhaustion | Docker creation applies CPU, memory, and PID limits; the runtime also applies a host-observed wall timeout, bounds captured output and terminates timed-out process groups. | `test_container_defaults_encode_network_identity_and_resource_limits`, `test_timeout_is_enforced_and_process_is_terminated` |
| Sandbox output escape | Container output uses a dedicated `/artifacts` mount, is size-bounded, rejects links/non-regular files, copies to a unique workspace artifact directory without overwriting, then removes the container and staging directory. | `test_container_artifacts_are_recovered_before_cleanup` |
| Forged tool results | Provider responses must be assistant messages and cannot carry `tool_call_id`; only the runtime creates tool-role messages, and sessions validate every tool result against a unique prior call. | `test_oversized_response_and_forged_tool_result_are_rejected`, `test_tool_result_must_reference_an_existing_call` |

## End-to-end recovery and provider behavior

The integration suite exercises a real tool registry and filesystem for:

1. reading a file, patching it, and running its tests;
2. a failed patch followed by a fresh read and corrected patch;
3. Shell timeout and process-group termination;
4. interactive refusal of a mutating command;
5. bounded retry after HTTP 429, including `Retry-After`;
6. session resume after an interrupted tool call (the pending call is closed with
   an explicit interrupted result and is never silently replayed);
7. preservation and separate attribution of existing uncommitted changes; and
8. rejection of closed-schema injection, traversal, and outside-Shell-read
   arguments.

## Residual boundary

`LocalSandboxRuntime` remains an application-only compatibility boundary.
Approved test runners, compilers, interpreters, hooks, and repository code can
perform behavior that argv inspection cannot infer, and Local cannot enforce
network or resource policy at the kernel boundary.

`ContainerSandboxRuntime` materially reduces that exposure but is an
experimental Docker boundary, not a multi-tenant isolation service. Docker
daemon compromise, kernel vulnerabilities, hostile images, and Docker Desktop
platform differences remain out of scope. Kubernetes, micro-VMs, remote
execution clusters, Secrets Brokers, and multi-tenant sandbox services are not
implemented.
