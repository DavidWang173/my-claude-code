from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.context import (
    ContextManager,
    ContextState,
    context_state_from_dict,
    context_state_to_dict,
)
from src.git_runtime import GitRunTracker
from src.harness.project_knowledge import (
    ContextBuilder,
    ContextRetrievalService,
    InstructionResolver,
    ProjectIndexer,
    RetrievalCandidate,
    RetrievalQuery,
)
from src.models import SystemMessage, UserMessage


class ContextRetrievalTests(unittest.TestCase):
    def test_workspace_tree_and_readme_are_bootstrap_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            readme_body = "BOOTSTRAP_README_BODY"
            (workspace / "README.md").write_text(readme_body, encoding="utf-8")
            (workspace / "only_at_bootstrap.py").write_text("value = 1\n", encoding="utf-8")
            manager = ContextManager(max_tokens=4_000)
            state = ContextState()
            messages = (
                SystemMessage(manager.initial_system_prompt(workspace, "base rules")),
                UserMessage("inspect value"),
            )

            first = manager.select(
                messages,
                (),
                state,
                session_id="b" * 32,
            )
            second = manager.select(
                messages,
                (),
                state,
                session_id="b" * 32,
            )

        first_text = "\n".join(message.content or "" for message in first.messages)
        second_text = "\n".join(message.content or "" for message in second.messages)
        self.assertIn(readme_body, first_text)
        self.assertIn("only_at_bootstrap.py", first_text)
        self.assertNotIn(readme_body, second_text)
        self.assertNotIn("only_at_bootstrap.py", second_text)
        self.assertIn("retrieved dynamically", second_text)

    def test_task_and_plan_step_find_filename_symbol_and_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            (workspace / "src" / "payment_service.py").write_text(
                "class PaymentProcessor:\n"
                "    def charge(self, amount):\n"
                "        return amount\n",
                encoding="utf-8",
            )
            (workspace / "src" / "api.py").write_text(
                "from src.payment_service import PaymentProcessor\n\n"
                "def submit_payment(amount):\n"
                "    return PaymentProcessor().charge(amount)\n",
                encoding="utf-8",
            )
            (workspace / "src" / "weather.py").write_text(
                "def forecast():\n    return 'sunny'\n",
                encoding="utf-8",
            )
            service = ContextRetrievalService(workspace)
            state = ContextState()

            selection = service.retrieve(
                RetrievalQuery(
                    task="Repair the charge flow",
                    plan_step="Update PaymentProcessor in src/payment_service.py",
                    target_paths=("src/payment_service.py",),
                    token_budget=5_000,
                ),
                state,
            )

        payment = [
            item for item in selection.candidates if item.source_path == "src/payment_service.py"
        ]
        dependencies = [
            item
            for item in selection.candidates
            if item.source_path == "src/api.py" and "dependency_of" in item.reason
        ]
        weather = [
            item for item in selection.candidates if item.source_path == "src/weather.py"
        ]
        self.assertTrue(payment)
        self.assertTrue(
            any(
                "filename_match" in item.reason or "python_symbol" in item.reason
                for item in payment
            )
        )
        self.assertTrue(dependencies)
        self.assertTrue(weather)
        self.assertIn("Update PaymentProcessor", selection.prompt)
        self.assertTrue(
            any(item.reason == "current_plan_step" for item in selection.candidates)
        )
        self.assertGreater(
            max(item.relevance_score for item in payment),
            max(item.relevance_score for item in weather),
        )

    def test_instruction_resolution_is_root_to_nearest_with_explicit_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target_dir = workspace / "packages" / "api"
            target_dir.mkdir(parents=True)
            target = target_dir / "handler.py"
            target.write_text("def handle():\n    return 1\n", encoding="utf-8")
            (workspace / "AGENTS.md").write_text("root rule", encoding="utf-8")
            (workspace / "packages" / "AGENTS.md").write_text(
                "package rule", encoding="utf-8"
            )
            (target_dir / "AGENTS.md").write_text("adjacent rule", encoding="utf-8")
            indexer = ProjectIndexer(workspace)

            candidates = InstructionResolver(indexer).resolve(
                current_directory="packages",
                target_paths=("packages/api/handler.py",),
            )
            prompt = ContextBuilder().build(
                candidates,
                token_budget=2_000,
            ).prompt

        by_path = {item.source_path: item for item in candidates}
        self.assertLess(
            by_path["AGENTS.md"].instruction_priority,
            by_path["packages/api/AGENTS.md"].instruction_priority,
        )
        self.assertGreater(
            by_path["packages/api/AGENTS.md"].relevance_score,
            by_path["AGENTS.md"].relevance_score,
        )
        self.assertIn("instruction:target-adjacent", by_path["packages/api/AGENTS.md"].reason)
        self.assertLess(prompt.index("root rule"), prompt.index("package rule"))
        self.assertLess(prompt.index("package rule"), prompt.index("adjacent rule"))
        self.assertIn("source=packages/api/AGENTS.md", prompt)

    def test_file_change_marks_old_result_stale_and_resume_adds_fresh_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            source = workspace / "src" / "service.py"
            source.write_text(
                "class Service:\n    value = 'old'\n",
                encoding="utf-8",
            )
            service = ContextRetrievalService(workspace)
            state = ContextState()
            query = RetrievalQuery(
                task="Update Service",
                target_paths=("src/service.py",),
                token_budget=2_000,
            )
            service.retrieve(query, state)
            old_hashes = {
                item.content_hash
                for item in state.retrieval_results
                if item.source_path == "src/service.py"
            }
            self.assertTrue(old_hashes)

            source.write_text(
                "class Service:\n    value = 'new and changed'\n",
                encoding="utf-8",
            )
            stale_on_resume = service.refresh_session(state)
            selection = service.retrieve(query, state)

        self.assertEqual(stale_on_resume, ("src/service.py",))
        old = [
            item
            for item in state.retrieval_results
            if item.source_path == "src/service.py" and item.content_hash in old_hashes
        ]
        fresh = [
            item
            for item in state.retrieval_results
            if item.source_path == "src/service.py" and not item.stale
        ]
        self.assertTrue(old)
        self.assertTrue(all(item.stale for item in old))
        self.assertTrue(fresh)
        self.assertTrue(all(item.content_hash not in old_hashes for item in fresh))
        self.assertIn("Stale retrievals omitted", selection.prompt)

    def test_token_budget_clips_candidates(self) -> None:
        large = RetrievalCandidate(
            source_path="src/large.py",
            start_line=1,
            end_line=500,
            content="important_value = 1\n" * 500,
            content_hash="a" * 64,
            reason="python_symbol:important_value",
            relevance_score=0.95,
        )
        irrelevant = RetrievalCandidate(
            source_path="docs/unrelated.md",
            start_line=1,
            end_line=100,
            content="unrelated\n" * 100,
            content_hash="b" * 64,
            reason="recently_modified",
            relevance_score=0.1,
        )

        selection = ContextBuilder().build(
            (large, irrelevant),
            token_budget=180,
        )

        self.assertLessEqual(selection.estimated_tokens, 180)
        self.assertIn("src/large.py", selection.prompt)
        self.assertIn("truncated", selection.prompt)
        self.assertNotIn("docs/unrelated.md", selection.prompt)

    def test_git_baseline_is_injected_into_provider_facing_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            (workspace / "src" / "owned.py").write_text("value = 1\n", encoding="utf-8")
            tracker = GitRunTracker(
                workspace,
                is_repository=True,
                baseline_paths={"src/owned.py"},
            )
            state = ContextState()
            retrieved = ContextRetrievalService(workspace).retrieve(
                RetrievalQuery(
                    task="Inspect owned.py",
                    git_baseline=tracker.baseline_prompt(),
                    token_budget=1_000,
                ),
                state,
            )
            selected = ContextManager(max_tokens=2_000).select(
                (SystemMessage("base"), UserMessage("Inspect owned.py")),
                (),
                state,
                session_id="a" * 32,
                dynamic_context=retrieved.prompt,
            )

        provider_text = "\n".join(message.content or "" for message in selected.messages)
        self.assertIn("Git baseline:", provider_text)
        self.assertIn("src/owned.py", provider_text)
        self.assertTrue(
            any(item.reason == "git_baseline" for item in state.retrieval_results)
        )

    def test_error_stack_targets_exact_file_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            source = workspace / "src" / "crash.py"
            source.write_text(
                "\n".join(f"line_{line} = {line}" for line in range(1, 61)) + "\n",
                encoding="utf-8",
            )
            selection = ContextRetrievalService(workspace).retrieve(
                RetrievalQuery(
                    task="Fix the failing test",
                    error_stack=f'File "{source}", line 37, in explode',
                    token_budget=2_000,
                ),
                ContextState(),
            )

        stack_candidates = [
            item for item in selection.candidates if item.reason == "error_stack"
        ]
        self.assertTrue(stack_candidates)
        self.assertEqual(stack_candidates[0].source_path, "src/crash.py")
        self.assertLessEqual(stack_candidates[0].start_line, 37)
        self.assertGreaterEqual(stack_candidates[0].end_line, 37)
        self.assertGreaterEqual(stack_candidates[0].relevance_score, 0.99)

    def test_session_metadata_round_trip_omits_source_body(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "account.py"
            secret_source_body = "PRIVATE_PROJECT_SOURCE = 'do not persist'\n"
            source.write_text(secret_source_body, encoding="utf-8")
            state = ContextState()
            ContextRetrievalService(workspace).retrieve(
                RetrievalQuery(
                    task="Inspect account.py",
                    target_paths=("account.py",),
                    token_budget=1_000,
                ),
                state,
            )

            payload = context_state_to_dict(state)
            restored = context_state_from_dict(payload)

        self.assertTrue(restored.retrieval_results)
        self.assertNotIn(secret_source_body.strip(), repr(payload))
        record = restored.retrieval_results[0]
        self.assertTrue(record.source_path)
        self.assertGreaterEqual(record.start_line, 1)
        self.assertTrue(record.content_hash)
        self.assertTrue(record.reason)
        self.assertGreaterEqual(record.relevance_score, 0)


if __name__ == "__main__":
    unittest.main()
