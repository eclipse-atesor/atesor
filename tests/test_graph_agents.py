"""Tests for the graph nodes in src/graph.py.

Covers the analyst node (consumed PackageAnalysis + real cost logging),
the fixer's read-only investigation helper, expected-artifact
verification, the init, scout, fixer, builder, cached-recipe, optimize
and terminal nodes, and the auto-learning step. The LLM and the
sandbox are mocked.
"""

import os
import unittest
from unittest import mock

from src.llm_helpers import LLMCallOutcome
from src.state import (
    AgentRole,
    ArchSpecificCode,
    BuildPhase,
    BuildPlan,
    BuildStatus,
    BuildSystemInfo,
    CommandResult,
    DependencyInfo,
    ErrorCategory,
    FailureSeverity,
    FixAttempt,
    PackageAnalysis,
    create_error_record,
    create_initial_state,
)


def _state(**overrides):
    """Build a base AgentState with field overrides."""
    s = create_initial_state("https://github.com/a/b.git")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


_ANALYST_DATA = {
    "purpose": "A DNS enumeration CLI tool",
    "language": "Go",
    "build_system": {
        "type": "go",
        "confidence": 0.95,
        "reasoning": "go.mod declares module, main.go at root",
    },
    "dependencies": [{"name": "git", "reason": "go mod fetch"}],
    "riscv_risks": ["cgo disabled path unclear"],
    "build_strategy": "go build the root module with -buildvcs=false.",
    "expected_artifacts": ["assetfinder"],
    "needs_custom_plan": False,
    "complexity": 3,
}


class TestAnalystNode(unittest.TestCase):
    """Tests for analyst_node."""

    def _run(self, outcome):
        """Run analyst_node with the LLM layer mocked out."""
        from src import graph

        s = _state()
        with (
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ),
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="### go.mod\n..."
            ),
        ):
            return graph.analyst_node(s)

    def test_valid_analysis_is_stored_and_consumable(self) -> None:
        """A valid LLM response becomes an LLM-grounded analysis."""
        out = self._run(
            LLMCallOutcome(
                data=dict(_ANALYST_DATA),
                used_fallback=False,
                attempts=1,
                input_tokens=1200,
                output_tokens=300,
                cost_usd=0.0,
            )
        )
        pa = out.package_analysis
        self.assertIsNotNone(pa)
        self.assertTrue(pa.llm_grounded)
        self.assertEqual(pa.build_system, "go")
        self.assertEqual(pa.expected_artifacts, ["assetfinder"])
        self.assertFalse(pa.needs_custom_plan)
        # Structural task plan exists without spending LLM tokens on it
        self.assertIsNotNone(out.task_plan)

    def test_real_usage_is_logged(self) -> None:
        """The node logs the outcome's REAL tokens/cost, not a flat fee."""
        out = self._run(
            LLMCallOutcome(
                data=dict(_ANALYST_DATA),
                used_fallback=False,
                attempts=2,
                input_tokens=2000,
                output_tokens=500,
                cost_usd=0.0125,
            )
        )
        self.assertEqual(out.api_tokens_in, 2000)
        self.assertEqual(out.api_tokens_out, 500)
        self.assertAlmostEqual(out.api_cost_usd, 0.0125)
        self.assertEqual(out.api_calls_made, 2)

    def test_llm_starvation_falls_back_deterministically(self) -> None:
        """No LLM data → deterministic, non-grounded analysis."""
        out = self._run(
            LLMCallOutcome(
                data=None,
                used_fallback=False,
                attempts=3,
                last_error="429 everywhere",
            )
        )
        pa = out.package_analysis
        self.assertIsNotNone(pa)
        self.assertFalse(pa.llm_grounded)
        self.assertIsNotNone(out.task_plan)

    def test_analysis_validator_rejects_bad_shapes(self) -> None:
        """The nested analyst validator rejects malformed JSON shapes."""
        from src import graph

        def fake_llm_call(**kwargs):
            validator = kwargs["validator"]
            self.assertFalse(validator({}).ok)
            self.assertFalse(
                validator({"purpose": "x", "build_system": "go"}).ok
            )
            self.assertFalse(
                validator(
                    {
                        "purpose": "x",
                        "build_system": {"type": "go"},
                        "dependencies": "git",
                        "needs_custom_plan": False,
                    }
                ).ok
            )
            self.assertFalse(
                validator(
                    {
                        "purpose": "x",
                        "build_system": {"type": "go"},
                        "dependencies": [],
                        "needs_custom_plan": "false",
                    }
                ).ok
            )
            self.assertTrue(validator(dict(_ANALYST_DATA)).ok)
            return LLMCallOutcome(
                data=dict(_ANALYST_DATA),
                used_fallback=False,
                attempts=1,
            )

        state = _state()
        with (
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="evidence"
            ),
        ):
            out = graph.analyst_node(state)

        self.assertTrue(out.package_analysis.llm_grounded)


class TestUnsupportedPackage(unittest.TestCase):
    """A package that builds no riscv64 binary stops before the build."""

    def setUp(self) -> None:
        import shutil
        import tempfile

        self.repo = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.repo, True)
        self._write("package.json", "{}")
        self._write("src/index.ts", "export {}")

    def _write(self, rel: str, text: str = "") -> None:
        path = os.path.join(self.repo, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)

    @staticmethod
    def _nginx_data(**changes):
        """The analyst answer of nginx-binaries (2026-10-16)."""
        data = dict(
            _ANALYST_DATA,
            language="TypeScript (compiled to CommonJS JavaScript)",
            build_system={"type": "other", "confidence": 0.9},
            expected_artifacts=["lib/index.js", "lib/index.d.ts"],
        )
        data.update(changes)
        return data

    def _analyst(self, data):
        from src import graph

        with (
            mock.patch.object(
                graph,
                "llm_call_with_validation",
                return_value=LLMCallOutcome(
                    data=data, used_fallback=False, attempts=1
                ),
            ),
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="### x\n"
            ),
            mock.patch.object(graph, "_to_host_path", return_value=self.repo),
        ):
            return graph.analyst_node(_state())

    def test_script_outputs_without_native_code_stop_the_run(self) -> None:
        from src import graph

        out = self._analyst(self._nginx_data())
        reason = out.context_cache.get("unsupported", "")
        self.assertIn("lib/index.js, lib/index.d.ts", reason)
        self.assertEqual(graph.route_analyst_to_next(out), "escalate_node")
        self.assertEqual(out.outcome_label, "UNSUPPORTED")
        with mock.patch.object(graph, "log_data_effect"):
            out = graph.escalate_node(out)
        self.assertIn(reason, out.context_cache["escalation_report"])

    def test_a_native_file_or_output_keeps_the_pipeline(self) -> None:
        from src import graph

        cases = (
            # A Node addon builds an ELF .node file.
            ("binding.gyp", ["lib/index.js"]),
            ("src/addon.c", ["lib/index.js"]),
            ("", ["lib/index.js", "build/Release/addon.node"]),
            # static-ffmpeg-binaries: names without a suffix.
            ("", ["ffmpeg", "ffprobe"]),
        )
        for native_file, outputs in cases:
            with self.subTest(native_file=native_file, outputs=outputs):
                if native_file:
                    self._write(native_file)
                out = self._analyst(
                    self._nginx_data(expected_artifacts=outputs)
                )
                self.assertNotIn("unsupported", out.context_cache)
                self.assertEqual(
                    graph.route_analyst_to_next(out), "heuristic_plan_node"
                )
                if native_file:
                    os.remove(os.path.join(self.repo, native_file))

    def test_no_label_without_a_grounded_answer_or_outputs(self) -> None:
        from src import graph

        out = self._analyst(self._nginx_data(expected_artifacts=[]))
        self.assertNotIn("unsupported", out.context_cache)
        state = _state()
        state.package_analysis = graph._fallback_analysis(state)
        state.package_analysis.expected_artifacts = ["lib/index.js"]
        with mock.patch.object(graph, "_to_host_path", return_value=self.repo):
            self.assertEqual(graph._unsupported_reason(state), "")

    def test_an_unreadable_tree_counts_as_native(self) -> None:
        from src import graph

        missing = os.path.join(self.repo, "missing")
        self.assertEqual(graph._native_build_input(missing), "(no tree)")
        with mock.patch.object(graph, "_NATIVE_SCAN_LIMIT", 1):
            self.assertEqual(
                graph._native_build_input(self.repo), "(scan limit)"
            )


class TestAnalystRefusal(unittest.TestCase):
    """A refused analyst call marks the package and uses the fallback."""

    def test_refusal_marks_the_package(self) -> None:
        out = TestAnalystNode._run(
            self,
            LLMCallOutcome(
                data=None, used_fallback=False, attempts=1, refused=True
            ),
        )
        self.assertEqual(out.context_cache["llm_refused"], "planner")
        self.assertFalse(out.package_analysis.llm_grounded)


class TestInitNode(unittest.TestCase):
    """Tests for init_node with clone and analysis operations mocked."""

    def test_successful_init_populates_scripted_analysis(self) -> None:
        """Init stores quick-analysis facts and missing tool context."""
        from src import graph

        state = _state()
        deps = DependencyInfo(build_tools=["cmake"], libraries=["zlib"])
        analysis = {
            "build_system": BuildSystemInfo(
                type="unknown",
                confidence=0.1,
                primary_file="",
            ),
            "dependencies": deps,
            "arch_specific_code": [
                ArchSpecificCode("src/simd.c", 1, "_mm", "x86", "high")
            ],
            "optimized_tree": "repo tree",
            "documentation": ["README.md"],
            "go_main_info": {"has_main": True},
            "arch_build_files": {
                "has_arch_specific": True,
                "archs_found": ["x86"],
                "riscv_exists": False,
            },
        }

        with (
            mock.patch.object(
                graph.scripted_ops,
                "clone_or_update_repository",
                return_value=CommandResult("git clone", 0, "", "", 1),
            ),
            mock.patch.object(graph, "quick_analysis", return_value=analysis),
            mock.patch.object(
                graph.scripted_ops,
                "read_file",
                return_value="readme contents",
            ) as read_file,
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={
                    "gcc": "Installed",
                    "make": "Installed",
                    "unknown": "Installed",
                    "cmake": "Not installed",
                },
            ),
        ):
            out = graph.init_node(state)

        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertEqual(out.current_phase, "initialized")
        self.assertEqual(out.repo_tree, "repo tree")
        self.assertEqual(out.context_cache["missing_tools"], ["cmake"])
        self.assertEqual(out.context_cache["go_main_info"], {"has_main": True})
        read_file.assert_called_once_with("README.md", max_lines=500)

    def test_clone_failure_without_recovery_escalates(self) -> None:
        """Init turns unrecoverable clone failures into state errors."""
        from src import graph

        state = _state()
        failure = CommandResult(
            "git clone",
            128,
            "",
            "remote: Repository not found.",
            1,
        )

        with (
            mock.patch.object(
                graph.scripted_ops,
                "clone_or_update_repository",
                return_value=failure,
            ),
            mock.patch.object(graph, "_try_clone_recovery", return_value=None),
        ):
            out = graph.init_node(state)

        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertEqual(out.current_phase, "escalate")
        self.assertEqual(out.last_error_category, ErrorCategory.CONFIGURATION)

    def test_clone_recovery_continues_to_quick_analysis(self) -> None:
        """A successful clone URL variant lets init continue normally."""
        from src import graph

        state = _state()
        failure = CommandResult("git clone bad", 128, "", "bad url", 1)
        recovered = CommandResult("git clone good", 0, "", "", 1)
        analysis = {
            "build_system": BuildSystemInfo(
                type="make",
                confidence=0.8,
                primary_file="Makefile",
            ),
            "dependencies": DependencyInfo(),
            "arch_specific_code": [],
            "optimized_tree": "",
            "documentation": [],
        }

        with (
            mock.patch.object(
                graph.scripted_ops,
                "clone_or_update_repository",
                return_value=failure,
            ),
            mock.patch.object(
                graph, "_try_clone_recovery", return_value=recovered
            ),
            mock.patch.object(graph, "quick_analysis", return_value=analysis),
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={"gcc": "Installed", "make": "Installed"},
            ),
        ):
            out = graph.init_node(state)

        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertEqual(out.build_system_info.type, "make")


class TestScoutNode(unittest.TestCase):
    """Tests for scout_node with LLM and scripted operations mocked."""

    def _scout_state(self):
        """Build a state with enough context to render the scout prompt."""
        state = _state()
        state.build_system_info = BuildSystemInfo(
            type="cmake",
            confidence=0.8,
            primary_file="CMakeLists.txt",
            module_dir="",
        )
        state.dependencies = DependencyInfo(
            build_tools=["cmake"],
            system_packages=["zlib-dev"],
            libraries=["zlib"],
        )
        state.file_content_cache["README.md"] = "Build with cmake."
        state.arch_specific_code = [
            ArchSpecificCode(
                file="src/simd.c",
                line=12,
                code_snippet="_mm_add_ps",
                arch_type="x86",
                severity="high",
            )
        ]
        state.package_analysis = PackageAnalysis(
            purpose="test package",
            build_system="cmake",
            build_system_confidence=0.9,
            llm_grounded=True,
        )
        state.last_error = "fatal error: zlib.h: No such file"
        state.error_history = [
            create_error_record(
                "fatal error: zlib.h: No such file",
                ErrorCategory.DEPENDENCY,
                command="make",
            )
        ]
        return state

    def _patch_scout_environment(self, graph, outcome):
        """Patch scout dependencies that would otherwise touch the sandbox."""
        return (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ),
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={"architecture": "riscv64", "gcc": "installed"},
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="build evidence"
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="knowledge"
            ),
        )

    def test_scout_validator_rejects_bad_shapes_and_builds_plan(self) -> None:
        from src import graph

        valid_plan = {
            "build_system": "cmake",
            "build_system_confidence": 0.91,
            "phases": [
                {
                    "id": 1,
                    "name": "build",
                    "commands": ["cmake -B build -S ."],
                    "can_parallelize": False,
                    "expected_duration": "1m",
                }
            ],
            "total_estimated_duration": "1m",
            "notes": ["ok"],
        }

        def fake_llm_call(**kwargs):
            validator = kwargs["validator"]
            self.assertFalse(validator({}).ok)
            self.assertFalse(validator({"phases": ["bad"]}).ok)
            self.assertFalse(
                validator({"phases": [{"name": "", "commands": ["make"]}]}).ok
            )
            self.assertFalse(
                validator({"phases": [{"name": "b", "commands": []}]}).ok
            )
            self.assertFalse(
                validator({"phases": [{"name": "b", "commands": ["   "]}]}).ok
            )
            self.assertFalse(
                validator(
                    {
                        "phases": [
                            {
                                "name": "b",
                                "commands": ["cp /path/to/foo bar"],
                            }
                        ]
                    }
                ).ok
            )
            self.assertTrue(validator(valid_plan).ok)
            return LLMCallOutcome(
                data=valid_plan,
                used_fallback=False,
                attempts=2,
                input_tokens=100,
                output_tokens=25,
                cost_usd=0.01,
            )

        state = self._scout_state()
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={"architecture": "riscv64", "gcc": "installed"},
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="build evidence"
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="knowledge"
            ),
        ):
            out = graph.scout_node(state)

        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertEqual(out.build_plan.build_system, "cmake")
        self.assertEqual(out.api_calls_made, 2)

    def test_scout_falls_back_when_llm_returns_no_plan(self) -> None:
        from src import graph

        outcome = LLMCallOutcome(
            data=None,
            used_fallback=False,
            attempts=1,
            last_error="invalid json",
        )
        state = self._scout_state()
        patches = self._patch_scout_environment(graph, outcome)

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
        ):
            out = graph.scout_node(state)

        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertIsNotNone(out.build_plan)
        self.assertTrue(out.error_history)

    def _run_scout(self, state, outcome):
        """Run scout_node with the standard environment patches."""
        from src import graph

        patches = self._patch_scout_environment(graph, outcome)
        with (
            patches[0],
            patches[1] as llm_call,
            patches[2],
            patches[3],
            patches[4],
            patches[5],
        ):
            out = graph.scout_node(state)
        return out, llm_call

    def test_scout_validator_rejects_a_whitelist_blocked_command(
        self,
    ) -> None:
        """A command the sandbox validator blocks never reaches the build."""
        from src import graph

        captured = {}

        def fake_llm_call(**kwargs):
            captured["validator"] = kwargs["validator"]
            return LLMCallOutcome(data=None, used_fallback=False, attempts=1)

        state = self._scout_state()
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={"architecture": "riscv64"},
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="evidence"
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="k"
            ),
        ):
            graph.scout_node(state)

        verdict = captured["validator"](
            {
                "phases": [
                    {"name": "setup", "commands": ["nix-build default.nix"]}
                ]
            }
        )
        self.assertFalse(verdict.ok)
        self.assertIn("nix-build default.nix", verdict.reason)
        ok = captured["validator"](
            {"phases": [{"name": "build", "commands": ["make -j4"]}]}
        )
        self.assertTrue(ok.ok)

    def test_scout_refusal_uses_fallback_without_an_attempt(self) -> None:
        outcome = LLMCallOutcome(
            data=None, used_fallback=False, attempts=1, refused=True
        )
        state = self._scout_state()
        errors_before = len(state.error_history)
        attempts_before = state.attempt_count

        out, _ = self._run_scout(state, outcome)

        self.assertEqual(out.context_cache["llm_refused"], "scout")
        self.assertIsNotNone(out.build_plan)
        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertEqual(len(out.error_history), errors_before)
        self.assertEqual(out.attempt_count, attempts_before)

    def test_scout_skips_the_llm_after_a_refusal(self) -> None:
        state = self._scout_state()
        state.context_cache["llm_refused"] = "planner"

        out, llm_call = self._run_scout(state, None)

        llm_call.assert_not_called()
        self.assertIsNotNone(out.build_plan)
        self.assertEqual(out.build_status, BuildStatus.PENDING)

    def test_scout_escalates_when_only_blocked_plans_exist(self) -> None:
        """No guessed fallback plan when every plan needs a blocked tool."""
        from src import graph

        outcome = LLMCallOutcome(
            data=None,
            used_fallback=False,
            attempts=3,
            last_error=(
                "schema validation failed: phase 1 (build): the sandbox "
                "command validator blocks `cabal build` (Unknown command "
                "pattern)"
            ),
        )
        state = self._scout_state()
        state.build_plan = None

        out, _ = self._run_scout(state, outcome)

        self.assertIsNone(out.build_plan)
        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertIn(
            "Command blocked by safety validation",
            out.error_history[-1].message,
        )
        self.assertEqual(out.context_cache["validator_block_replans"], 1)
        self.assertEqual(graph.route_supervisor_to_next(out), "escalate_node")

    def test_rescout_after_a_validator_block_is_counted(self) -> None:
        state = self._scout_state()
        state.add_error(
            create_error_record(
                "Build command failed: zig version (exit 1) - Command "
                "blocked by safety validation: Unknown command pattern",
                command="zig version",
            )
        )
        outcome = LLMCallOutcome(data=None, used_fallback=False, attempts=1)

        out, _ = self._run_scout(state, outcome)

        self.assertEqual(out.context_cache["validator_block_replans"], 1)


class TestFixerInvestigation(unittest.TestCase):
    """Tests for _run_fixer_investigation."""

    def _fake_exec(self, stdout="output", stderr="", exit_code=0):
        """Build a fake execute_command result."""
        return mock.MagicMock(
            stdout=stdout, stderr=stderr, exit_code=exit_code, success=True
        )

    def test_read_only_commands_execute(self) -> None:
        """Whitelisted commands run and their output is captured."""
        from src import graph

        s = _state()
        with mock.patch.object(
            graph, "execute_command", return_value=self._fake_exec("hello")
        ) as exec_mock:
            out = graph._run_fixer_investigation(
                s, ["cat Makefile", "grep -rn simd src/"]
            )
        self.assertEqual(exec_mock.call_count, 2)
        self.assertIn("$ cat Makefile", out)
        self.assertIn("hello", out)

    def test_mutating_commands_are_rejected(self) -> None:
        """Non-whitelisted or redirecting commands never execute."""
        from src import graph

        s = _state()
        with mock.patch.object(graph, "execute_command") as exec_mock:
            out = graph._run_fixer_investigation(
                s,
                [
                    "rm -rf /",
                    "sed -i s/a/b/ Makefile",
                    "cat foo > bar",
                ],
            )
        exec_mock.assert_not_called()
        self.assertEqual(out.count("[rejected"), 3)

    def test_command_count_is_capped(self) -> None:
        """At most 4 commands run per investigation round."""
        from src import graph

        s = _state()
        with mock.patch.object(
            graph, "execute_command", return_value=self._fake_exec()
        ) as exec_mock:
            graph._run_fixer_investigation(s, ["ls"] * 10)
        self.assertEqual(exec_mock.call_count, 4)

    def test_output_is_truncated(self) -> None:
        """Command output is capped so the prompt stays small."""
        from src import graph

        s = _state()
        with mock.patch.object(
            graph,
            "execute_command",
            return_value=self._fake_exec("x" * 5000),
        ):
            out = graph._run_fixer_investigation(s, ["cat big.txt"])
        self.assertIn("[... truncated ...]", out)
        self.assertLess(len(out), 2500)

    def test_empty_investigation_commands_return_placeholder(self) -> None:
        from src import graph

        s = _state()
        with mock.patch.object(graph, "execute_command") as exec_mock:
            out = graph._run_fixer_investigation(s, ["", "   "])

        exec_mock.assert_not_called()
        self.assertEqual(out, "(no valid commands to run)")


class TestExpectedArtifactVerification(unittest.TestCase):
    """Tests for verify_node on top of the ELF-header scanner."""

    RVC_DOUBLE = 0x5

    def _entry(self, path, machine, tracked=False):
        """Return one scan entry for an executable ELF file."""
        return {
            "path": path,
            "size": 4096,
            "exec": True,
            "tracked": tracked,
            "elf": [64, 1, 2, machine, self.RVC_DOUBLE],
        }

    def _verdict(self, files=(), outside=(), expected=()):
        """Build a real VerificationResult from a scan payload."""
        from src.artifact_scanner import evaluate_scan

        payload = {
            "files": list(files),
            "outside": list(outside),
            "examined": len(files),
            "commit": "a" * 40,
        }
        return evaluate_scan(payload, "/workspace/repos/b", expected)

    def _verify(self, verdict, state=None):
        """Run verify_node with the scanner returning ``verdict``."""
        from src import graph

        state = state or _state()
        scanner = mock.MagicMock()
        scanner.scan.return_value = verdict
        with mock.patch.object(
            graph, "ArtifactScanner", return_value=scanner
        ) as cls:
            out = graph.verify_node(state)
        return out, cls, scanner

    def test_scanner_valid_artifacts_are_recorded(self) -> None:
        """A verified scan records riscv64 artifacts and succeeds."""
        out, _cls, _scanner = self._verify(
            self._verdict(files=[self._entry("build/tool", 243)])
        )

        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        self.assertEqual(out.verification_status, "verified")
        self.assertTrue(out.is_verified_success)
        self.assertEqual(
            out.build_artifacts[0]["filepath"],
            "/workspace/repos/b/build/tool",
        )
        self.assertEqual(out.build_artifacts[0]["architecture"], "riscv64")

    def test_scanner_wrong_architecture_fails_verification(self) -> None:
        """One x86-64 output beside riscv64 ones is a hard failure."""
        out, _cls, _scanner = self._verify(
            self._verdict(
                files=[
                    self._entry("build/tool", 243),
                    self._entry("build/helper", 62),
                ]
            )
        )

        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertEqual(out.last_error_category, ErrorCategory.ARCHITECTURE)
        self.assertEqual(out.verification_status, "wrong_arch")
        self.assertEqual(out.build_artifacts, [])

    def test_riscv_expected_artifact_seals_success(self) -> None:
        """Expected names and the start time reach the scanner."""
        state = _state()
        state.package_analysis = PackageAnalysis(
            purpose="x", expected_artifacts=["tool"]
        )
        verdict = self._verdict(
            outside=[self._entry("/usr/local/bin/tool", 243)],
            expected=["tool"],
        )

        out, cls, scanner = self._verify(verdict, state)

        cls.assert_called_once_with(state.repo_path)
        kwargs = scanner.scan.call_args.kwargs
        self.assertEqual(kwargs["expected_names"], ["tool"])
        self.assertEqual(
            kwargs["since"], state.execution_start_time.timestamp()
        )
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        self.assertTrue(out.is_verified_success)

    def test_wrong_arch_expected_artifact_fails(self) -> None:
        """An installed expected artifact that is x86 must FAIL."""
        verdict = self._verdict(
            outside=[self._entry("/usr/local/bin/tool", 62)],
            expected=["tool"],
        )

        out, _cls, _scanner = self._verify(verdict)

        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertEqual(out.last_error_category, ErrorCategory.ARCHITECTURE)

    def test_missing_expected_artifacts_recorded_in_caveat(self) -> None:
        """An unverified build names the expected outputs it never saw."""
        out, _cls, _scanner = self._verify(
            self._verdict(expected=["tool", "libtool.so"])
        )

        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        self.assertFalse(out.is_verified_success)
        self.assertEqual(out.verification_status, "unverified")
        self.assertEqual(
            out.artifact_verification["expected_missing"],
            ["tool", "libtool.so"],
        )

    def test_expectation_names_are_sanitized(self) -> None:
        """Path-ish or globby expectations never reach the scan program."""
        import json

        from src import artifact_scanner

        ok = CommandResult(
            "python3",
            0,
            'ATESOR_SCAN_JSON {"files": [], "outside": [], "examined": 0}',
            "",
            0.1,
        )
        with mock.patch.object(
            artifact_scanner, "execute_command", return_value=ok
        ) as execute:
            artifact_scanner.ArtifactScanner("/workspace/repos/b").scan(
                ["tool", "../etc/passwd", "*.so", "a b", "tool"]
            )

        names = json.loads(execute.call_args.args[0][5])
        self.assertEqual(names, ["tool"])

    def test_rescan_replaces_the_previous_artifact_list(self) -> None:
        """A rebuild after a fix is judged by its own outputs only."""
        state = _state()
        state.add_build_artifact("/old/x86-tool", "binary", "x86-64")

        out, _cls, _scanner = self._verify(
            self._verdict(files=[self._entry("build/tool", 243)]), state
        )

        self.assertEqual(
            [a["filepath"] for a in out.build_artifacts],
            ["/workspace/repos/b/build/tool"],
        )

    def test_recorded_artifacts_are_capped_expected_first(self) -> None:
        """A big build keeps 60 outputs in state; the summary counts all.

        The curator sends the recorded list to an LLM, so it must stay
        small. The expected output is kept even when it sorts last.
        """
        from src import graph

        state = _state()
        state.package_analysis = PackageAnalysis(
            purpose="x", expected_artifacts=["zz-tool"]
        )
        files = [
            self._entry(f"build/tests/t{i:03d}", 243) for i in range(100)
        ] + [self._entry("build/zz-tool", 243)]

        out, _cls, _scanner = self._verify(self._verdict(files=files), state)

        self.assertEqual(
            len(out.build_artifacts), graph._MAX_RECORDED_ARTIFACTS
        )
        self.assertEqual(
            out.build_artifacts[0]["filepath"],
            "/workspace/repos/b/build/zz-tool",
        )
        self.assertEqual(out.artifact_verification["counts"]["riscv64"], 101)
        self.assertTrue(out.is_verified_success)


class TestFixerNode(unittest.TestCase):
    """Tests for fixer_node paths that can be fully mocked."""

    def _fixer_state(self):
        """Build a state containing the minimum error context."""
        state = _state()
        state.last_error = "fatal error: zlib.h: No such file"
        state.last_error_category = ErrorCategory.DEPENDENCY
        state.build_plan = BuildPlan(
            build_system="make",
            build_system_confidence=0.8,
            phases=[BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )
        state.error_history = [
            create_error_record(
                state.last_error,
                ErrorCategory.DEPENDENCY,
                command="make",
            )
        ]
        return state

    def test_no_error_returns_to_pending(self) -> None:
        """Fixer does nothing when there is no error to fix."""
        from src import graph

        out = graph.fixer_node(_state())

        self.assertEqual(out.build_status, BuildStatus.PENDING)

    def _run_fixer_with(self, state, actions):
        """Run fixer_node with one mocked strategy made of ``actions``."""
        from src import graph

        strategy = {
            "strategies": [
                {"id": 1, "description": "rebuild fix", "actions": actions}
            ],
            "recommended_strategy_id": 1,
            "reflection": {
                "root_cause": "objects built for the wrong arch",
                "this_fix_will_work_because": "it rebuilds from source",
            },
        }
        outcome = LLMCallOutcome(
            data=strategy, used_fallback=False, attempts=1
        )
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ),
            mock.patch.object(
                graph,
                "execute_command",
                return_value=mock.MagicMock(success=True),
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="k"
            ),
        ):
            return graph.fixer_node(state)

    def test_refused_fixer_call_escalates_the_package(self) -> None:
        """A refusal is not retried and ends the loop for this package."""
        from src import graph

        state = self._fixer_state()
        errors_before = len(state.error_history)
        outcome = LLMCallOutcome(
            data=None, used_fallback=False, attempts=1, refused=True
        )
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ) as llm_call,
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="k"
            ),
        ):
            out = graph.fixer_node(state)

        self.assertEqual(llm_call.call_count, 1)
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)
        self.assertEqual(out.context_cache["llm_refused"], "fixer")
        self.assertEqual(len(out.error_history), errors_before)

    def test_fixer_skips_the_llm_after_a_refusal(self) -> None:
        from src import graph

        state = self._fixer_state()
        state.context_cache["llm_refused"] = "planner"
        with mock.patch.object(graph, "llm_call_with_validation") as call:
            out = graph.fixer_node(state)

        call.assert_not_called()
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)

    def test_fix_without_effective_change_skips_the_rebuild(self) -> None:
        """A fix whose commands all failed goes back to the fixer."""
        from src import graph

        state = self._fixer_state()
        state.attempt_count = 1
        strategy = {
            "strategies": [
                {
                    "id": 1,
                    "description": "install headers",
                    "actions": [
                        {"type": "command", "command": "apt-get install x"}
                    ],
                }
            ],
            "recommended_strategy_id": 1,
        }
        outcome = LLMCallOutcome(
            data=strategy, used_fallback=False, attempts=1
        )
        failed = mock.MagicMock(success=False, stderr="E: no package")
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ),
            mock.patch.object(graph, "execute_command", return_value=failed),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="k"
            ),
        ):
            out = graph.fixer_node(state)

        self.assertEqual(out.attempt_count, 2)
        self.assertEqual(out.build_status, BuildStatus.FIXING)
        self.assertEqual(
            out.fixes_attempted[-1].build_result, "no change applied"
        )
        self.assertEqual(graph.route_fix_result(out), "fix_node")

    def test_three_answers_without_a_change_end_the_loop(self) -> None:
        """The third no-change answer since a build stops the re-asks."""
        from src import graph

        state = self._fixer_state()
        state.attempt_count = 1
        strategy = {
            "strategies": [
                {
                    "id": 1,
                    "description": "install headers",
                    "actions": [
                        {"type": "command", "command": "apt-get install x"}
                    ],
                }
            ],
            "recommended_strategy_id": 1,
        }
        outcome = LLMCallOutcome(
            data=strategy, used_fallback=False, attempts=1
        )
        failed = mock.MagicMock(success=False, stderr="E: no package")
        routes = []
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ),
            mock.patch.object(graph, "execute_command", return_value=failed),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="k"
            ),
        ):
            for _ in range(3):
                state = graph.fixer_node(state)
                routes.append(graph.route_fix_result(state))

        # Attempt 4 of 5: the attempt limit did not stop the loop.
        self.assertEqual(state.attempt_count, 4)
        self.assertEqual(routes, ["fix_node", "fix_node", "__end__"])
        self.assertIn("changed nothing", graph._escalation_reason(state))

    def test_a_build_counts_and_resets_the_reasks(self) -> None:
        from src import graph

        state = _state()
        state.context_cache["reasks_since_build"] = 3
        state.build_plan = BuildPlan(
            build_system="make",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )
        ok = CommandResult("make", 0, "", "", 1.0)
        with (
            mock.patch.object(graph, "execute_command", return_value=ok),
            mock.patch.object(graph, "predict_build_issues", return_value=[]),
        ):
            out = graph.builder_node(state)
        self.assertEqual(out.build_count, 1)
        self.assertEqual(out.context_cache["reasks_since_build"], 0)

    def test_the_builder_runs_cargo_with_this_workers_jobs(self) -> None:
        from src import graph
        from src.machine import MachineProfile

        state = _state()
        state.build_plan = BuildPlan(
            build_system="cargo",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["cargo build --release"])],
            total_estimated_duration="1m",
        )
        board = MachineProfile(
            kind="native",
            emulated_cpu=None,
            extensions=["i"],
            cores=4,
            mem_kib=25 * 1024 * 1024,
        )
        ok = CommandResult("cargo", 0, "", "", 1.0)
        with (
            mock.patch.dict(os.environ, {"ATESOR_BATCH_WORKERS": "2"}),
            mock.patch.object(graph, "_machine", return_value=board),
            mock.patch.object(
                graph, "execute_command", return_value=ok
            ) as run,
            mock.patch.object(graph, "predict_build_issues", return_value=[]),
        ):
            graph.builder_node(state)
        self.assertEqual(
            run.call_args.args[0],
            "env CARGO_BUILD_JOBS=2 cargo build --release",
        )

    def test_the_builder_installs_the_go_of_go_mod_first(self) -> None:
        import shutil
        import tempfile

        from src import graph

        repo = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, repo, True)
        with open(os.path.join(repo, "go.mod"), "w") as fh:
            fh.write("module example.com/x\n\ngo 1.27.0\n")
        state = _state()
        state.build_plan = BuildPlan(
            build_system="go",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["go build ./..."])],
            total_estimated_duration="1m",
        )

        def fake_execute(cmd, *args, **kwargs):
            out = "go1.26.5" if cmd == "go env GOVERSION" else ""
            return CommandResult(cmd, 0, out, "", 0.1)

        with (
            mock.patch.object(graph, "_to_host_path", return_value=repo),
            mock.patch.object(
                graph, "execute_command", side_effect=fake_execute
            ) as run,
            mock.patch.object(graph, "predict_build_issues", return_value=[]),
        ):
            graph.builder_node(state)
        ran = [c.args[0] for c in run.call_args_list]
        self.assertEqual(ran[0], "go env GOVERSION")
        self.assertIn("go1.27.0.linux-riscv64.tar.gz", ran[1])
        self.assertEqual(ran[2], "go build ./...")

    def test_applied_fix_forces_a_full_rebuild(self) -> None:
        """A fix after a verify failure is compiled, not skipped."""
        state = self._fixer_state()
        # Every phase already succeeded; verify then failed.
        state.last_successful_phase = 1
        state.cache_command_result(
            "make", CommandResult("make", 0, "", "", 1.0)
        )

        out = self._run_fixer_with(
            state, [{"type": "command", "command": "make clean"}]
        )

        self.assertEqual(out.last_successful_phase, 0)
        self.assertIsNone(out.get_cached_command_result("make"))
        self.assertEqual(out.build_status, BuildStatus.PENDING)

    def _two_phase_state(self, done_phase: int):
        """A Go plan whose first phase is not safe to repeat."""
        state = self._fixer_state()
        state.build_plan = BuildPlan(
            build_system="go",
            build_system_confidence=0.9,
            phases=[
                BuildPhase(1, "setup", ["go mod init example.com/x"]),
                BuildPhase(2, "build", ["go build ./..."]),
            ],
            total_estimated_duration="1m",
        )
        state.last_successful_phase = done_phase
        for cmd in ("go mod init example.com/x", "go build ./..."):
            state.cache_command_result(cmd, CommandResult(cmd, 0, "", "", 1.0))
        return state

    def test_verify_failure_fix_reruns_build_steps_only(self) -> None:
        """The go mod init step stays done; go build runs again."""
        out = self._run_fixer_with(
            self._two_phase_state(done_phase=2),
            [{"type": "command", "command": "make clean"}],
        )

        self.assertEqual(out.last_successful_phase, 1)
        self.assertIsNotNone(
            out.get_cached_command_result("go mod init example.com/x")
        )
        self.assertIsNone(out.get_cached_command_result("go build ./..."))

    def test_build_failure_fix_resumes_at_the_failed_phase(self) -> None:
        """After a build failure the fixer does not rewind the progress."""
        out = self._run_fixer_with(
            self._two_phase_state(done_phase=1),
            [{"type": "command", "command": "make clean"}],
        )

        self.assertEqual(out.last_successful_phase, 1)
        self.assertIsNotNone(out.get_cached_command_result("go build ./..."))

    def test_rejected_fix_keeps_the_build_progress(self) -> None:
        """A strategy that changed nothing does not restart the build."""
        state = self._fixer_state()
        state.last_successful_phase = 1
        state.cache_command_result(
            "make", CommandResult("make", 0, "", "", 1.0)
        )

        out = self._run_fixer_with(
            state,
            [
                {
                    "type": "create_file",
                    "path": "../../etc/profile",
                    "content": "x",
                }
            ],
        )

        self.assertEqual(out.last_successful_phase, 1)
        self.assertIsNotNone(out.get_cached_command_result("make"))

    def test_investigation_round_then_command_fix(self) -> None:
        """Fixer can request one read-only investigation then emit a fix."""
        from src import graph

        strategy = {
            "strategies": [
                {
                    "id": 1,
                    "description": "install missing zlib",
                    "actions": [{"type": "command", "command": "make clean"}],
                }
            ],
            "recommended_strategy_id": 1,
            "reflection": {
                "root_cause": "zlib headers missing",
                "this_fix_will_work_because": "it reruns a clean build",
            },
        }
        calls = []

        def fake_llm_call(**kwargs):
            validator = kwargs["validator"]
            if not calls:
                self.assertTrue(
                    validator(
                        {"investigate": {"commands": ["cat Makefile"]}}
                    ).ok
                )
                calls.append("investigate")
                return LLMCallOutcome(
                    data={"investigate": {"commands": ["cat Makefile"]}},
                    used_fallback=False,
                    attempts=1,
                )

            self.assertFalse(
                validator({"investigate": {"commands": ["ls"]}}).ok
            )
            self.assertTrue(validator(strategy).ok)
            calls.append("fix")
            return LLMCallOutcome(
                data=strategy,
                used_fallback=False,
                attempts=1,
            )

        state = self._fixer_state()
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph, "_run_fixer_investigation", return_value="read output"
            ) as investigate,
            mock.patch.object(
                graph,
                "execute_command",
                return_value=mock.MagicMock(success=True),
            ),
            mock.patch.object(
                graph,
                "error_context_excerpts",
                return_value="src/main.c:1",
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="knowledge"
            ),
        ):
            out = graph.fixer_node(state)

        investigate.assert_called_once_with(state, ["cat Makefile"])
        self.assertEqual(calls, ["investigate", "fix"])
        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertEqual(
            out.fixes_attempted[-1].strategy,
            "install missing zlib",
        )

    def test_invalid_fixer_outcome_records_failure(self) -> None:
        """A missing final fixer strategy records a configuration error."""
        from src import graph

        outcome = LLMCallOutcome(
            data=None,
            used_fallback=False,
            attempts=2,
            last_error="schema invalid",
        )
        state = self._fixer_state()
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
        ):
            out = graph.fixer_node(state)

        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertIn("schema invalid", out.error_history[-1].message)

    def test_create_file_patch_and_command_actions_are_applied(self) -> None:
        from src import graph

        state = self._fixer_state()
        state.repo_path = "workspace/test_graph_agents/repo"
        strategy = {
            "strategies": [
                {
                    "id": 1,
                    "description": "add config and patch",
                    "actions": [
                        {
                            "type": "create_file",
                            "path": "generated/config.h",
                            "content": "#define HAVE_ZLIB 1\n",
                        },
                        {
                            "type": "patch",
                            "file": "src/main.c",
                            "content": "--- a/src/main.c\n+++ b/src/main.c\n",
                        },
                        {"type": "command", "command": "make clean"},
                    ],
                }
            ],
            "recommended_strategy_id": 1,
            "reflection": {
                "root_cause": "configuration missing",
                "this_fix_will_work_because": "files now exist",
            },
        }
        outcome = LLMCallOutcome(
            data=strategy,
            used_fallback=False,
            attempts=1,
        )

        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ),
            mock.patch.object(
                graph,
                "execute_command",
                return_value=mock.MagicMock(success=True, stderr=""),
            ) as execute,
            mock.patch.object(
                graph, "apply_patch", return_value=True
            ) as patch,
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
        ):
            out = graph.fixer_node(state)

        # One sandbox call for the checked create_file write (it
        # makes the directory too) and one for the command.
        self.assertEqual(execute.call_count, 2)
        patch.assert_called_once()
        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertEqual(len(out.fixes_attempted[-1].changes_made), 3)


class TestTerminalNodes(unittest.TestCase):
    """Tests for supervisor, escalation, learning, and finish nodes."""

    def _plan(self):
        """Return a minimal successful build plan."""
        return BuildPlan(
            build_system="make",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )

    def test_supervisor_node_logs_scripted_eval(self) -> None:
        """Supervisor node is a lightweight state-preserving audit pass."""
        from src import graph

        state = _state(build_plan=self._plan())
        out = graph.supervisor_node(state)

        self.assertIs(out, state)
        self.assertEqual(out.current_agent, AgentRole.SUPERVISOR)
        self.assertGreater(out.scripted_ops_count, 0)

    def test_escalate_node_builds_report_with_recent_context(self) -> None:
        """Escalation report includes fixes, arch issues, and failures."""
        from src import graph

        state = _state(build_plan=self._plan())
        state.attempt_count = 2
        state.api_cost_usd = 0.25
        state.arch_specific_code = [
            ArchSpecificCode("src/simd.c", 8, "_mm_add_ps", "x86", "high")
        ]
        state.add_fix_attempt(
            FixAttempt(
                error_category=ErrorCategory.COMPILATION,
                strategy="disable simd",
                changes_made=["Executed: make"],
                success=False,
            )
        )
        state.add_error(
            create_error_record(
                "compiler failed",
                ErrorCategory.COMPILATION,
                FailureSeverity.HIGH,
                command="make",
            )
        )

        out = graph.escalate_node(state)

        report = out.context_cache["escalation_report"]
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)
        self.assertIn("disable simd", report)
        self.assertIn("src/simd.c:8", report)
        self.assertIn("Recent Failures", report)

    def test_finish_node_uses_summarizer_response(self) -> None:
        """Finish node stores the summarizer recipe and logs real usage."""
        from src import graph

        response = mock.MagicMock(content="## Executive Summary\nDone.")
        llm = mock.MagicMock(model_name="mock-summary")
        state = _state(build_plan=self._plan())
        state.curated_artifacts = [
            {
                "role": "primary",
                "type": "binary",
                "architecture": "RISC-V",
                "filepath": "bin/tool",
            },
            {
                "role": "secondary",
                "type": "binary",
                "architecture": "RISC-V",
                "filepath": "bin/test-tool",
            },
        ]

        with (
            mock.patch.object(graph, "get_model_for_role", return_value=llm),
            mock.patch.object(graph, "invoke_llm", return_value=response),
            mock.patch.object(graph, "response_usage", return_value=(10, 5)),
            mock.patch.object(graph, "response_cost", return_value=0.02),
            mock.patch.object(graph, "log_llm_call") as log_call,
            mock.patch.object(graph, "_save_learning_data") as save_learning,
        ):
            out = graph.finish_node(state)

        self.assertEqual(out.porting_recipe, response.content)
        self.assertEqual(out.api_tokens_in, 10)
        self.assertEqual(out.api_tokens_out, 5)
        self.assertAlmostEqual(out.api_cost_usd, 0.02)
        log_call.assert_called_once()
        save_learning.assert_called_once_with(state)

    def test_finish_node_runs_package_tests_once_without_gating(
        self,
    ) -> None:
        """A failing test suite is recorded and the port stays SUCCESS."""
        from src import graph

        state = _state(build_plan=self._plan())
        failed = {"status": "failed", "framework": "make check"}
        with (
            mock.patch.object(
                graph, "_run_package_tests", return_value=failed
            ) as run_tests,
            mock.patch.object(
                graph, "get_model_for_role", side_effect=RuntimeError("x")
            ),
            mock.patch.object(graph, "_save_learning_data"),
        ):
            out = graph.finish_node(state)
            graph.finish_node(out)

        run_tests.assert_called_once()
        self.assertEqual(out.package_tests, failed)
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        self.assertIn(
            "Package tests (not gating):** failed", out.porting_recipe
        )

    def test_finish_node_falls_back_when_summarizer_fails(self) -> None:
        """Finish node emits a deterministic recipe if the LLM fails."""
        from src import graph

        state = _state(build_plan=self._plan())
        with (
            mock.patch.object(
                graph, "get_model_for_role", return_value=mock.MagicMock()
            ),
            mock.patch.object(
                graph, "invoke_llm", side_effect=RuntimeError("offline")
            ),
            mock.patch.object(graph, "_save_learning_data") as save_learning,
        ):
            out = graph.finish_node(state)

        self.assertIn("# RISC-V Porting Recipe", out.porting_recipe)
        self.assertIn("make", out.porting_recipe)
        save_learning.assert_called_once_with(state)

    def test_finish_node_falls_back_on_a_refusal(self) -> None:
        """A refused or empty summary leaves the deterministic recipe."""
        from src import graph

        for meta in ({"stop_reason": "refusal"}, {"stop_reason": "end_turn"}):
            state = _state(build_plan=self._plan())
            response = mock.MagicMock(content="", response_metadata=meta)
            with (
                mock.patch.object(
                    graph, "get_model_for_role", return_value=mock.MagicMock()
                ),
                mock.patch.object(graph, "invoke_llm", return_value=response),
                mock.patch.object(
                    graph, "response_usage", return_value=(10, 0)
                ),
                mock.patch.object(graph, "response_cost", return_value=0.0),
                mock.patch.object(graph, "log_llm_call"),
                mock.patch.object(graph, "_save_learning_data"),
            ):
                out = graph.finish_node(state)

            self.assertIn("# RISC-V Porting Recipe", out.porting_recipe)
            refused = meta["stop_reason"] == "refusal"
            self.assertEqual("llm_refused" in out.context_cache, refused)

    def test_finish_node_skips_the_llm_after_a_refusal(self) -> None:
        from src import graph

        state = _state(build_plan=self._plan())
        state.context_cache["llm_refused"] = "fixer"
        with (
            mock.patch.object(graph, "invoke_llm") as invoke,
            mock.patch.object(graph, "_save_learning_data"),
        ):
            out = graph.finish_node(state)

        invoke.assert_not_called()
        self.assertIn("# RISC-V Porting Recipe", out.porting_recipe)

    def test_learned_fixer_example_needs_actions_and_the_error_line(
        self,
    ) -> None:
        from src import graph

        state = _state(build_plan=self._plan())
        state.build_status = BuildStatus.SUCCESS
        state.artifact_verification = {"status": "verified"}
        state.curated_artifacts = [{"filepath": "bin/tool", "role": "primary"}]
        for strategy, change in (
            ("add stub", "Created file: doc.go"),
            ("link support", "Executed: make -C support"),
        ):
            state.add_fix_attempt(
                FixAttempt(
                    error_category=ErrorCategory.COMPILATION,
                    strategy=strategy,
                    changes_made=[change],
                    success=True,
                    error_message=(
                        "Build command failed: make gawk (exit 2) - "
                        "awkgram.y: warning: 36 shift/reduce conflicts\n"
                        "make: *** No rule to make target 'support/x.a'."
                    ),
                )
            )
        with (
            mock.patch.object(graph, "save_learned_example") as save_example,
            mock.patch.object(graph, "save_to_recipe_cache"),
        ):
            graph._save_learning_data(state)
        fixer = [
            c.args[1]
            for c in save_example.call_args_list
            if c.args[0] == "fixer"
        ]
        self.assertEqual(
            [f["fix"]["strategy"] for f in fixer], ["link support"]
        )
        self.assertEqual(
            fixer[0]["error_context"]["error_message"],
            "make: *** No rule to make target 'support/x.a'.",
        )

    def test_save_learning_data_writes_examples_and_recipe_cache(self) -> None:
        """Successful states are learned as scout/fixer/builder examples."""
        from src import graph

        state = _state(build_plan=self._plan())
        # Only a verified port is learned (the learning gate).
        state.build_status = BuildStatus.SUCCESS
        state.artifact_verification = {"status": "verified"}
        state.build_artifacts = [
            {
                "filepath": "bin/tool",
                "type": "binary",
                "architecture": "riscv64",
            }
        ]
        state.context_cache["go_main_info"] = {"has_main": True}
        state.dependencies = DependencyInfo(
            system_packages=["zlib-dev", "cmake", "zlib-dev"]
        )
        state.patches_generated = ["--- a/file.c\n+++ b/file.c\n"]
        # A later, unrelated error: the fixer example must keep the
        # error its own fix solved.
        state.last_error = "later unrelated error"
        state.add_fix_attempt(
            FixAttempt(
                error_category=ErrorCategory.COMPILATION,
                strategy="patch header",
                changes_made=["Executed: sed -i s/a/b/ file.c"],
                success=True,
                error_message="fatal error: a.h (x)",
            )
        )

        curated = [
            {
                "filepath": "bin/tool",
                "type": "binary",
                "architecture": "RISC-V",
                "role": "primary",
            }
        ]
        with (
            mock.patch.object(
                graph, "get_model_for_role", return_value=mock.MagicMock()
            ),
            mock.patch(
                "src.artifact_curator.curate_artifacts", return_value=curated
            ) as curate,
            mock.patch.object(graph, "save_learned_example") as save_example,
            mock.patch.object(graph, "save_to_recipe_cache") as save_cache,
        ):
            graph._save_learning_data(state)

        curate.assert_called_once()
        self.assertEqual(state.curated_artifacts, curated)
        self.assertEqual(save_example.call_count, 3)
        save_cache.assert_called_once()
        fixer_payload = save_example.call_args_list[1].args[1]
        self.assertEqual(
            fixer_payload["fix"]["actions"][0]["command"],
            "sed -i s/a/b/ file.c",
        )
        self.assertEqual(
            fixer_payload["error_pattern"], r"fatal\ error:\ a\.h\ \(x\)"
        )
        cache_kwargs = save_cache.call_args.kwargs
        self.assertEqual(cache_kwargs["dependencies"], ["cmake", "zlib-dev"])
        self.assertEqual(
            cache_kwargs["patches"], ["--- a/file.c\n+++ b/file.c\n"]
        )
        self.assertIs(
            cache_kwargs["verification"], state.artifact_verification
        )

    def test_unverified_success_is_not_learned(self) -> None:
        """An unverified or wrong-arch build never reaches the caches."""
        from src import graph

        for status in ("unverified", "wrong_arch", None):
            with self.subTest(status=status):
                state = _state(build_plan=self._plan())
                state.build_status = BuildStatus.SUCCESS
                state.artifact_verification = (
                    {"status": status} if status else None
                )
                with (
                    mock.patch.object(
                        graph, "save_learned_example"
                    ) as save_example,
                    mock.patch.object(
                        graph, "save_to_recipe_cache"
                    ) as save_cache,
                ):
                    graph._save_learning_data(state)

                save_example.assert_not_called()
                save_cache.assert_not_called()

    def test_save_learning_data_is_non_fatal(self) -> None:
        """Auto-learning swallows persistence failures."""
        from src import graph

        state = _state(build_plan=self._plan())
        # Verified, so the save (and its failure) is really reached.
        state.build_status = BuildStatus.SUCCESS
        state.artifact_verification = {"status": "verified"}
        with mock.patch.object(
            graph,
            "save_learned_example",
            side_effect=RuntimeError("disk full"),
        ):
            graph._save_learning_data(state)


class TestFixerRoundOneContinues(unittest.TestCase):
    """Round 1 continues round 0: the model sees its own request."""

    def test_round_one_sends_history_and_only_the_results(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        from src import graph

        calls = []
        thread = [
            HumanMessage(content="round 0 task"),
            AIMessage(content='{"investigate": {"commands": ["ls"]}}'),
        ]
        fix = {
            "strategies": [
                {
                    "id": 1,
                    "description": "install headers",
                    "actions": [{"type": "command", "command": "make"}],
                }
            ],
            "recommended_strategy_id": 1,
        }

        def fake_llm_call(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return LLMCallOutcome(
                    data={"investigate": {"commands": ["ls"]}},
                    used_fallback=False,
                    attempts=1,
                    messages=thread,
                )
            return LLMCallOutcome(data=fix, used_fallback=False, attempts=1)

        state = TestFixerNode._fixer_state(self)
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph, "_run_fixer_investigation", return_value="$ ls\nsrc"
            ),
            mock.patch.object(
                graph,
                "execute_command",
                return_value=mock.MagicMock(success=True),
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
        ):
            graph.fixer_node(state)

        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["history"])
        self.assertIs(calls[1]["history"], thread)
        self.assertIn("Investigation results", calls[1]["prompt"])
        self.assertNotIn("## Failure", calls[1]["prompt"])
        self.assertEqual(calls[1]["system_prompt"], calls[0]["system_prompt"])


class TestFixerEvidence(unittest.TestCase):
    """The fixer prompt carries the real exit code, phase, plan and fixes."""

    def _state(self):
        state = _state()
        state.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[
                BuildPhase(1, "configure", ["cmake -S . -B build"]),
                BuildPhase(2, "build", ["cmake --build build"]),
            ],
            total_estimated_duration="1m",
        )
        state.last_successful_phase = 1
        message = (
            "Build command failed: cmake --build build (exit 137) - "
            "In file included from x.h:1\nwarning: a\nKilled"
        )
        state.add_error(
            create_error_record(
                message,
                ErrorCategory.COMPILATION,
                command="cmake --build build",
            )
        )
        return state

    def test_exit_code_and_phase(self) -> None:
        from src import graph

        state = self._state()
        self.assertEqual(
            graph._exit_code_text(state.last_error),
            "137 (SIGKILL: the process was killed, usually out of memory)",
        )
        self.assertEqual(graph._exit_code_text("no code"), "not reported")
        self.assertEqual(
            graph._failed_phase_text(state, "cmake --build build"),
            "build (phase 2 of 2)",
        )
        plan = graph._build_plan_text(state, "cmake --build build")
        self.assertIn("1. configure (completed)", plan)
        self.assertIn("`cmake --build build`   <- failed here", plan)

    def test_previous_fixes_show_actions_and_what_followed(self) -> None:
        from src import graph

        state = self._state()
        fix = FixAttempt(
            error_category=ErrorCategory.COMPILATION,
            strategy="install headers",
            changes_made=[],
            success=False,
            action_results=["`apt-get install x` failed (exit 100): E: none"],
        )
        state.add_fix_attempt(fix)
        state.add_error(
            create_error_record(
                "Build command failed: make (exit 2) - a.c:1: error: next",
                ErrorCategory.COMPILATION,
                command="make",
            )
        )
        text = graph._previous_fixes_text(state)
        self.assertIn("Fix 1: install headers (Failed)", text)
        self.assertIn("failed (exit 100): E: none", text)
        self.assertIn("then: next error: a.c:1: error: next", text)

    def test_fixer_prompt_has_the_facts(self) -> None:
        from src import graph

        state = self._state()
        seen = {}

        def fake_llm_call(**kwargs):
            seen.update(kwargs)
            return LLMCallOutcome(data=None, used_fallback=False, attempts=1)

        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "top_level_listing", return_value="CMakeLists.txt"
            ),
        ):
            graph.fixer_node(state)
        prompt = seen["prompt"]
        self.assertIn("Exit code: 137 (SIGKILL", prompt)
        self.assertIn("Failed phase: build (phase 2 of 2)", prompt)
        self.assertIn("Killed", prompt)
        self.assertIn("### Top-level files\nCMakeLists.txt", prompt)
        self.assertNotIn("Phase: fixing", prompt)
        self.assertNotIn("N/A", prompt)
        rules = seen["system_prompt"][0]
        self.assertIn("re-runs the build plan commands", rules)

    def test_stdout_error_is_kept_when_stderr_holds_the_message(self) -> None:
        from src import graph

        configure = mock.MagicMock(
            stderr="+ cd libvpx\n+ ./configure --enable-vp8\n+ exit 1",
            stdout="Configuring for target 'riscv64-linux-gcc'\n"
            "Unrecognized toolchain 'riscv64-linux-gcc'\n"
            "Configuration failed. This could reflect a misconfiguration.",
        )
        self.assertIn(
            "Configuration failed", graph._stdout_evidence(configure)
        )
        progress = mock.MagicMock(
            stderr="a.c:1: error: x", stdout="CC a.o\nCC b.o"
        )
        self.assertEqual(graph._stdout_evidence(progress), "")
        no_stderr = mock.MagicMock(stderr="", stdout="error: in message")
        self.assertEqual(graph._stdout_evidence(no_stderr), "")

    def test_builder_records_the_stdout_tail(self) -> None:
        from src import graph

        state = _state()
        state.build_plan = BuildPlan(
            build_system="autotools",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "configure", ["./configure"])],
            total_estimated_duration="1m",
        )

        def fake_execute(cmd, *args, **kwargs):
            if cmd == "./configure":
                return CommandResult(
                    cmd, 1, "Configuration failed.", "+ ./configure", 1.0
                )
            return CommandResult(str(cmd), 0, "", "", 1.0)

        with (
            mock.patch.object(
                graph, "execute_command", side_effect=fake_execute
            ),
            mock.patch.object(graph, "predict_build_issues", return_value=[]),
        ):
            out = graph.builder_node(state)
        record = out.error_history[-1]
        self.assertEqual(record.stdout_tail, "Configuration failed.")
        self.assertNotIn("Configuration failed.", record.message)

    def test_fixer_prompt_shows_the_stdout_tail(self) -> None:
        from src import graph

        state = self._state()
        state.error_history[-1].stdout_tail = "Configuration failed."
        seen = {}

        def fake_llm_call(**kwargs):
            seen.update(kwargs)
            return LLMCallOutcome(data=None, used_fallback=False, attempts=1)

        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(graph, "top_level_listing", return_value=""),
        ):
            graph.fixer_node(state)
        self.assertIn(
            "### Standard output (the end; the output above is stderr)\n"
            "```\nConfiguration failed.\n```",
            seen["prompt"],
        )

    def test_failed_fix_command_is_recorded_with_its_output(self) -> None:
        from src import graph

        state = self._state()
        strategy = {
            "strategies": [
                {
                    "id": 1,
                    "description": "two steps",
                    "actions": [
                        {"type": "command", "command": "apt-get install x"},
                        {"type": "command", "command": "make clean"},
                    ],
                }
            ],
            "recommended_strategy_id": 1,
        }
        results = [
            mock.MagicMock(
                success=False, stderr="E: Unable to locate x", exit_code=100
            ),
            mock.MagicMock(success=True),
        ]
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph,
                "llm_call_with_validation",
                return_value=LLMCallOutcome(
                    data=strategy, used_fallback=False, attempts=1
                ),
            ),
            mock.patch.object(graph, "execute_command", side_effect=results),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(graph, "top_level_listing", return_value=""),
        ):
            out = graph.fixer_node(state)
        self.assertEqual(
            out.fixes_attempted[-1].action_results,
            [
                "`apt-get install x` failed (exit 100): E: Unable to locate x",
                "`make clean` ok",
            ],
        )


class TestStructuredFixer(unittest.TestCase):
    """The fixer schema: escalate and investigate fields, no comments."""

    def test_every_schema_object_is_closed_and_fully_required(self) -> None:
        from src import graph
        from src.artifact_curator import _CURATOR_SCHEMA

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(
                        sorted(node["required"]), sorted(node["properties"])
                    )
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        for schema in (
            graph.ANALYST_SCHEMA,
            graph.SCOUT_SCHEMA,
            graph.FIXER_SCHEMA,
            _CURATOR_SCHEMA,
        ):
            walk(schema)

    def test_fixer_prompt_has_no_json_comments(self) -> None:
        from src import graph

        text = graph.FIXER_PROMPT.format(platform_banner="b")
        self.assertNotIn("//", text)
        self.assertIn('"escalate": null', text)
        self.assertNotIn("propose\n  ESCALATE", text)
        self.assertIn("Never add a stub or placeholder source file", text)

    def _run(self, data):
        from src import graph

        state = TestFixerNode._fixer_state(self)
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph,
                "llm_call_with_validation",
                return_value=LLMCallOutcome(
                    data=data, used_fallback=False, attempts=1
                ),
            ) as call,
            mock.patch.object(
                graph,
                "execute_command",
                return_value=mock.MagicMock(success=True),
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(graph, "top_level_listing", return_value=""),
            mock.patch.object(
                graph, "apply_patch", return_value=True
            ) as patch,
        ):
            out = graph.fixer_node(state)
        return out, call, patch

    def test_escalate_ends_the_run_with_the_reason(self) -> None:
        out, call, _ = self._run(
            {
                "investigate": None,
                "escalate": {"reason": "cabal is not available"},
                "strategies": [],
                "recommended_strategy_id": 0,
                "reflection": {"root_cause": "x"},
            }
        )
        self.assertEqual(call.call_count, 1)
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)
        self.assertIn(
            "FIXER escalated: cabal is not available",
            out.error_history[-1].message,
        )
        validator = call.call_args.kwargs["validator"]
        self.assertFalse(
            validator({"escalate": {"reason": " "}, "strategies": []}).ok
        )

    def test_null_investigate_with_strategies_is_a_fix(self) -> None:
        out, call, _ = self._run(
            {
                "investigate": None,
                "escalate": None,
                "strategies": [
                    {
                        "id": 1,
                        "description": "make clean",
                        "actions": [
                            {
                                "type": "command",
                                "command": "make clean",
                                "path": "",
                                "content": "",
                            }
                        ],
                    }
                ],
                "recommended_strategy_id": 1,
                "reflection": {"root_cause": "stale objects"},
            }
        )
        self.assertEqual(call.call_count, 1)
        self.assertEqual(out.build_status, BuildStatus.PENDING)

    def test_patch_names_its_file_in_path(self) -> None:
        _, _, patch = self._run(
            {
                "investigate": None,
                "escalate": None,
                "strategies": [
                    {
                        "id": 1,
                        "description": "guard",
                        "actions": [
                            {
                                "type": "patch",
                                "command": "",
                                "path": "src/a.c",
                                "content": "--- a/src/a.c\n+++ b/src/a.c\n",
                            }
                        ],
                    }
                ],
                "recommended_strategy_id": 1,
                "reflection": {"root_cause": "x86 only"},
            }
        )
        self.assertTrue(patch.call_args.kwargs["filepath"].endswith("src/a.c"))


class TestShortFields(unittest.TestCase):
    """Analyst fields are bounded at word boundaries; unused fields go."""

    def test_shorten_keeps_whole_words(self) -> None:
        from src.graph import _shorten

        text = "the agent should locate them before assuming x86 only paths"
        out = _shorten(text, 40)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 40)
        self.assertIn(out[:-1].split()[-1], text.split())
        self.assertEqual(_shorten("  a  b ", 10), "a b")

    def test_schemas_ask_only_for_read_fields(self) -> None:
        from src import graph

        self.assertNotIn("complexity", graph.ANALYST_SCHEMA["properties"])
        self.assertEqual(
            sorted(graph.SCOUT_SCHEMA["properties"]),
            ["build_system", "phases"],
        )
        self.assertEqual(
            list(graph.FIXER_SCHEMA["properties"]["reflection"]["properties"]),
            ["root_cause"],
        )

    def test_root_cause_reaches_the_next_round(self) -> None:
        from src import graph

        state = _state()
        state.add_fix_attempt(
            FixAttempt(
                error_category=ErrorCategory.COMPILATION,
                strategy="add guard",
                changes_made=["Executed: sed -i x a.c"],
                success=False,
                root_cause="x86 intrinsics in a.c",
            )
        )
        self.assertIn(
            "believed cause: x86 intrinsics in a.c",
            graph._previous_fixes_text(state),
        )


class TestScoutWithoutDuplicates(unittest.TestCase):
    """The scout reuses the analyst evidence and sends each doc once."""

    def test_scout_reuses_evidence_and_skips_the_readme_copy(self) -> None:
        from src import graph

        state = TestScoutNode._scout_state(self)
        state.error_history = []
        state.last_error = None
        state.file_content_cache = {
            "/workspace/repos/b/README.md": "readme text",
            "/workspace/repos/b/INSTALL": "install text",
        }
        state.context_cache["build_evidence"] = (
            "### Top-level files\nREADME.md\n\n### README.md (excerpt)\nreadme"
        )
        seen = {}

        def fake_llm_call(**kwargs):
            seen.update(kwargs)
            return LLMCallOutcome(data=None, used_fallback=False, attempts=1)

        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={"architecture": "riscv64"},
            ),
            mock.patch.object(graph, "collect_build_evidence") as collect,
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
        ):
            graph.scout_node(state)
        collect.assert_not_called()
        self.assertIn("## INSTALL\ninstall text", seen["prompt"])
        self.assertNotIn("## README.md\nreadme text", seen["prompt"])

    def test_rescout_after_a_failure_reads_the_tree_again(self) -> None:
        from src import graph

        state = TestScoutNode._scout_state(self)
        state.context_cache["build_evidence"] = "old evidence"
        state.add_error(
            create_error_record(
                "Build command failed: make (exit 2) - x",
                ErrorCategory.UNKNOWN,
            )
        )
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph,
                "llm_call_with_validation",
                return_value=LLMCallOutcome(
                    data=None, used_fallback=False, attempts=1
                ),
            ),
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={"architecture": "riscv64"},
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="new"
            ) as collect,
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
        ):
            graph.scout_node(state)
        collect.assert_called_once()


class TestLearningOnlyRealFixes(unittest.TestCase):
    """A fix that did not change the error is not learned as a success."""

    def test_only_fixes_that_moved_the_error_count(self) -> None:
        from datetime import datetime, timedelta

        from src import graph

        state = _state()
        same = "Build command failed: make (exit 2) - a.c:1: error: zlib.h"
        t0 = datetime(2026, 10, 16, 12, 0, 0)

        def fix(minute, message):
            f = FixAttempt(
                error_category=ErrorCategory.COMPILATION,
                strategy=f"fix {minute}",
                changes_made=["Executed: true"],
                success=False,
                error_message=message,
            )
            f.timestamp = t0 + timedelta(minutes=minute)
            return f

        def error(minute, message):
            e = create_error_record(message, ErrorCategory.COMPILATION)
            e.timestamp = t0 + timedelta(minutes=minute)
            return e

        state.fixes_attempted = [fix(1, same), fix(3, same)]
        state.error_history = [error(0, same), error(2, same)]
        state.build_plan = BuildPlan(
            build_system="make",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["true"])],
            total_estimated_duration="1m",
        )
        state.last_successful_phase = 1
        with mock.patch.object(
            graph,
            "execute_command",
            return_value=mock.MagicMock(success=True, stdout="", stderr=""),
        ):
            graph.builder_node(state)
        first, last = state.fixes_attempted
        self.assertFalse(first.success)
        self.assertEqual(
            first.build_result, "the error did not change after this fix"
        )
        self.assertTrue(last.success)

    def test_error_regex_tolerates_paths_and_numbers(self) -> None:
        import re

        from src import graph

        pattern = graph._error_regex(
            "/workspace/repos/a/src/x.c:12:3: error: zlib.h: No such file"
        )
        self.assertTrue(
            re.search(
                pattern,
                "/workspace/repos/b/lib/y.c:99:1: error: zlib.h: no such file",
                re.IGNORECASE,
            )
        )


class TestCurateBeforeSummary(unittest.TestCase):
    """The summarizer sees the curated list, without build probes."""

    def test_summary_prompt_has_no_cmake_probes(self) -> None:
        from src import graph

        state = _state(
            build_plan=BuildPlan(
                build_system="cmake",
                build_system_confidence=0.9,
                phases=[BuildPhase(1, "build", ["make"])],
                total_estimated_duration="1m",
            )
        )
        state.build_artifacts = [
            {"filepath": "/r/build/libz.a", "type": "library_static"},
            {
                "filepath": "/r/build/CMakeFiles/3.22/CompilerIdC/a.out",
                "type": "binary",
            },
        ]
        seen = {}

        def fake_invoke(llm, messages, timeout=None):
            seen["prompt"] = messages[0].content
            return mock.MagicMock(content="## Executive Summary\nok")

        with (
            mock.patch.object(
                graph, "get_model_for_role", return_value=mock.MagicMock()
            ),
            mock.patch.object(graph, "invoke_llm", side_effect=fake_invoke),
            mock.patch.object(graph, "response_usage", return_value=(1, 1)),
            mock.patch.object(graph, "response_cost", return_value=0.0),
            mock.patch.object(graph, "log_llm_call"),
            mock.patch.object(
                graph, "_run_package_tests", return_value={"status": "skipped"}
            ),
            mock.patch.object(graph, "_save_learning_data"),
        ):
            graph.finish_node(state)
        self.assertIn("libz.a", seen["prompt"])
        self.assertNotIn("CompilerIdC", seen["prompt"])

    def test_curation_runs_once_even_when_it_keeps_nothing(self) -> None:
        from src import graph

        state = _state()
        state.build_artifacts = [{"filepath": "/r/x", "type": "binary"}]
        with (
            mock.patch.object(graph, "get_model_for_role", return_value=None),
            mock.patch(
                "src.artifact_curator.curate_artifacts", return_value=[]
            ) as curate,
        ):
            graph._curate_artifacts(state)
            graph._curate_artifacts(state)
        curate.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class TestBuilderCMakeCacheCleanup(unittest.TestCase):
    """The stale build/ removal must target the container path."""

    def test_rm_uses_container_path_not_host_path(self) -> None:
        """The rm runs in the sandbox, so it needs the /workspace path."""
        from src import graph

        state = _state()
        state.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "configure", ["cmake -S . -B build"])],
            total_estimated_duration="1m",
        )
        host_build = "/host/repos/b/build"
        calls = []

        def fake_execute(cmd, *args, **kwargs):
            calls.append(cmd)
            text = cmd if isinstance(cmd, str) else " ".join(cmd)
            if text.startswith("cmake -S . -B build") and (
                "CMAKE_C_COMPILER" not in text
            ):
                return CommandResult(
                    text,
                    1,
                    "",
                    "CMAKE_C_COMPILER: gcc is not a full path to an "
                    "existing compiler tool",
                    1.0,
                )
            return CommandResult(text, 0, "", "", 1.0)

        real_isdir = graph.os.path.isdir
        with (
            mock.patch.object(
                graph, "execute_command", side_effect=fake_execute
            ),
            mock.patch.object(
                graph,
                "_to_host_path",
                side_effect=lambda p: p.replace("/workspace", "/host", 1),
            ),
            mock.patch.object(
                graph.os.path,
                "isdir",
                side_effect=lambda p: p == host_build or real_isdir(p),
            ),
            mock.patch.object(graph, "predict_build_issues", return_value=[]),
        ):
            graph.builder_node(state)

        rm_calls = [c for c in calls if str(c).startswith("rm -rf")]
        self.assertEqual(rm_calls, ["rm -rf /workspace/repos/b/build"])


class TestCachedRecipeNode(unittest.TestCase):
    """A verified cached recipe becomes the build plan (no LLM)."""

    _RECIPE = {
        "build_system": "cmake",
        "last_built": "2026-10-02T15:35:35",
        "build_duration_seconds": 31.2,
        "build_plan": {
            "phases": [
                {"name": "deps", "commands": ["apt-get install -y cmake"]},
                {"name": "empty", "commands": ["  "]},
                {
                    "name": "build",
                    "commands": [
                        "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release",
                        "cmake --build build",
                    ],
                },
            ]
        },
        "dependencies": ["cmake"],
        "patches": ["--- a/x.c\n+++ b/x.c\n@@ -1 +1 @@\n-a\n+b\n"],
        "artifacts": [
            {
                "type": "library_shared",
                "path": "/workspace/repos/b/build/libz.so.1",
            }
        ],
        "recipe_markdown": "# Recipe\n\n## Executive Summary\nThe zlib "
        "build succeeded on RISC-V. No patch was needed.",
        "verification": {
            "status": "verified",
            "source_commit": "767c4c947852e143f582",
        },
    }

    def test_recipe_becomes_the_plan(self) -> None:
        """Phases, analysis and patches come from the cache entry."""
        from src import graph

        state = _state(cached_recipe=dict(self._RECIPE))
        head = CommandResult("git", 0, "aaaaaaaaaaaaaaaa\n", "", 0.1)
        with (
            mock.patch.object(
                graph, "apply_patch", return_value=True
            ) as patch,
            mock.patch.object(graph, "execute_command", return_value=head),
            mock.patch.object(graph, "invoke_llm") as llm,
        ):
            out = graph.cached_recipe_node(state)

        llm.assert_not_called()
        plan = out.build_plan
        self.assertEqual(plan.build_system, "cmake")
        self.assertEqual([p.name for p in plan.phases], ["deps", "build"])
        self.assertEqual([p.id for p in plan.phases], [1, 2])
        self.assertEqual(
            plan.phases[1].commands[0],
            "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release",
        )
        analysis = out.package_analysis
        self.assertEqual(
            analysis.purpose, "The zlib build succeeded on RISC-V."
        )
        self.assertFalse(analysis.llm_grounded)
        self.assertFalse(analysis.needs_custom_plan)
        self.assertEqual(analysis.expected_artifacts, ["libz.so.1"])
        patch.assert_called_once_with(
            self._RECIPE["patches"][0], cwd=state.repo_path, use_docker=True
        )
        self.assertEqual(out.patches_generated, self._RECIPE["patches"])
        note = out.context_cache["cached_recipe_note"]
        self.assertIn("3 commands in 2 phases, 1 of 1 patches applied", note)
        self.assertIn("verified on commit 767c4c947852", note)

    def test_recipe_without_commands_plans_again(self) -> None:
        """No command: the cache entry is dropped and nothing is set."""
        from src import graph

        state = _state(
            cached_recipe={
                "build_plan": {"phases": [{"name": "x", "commands": []}]}
            }
        )
        out = graph.cached_recipe_node(state)
        self.assertIsNone(out.cached_recipe)
        self.assertIsNone(out.build_plan)
        self.assertIsNone(out.package_analysis)


class TestOptimizeNode(unittest.TestCase):
    """optimize_node never turns a finished port into a failure."""

    def _state(self):
        """Return a verified SUCCESS state with --optimize and ctest passed."""
        state = _state(optimize_requested=True)
        state.build_status = BuildStatus.SUCCESS
        state.artifact_verification = {"status": "verified"}
        state.package_tests = {
            "status": "passed",
            "framework": "ctest",
            "command": "ctest --test-dir build",
        }
        return state

    def test_records_the_result_and_tests_variant_trees(self) -> None:
        """The record lands in state; gate 2 reuses the baseline command."""
        from src import graph

        state = self._state()
        seen = {}

        def fake_run(run_state, agent, run_tests, progress):
            seen["tests"] = run_tests("/w/b.opt/v1", {"CC": "cc"})
            seen["agent"] = agent
            return {"status": "checked", "kind": "qemu", "variants": []}

        with (
            mock.patch("src.optimize.run_optimization", side_effect=fake_run),
            mock.patch.object(
                graph,
                "_run_package_tests",
                return_value={"status": "passed"},
            ) as run_tests,
            mock.patch.object(graph, "record_optimization") as record,
        ):
            out = graph.optimize_node(state)

        self.assertEqual(out.optimization["status"], "checked")
        self.assertTrue(callable(seen["agent"]))
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        self.assertEqual(seen["tests"], {"status": "passed"})
        run_tests.assert_called_once_with(
            state,
            repo_path="/w/b.opt/v1",
            extra_env={"CC": "cc"},
            detected=("ctest", "ctest --test-dir build"),
        )
        record.assert_called_once_with(
            state.repo_name, out.optimization, state.machine_profile
        )

    def test_a_crash_is_recorded_not_escalated(self) -> None:
        """An exception in performance mode keeps the port a SUCCESS."""
        from src import graph

        state = self._state()
        with (
            mock.patch(
                "src.optimize.run_optimization",
                side_effect=RuntimeError("disk full"),
            ),
            mock.patch.object(graph, "record_optimization"),
        ):
            out = graph.optimize_node(state)

        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        self.assertEqual(out.optimization["status"], "error")
        self.assertIn("disk full", out.optimization["reason"])
        self.assertEqual(out.error_history, [])

    def test_agent_asks_the_other_role_after_a_timeout(self) -> None:
        """Harness code gets a long timeout and a second set of models."""
        from src import graph
        from src.llm_helpers import LLMCallOutcome

        state = self._state()
        pools = {
            AgentRole.BUILDER: ["coder"],
            AgentRole.PLANNER: ["planner"],
        }
        late = LLMCallOutcome(
            data=None,
            used_fallback=False,
            attempts=1,
            last_error="LLM invocation raised: LLM call timed out after 240s",
        )
        answer = LLMCallOutcome(
            data={"source": "x"}, used_fallback=False, attempts=1
        )
        wrong = LLMCallOutcome(
            data=None,
            used_fallback=False,
            attempts=3,
            last_error="the source calls no function",
        )
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                side_effect=lambda role: pools[role],
            ),
            mock.patch.object(
                graph,
                "llm_call_with_validation",
                side_effect=[late, answer, wrong],
            ) as call,
        ):
            agent = graph._optimizer_agent(state)
            data, _ = agent("harness-c", "write a harness", lambda data: None)
            missed, error = agent(
                "harness-c", "write a harness", lambda data: None
            )
        self.assertEqual(data, {"source": "x"})
        first, second, third = [c.kwargs for c in call.call_args_list]
        self.assertEqual(first["llm"], "coder")
        self.assertEqual(first["timeout"], graph.OPTIMIZER_CODE_TIMEOUT)
        self.assertTrue(first["rotate_on_timeout"])
        self.assertEqual(second["llm"], "planner")
        self.assertEqual(second["fallback_llms"], [])
        # A wrong answer is no timeout: the other role is not asked.
        self.assertEqual(third["llm"], "coder")
        self.assertEqual(
            (missed, error), (None, "the source calls no function")
        )
        self.assertEqual(call.call_count, 3)
        self.assertEqual(state.api_calls_made, 5)

    def test_agent_uses_the_validated_llm_call(self) -> None:
        """The agent's question goes through llm_call_with_validation."""
        from src import graph
        from src.llm_helpers import LLMCallOutcome

        state = self._state()
        pools = {
            AgentRole.BUILDER: ["coder", "coder-2"],
            AgentRole.PLANNER: ["planner"],
        }
        outcome = LLMCallOutcome(
            data={"source": "x"},
            used_fallback=False,
            attempts=2,
            input_tokens=7,
            output_tokens=3,
            cost_usd=0.01,
        )
        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                side_effect=lambda role: pools[role],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ) as call,
        ):
            agent = graph._optimizer_agent(state)
            data, error = agent(
                "harness-c", "write a harness", lambda data: None
            )
            agent(
                "workload-program", "choose a run", lambda data: "bad binary"
            )

        self.assertEqual((data, error), ({"source": "x"}, ""))
        first = call.call_args_list[0].kwargs
        self.assertEqual(first["llm"], "coder")
        self.assertEqual(first["fallback_llms"], ["coder-2"])
        self.assertEqual(first["role"], "optimizer")
        self.assertEqual(
            first["audit_metadata"]["phase"], "optimize-harness-c"
        )
        self.assertTrue(first["validator"]({}).ok)
        second = call.call_args_list[1].kwargs
        self.assertEqual(second["llm"], "planner")
        result = second["validator"]({})
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "bad binary")
        self.assertEqual(state.api_tokens_in, 14)
        self.assertEqual(state.api_calls_made, 4)

    def test_the_agent_stops_at_the_cost_cap(self) -> None:
        """One workload source can ask 8 rounds; the step has a cap."""
        from src import graph
        from src.llm_helpers import LLMCallOutcome

        state = self._state()
        outcome = LLMCallOutcome(
            data={"source": "x"},
            used_fallback=False,
            attempts=1,
            cost_usd=0.3,
        )
        with (
            mock.patch.dict(os.environ, {"ATESOR_OPTIMIZE_COST_CAP": "0.5"}),
            mock.patch.object(
                graph, "get_model_pool_for_role", return_value=["coder"]
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", return_value=outcome
            ) as call,
        ):
            agent = graph._optimizer_agent(state)
            answers = [
                agent("harness-c", "write a harness", lambda data: None)
                for _ in range(3)
            ]
        # $0.30, then $0.60: the third question is not sent.
        self.assertEqual(call.call_count, 2)
        data, error = answers[2]
        self.assertIsNone(data)
        self.assertIn("cost cap ($0.60 of $0.50", error)
        self.assertAlmostEqual(state.api_cost_usd, 0.6)


class TestPromptSplit(unittest.TestCase):
    """Static text goes to the system prompt, package data to the user."""

    def _capture(self, node, state, extra_patches=()):
        """Run ``node`` and return the llm_call_with_validation kwargs."""
        from src import graph

        seen = {}

        def fake_llm_call(**kwargs):
            seen.update(kwargs)
            return LLMCallOutcome(data=None, used_fallback=False, attempts=1)

        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_llm_call
            ),
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={"architecture": "riscv64"},
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="EVIDENCE"
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
        ):
            node(state)
        return seen

    def test_analyst(self) -> None:
        from src import graph

        seen = self._capture(graph.analyst_node, _state())
        self.assertIn("## Your task", seen["system_prompt"])
        self.assertIn("## Active sandbox", seen["system_prompt"])
        self.assertNotIn("EVIDENCE", seen["system_prompt"])
        self.assertIn("EVIDENCE", seen["prompt"])
        self.assertNotIn("## Your task", seen["prompt"])

    def test_scout(self) -> None:
        from src import graph

        state = TestScoutNode._scout_state(self)
        state.repo_name = "quux-tool"
        seen = self._capture(graph.scout_node, state)
        rules, knowledge = seen["system_prompt"]
        self.assertIn("## Build plan rules", rules)
        self.assertNotIn(state.repo_name, rules)
        self.assertTrue(knowledge.startswith("## Platform knowledge"))
        self.assertIn("EVIDENCE", seen["prompt"])
        self.assertIn(state.repo_name, seen["prompt"])
        self.assertNotIn("## Build plan rules", seen["prompt"])

    def test_fixer(self) -> None:
        from src import graph

        state = TestFixerNode._fixer_state(self)
        seen = self._capture(graph.fixer_node, state)
        rules, knowledge = seen["system_prompt"]
        self.assertIn("## Critical rules", rules)
        self.assertNotIn("zlib.h", rules)
        self.assertTrue(knowledge.startswith("## Platform knowledge"))
        self.assertIn("zlib.h", seen["prompt"])
        self.assertNotIn("## Critical rules", seen["prompt"])
        # Same build system → same knowledge text → a cache hit.
        again = self._capture(
            graph.fixer_node, TestFixerNode._fixer_state(self)
        )
        self.assertEqual(again["system_prompt"], seen["system_prompt"])


class TestPlanCommandAction(unittest.TestCase):
    """The fixer can change a plan command (run 80: 17 wrong targets)."""

    def _state(self):
        state = TestFixerNode._fixer_state(self)
        state.build_plan = BuildPlan(
            build_system="go",
            build_system_confidence=0.8,
            phases=[
                BuildPhase(1, "setup", ["apt-get update"]),
                BuildPhase(2, "build", ["go build ./cmd/tinygo"]),
            ],
            total_estimated_duration="1m",
        )
        state.last_successful_phase = 2
        return state

    def test_plan_command_replaces_the_command_and_reruns_its_phase(
        self,
    ) -> None:
        state = self._state()
        out = TestFixerNode._run_fixer_with(
            self,
            state,
            [
                {
                    "type": "plan_command",
                    "replaces": "go build ./cmd/tinygo",
                    "command": "cd v2 && go build ./cmd/tinygo",
                    "path": "",
                    "content": "",
                }
            ],
        )
        self.assertEqual(
            out.build_plan.phases[1].commands,
            ["cd v2 && go build ./cmd/tinygo"],
        )
        self.assertEqual(out.last_successful_phase, 1)
        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertIn("plan:", out.fixes_attempted[-1].action_results[0])

    def test_the_same_command_text_is_no_change(self) -> None:
        """nushell: "rerun unchanged" gave a third, doomed build."""
        from src import graph

        state = self._state()
        state.last_successful_phase = 1
        out = TestFixerNode._run_fixer_with(
            self,
            state,
            [
                {
                    "type": "plan_command",
                    "replaces": "go build ./cmd/tinygo",
                    "command": "go build  ./cmd/tinygo",
                    "path": "",
                    "content": "",
                }
            ],
        )
        fix = out.fixes_attempted[-1]
        self.assertEqual(fix.build_result, "no change applied")
        self.assertIn("unchanged (the same text)", fix.action_results[0])
        self.assertEqual(out.build_status, BuildStatus.FIXING)
        self.assertEqual(graph.route_fix_result(out), "fix_node")
        self.assertEqual(out.context_cache["reasks_since_build"], 1)

    def test_the_same_text_with_a_real_change_reruns_its_phase(self) -> None:
        """Install a library, then run the completed configure again."""
        state = self._state()
        state.build_plan = BuildPlan(
            build_system="autotools",
            build_system_confidence=0.8,
            phases=[
                BuildPhase(1, "setup", ["apt-get update"]),
                BuildPhase(2, "configure", ["./configure"]),
                BuildPhase(3, "build", ["make"]),
            ],
            total_estimated_duration="1m",
        )
        # configure passed; make failed.
        state.last_successful_phase = 2
        out = TestFixerNode._run_fixer_with(
            self,
            state,
            [
                {
                    "type": "command",
                    "command": "apt-get install -y libfoo-dev",
                    "path": "",
                    "content": "",
                },
                {
                    "type": "plan_command",
                    "replaces": "./configure",
                    "command": "./configure",
                    "path": "",
                    "content": "",
                },
            ],
        )
        self.assertIsNone(out.fixes_attempted[-1].build_result)
        self.assertEqual(out.build_status, BuildStatus.PENDING)
        self.assertEqual(out.last_successful_phase, 1)

    def test_validator_checks_replaces_against_the_plan(self) -> None:
        from src.graph import validate_fixer_response

        def reply(old, new):
            return {
                "strategies": [
                    {
                        "id": 1,
                        "description": "x",
                        "actions": [
                            {
                                "type": "plan_command",
                                "replaces": old,
                                "command": new,
                            }
                        ],
                    }
                ],
                "recommended_strategy_id": 1,
            }

        plan = ["make", "make install"]
        self.assertTrue(
            validate_fixer_response(reply("make", "make -C src"), plan)[0]
        )
        ok, why = validate_fixer_response(reply("make all", "make"), plan)
        self.assertFalse(ok)
        self.assertIn("`make install`", why)
        ok, why = validate_fixer_response(reply("make", "curl x | sh"), plan)
        self.assertFalse(ok)
        self.assertFalse(
            validate_fixer_response(reply("make", "make"), None)[0]
        )

    def test_schema_and_prompt_offer_plan_command(self) -> None:
        from src import graph

        action = graph.FIXER_SCHEMA["properties"]["strategies"]["items"][
            "properties"
        ]["actions"]["items"]
        self.assertIn("plan_command", action["properties"]["type"]["enum"])
        self.assertIn("replaces", action["required"])
        self.assertIn("`plan_command` action", graph.FIXER_PROMPT)


class TestTimeLimitStop(unittest.TestCase):
    """A stop at the command time limit is resumed, not run at -j1."""

    def test_builder_resumes_the_command_that_was_stopped(self) -> None:
        from src import graph
        from src.state import TIME_LIMIT_MARK

        state = _state()
        state.build_plan = BuildPlan(
            build_system="cargo",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["cargo build --release"])],
            total_estimated_duration="1m",
        )
        runs = []

        def fake_execute(cmd, *args, **kwargs):
            runs.append(cmd)
            if len(runs) == 1:
                return CommandResult(
                    cmd,
                    137,
                    "",
                    TIME_LIMIT_MARK + " (1770 s).",
                    1770.0,
                    timed_out=True,
                )
            return CommandResult(cmd, 0, "", "", 60.0)

        with (
            mock.patch.object(
                graph, "execute_command", side_effect=fake_execute
            ),
            mock.patch.object(graph, "predict_build_issues", return_value=[]),
        ):
            out = graph.builder_node(state)
        self.assertEqual(runs[:2], ["cargo build --release"] * 2)
        self.assertFalse(
            any("-j 1" in r or "CARGO_BUILD_JOBS=1" in r for r in runs)
        )
        self.assertNotEqual(out.build_status, BuildStatus.FAILED)

    def test_resume_gets_a_longer_time_limit(self) -> None:
        """One crate can compile longer than the normal limit."""
        from src import graph
        from src.state import TIME_LIMIT_MARK

        state = _state()
        state.build_plan = BuildPlan(
            build_system="cargo",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["cargo build --release"])],
            total_estimated_duration="1m",
        )
        limits = []

        def fake_execute(cmd, *args, **kwargs):
            limits.append(kwargs.get("timeout"))
            if len(limits) == 1:
                return CommandResult(
                    cmd,
                    137,
                    "",
                    TIME_LIMIT_MARK + " (1770 s).",
                    1770.0,
                    timed_out=True,
                )
            return CommandResult(cmd, 0, "", "", 60.0)

        with (
            mock.patch.object(
                graph, "execute_command", side_effect=fake_execute
            ),
            mock.patch.object(graph, "predict_build_issues", return_value=[]),
        ):
            graph.builder_node(state)
        self.assertEqual(limits[1], graph._RESUME_TIMEOUT_SECONDS)
        self.assertGreater(graph._RESUME_TIMEOUT_SECONDS, 1800)

    def test_time_limit_is_not_a_memory_kill(self) -> None:
        from src import graph
        from src.state import TIME_LIMIT_MARK, classify_error

        msg = (
            "Build command failed: make (exit 137) - cc ...\n"
            + TIME_LIMIT_MARK
            + " (1770 s)."
        )
        self.assertEqual(classify_error(msg), ErrorCategory.UNKNOWN)
        self.assertIn("time limit", graph._exit_code_text(msg))
        timed = CommandResult("make", 137, "", "", 1770.0, timed_out=True)
        self.assertFalse(graph._is_suspected_oom(timed, "make -j4"))
        killed = CommandResult("make", 137, "", "", 30.0)
        self.assertTrue(graph._is_suspected_oom(killed, "make -j4"))


class TestEscalationReasonFirst(unittest.TestCase):
    """The report names the real stop reason before the category."""

    def test_refusal_reason_beats_the_category(self) -> None:
        from src import graph

        state = _state()
        state.add_error(
            create_error_record(
                "Build command failed: cargo build (exit 137) - x",
                ErrorCategory.DISK_SPACE,
            )
        )
        state.context_cache["llm_refused"] = "fixer"
        out = graph.escalate_node(state)
        report = out.context_cache["escalation_report"]
        rec = report.split("## Recommendation", 1)[1]
        self.assertIn("refused", rec.split("##", 1)[0])


class TestLearningLoop(unittest.TestCase):
    """A verified run teaches every fix that worked, keyed on its error."""

    _GO_ERROR = (
        "Build command failed: go build -buildvcs=false ./cmd/tool (exit 1) "
        "- go: downloading github.com/hashicorp/go-multierror v1.1.1\n"
        "# example.com/tool/pkg\n"
        "pkg/paths.go:70:62: undefined: platform.Paths"
    )
    _PCAP_ERROR = (
        "Build command failed: go build -buildvcs=false ./cmd/tool (exit 1) "
        "- pcap.go:30:10: fatal error: pcap.h: No such file or directory"
    )

    def _verified(self, build="go build -buildvcs=false ./cmd/tool"):
        state = _state(
            build_plan=BuildPlan(
                build_system="go",
                build_system_confidence=0.9,
                phases=[
                    BuildPhase(
                        1,
                        "setup",
                        [
                            "apt-get update && apt-get install -y "
                            "--no-install-recommends git"
                        ],
                    ),
                    BuildPhase(2, "build", [build]),
                ],
                total_estimated_duration="1m",
            )
        )
        state.build_status = BuildStatus.SUCCESS
        state.artifact_verification = {"status": "verified"}
        state.curated_artifacts = [{"filepath": "tool", "role": "primary"}]
        return state

    @staticmethod
    def _fix(strategy, actions, error):
        return FixAttempt(
            error_category=ErrorCategory.COMPILATION,
            strategy=strategy,
            changes_made=[
                f"Executed: {a['command']}"
                for a in actions
                if a["type"] == "command"
            ],
            success=True,
            error_message=error,
            actions=actions,
        )

    @staticmethod
    def _learn(state):
        from src import graph

        with (
            mock.patch.object(graph, "save_learned_example") as save_example,
            mock.patch.object(graph, "save_to_recipe_cache") as save_cache,
            mock.patch.object(
                graph, "get_model_for_role", return_value=mock.MagicMock()
            ),
            mock.patch(
                "src.artifact_curator.curate_artifacts",
                return_value=[{"filepath": "tool", "role": "primary"}],
            ),
        ):
            graph._save_learning_data(state)
        payloads = {}
        for call in save_example.call_args_list:
            payloads.setdefault(call.args[0], []).append(call.args[1])
        cache = save_cache.call_args.kwargs if save_cache.called else {}
        return payloads, cache

    def test_every_working_fix_is_learned_on_its_error_line(self) -> None:
        state = self._verified()
        state.add_fix_attempt(
            self._fix(
                "add the riscv64 paths file",
                [
                    {
                        "type": "command",
                        "command": "cp pkg/paths_amd64.go "
                        "pkg/paths_riscv64.go",
                        "ok": True,
                    }
                ],
                self._GO_ERROR,
            )
        )
        state.add_fix_attempt(
            self._fix(
                "install libpcap headers",
                [
                    {
                        "type": "command",
                        "command": "apt-get install -y libpcap-dev",
                        "ok": True,
                    }
                ],
                self._PCAP_ERROR,
            )
        )
        fixer = self._learn(state)[0]["fixer"]
        self.assertEqual(
            [f["error_context"]["error_message"] for f in fixer],
            [
                "pkg/paths.go:70:62: undefined: platform.Paths",
                "pcap.go:30:10: fatal error: pcap.h: "
                "No such file or directory",
            ],
        )
        for payload in fixer:
            self.assertNotIn("downloading", payload["error_pattern"])

    def test_read_only_and_stub_fixes_teach_nothing(self) -> None:
        state = self._verified()
        state.add_fix_attempt(
            self._fix(
                "look at the Makefile",
                [{"type": "command", "command": "cat Makefile", "ok": True}],
                self._GO_ERROR,
            )
        )
        state.add_fix_attempt(
            self._fix(
                "add a root package",
                [
                    {
                        "type": "create_file",
                        "path": "doc.go",
                        "content": "// Package tool.\npackage tool\n",
                        "ok": True,
                    }
                ],
                self._PCAP_ERROR,
            )
        )
        self.assertNotIn("fixer", self._learn(state)[0])

    def test_plan_and_real_file_fixes_are_learned(self) -> None:
        state = self._verified()
        state.add_fix_attempt(
            self._fix(
                "build in the module directory",
                [
                    {
                        "type": "plan_command",
                        "replaces": "go build .",
                        "command": "cd src && go build .",
                        "ok": True,
                    }
                ],
                self._GO_ERROR,
            )
        )
        content = (
            "//go:build riscv64\n\npackage platform\n\n"
            'var Paths = []string{"/usr"}\n'
        )
        state.add_fix_attempt(
            self._fix(
                "add the riscv64 file",
                [
                    {
                        "type": "create_file",
                        "path": "pkg/platform/paths_linux_riscv64.go",
                        "content": content,
                        "ok": True,
                    }
                ],
                self._PCAP_ERROR,
            )
        )
        fixer = self._learn(state)[0]["fixer"]
        self.assertEqual(
            [f["fix"]["actions"][0]["type"] for f in fixer],
            ["plan_command", "create_file"],
        )
        self.assertEqual(fixer[1]["fix"]["actions"][0]["content"], content)

    def test_download_noise_alone_is_no_key(self) -> None:
        state = self._verified()
        state.add_fix_attempt(
            self._fix(
                "tidy",
                [{"type": "command", "command": "go mod tidy", "ok": True}],
                "Build command failed: go build . (exit 1) - "
                "go: downloading example.com/errors v1.0.0\n",
            )
        )
        self.assertNotIn("fixer", self._learn(state)[0])

    def test_recipe_replays_the_fix_commands(self) -> None:
        state = self._verified()
        state.add_fix_attempt(
            self._fix(
                "install headers and patch the source",
                [
                    {
                        "type": "command",
                        "command": "apt-get update",
                        "ok": True,
                    },
                    {
                        "type": "command",
                        "command": "apt-get install -y "
                        "--no-install-recommends libpcap-dev",
                        "ok": True,
                    },
                    {"type": "command", "command": "cat go.mod", "ok": True},
                    {
                        "type": "command",
                        "command": "sed -i s/a/b/ x.go",
                        "ok": True,
                    },
                    {
                        "type": "command",
                        "command": "go build ./...",
                        "ok": True,
                    },
                    {"type": "command", "command": "make broken", "ok": False},
                ],
                self._PCAP_ERROR,
            )
        )
        cache = self._learn(state)[1]
        phases = cache["build_plan"]["phases"]
        self.assertEqual(
            [p["name"] for p in phases], ["setup", "pre_build", "build"]
        )
        self.assertEqual(
            phases[1]["commands"],
            [
                "apt-get update",
                "apt-get install -y --no-install-recommends libpcap-dev",
                "sed -i s/a/b/ x.go",
            ],
        )
        self.assertIn("libpcap-dev", cache["dependencies"])
        self.assertEqual(cache["layout"]["main_path"], "./cmd/tool")

    def test_scout_trigger_keeps_the_layout(self) -> None:
        state = self._verified(
            build="cd v1 && CGO_ENABLED=1 go build -buildvcs=false "
            "-o tool ./cmd/tool"
        )
        scout = self._learn(state)[0]["scout"][0]
        trigger = scout["trigger"]
        self.assertEqual(trigger["main_path"], "./cmd/tool")
        self.assertEqual(trigger["module_dir"], "v1")
        self.assertTrue(trigger["has_cgo"])
        self.assertEqual(trigger["build_env"], {"CGO_ENABLED": "1"})

    def test_fix_that_removed_one_error_changed_the_error(self) -> None:
        from datetime import datetime

        from src import graph

        for after, changed in (
            ("b.c:2:1: error: y", True),
            ("a.c:1:1: error: x\nb.c:2:1: error: y", False),
        ):
            with self.subTest(after=after):
                state = _state()
                fix = self._fix(
                    "f", [], "a.c:1:1: error: x\nb.c:2:1: error: y"
                )
                fix.timestamp = datetime(2026, 1, 1)
                state.add_fix_attempt(fix)
                state.add_error(
                    create_error_record(after, ErrorCategory.COMPILATION)
                )
                self.assertEqual(
                    graph._fix_changed_the_error(state, fix), changed
                )


class TestFailureMemory(unittest.TestCase):
    """A fix after which the error stayed is remembered once."""

    def _state_with_fix(self, error_after):
        from datetime import datetime

        state = _state()
        fix = FixAttempt(
            error_category=ErrorCategory.COMPILATION,
            strategy="install x",
            changes_made=["Executed: apt-get install -y x"],
            success=False,
            error_message="Build command failed: make (exit 2) - "
            "a.c:1:10: fatal error: x.h: No such file or directory",
            actions=[
                {
                    "type": "command",
                    "command": "apt-get install -y x",
                    "ok": True,
                }
            ],
        )
        fix.timestamp = datetime(2026, 1, 1)
        state.add_fix_attempt(fix)
        state.add_error(
            create_error_record(error_after, ErrorCategory.DEPENDENCY)
        )
        return state

    def test_a_fix_that_left_the_error_is_remembered(self) -> None:
        from src import graph

        state = self._state_with_fix(
            "Build command failed: make (exit 2) - "
            "a.c:1:10: fatal error: x.h: No such file or directory"
        )
        with mock.patch.object(graph, "save_failure_record") as save:
            graph._remember_failed_fixes(state)
            graph._remember_failed_fixes(state)
        save.assert_called_once()
        repo, key, record = save.call_args.args
        self.assertEqual(repo, state.repo_name)
        self.assertIn("x.h", key)
        self.assertEqual(record["strategy"], "install x")
        self.assertEqual(record["commands"], ["apt-get install -y x"])

    def test_a_fix_that_changed_the_error_is_not_remembered(self) -> None:
        from src import graph

        state = self._state_with_fix("b.c:2:1: error: y")
        with mock.patch.object(graph, "save_failure_record") as save:
            graph._remember_failed_fixes(state)
        save.assert_not_called()


class TestRecall(unittest.TestCase):
    """Every run reuses what earlier runs of the package left."""

    _RECIPE = {
        "repo_url": "https://github.com/a/b.git",
        "build_system": "go",
        "architecture": "riscv64",
        "build_plan": {
            "phases": [
                {
                    "name": "build",
                    "commands": ["cd v1 && go build -buildvcs=false ./cmd/b"],
                }
            ]
        },
        "artifacts": [{"type": "binary", "path": "/workspace/repos/b/b"}],
        "verification": {"status": "verified"},
        "last_built": "2026-10-13T18:00:00",
    }

    def _recall(self, entries, failures=None, replan=False, expected=None):
        from src import graph

        state = _state()
        state.replan = replan
        state.package_analysis = PackageAnalysis(
            purpose="a tool",
            build_system="go",
            build_system_confidence=0.9,
            expected_artifacts=list(expected or []),
            llm_grounded=True,
        )

        def lookup(name, sandbox=None, repo_url=None):
            return entries.get(sandbox)

        with (
            mock.patch.object(
                graph,
                "recall_sandboxes",
                return_value=["debian-riscv64-native", "debian-riscv64"],
            ),
            mock.patch.object(graph, "get_cached_recipe", side_effect=lookup),
            mock.patch.object(
                graph,
                "load_recipe_cache",
                return_value={"packages": {"b": dict(entries)}},
            ),
            mock.patch.object(
                graph, "load_failure_records", return_value=failures or {}
            ),
        ):
            return state, graph._recall(state)

    def test_the_active_sandbox_comes_first(self) -> None:
        native = {**self._RECIPE, "last_built": "2026-10-20T10:00:00"}
        _, recall = self._recall(
            {"debian-riscv64-native": native, "debian-riscv64": self._RECIPE}
        )
        self.assertEqual(recall["sandbox"], "debian-riscv64-native")

    def test_the_other_target_is_the_fallback(self) -> None:
        _, recall = self._recall({"debian-riscv64": self._RECIPE})
        self.assertEqual(recall["sandbox"], "debian-riscv64")

    def test_another_distro_gives_the_layout_only(self) -> None:
        _, recall = self._recall({"alpine-riscv64": self._RECIPE})
        self.assertNotIn("recipe", recall)
        self.assertEqual(
            recall["layout"], {"main_path": "./cmd/b", "module_dir": "v1"}
        )
        self.assertEqual(recall["layout_sandbox"], "alpine-riscv64")

    def test_replan_recalls_nothing(self) -> None:
        _, recall = self._recall({"debian-riscv64": self._RECIPE}, replan=True)
        self.assertEqual(recall, {})

    def test_a_verified_recipe_is_the_plan(self) -> None:
        from src import graph

        state, _ = self._recall(
            {"debian-riscv64": self._RECIPE}, expected=["b"]
        )
        head = CommandResult("git", 0, "abc\n", "", 0.1)
        with (
            mock.patch.object(graph, "execute_command", return_value=head),
            mock.patch.object(graph, "invoke_llm") as llm,
        ):
            out = graph.heuristic_plan_node(state)
        llm.assert_not_called()
        self.assertEqual(
            [c for p in out.build_plan.phases for c in p.commands],
            ["cd v1 && go build -buildvcs=false ./cmd/b"],
        )
        self.assertTrue(out.package_analysis.llm_grounded)

    def test_unverified_or_other_target_recipe_is_only_a_hint(self) -> None:
        from src import graph

        unverified = {**self._RECIPE, "verification": None}
        state, _ = self._recall({"debian-riscv64": unverified})
        self.assertIsNone(graph._recall_seed(state))
        state, _ = self._recall(
            {"debian-riscv64": self._RECIPE}, expected=["netbird"]
        )
        self.assertIsNone(graph._recall_seed(state))

    def test_prompts_show_the_plan_and_the_failed_fixes(self) -> None:
        from src import graph

        failures = {
            "a.c:<n>: fatal error: x.h": [
                {
                    "strategy": "install x",
                    "commands": ["apt-get install -y x"],
                    "error_after": "a.c:1: fatal error: x.h",
                }
            ]
        }
        state, _ = self._recall({"debian-riscv64": self._RECIPE}, failures)
        section = graph._recall_prompt_section(state)
        self.assertIn("Verified plan of this package", section)
        self.assertIn("cd v1 && go build -buildvcs=false ./cmd/b", section)
        self.assertIn("Tried before on this package", section)
        self.assertIn("install x", section)
        self.assertEqual(graph._recall_label(state), "b[debian-riscv64]")


class TestHeuristicPlanFacts(unittest.TestCase):
    """The default plan uses what init and the analyst found."""

    def _plan(
        self,
        build_type,
        module_dir="",
        arch_codes=(),
        analysis=None,
        go_main_info=None,
    ):
        from src import graph

        state = _state()
        state.context_cache["recall"] = {}
        state.build_system_info = BuildSystemInfo(
            type=build_type,
            confidence=0.9,
            primary_file="CMakeLists.txt",
            module_dir=module_dir,
        )
        state.arch_specific_code = list(arch_codes)
        if analysis is not None:
            state.package_analysis = analysis
        if go_main_info is not None:
            state.context_cache["go_main_info"] = go_main_info
        return graph.heuristic_plan_node(state).build_plan

    def test_nested_cmake_root(self) -> None:
        plan = self._plan("cmake", module_dir="expat")
        self.assertEqual(
            plan.phases[1].commands,
            ["cmake -S expat -B build -DCMAKE_BUILD_TYPE=Release"],
        )
        self.assertEqual(
            plan.phases[2].commands, ["cmake --build build -j$(nproc)"]
        )

    def test_go_files_for_other_arches_defer_to_the_scout(self) -> None:
        finding = ArchSpecificCode(
            file="/workspace/repos/b/pkg/paths_linux_amd64.go",
            line=1,
            code_snippet="arch-specific Go file",
            arch_type="go_arch_files",
            severity="medium",
        )
        self.assertIsNone(self._plan("go", arch_codes=[finding]))

    def test_asm_in_bundled_code_does_not_defer(self) -> None:
        findings = [
            ArchSpecificCode(
                file=f"/workspace/repos/b/gnulib/lib/x{i}.c",
                line=1,
                code_snippet="asm",
                arch_type="inline_asm",
                severity="high",
            )
            for i in range(3)
        ]
        self.assertIsNotNone(self._plan("cmake", arch_codes=findings))

    def test_analyst_build_settings_reach_the_go_command(self) -> None:
        analysis = PackageAnalysis(
            purpose="a tool",
            build_system="go",
            build_system_confidence=0.9,
            build_strategy=(
                "The GNUmakefile builds with CGO_ENABLED=0 "
                "GOEXPERIMENT=jsonv2 and -tags noweb."
            ),
            llm_grounded=True,
        )
        plan = self._plan(
            "go",
            analysis=analysis,
            go_main_info={
                "has_main": True,
                "has_go_sum": True,
                "plan_command": "go build ./cmd/b",
            },
        )
        command = plan.phases[1].commands[-1]
        self.assertTrue(
            command.startswith("CGO_ENABLED=0 GOEXPERIMENT=jsonv2 go build"),
            command,
        )
        self.assertIn("-tags noweb ", command)
        self.assertNotIn("noweb.", command)
        self.assertIn("-buildvcs=false", command)


class TestFixOutcomeAndReplay(unittest.TestCase):
    """A fix is judged on the lines it removed and replayed where it ran."""

    _MAKE_ERROR = (
        "Build command failed: make -j4 (exit 2) - "
        "src/b.c:20:3: error: 'FOO' undeclared\n"
        "make[1]: *** [Makefile:400: src/b.o] Error 1\n"
        "make[1]: *** Waiting for unfinished jobs....\n"
        "make: *** [Makefile:300: all] Error 2"
    )

    @staticmethod
    def _at(seconds):
        from datetime import datetime, timedelta

        return datetime(2026, 1, 1) + timedelta(seconds=seconds)

    def _error(self, state, seconds, message, command="make -j4"):
        error = create_error_record(
            message, ErrorCategory.COMPILATION, command=command
        )
        error.timestamp = self._at(seconds)
        state.add_error(error)

    def _fix(self, state, seconds, error, commands, success=True):
        fix = FixAttempt(
            error_category=ErrorCategory.DEPENDENCY,
            strategy=f"run {commands[0]}",
            changes_made=[f"Executed: {c}" for c in commands],
            success=success,
            error_message=error,
            actions=[
                {"type": "command", "command": c, "ok": True} for c in commands
            ],
        )
        fix.timestamp = self._at(seconds)
        state.add_fix_attempt(fix)
        return fix

    def test_make_summary_lines_are_no_progress(self) -> None:
        """A parallel make prints a different number of summary lines."""
        from src import graph

        state = _state()
        self._error(state, 0, self._MAKE_ERROR)
        fix = self._fix(state, 1, self._MAKE_ERROR, ["true"], success=False)
        self._error(
            state,
            2,
            self._MAKE_ERROR.replace(
                "make[1]: *** Waiting for unfinished jobs....\n", ""
            ),
        )
        self.assertFalse(graph._fix_changed_the_error(state, fix))

    def test_a_new_line_alone_is_no_progress(self) -> None:
        from src import graph

        state = _state()
        before = "Build command failed: make (exit 2) - a.c:1:1: error: x"
        self._error(state, 0, before)
        fix = self._fix(state, 1, before, ["true"], success=False)
        self._error(state, 2, before + "\nb.c:2:1: error: y")
        self.assertFalse(graph._fix_changed_the_error(state, fix))

    def test_the_key_is_the_line_the_fix_removed(self) -> None:
        """A fix that removed the second of two errors is keyed on it."""
        from src import graph

        both = (
            "Build command failed: make -j4 (exit 2) - "
            "a.c:1:10: fatal error: pcap.h: No such file or directory\n"
            "b.c:1:10: fatal error: sys/capability.h: No such file or "
            "directory"
        )
        pcap = both.split("\nb.c")[0]
        state = _state()
        self._error(state, 0, both)
        libcap = self._fix(state, 1, both, ["apt-get install -y libcap-dev"])
        self._error(state, 2, pcap)
        libpcap = self._fix(state, 3, pcap, ["apt-get install -y libpcap-dev"])
        self.assertIn("sys/capability.h", graph._fix_key_line(state, libcap))
        self.assertIn("pcap.h", graph._fix_key_line(state, libpcap))
        self.assertEqual(
            graph._fix_key_line(
                state,
                self._fix(state, 1, pcap, ["true"]),
            ),
            "",
        )

    def _replay_state(self):
        state = _state(
            build_plan=BuildPlan(
                build_system="autotools",
                build_system_confidence=0.9,
                phases=[
                    BuildPhase(1, "setup", ["apt-get update"]),
                    BuildPhase(2, "configure", ["./configure"]),
                    BuildPhase(3, "build", ["make -j4"]),
                ],
                total_estimated_duration="1m",
            )
        )
        return state

    def test_fix_commands_run_before_the_phase_that_failed(self) -> None:
        """A sed of a generated Makefile runs after configure."""
        from src import graph

        state = self._replay_state()
        warning = (
            "Build command failed: make -j4 (exit 2) - "
            "cc1: all warnings being treated as errors"
        )
        self._error(state, 0, warning)
        self._fix(
            state,
            1,
            warning,
            [
                "sed -i 's/-Werror//' src/Makefile",
                "CGO_ENABLED=0 go build -o x .",
            ],
        )
        phases, _ = graph._recipe_phases(state)
        self.assertEqual(
            [p["name"] for p in phases],
            ["setup", "configure", "pre_build", "build"],
        )
        self.assertEqual(
            phases[2]["commands"], ["sed -i 's/-Werror//' src/Makefile"]
        )

    def test_artifact_check_fixes_run_after_the_plan(self) -> None:
        """A fix of the artifact check builds what the plan missed."""
        from src import graph

        state = self._replay_state()
        missing = "No riscv64 ELF artifact found after the build"
        self._error(state, 0, missing, command=None)
        self._fix(state, 1, missing, ["go build -buildvcs=false -o x ."])
        self._error(
            state,
            2,
            "Build command failed: ./configure (exit 1) "
            "- configure: error: no C compiler",
            command="./configure",
        )
        self._fix(
            state,
            3,
            "configure: error: no C compiler",
            ["apt-get install -y gcc"],
        )
        phases, packages = graph._recipe_phases(state)
        self.assertEqual(
            [p["name"] for p in phases],
            ["setup", "pre_configure", "configure", "build", "post_build"],
        )
        self.assertEqual(
            phases[-1]["commands"], ["go build -buildvcs=false -o x ."]
        )
        self.assertEqual(packages, ["gcc"])

    def test_header_with_preprocessor_lines_is_real_code(self) -> None:
        from src import graph

        header = "#ifndef CAP_H\n#define CAP_H\n#include <x.h>\n#endif\n"
        self.assertFalse(graph._is_stub_content(header, "include/cap.h"))
        self.assertTrue(graph._is_stub_content("# a comment\n", "fix.sh"))
        self.assertTrue(
            graph._is_stub_content("// doc\npackage tool\n", "doc.go")
        )

    def test_error_regex_matches_its_line_and_column(self) -> None:
        import re

        from src import graph

        line = "pcap.go:30:10: fatal error: pcap/pcap.h: No such file"
        pattern = graph._error_regex(line)
        self.assertRegex(line, re.compile(pattern, re.IGNORECASE))
        self.assertRegex(
            "pcap.go:7: fatal error: pcap/pcap.h: No such file",
            re.compile(pattern, re.IGNORECASE),
        )


class TestInfrastructureGuard(unittest.TestCase):
    """A dead container or ssh session never reaches the fixer."""

    def _build(self, results):
        from src import graph

        state = _state(
            build_plan=BuildPlan(
                build_system="cmake",
                build_system_confidence=0.9,
                phases=[BuildPhase(1, "build", ["ninja -C build"])],
                total_estimated_duration="1m",
            )
        )
        queue = list(results)
        runs = []

        def fake_execute(cmd, *args, **kwargs):
            runs.append(cmd)
            return queue.pop(0)

        with (
            mock.patch.object(
                graph, "execute_command", side_effect=fake_execute
            ),
            mock.patch.object(graph, "predict_build_issues", return_value=[]),
            mock.patch.object(
                graph, "ensure_build_container", return_value="started"
            ) as ensure,
        ):
            out = graph.builder_node(state)
        return out, runs, ensure

    def test_one_rerun_after_a_dropped_session(self) -> None:
        dropped = CommandResult(
            "ninja -C build", 255, "", "go: downloading a.b/c v1.0.0\n", 30.0
        )
        done = CommandResult("ninja -C build", 0, "", "", 60.0)
        out, runs, ensure = self._build([dropped, done])
        self.assertEqual(runs, ["ninja -C build", "ninja -C build"])
        ensure.assert_called_once()
        self.assertEqual(out.error_history, [])
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)

    def test_a_fault_that_stays_escalates_without_the_fixer(self) -> None:
        from src import graph

        dead = CommandResult("ninja -C build", 137, "", "", 2.0)
        out, runs, _ = self._build([dead, dead])
        self.assertEqual(len(runs), 2)
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)
        self.assertEqual(out.last_error_category, ErrorCategory.INFRASTRUCTURE)
        self.assertTrue(out.last_error.startswith(graph._INFRA_FAULT))
        self.assertEqual(graph.route_build_result(out), "__end__")

    def test_a_build_error_after_the_rerun_goes_on_as_usual(self) -> None:
        dead = CommandResult("ninja -C build", 137, "", "", 2.0)
        broken = CommandResult(
            "ninja -C build", 1, "", "a.c:1:1: error: x undeclared", 9.0
        )
        out, _, _ = self._build([dead, broken])
        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertEqual(out.last_error_category, ErrorCategory.COMPILATION)


class TestFixerLoopRules(unittest.TestCase):
    """Chains split, reading is investigation, history names refusals."""

    def _state(self, plan=None, command="make"):
        state = _state()
        state.last_error = (
            "Build command failed: make (exit 2) - a.c: error: x"
        )
        state.last_error_category = ErrorCategory.COMPILATION
        state.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.8,
            phases=plan or [BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )
        state.error_history = [
            create_error_record(
                state.last_error, ErrorCategory.COMPILATION, command=command
            )
        ]
        return state

    @staticmethod
    def _answer(*commands, description="fix it"):
        return {
            "strategies": [
                {
                    "id": 1,
                    "description": description,
                    "actions": [
                        {"type": "command", "command": c} for c in commands
                    ],
                }
            ],
            "recommended_strategy_id": 1,
            "reflection": {"root_cause": "x"},
        }

    def _run(self, state, answers, results=None, investigate=None):
        """Run fixer_node; each answer goes through the real validator."""
        from src import graph

        queue = list(answers)
        runs = []

        def fake_call(**kwargs):
            data = queue.pop(0)
            if isinstance(data, LLMCallOutcome):
                return data
            verdict = kwargs["validator"](data)
            if not verdict.ok:
                return LLMCallOutcome(
                    data=None,
                    used_fallback=False,
                    attempts=1,
                    last_error=verdict.reason,
                )
            return LLMCallOutcome(data=data, used_fallback=False, attempts=1)

        def fake_execute(cmd, *args, **kwargs):
            runs.append(cmd)
            if results and cmd in results:
                return results[cmd]
            return CommandResult(cmd, 0, "", "", 1.0)

        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_call
            ),
            mock.patch.object(
                graph, "execute_command", side_effect=fake_execute
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="k"
            ),
            mock.patch.object(
                graph,
                "_run_fixer_investigation",
                return_value="results",
            ) as inspect,
        ):
            out = graph.fixer_node(state)
        return out, runs, inspect

    def test_a_plain_chain_runs_as_single_commands(self) -> None:
        """In snappy run 1 the right answer was refused twice for &&."""
        out, runs, _ = self._run(
            self._state(),
            [self._answer("mkdir -p build && cd build && cmake ..")],
        )
        self.assertEqual(runs, ["mkdir -p build", "cd build && cmake .."])
        self.assertEqual(out.build_status, BuildStatus.PENDING)

    def test_a_failed_step_ends_its_chain(self) -> None:
        failed = CommandResult("apt-get update", 100, "", "E: no net", 1.0)
        out, runs, _ = self._run(
            self._state(),
            [self._answer("apt-get update && apt-get install -y libx-dev")],
            results={"apt-get update": failed},
        )
        self.assertEqual(runs, ["apt-get update"])
        self.assertTrue(
            any("skipped" in r for r in out.fixes_attempted[-1].action_results)
        )

    def test_a_reading_strategy_is_the_investigation_round(self) -> None:
        out, runs, inspect = self._run(
            self._state(),
            [
                self._answer("cat Makefile", description="look"),
                self._answer("apt-get install -y libx-dev"),
            ],
        )
        inspect.assert_called_once()
        self.assertEqual(inspect.call_args.args[1], ["cat Makefile"])
        self.assertEqual(runs, ["apt-get install -y libx-dev"])

    def test_a_reading_fix_in_the_last_round_changes_nothing(self) -> None:
        from src import graph

        state = self._state()
        state.attempt_count = 1
        out, runs, _ = self._run(
            state,
            [{"investigate": {"commands": ["ls"]}}, self._answer("cat x")],
        )
        self.assertEqual(runs, ["cat x"])
        self.assertEqual(
            out.fixes_attempted[-1].build_result, "no change applied"
        )
        self.assertEqual(out.attempt_count, 2)
        self.assertEqual(graph.route_fix_result(out), "fix_node")

    def test_a_failed_phase_command_the_fix_ran_is_done(self) -> None:
        state = self._state(
            plan=[
                BuildPhase(1, "configure", ["./configure"]),
                BuildPhase(2, "build", ["make"]),
            ],
            command="./configure",
        )
        out, _, _ = self._run(
            state, [self._answer("apt-get install -y libx-dev", "./configure")]
        )
        self.assertTrue(out.get_cached_command_result("./configure").success)
        self.assertIsNone(out.get_cached_command_result("make"))

    def test_refused_answers_reach_the_next_prompt(self) -> None:
        from src import graph

        refused = LLMCallOutcome(
            data=None,
            used_fallback=False,
            attempts=3,
            last_error="Unsafe command: Chained commands are not allowed",
        )
        state = self._state()
        before = state.attempt_count
        out, _, _ = self._run(state, [refused])
        self.assertEqual(out.attempt_count, before + 1)
        text = graph._previous_fixes_text(out)
        self.assertIn("Refused answer", text)
        self.assertIn("Chained commands", text)

    def test_a_provider_fault_is_no_attempt(self) -> None:
        fault = LLMCallOutcome(
            data=None,
            used_fallback=False,
            attempts=3,
            last_error="LLM invocation raised TimeoutError: read timed out",
        )
        state = self._state()
        before = state.attempt_count
        out, _, _ = self._run(state, [fault])
        self.assertEqual(out.attempt_count, before)

    def test_the_prompt_names_the_command_that_ran(self) -> None:
        state = self._state(command="./configure")
        state.cache_command_result(
            "./configure",
            CommandResult("sh bootstrap && ./configure", 1, "", "x", 1.0),
        )
        seen = {}

        def fake_call(**kwargs):
            seen.update(kwargs)
            return LLMCallOutcome(data=None, used_fallback=False, attempts=1)

        from src import graph

        with (
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_call
            ),
            mock.patch.object(
                graph, "error_context_excerpts", return_value=""
            ),
            mock.patch.object(graph, "top_level_listing", return_value="x"),
        ):
            graph.fixer_node(state)
        self.assertIn("(run as `sh bootstrap && ./configure`)", seen["prompt"])


class TestVerificationRunsTheBuild(unittest.TestCase):
    """A verified build must also run on the native machine."""

    _NO_V = ["i", "m", "a", "f", "d", "c", "zba", "zbb"]

    def _profile(self):
        from src.machine import MachineProfile

        return MachineProfile(
            kind="native",
            emulated_cpu=None,
            extensions=list(self._NO_V),
            label="rv64gc+zba+zbb",
        )

    def _verify(self, run_code, counts=None, march=""):
        from src import graph
        from src.artifact_scanner import (
            VERIFIED,
            Artifact,
            VerificationResult,
        )

        state = _state(
            build_plan=BuildPlan(
                build_system="cmake",
                build_system_confidence=0.9,
                phases=[
                    BuildPhase(
                        1,
                        "build",
                        [f"cmake -DCMAKE_C_FLAGS={march} -S . -B b"],
                    )
                ],
                total_estimated_duration="1m",
            )
        )
        tool = Artifact(
            f"{state.repo_path}/b/cwebp", "binary", "riscv64", "lp64d", 9
        )
        verdict = VerificationResult(
            status=VERIFIED, reason="1 riscv64 output", riscv64=[tool]
        )
        facts = {"artifacts": {"b/cwebp": {"exists": True, "counts": counts}}}
        runs = []

        def fake_execute(argv, *args, **kwargs):
            runs.append(argv)
            return CommandResult(" ".join(argv), run_code, "", "", 0.1)

        with (
            mock.patch.object(graph, "is_native", return_value=True),
            mock.patch.object(graph, "_machine", return_value=self._profile()),
            mock.patch.object(
                graph.ArtifactScanner, "scan", return_value=verdict
            ),
            mock.patch("src.optimize.inspect_tree", return_value=(facts, "")),
            mock.patch.object(
                graph, "execute_command", side_effect=fake_execute
            ),
        ):
            out = graph.verify_node(state)
        return out, runs

    def test_a_crash_signal_fails_the_build(self) -> None:
        out, runs = self._verify(132)
        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertEqual(out.artifact_verification["status"], "crashes")
        self.assertIn("SIGILL", out.last_error)
        self.assertEqual(out.last_error_category, ErrorCategory.ARCHITECTURE)
        self.assertEqual(runs[0][1], "--version")

    def test_a_program_that_runs_passes(self) -> None:
        out, _ = self._verify(0)
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)

    def test_missing_extension_instructions_warn_or_fail(self) -> None:
        """Without the plan's -march they can be runtime dispatch."""
        out, _ = self._verify(0, counts={"v": 120})
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)
        self.assertTrue(out.artifact_verification["isa_warnings"])
        out, _ = self._verify(0, counts={"v": 120}, march="-march=rv64gcv")
        self.assertEqual(out.build_status, BuildStatus.FAILED)
        self.assertIn("-march asks for them", out.last_error)

    def test_qemu_runs_no_check(self) -> None:
        from src import graph
        from src.artifact_scanner import VERIFIED, Artifact, VerificationResult

        state = _state()
        tool = Artifact(
            f"{state.repo_path}/t", "binary", "riscv64", "lp64d", 9
        )
        verdict = VerificationResult(
            status=VERIFIED, reason="ok", riscv64=[tool]
        )
        with (
            mock.patch.object(graph, "is_native", return_value=False),
            mock.patch.object(
                graph.ArtifactScanner, "scan", return_value=verdict
            ),
            mock.patch.object(graph, "execute_command") as run,
        ):
            out = graph.verify_node(state)
        run.assert_not_called()
        self.assertEqual(out.build_status, BuildStatus.SUCCESS)


class TestMachineFlags(unittest.TestCase):
    """A -march the machine lacks never becomes or stays the plan."""

    def _profile(self):
        from src.machine import MachineProfile

        return MachineProfile(
            kind="native",
            emulated_cpu=None,
            extensions=["i", "m", "a", "f", "d", "c", "zba", "zbb"],
            label="rv64gc+zba+zbb",
        )

    def test_the_check_names_the_missing_extensions(self) -> None:
        from src import graph

        profile = self._profile()
        self.assertEqual(
            graph._isa_flags_the_machine_lacks(
                ["cmake -DCMAKE_C_FLAGS=-march=rv64gcv ."], profile
            ),
            ["-march=rv64gcv (the machine lacks v)"],
        )
        self.assertEqual(
            graph._isa_flags_the_machine_lacks(
                ["gcc -march=rv64gc_zba_zbb x.c"], profile
            ),
            [],
        )
        self.assertEqual(
            graph._isa_flags_the_machine_lacks(["gcc -march=rv64gcv"], None),
            [],
        )

    def test_a_cached_recipe_for_another_isa_is_only_a_hint(self) -> None:
        """The native libwebp recipe compiles for rv64gcv."""
        from src import graph

        state = _state()
        state.context_cache["recall"] = {
            "recipe": {
                "build_system": "cmake",
                "build_plan": {
                    "phases": [
                        {
                            "name": "build",
                            "commands": [
                                "cmake -S . -B build "
                                "-DCMAKE_C_FLAGS=-march=rv64gcv"
                            ],
                        }
                    ]
                },
                "verification": {"status": "verified"},
            },
            "sandbox": "debian-riscv64-native",
        }
        with mock.patch.object(
            graph, "_machine", return_value=self._profile()
        ):
            self.assertIsNone(graph._recall_seed(state))

    def test_the_prompt_says_a_recipe_crashed(self) -> None:
        from src import graph

        state = _state()
        state.context_cache["recall"] = {
            "recipe": {
                "build_plan": {
                    "phases": [{"name": "b", "commands": ["make"]}]
                },
                "verification": {"status": "crashes"},
            },
            "sandbox": "debian-riscv64-native",
        }
        self.assertIn("crashed", graph._recall_prompt_section(state))

    def test_a_baseline_crash_marks_the_recipe(self) -> None:
        from src import graph

        state = _state()
        state.machine_profile = {"label": "rv64gc+zba+zbb"}
        crash = "the baseline run crashed with SIGILL in the timed phase"
        with (
            mock.patch(
                "src.optimize.run_optimization",
                return_value={
                    "status": "not measured",
                    "baseline_crash": crash,
                },
            ),
            mock.patch.object(graph, "record_optimization"),
            mock.patch.object(graph, "mark_recipe_verification") as mark,
        ):
            graph.optimize_node(state)
        mark.assert_called_once()
        self.assertEqual(mark.call_args.args[1], "crashes")
        self.assertIn("SIGILL", mark.call_args.args[2])

    def test_smoke_run_names_only_crash_signals(self) -> None:
        from src import graph

        for codes, expected in (
            ([132], "died with SIGILL"),
            ([0], ""),
            ([2, 0], ""),
            ([137, 137], ""),
        ):
            with self.subTest(codes=codes):
                results = [CommandResult("x", c, "", "", 0.1) for c in codes]
                with mock.patch.object(
                    graph, "execute_command", side_effect=results
                ):
                    found = graph._smoke_run("/workspace/repos/b/tool")
                if expected:
                    self.assertIn(expected, found)
                else:
                    self.assertEqual(found, "")


class TestScoutMachineFlags(unittest.TestCase):
    """The scout gets no plan through with a -march the machine lacks."""

    def test_the_plan_is_refused_and_asked_again(self) -> None:
        from src import graph
        from src.machine import MachineProfile

        profile = MachineProfile(
            kind="native",
            emulated_cpu=None,
            extensions=["i", "m", "a", "f", "d", "c"],
            label="rv64gc",
        )

        def plan(command):
            return {
                "build_system": "cmake",
                "phases": [{"id": 1, "name": "build", "commands": [command]}],
            }

        answers = [
            plan("cmake -S . -B b -DCMAKE_C_FLAGS=-march=rv64gcv"),
            plan("cmake -S . -B b"),
        ]
        verdicts = []

        def fake_call(**kwargs):
            for data in answers:
                verdict = kwargs["validator"](data)
                verdicts.append(verdict)
                if verdict.ok:
                    return LLMCallOutcome(
                        data=data, used_fallback=False, attempts=2
                    )
            return LLMCallOutcome(data=None, used_fallback=False, attempts=2)

        state = _state()
        with (
            mock.patch.object(graph, "_machine", return_value=profile),
            mock.patch.object(
                graph,
                "get_model_pool_for_role",
                return_value=[mock.MagicMock()],
            ),
            mock.patch.object(
                graph, "llm_call_with_validation", side_effect=fake_call
            ),
            mock.patch.object(
                graph.scripted_ops,
                "get_system_info",
                return_value={"architecture": "riscv64"},
            ),
            mock.patch.object(
                graph, "collect_build_evidence", return_value="evidence"
            ),
            mock.patch.object(
                graph, "get_system_knowledge_summary", return_value="k"
            ),
            mock.patch.object(
                graph, "format_few_shot_examples", return_value=""
            ),
        ):
            out = graph.scout_node(state)
        self.assertFalse(verdicts[0].ok)
        self.assertIn("lacks v", verdicts[0].reason)
        self.assertEqual(
            out.build_plan.phases[0].commands, ["cmake -S . -B b"]
        )


class TestExpectedGoPrograms(unittest.TestCase):
    """Each named program with its own cmd/ directory is built."""

    def test_age_keygen_is_built_next_to_age(self) -> None:
        import tempfile

        from src import graph

        with tempfile.TemporaryDirectory() as root:
            for name in ("age", "age-keygen", "internal"):
                os.makedirs(os.path.join(root, "cmd", name))
            state = _state()
            state.package_analysis = PackageAnalysis(
                purpose="encryption tool",
                expected_artifacts=["age", "age-keygen", "libnothing.so"],
                llm_grounded=True,
            )
            with mock.patch.object(graph, "_to_host_path", return_value=root):
                extra = graph._extra_go_mains(
                    state, {}, "go build -buildvcs=false ./cmd/age"
                )
        self.assertEqual(
            extra, ["go build -buildvcs=false -o age-keygen ./cmd/age-keygen"]
        )


class TestCachedRecipeMachineFlags(unittest.TestCase):
    """--optimize does not replay a recipe built for another ISA."""

    def test_the_run_plans_again(self) -> None:
        from src import graph
        from src.machine import MachineProfile

        profile = MachineProfile(
            kind="native",
            emulated_cpu=None,
            extensions=["i", "m", "a", "f", "d", "c"],
            label="rv64gc",
        )
        state = _state(
            cached_recipe={
                "build_system": "cmake",
                "build_plan": {
                    "phases": [
                        {
                            "name": "build",
                            "commands": ["gcc -march=rv64gcv -o x x.c"],
                        }
                    ]
                },
                "verification": {"status": "verified"},
            }
        )
        with mock.patch.object(graph, "_machine", return_value=profile):
            out = graph.cached_recipe_node(state)
        self.assertIsNone(out.build_plan)
        self.assertIsNone(out.cached_recipe)
        self.assertEqual(
            graph.route_cached_recipe_to_next(out), "analyst_node"
        )


class TestDataEffectRecord(unittest.TestCase):
    """Each run says what the learned data gave it."""

    def _state_with_fix(self, command):
        state = _state()
        state.context_cache["examples_shown"] = {"fixer": ["fx-1"]}
        state.add_fix_attempt(
            FixAttempt(
                error_category=ErrorCategory.DEPENDENCY,
                strategy="install",
                changes_made=[f"Executed: {command}"],
                success=True,
                actions=[{"type": "command", "command": command, "ok": True}],
            )
        )
        return state

    def _copied(self, state):
        from src import graph
        from src.memory import AgentExample

        example = AgentExample(
            id="fx-1",
            name="pcap",
            tags=[],
            fix={
                "actions": [
                    {"type": "command", "command": "apt-get update"},
                    {
                        "type": "command",
                        "command": "apt-get install -y libpcap-dev",
                    },
                ]
            },
        )
        memory = mock.MagicMock(examples=[example])
        with mock.patch.object(graph, "get_agent_memory", return_value=memory):
            return graph._copied_examples(state)

    def test_a_fix_that_repeats_an_example_copied_it(self) -> None:
        state = self._state_with_fix("apt-get install  -y libpcap-dev")
        self.assertEqual(self._copied(state), ["fx-1"])

    def test_the_index_update_alone_is_no_copy(self) -> None:
        self.assertEqual(
            self._copied(self._state_with_fix("apt-get update")), []
        )

    def test_the_record_names_the_plan_and_the_outcome(self) -> None:
        from src import graph

        state = _state()
        state.context_cache["plan_source"] = "recipe"
        state.build_status = BuildStatus.SUCCESS
        state.artifact_verification = {"status": "verified"}
        with mock.patch.object(graph, "log_data_effect") as log:
            graph._log_data_effect(state)
        record = log.call_args.args[0]
        self.assertEqual(record["plan_source"], "recipe")
        self.assertEqual(record["outcome"], "verified")
        self.assertEqual(record["fixer_rounds"], 0)

    def test_a_stopped_run_gets_one_interrupted_record(self) -> None:
        """nushell: the batch's kill left the run without a record."""
        from src import graph

        state = _state()
        with mock.patch.object(graph, "log_data_effect") as log:
            graph._log_data_effect(state, outcome="interrupted")
            graph._log_data_effect(state)
        self.assertEqual(log.call_count, 1)
        self.assertEqual(log.call_args.args[0]["outcome"], "interrupted")

    def test_the_record_counts_the_builds(self) -> None:
        from src import graph

        state = _state()
        state.build_count = 2
        with mock.patch.object(graph, "log_data_effect") as log:
            graph._log_data_effect(state)
        self.assertEqual(log.call_args.args[0]["builds"], 2)
