"""Tests for the graph nodes in src/graph.py.

Covers the analyst node (consumed PackageAnalysis + real cost logging),
the fixer's read-only investigation helper, expected-artifact
verification, the init, scout, fixer, builder, cached-recipe, optimize
and terminal nodes, and the auto-learning step. The LLM and the
sandbox are mocked.
"""

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
            {"phases": [{"name": "setup", "commands": ["cabal --version"]}]}
        )
        self.assertFalse(verdict.ok)
        self.assertIn("cabal --version", verdict.reason)
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
