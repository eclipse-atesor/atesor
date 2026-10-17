"""Tests for graph routing and the heuristic plan node.

The tests cover the routers for init, cached recipe, analyst,
heuristic plan, supervisor, finish, and the build-fix subgraph.
"""

import unittest

from src.graph import (
    _fix_loop_stop_reason,
    create_workflow,
    route_analyst_to_next,
    route_build_result,
    route_cached_recipe_to_next,
    route_finish_to_next,
    route_fix_result,
    route_heuristic_plan_to_next,
    route_init_to_next,
    route_supervisor_to_next,
    route_verify_result,
)
from src.state import (
    AgentState,
    BuildPhase,
    BuildPlan,
    BuildStatus,
    ErrorCategory,
    FixAttempt,
    PackageAnalysis,
    create_error_record,
    create_initial_state,
)


def _base_state(**overrides) -> AgentState:
    """Build a base AgentState with field overrides."""
    s = create_initial_state("https://github.com/a/b.git")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _analysis(**overrides) -> PackageAnalysis:
    """Build a PackageAnalysis with field overrides."""
    pa = PackageAnalysis(purpose="test package", llm_grounded=True)
    for k, v in overrides.items():
        setattr(pa, k, v)
    return pa


class TestRouteInit(unittest.TestCase):
    """Tests for route_init_to_next."""

    def test_successful_init_routes_to_analyst(self) -> None:
        s = _base_state(build_status=BuildStatus.PENDING)
        self.assertEqual(route_init_to_next(s), "analyst_node")

    def test_failed_init_routes_to_escalate(self) -> None:
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_init_to_next(s), "escalate_node")


class TestRouteAnalyst(unittest.TestCase):
    """Tests for route_analyst_to_next."""

    def test_analysis_routes_to_heuristic_plan(self) -> None:
        s = _base_state(package_analysis=_analysis())
        self.assertEqual(route_analyst_to_next(s), "heuristic_plan_node")

    def test_missing_analysis_escalates(self) -> None:
        """The analyst always leaves an analysis; None means it crashed."""
        s = _base_state(package_analysis=None)
        self.assertEqual(route_analyst_to_next(s), "escalate_node")


class TestRouteHeuristicPlan(unittest.TestCase):
    """Tests for route_heuristic_plan_to_next."""

    def test_materialized_plan_routes_to_supervisor(self) -> None:
        bp = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )
        s = _base_state(build_plan=bp)
        self.assertEqual(route_heuristic_plan_to_next(s), "supervisor_node")

    def test_no_plan_defers_to_llm_scout(self) -> None:
        s = _base_state(build_plan=None)
        self.assertEqual(route_heuristic_plan_to_next(s), "scout_node")


class TestHeuristicPlanNode(unittest.TestCase):
    """Tests for heuristic_plan_node's Go recipe and analyst consumption."""

    def test_go_plan_initializes_module_when_missing(self) -> None:
        """Go default plan adds go mod init for GOPATH-style repos."""
        from src.graph import heuristic_plan_node
        from src.state import BuildSystemInfo

        s = _base_state()
        s.build_system_info = BuildSystemInfo(
            type="go", confidence=0.7, primary_file="main.go"
        )
        s.context_cache["go_main_info"] = {
            "needs_go_init": True,
            "build_command": "go build .",
        }
        out = heuristic_plan_node(s)
        self.assertIsNotNone(out.build_plan)
        cmds = out.build_plan.phases[1].commands
        self.assertTrue(any(cmd.startswith("go mod init ") for cmd in cmds))
        self.assertIn("go mod tidy", cmds)
        self.assertTrue(any(cmd.startswith("go build ") for cmd in cmds))

    def test_analyst_custom_plan_veto_defers_to_scout(self) -> None:
        """needs_custom_plan from an LLM-grounded analysis defers.

        This is the load-bearing consumption path: the analyst READ the
        build files and says a textbook recipe will fail, so the
        heuristic node must NOT emit one even when detection confidence
        is high.
        """
        from src.graph import heuristic_plan_node
        from src.state import BuildSystemInfo

        s = _base_state()
        s.build_system_info = BuildSystemInfo(
            type="cmake", confidence=0.9, primary_file="CMakeLists.txt"
        )
        s.package_analysis = _analysis(
            build_system="cmake",
            build_system_confidence=0.9,
            needs_custom_plan=True,
        )
        out = heuristic_plan_node(s)
        self.assertIsNone(out.build_plan)

    def test_fallback_analysis_does_not_veto(self) -> None:
        """A non-grounded fallback analysis must not veto the recipe.

        Otherwise every package would defer to the LLM scout exactly
        when the LLM layer is starved.
        """
        from src.graph import heuristic_plan_node
        from src.state import BuildSystemInfo

        s = _base_state()
        s.build_system_info = BuildSystemInfo(
            type="make", confidence=0.9, primary_file="Makefile"
        )
        s.package_analysis = _analysis(
            needs_custom_plan=True, llm_grounded=False
        )
        out = heuristic_plan_node(s)
        self.assertIsNotNone(out.build_plan)
        self.assertEqual(out.build_plan.build_system, "make")

    def test_analyst_build_system_overrides_weak_detection(self) -> None:
        """An LLM-grounded, more-confident build-system verdict wins."""
        from src.graph import heuristic_plan_node
        from src.state import BuildSystemInfo

        s = _base_state()
        s.build_system_info = BuildSystemInfo(
            type="make", confidence=0.55, primary_file="Makefile"
        )
        s.package_analysis = _analysis(
            build_system="cmake",
            build_system_confidence=0.95,
            needs_custom_plan=False,
        )
        out = heuristic_plan_node(s)
        self.assertIsNotNone(out.build_plan)
        self.assertEqual(out.build_plan.build_system, "cmake")


class TestRouteSupervisor(unittest.TestCase):
    """Tests for route_supervisor_to_next."""

    def test_success_routes_to_finish(self) -> None:
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_success_at_max_attempts_still_finishes(self) -> None:
        """A build that succeeded on its final attempt must FINISH.

        Regression: should_escalate() was consulted before the SUCCESS
        check, so a package whose last fix landed on attempt
        max_attempts was escalated and its recipe thrown away.
        """
        s = _base_state(
            build_status=BuildStatus.SUCCESS,
            attempt_count=5,
            max_attempts=5,
        )
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_success_over_cost_cap_still_finishes(self) -> None:
        s = _base_state(build_status=BuildStatus.SUCCESS, api_cost_usd=1.5)
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_max_attempts_escalates(self) -> None:
        s = _base_state(
            attempt_count=5, max_attempts=5, build_status=BuildStatus.FAILED
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_error_loop_escalates(self) -> None:
        from src.state import ErrorRecord

        err = ErrorRecord(message="fail", category=ErrorCategory.COMPILATION)
        s = _base_state(
            attempt_count=3,
            max_attempts=5,
            error_history=[err, err, err],
            build_status=BuildStatus.FAILED,
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_missing_deps_routes_to_scout(self) -> None:
        from src.state import ErrorRecord

        err = ErrorRecord(
            message="pkg not found",
            category=ErrorCategory.DEPENDENCY,
        )
        s = _base_state(
            package_analysis=_analysis(),
            build_status=BuildStatus.FAILED,
            last_error_category=ErrorCategory.DEPENDENCY,
            error_history=[err],
        )
        self.assertEqual(route_supervisor_to_next(s), "scout_node")

    def test_no_analysis_routes_to_analyst(self) -> None:
        s = _base_state(
            package_analysis=None, build_status=BuildStatus.PENDING
        )
        self.assertEqual(route_supervisor_to_next(s), "analyst_node")

    def test_pending_build_routes_to_subgraph(self) -> None:
        bp = BuildPlan(
            build_system="make",
            build_system_confidence=0.8,
            phases=[BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )
        s = _base_state(
            package_analysis=_analysis(),
            build_plan=bp,
            build_status=BuildStatus.PENDING,
        )
        self.assertEqual(route_supervisor_to_next(s), "build_fix_subgraph")

    def test_failed_compilation_routes_to_subgraph(self) -> None:
        """Failed non-replan errors route into the build-fix subgraph."""
        s = _base_state(
            package_analysis=_analysis(),
            build_status=BuildStatus.FAILED,
            last_error_category=ErrorCategory.COMPILATION,
            last_error="compiler error",
        )
        self.assertEqual(route_supervisor_to_next(s), "build_fix_subgraph")

    def test_failed_disk_space_routes_to_escalate(self) -> None:
        """Infrastructure capacity failures route to human handoff."""
        s = _base_state(
            package_analysis=_analysis(),
            build_status=BuildStatus.FAILED,
            last_error_category=ErrorCategory.DISK_SPACE,
            last_error="No space left on device",
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_failed_permission_routes_to_escalate(self) -> None:
        """Permission failures route to human handoff, not the fixer."""
        s = _base_state(
            package_analysis=_analysis(),
            build_status=BuildStatus.FAILED,
            last_error_category=ErrorCategory.PERMISSION,
            last_error="Permission denied",
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_pending_without_plan_routes_to_scout(self) -> None:
        """Pending analyzed state without a plan asks the scout to plan."""
        s = _base_state(
            package_analysis=_analysis(),
            build_plan=None,
            build_status=BuildStatus.PENDING,
        )
        self.assertEqual(route_supervisor_to_next(s), "scout_node")


class TestBuildFixSubgraphRouting(unittest.TestCase):
    """Tests for the build-fix subgraph routing."""

    def test_build_success_routes_to_verify(self) -> None:
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_build_result(s), "verify_node")

    def test_build_fail_routes_to_fix(self) -> None:
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_build_result(s), "fix_node")

    def test_build_fail_at_max_attempts_exits_subgraph(self) -> None:
        """A failed build at the retry ceiling exits to the parent graph."""
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=5,
            max_attempts=5,
        )
        self.assertEqual(route_build_result(s), "__end__")

    def test_verify_success_exits_subgraph(self) -> None:
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_verify_result(s), "__end__")

    def test_verify_fail_routes_to_fix(self) -> None:
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_verify_result(s), "fix_node")

    def test_verify_fail_at_max_attempts_exits_subgraph(self) -> None:
        """A verify failure at max attempts exits instead of fixing."""
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=5,
            max_attempts=5,
        )
        self.assertEqual(route_verify_result(s), "__end__")

    def test_verify_fail_after_non_build_fix_exits_subgraph(self) -> None:
        """Post-fix non-build verification failures do not loop forever."""
        s = _base_state(
            build_status=BuildStatus.FAILED,
            last_error_category=ErrorCategory.CONFIGURATION,
            fixes_attempted=[
                FixAttempt(
                    error_category=ErrorCategory.CONFIGURATION,
                    strategy="change plan",
                    changes_made=["Executed: true"],
                    success=False,
                )
            ],
        )
        self.assertEqual(route_verify_result(s), "__end__")

    def test_fix_success_retries_build(self) -> None:
        s = _base_state(build_status=BuildStatus.PENDING)
        self.assertEqual(route_fix_result(s), "build_node")

    def test_fix_still_failed_exits_subgraph(self) -> None:
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_fix_result(s), "__end__")

    def test_fix_at_max_attempts_exits_subgraph(self) -> None:
        """A nominally successful fix still exits at the retry ceiling."""
        s = _base_state(
            build_status=BuildStatus.PENDING,
            attempt_count=5,
            max_attempts=5,
        )
        self.assertEqual(route_fix_result(s), "__end__")


_BLOCKED = (
    "Build command failed: cabal --version (exit 1) - Command blocked by "
    "safety validation: Unknown command pattern (not in whitelist)"
)


def _fix(build_result=None) -> FixAttempt:
    """Build a failed FixAttempt with an optional build_result marker."""
    return FixAttempt(
        error_category=ErrorCategory.COMPILATION,
        strategy="patch the source",
        changes_made=["Executed: sed -i x a.c"],
        success=False,
        build_result=build_result,
    )


def _error(message: str):
    """Build a COMPILATION error record for ``message``."""
    return create_error_record(
        message, ErrorCategory.COMPILATION, command="make"
    )


class TestFixLoopStopRules(unittest.TestCase):
    """The fixer is not called when another round cannot help."""

    def test_validator_block_exits_subgraph_without_fixer(self) -> None:
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=1,
            error_history=[_error(_BLOCKED)],
        )
        self.assertEqual(route_build_result(s), "__end__")

    def test_supervisor_replans_once_after_validator_block(self) -> None:
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=1,
            package_analysis=_analysis(),
            error_history=[_error(_BLOCKED)],
        )
        self.assertEqual(route_supervisor_to_next(s), "scout_node")
        s.context_cache["validator_block_replans"] = 1
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_unchanged_error_after_two_fixes_stops(self) -> None:
        msg = "Build command failed: make (exit 2) - a.c:3: error: x"
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=3,
            package_analysis=_analysis(),
            fixes_attempted=[_fix(), _fix()],
            error_history=[_error("other failure"), _error(msg), _error(msg)],
        )
        self.assertEqual(route_build_result(s), "__end__")
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_error_category_alone_does_not_skip_the_fixer(self) -> None:
        """A noisy PERMISSION or DISK_SPACE category still gets fixes."""
        for category in (ErrorCategory.PERMISSION, ErrorCategory.DISK_SPACE):
            with self.subTest(category=category):
                s = _base_state(
                    build_status=BuildStatus.FAILED, attempt_count=1
                )
                s.add_error(
                    create_error_record(
                        "Build command failed: bash ./build.sh (exit 1) - "
                        "warning: unable to access '/root/.config/git/"
                        "ignore': Permission denied",
                        category,
                        command="bash ./build.sh",
                    )
                )
                self.assertEqual(_fix_loop_stop_reason(s), "")
                self.assertEqual(route_build_result(s), "fix_node")

    def test_changed_error_after_two_fixes_continues(self) -> None:
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=3,
            fixes_attempted=[_fix(), _fix()],
            error_history=[
                _error("Build command failed: make - a.c:3: error: x"),
                _error("Build command failed: make - b.c:9: error: y"),
            ],
        )
        self.assertEqual(route_build_result(s), "fix_node")

    def test_one_fix_with_unchanged_error_gets_another_round(self) -> None:
        msg = "Build command failed: make (exit 2) - a.c:3: error: x"
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=2,
            fixes_attempted=[_fix()],
            error_history=[_error(msg), _error(msg)],
        )
        self.assertEqual(route_build_result(s), "fix_node")

    def test_refused_package_stops_the_loop(self) -> None:
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=1,
            package_analysis=_analysis(),
            error_history=[_error("Build command failed: make - error")],
        )
        s.context_cache["llm_refused"] = "fixer"
        self.assertEqual(route_build_result(s), "__end__")
        self.assertEqual(route_verify_result(s), "__end__")
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_no_change_fix_goes_back_to_the_fixer(self) -> None:
        s = _base_state(
            build_status=BuildStatus.FIXING,
            attempt_count=2,
            fixes_attempted=[_fix(build_result="no change applied")],
        )
        self.assertEqual(route_fix_result(s), "fix_node")
        s.attempt_count = 5
        self.assertEqual(route_fix_result(s), "__end__")

    def test_invalid_fixer_round_stops_the_loop(self) -> None:
        """A fixer round without a valid answer is not rebuilt and retried."""
        invalid = (
            "FIXER proposed invalid fix: schema validation failed: Unsafe "
            "command: Unknown command pattern (not in whitelist)"
        )
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=2,
            package_analysis=_analysis(),
            error_history=[
                _error("Build command failed: make - x"),
                _error(invalid),
            ],
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_fixer_provider_fault_gets_another_round(self) -> None:
        """A timeout is a provider fault, so the loop goes on."""
        fault = (
            "FIXER proposed invalid fix: LLM invocation raised: LLM call "
            "timed out after 135s"
        )
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=2,
            package_analysis=_analysis(),
            build_plan=BuildPlan(
                build_system="make",
                build_system_confidence=0.9,
                phases=[BuildPhase(1, "build", ["make"])],
                total_estimated_duration="1m",
            ),
            error_history=[
                _error("Build command failed: make - x"),
                _error(fault),
            ],
        )
        self.assertEqual(route_supervisor_to_next(s), "build_fix_subgraph")

    def test_fix_node_loops_to_itself_in_the_graph(self) -> None:
        from src.graph import create_build_fix_subgraph

        graph = create_build_fix_subgraph().get_graph()
        targets = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("fix_node", "fix_node"), targets)
        self.assertIn(("fix_node", "build_node"), targets)


if __name__ == "__main__":
    unittest.main()


class TestEscalatedRouting(unittest.TestCase):
    """ESCALATED (a crashed node) must end the run, not loop back."""

    def test_supervisor_sends_escalated_to_escalate(self) -> None:
        """The supervisor has a terminal branch for ESCALATED."""
        s = _base_state(
            build_status=BuildStatus.ESCALATED,
            package_analysis=_analysis(),
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_subgraph_routers_exit_on_escalated(self) -> None:
        """Build, verify and fix routers leave the subgraph."""
        for router in (
            route_build_result,
            route_verify_result,
            route_fix_result,
        ):
            with self.subTest(router=router.__name__):
                s = _base_state(build_status=BuildStatus.ESCALATED)
                self.assertEqual(router(s), "__end__")

    def test_crashed_init_escalates(self) -> None:
        """A crashed init_node must not hand a missing clone on."""
        s = _base_state(build_status=BuildStatus.ESCALATED)
        self.assertEqual(route_init_to_next(s), "escalate_node")


class TestRouteCachedRecipe(unittest.TestCase):
    """--optimize with a verified cached recipe skips the planning."""

    def test_init_goes_to_the_cached_recipe(self) -> None:
        """A cached recipe replaces the analyst; a failed init escalates."""
        s = _base_state(cached_recipe={"build_plan": {"phases": []}})
        self.assertEqual(route_init_to_next(s), "cached_recipe_node")
        s.build_status = BuildStatus.FAILED
        self.assertEqual(route_init_to_next(s), "escalate_node")

    def test_cached_recipe_builds_or_falls_back(self) -> None:
        """A plan goes to the supervisor; no plan plans as usual."""
        plan = BuildPlan("cmake", 1.0, [BuildPhase(1, "b", ["make"])], "1m")
        self.assertEqual(
            route_cached_recipe_to_next(_base_state(build_plan=plan)),
            "supervisor_node",
        )
        self.assertEqual(
            route_cached_recipe_to_next(_base_state()), "analyst_node"
        )

    def test_the_node_is_wired(self) -> None:
        """The compiled graph has the node between init and supervisor."""
        graph = create_workflow().get_graph()
        self.assertIn("cached_recipe_node", graph.nodes)
        targets = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("init_node", "cached_recipe_node"), targets)
        self.assertIn(("cached_recipe_node", "supervisor_node"), targets)
        self.assertIn(("cached_recipe_node", "analyst_node"), targets)


class TestRouteFinish(unittest.TestCase):
    """--optimize adds one step after finish_node."""

    def test_finish_ends_unless_optimize_was_requested(self) -> None:
        """Only a run with --optimize goes on to optimize_node."""
        self.assertEqual(route_finish_to_next(_base_state()), "__end__")
        s = _base_state(optimize_requested=True)
        self.assertEqual(route_finish_to_next(s), "optimize_node")
