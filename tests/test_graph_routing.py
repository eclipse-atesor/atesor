"""Tests for graph routing: init, analyst, supervisor, build-fix subgraph."""

import unittest

from src.graph import (
    route_analyst_to_next,
    route_build_result,
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

    def test_successful_init_routes_to_analyst(self):
        """Successful init routes to the analyst."""
        s = _base_state(build_status=BuildStatus.PENDING)
        self.assertEqual(route_init_to_next(s), "analyst_node")

    def test_failed_init_routes_to_escalate(self):
        """Failed init routes to escalate."""
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_init_to_next(s), "escalate_node")


class TestRouteAnalyst(unittest.TestCase):
    """Tests for route_analyst_to_next."""

    def test_analysis_routes_to_heuristic_plan(self):
        """A produced analysis enters the heuristic plan node."""
        s = _base_state(package_analysis=_analysis())
        self.assertEqual(route_analyst_to_next(s), "heuristic_plan_node")

    def test_missing_analysis_escalates(self):
        """The analyst always leaves an analysis; None means it crashed."""
        s = _base_state(package_analysis=None)
        self.assertEqual(route_analyst_to_next(s), "escalate_node")


class TestRouteHeuristicPlan(unittest.TestCase):
    """Tests for route_heuristic_plan_to_next."""

    def test_materialized_plan_routes_to_supervisor(self):
        """A materialized default plan routes to the supervisor."""
        bp = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )
        s = _base_state(build_plan=bp)
        self.assertEqual(route_heuristic_plan_to_next(s), "supervisor_node")

    def test_no_plan_defers_to_llm_scout(self):
        """No plan defers to the LLM scout."""
        s = _base_state(build_plan=None)
        self.assertEqual(route_heuristic_plan_to_next(s), "scout_node")


class TestHeuristicPlanNode(unittest.TestCase):
    """Tests for heuristic_plan_node's analyst consumption."""

    def test_go_plan_initializes_module_when_missing(self):
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

    def test_analyst_custom_plan_veto_defers_to_scout(self):
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

    def test_fallback_analysis_does_not_veto(self):
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

    def test_analyst_build_system_overrides_weak_detection(self):
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

    def test_success_routes_to_finish(self):
        """Success routes to finish."""
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_success_at_max_attempts_still_finishes(self):
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

    def test_success_over_cost_cap_still_finishes(self):
        """Success over cost cap still finishes."""
        s = _base_state(build_status=BuildStatus.SUCCESS, api_cost_usd=1.5)
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_max_attempts_escalates(self):
        """Max attempts escalates."""
        s = _base_state(
            attempt_count=5, max_attempts=5, build_status=BuildStatus.FAILED
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_error_loop_escalates(self):
        """Error loop escalates."""
        from src.state import ErrorRecord

        err = ErrorRecord(message="fail", category=ErrorCategory.COMPILATION)
        s = _base_state(
            attempt_count=3,
            max_attempts=5,
            error_history=[err, err, err],
            build_status=BuildStatus.FAILED,
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_missing_deps_routes_to_scout(self):
        """Missing deps routes to scout."""
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

    def test_no_analysis_routes_to_analyst(self):
        """No package analysis routes back to the analyst."""
        s = _base_state(
            package_analysis=None, build_status=BuildStatus.PENDING
        )
        self.assertEqual(route_supervisor_to_next(s), "analyst_node")

    def test_pending_build_routes_to_subgraph(self):
        """Pending build routes to subgraph."""
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

    def test_failed_compilation_routes_to_subgraph(self):
        """Failed non-replan errors route into the build-fix subgraph."""
        s = _base_state(
            package_analysis=_analysis(),
            build_status=BuildStatus.FAILED,
            last_error_category=ErrorCategory.COMPILATION,
            last_error="compiler error",
        )
        self.assertEqual(route_supervisor_to_next(s), "build_fix_subgraph")

    def test_pending_without_plan_routes_to_scout(self):
        """Pending analyzed state without a plan asks the scout to plan."""
        s = _base_state(
            package_analysis=_analysis(),
            build_plan=None,
            build_status=BuildStatus.PENDING,
        )
        self.assertEqual(route_supervisor_to_next(s), "scout_node")


class TestBuildFixSubgraphRouting(unittest.TestCase):
    """Tests for the build-fix subgraph routing."""

    def test_build_success_routes_to_verify(self):
        """Build success routes to verify."""
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_build_result(s), "verify_node")

    def test_build_fail_routes_to_fix(self):
        """Build fail routes to fix."""
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_build_result(s), "fix_node")

    def test_build_fail_at_max_attempts_exits_subgraph(self):
        """A failed build at the retry ceiling exits to the parent graph."""
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=5,
            max_attempts=5,
        )
        self.assertEqual(route_build_result(s), "__end__")

    def test_verify_success_exits_subgraph(self):
        """Verify success exits subgraph."""
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_verify_result(s), "__end__")

    def test_verify_fail_routes_to_fix(self):
        """Verify fail routes to fix."""
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_verify_result(s), "fix_node")

    def test_verify_fail_at_max_attempts_exits_subgraph(self):
        """A verify failure at max attempts exits instead of fixing."""
        s = _base_state(
            build_status=BuildStatus.FAILED,
            attempt_count=5,
            max_attempts=5,
        )
        self.assertEqual(route_verify_result(s), "__end__")

    def test_verify_fail_after_non_build_fix_exits_subgraph(self):
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

    def test_fix_success_retries_build(self):
        """Fix success retries build."""
        s = _base_state(build_status=BuildStatus.PENDING)
        self.assertEqual(route_fix_result(s), "build_node")

    def test_fix_still_failed_exits_subgraph(self):
        """Fix still failed exits subgraph."""
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_fix_result(s), "__end__")

    def test_fix_at_max_attempts_exits_subgraph(self):
        """A nominally successful fix still exits at the retry ceiling."""
        s = _base_state(
            build_status=BuildStatus.PENDING,
            attempt_count=5,
            max_attempts=5,
        )
        self.assertEqual(route_fix_result(s), "__end__")


if __name__ == "__main__":
    unittest.main()
