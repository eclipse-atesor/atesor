"""Tests for the agent_node decorator.

Covers rate-limit retry and generic-error escalation.
"""

import unittest
from unittest import mock

from src.graph import agent_node
from src.state import AgentRole, BuildStatus, create_initial_state


class TestAgentNodeRateLimitRetry(unittest.TestCase):
    """The decorator retries on rate-limit errors with backoff."""

    def _make_node(self, side_effects):
        """Return a SCOUT node that replays ``side_effects``, and a counter.

        Each call takes the next effect: an exception is raised, and any
        other value becomes ``state.current_phase``.
        """
        calls = {"n": 0}

        @agent_node(AgentRole.SCOUT)
        def node(state):
            calls["n"] += 1
            eff = side_effects[calls["n"] - 1]
            if isinstance(eff, Exception):
                raise eff
            state.current_phase = eff
            return state

        return node, calls

    def test_rate_limit_then_success_returns_normal_state(self) -> None:
        node, calls = self._make_node(
            [RuntimeError("Rate limit exceeded"), "scouting"]
        )
        state = create_initial_state("https://x/y.git")
        with mock.patch("time.sleep") as msleep:
            out = node(state)
        self.assertEqual(out.current_phase, "scouting")
        self.assertEqual(calls["n"], 2)
        # First retry should sleep ~30s (plus jitter)
        first_wait = msleep.call_args_list[0].args[0]
        self.assertGreaterEqual(first_wait, 30)
        self.assertLess(first_wait, 45)

    def test_three_rate_limits_then_escalate(self) -> None:
        node, calls = self._make_node(
            [
                RuntimeError("429 Too Many Requests"),
                RuntimeError("rate limit"),
                RuntimeError("quota exceeded"),
            ]
        )
        state = create_initial_state("https://x/y.git")
        with mock.patch("time.sleep"):
            out = node(state)
        # After exhausting retries the decorator falls through to the generic
        # error handler on the final attempt → escalate
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)
        self.assertEqual(out.current_phase, "escalate")
        self.assertEqual(calls["n"], 3)

    def test_backoff_doubles_each_retry(self) -> None:
        node, _ = self._make_node(
            [
                RuntimeError("rate limit"),
                RuntimeError("rate limit"),
                "scouting",
            ]
        )
        with mock.patch("time.sleep") as msleep:
            node(create_initial_state("https://x/y.git"))
        # Exponential backoff with jitter: ~30s then ~60s
        calls = [c.args[0] for c in msleep.call_args_list]
        self.assertGreaterEqual(calls[0], 30)
        self.assertLess(calls[0], 45)
        self.assertGreaterEqual(calls[1], 60)
        self.assertLess(calls[1], 75)

    def test_all_rate_limit_terms_recognised(self) -> None:
        terms = [
            "rate limit",
            "429 too many",
            "Too Many Requests",
            "quota exceeded",
            "RESOURCE_EXHAUSTED",
            "402 payment required",
            "Spend limit exceeded",
            "USD spend limit",
        ]
        for term in terms:
            with self.subTest(term=term):
                node, calls = self._make_node([RuntimeError(term), "scouting"])
                with mock.patch("time.sleep"):
                    node(create_initial_state("https://x/y.git"))
                self.assertEqual(calls["n"], 2, f"Did not retry on: {term}")


class TestAgentNodeGenericError(unittest.TestCase):
    """Generic exceptions escalate; they don't bubble.

    The class also checks that the wrapper sets ``current_agent``.
    """

    def test_runtime_error_escalates(self) -> None:
        @agent_node(AgentRole.BUILDER)
        def node(state) -> None:
            raise ValueError("kaboom — invalid foo")

        state = create_initial_state("https://x/y.git")
        out = node(state)
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)
        self.assertEqual(out.current_phase, "escalate")
        self.assertIn("kaboom", out.last_error)
        self.assertTrue(len(out.error_history) >= 1)

    def test_exception_does_not_propagate(self) -> None:
        @agent_node(AgentRole.FIXER)
        def node(state) -> None:
            raise KeyError("missing 'plan'")

        try:
            out = node(create_initial_state("https://x/y.git"))
        except Exception as e:  # pragma: no cover - failure case
            self.fail(f"agent_node should swallow exceptions, got {e!r}")
        self.assertEqual(out.current_phase, "escalate")

    def test_current_agent_set_before_invocation(self) -> None:
        captured = {}

        @agent_node(AgentRole.PLANNER)
        def node(state):
            captured["role"] = state.current_agent
            return state

        node(create_initial_state("https://x/y.git"))
        self.assertEqual(captured["role"], AgentRole.PLANNER)


if __name__ == "__main__":
    unittest.main()
