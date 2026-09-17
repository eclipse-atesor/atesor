"""Aggressive tests for src/state.py.

src/state.py is the single source of truth for agent state.

These cover:
  * AgentState construction + mutation helpers (add_error,
    add_fix_attempt, caches, audit trail, command-key collisions).
  * Enum stability (string serialization, no accidental renames).
  * classify_error: parametrized over every known pattern, including:
      - ordering hazards ("rate limit" must beat "timeout"),
      - Go-specific signatures,
      - Python exception class names → CONFIGURATION,
      - package-manager lock contention → DEPENDENCY,
      - unknown messages → UNKNOWN.
  * infer_failure_severity: covers the LOW/HIGH overrides and the
    broad-MEDIUM bucket.
  * should_escalate: every escalation trigger (max attempts, error
    loop, fundamental blocker, $ cap).
  * get_next_action_recommendation: covers the full decision tree.
  * to_dict / save_to_json roundtrip.
"""

from __future__ import annotations

import json
import unittest

from src.state import (
    MAX_AUDIT_EVENTS,
    MAX_ERROR_HISTORY,
    Action,
    AgentRole,
    AgentState,
    BuildPlan,
    BuildStatus,
    CommandResult,
    ErrorCategory,
    ErrorRecord,
    FailureSeverity,
    FixAttempt,
    TaskPlan,
    classify_error,
    create_error_record,
    create_initial_state,
    derive_repo_name,
    get_next_action_recommendation,
    infer_failure_severity,
    should_escalate,
)

# ===========================================================================
# create_initial_state + AgentState construction
# ===========================================================================


class TestInitialState(unittest.TestCase):
    """Tests for InitialState."""

    def test_initial_state_basic(self) -> None:
        """Test initial state basic."""
        state = create_initial_state("https://github.com/madler/zlib")
        self.assertEqual(state.repo_name, "zlib")
        self.assertEqual(state.repo_path, "/workspace/repos/zlib")
        self.assertEqual(state.build_status, BuildStatus.PENDING)
        self.assertEqual(state.attempt_count, 0)
        self.assertEqual(state.max_attempts, 5)
        self.assertIsNone(state.task_plan)
        self.assertIsNone(state.build_plan)
        self.assertEqual(state.current_phase, "initialization")

    def test_repo_name_strips_dot_git(self) -> None:
        """Test repo name strips dot git."""
        state = create_initial_state("https://github.com/foo/bar.git")
        self.assertEqual(state.repo_name, "bar")
        self.assertEqual(state.repo_path, "/workspace/repos/bar")

    def test_repo_name_strips_trailing_slash(self) -> None:
        """Test repo name strips trailing slash."""
        state = create_initial_state("https://github.com/foo/bar/")
        self.assertEqual(state.repo_name, "bar")

    def test_custom_max_attempts_honored(self) -> None:
        """Test custom max attempts honored."""
        state = create_initial_state(
            "https://github.com/foo/bar", max_attempts=12
        )
        self.assertEqual(state.max_attempts, 12)

    def test_history_collections_are_independent_instances(self) -> None:
        """Regression: mutable defaults must use field(default_factory)."""
        s1 = create_initial_state("https://github.com/a/b")
        s2 = create_initial_state("https://github.com/c/d")
        s1.error_history.append(
            ErrorRecord(category=ErrorCategory.COMPILATION, message="x")
        )
        self.assertEqual(
            len(s2.error_history),
            0,
            "AgentState collections leaked across instances",
        )


# ===========================================================================
# add_error / log_api_call / cache / audit trail
# ===========================================================================


class TestStateMutation(unittest.TestCase):
    """Tests for StateMutation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.state = create_initial_state("https://github.com/x/y")

    def test_add_error_updates_all_fields_and_attempt(self) -> None:
        """Test add error updates all fields and attempt."""
        err = create_error_record(
            "Permission denied", ErrorCategory.PERMISSION
        )
        self.state.add_error(err)

        self.assertEqual(self.state.last_error, "Permission denied")
        self.assertEqual(
            self.state.last_error_category, ErrorCategory.PERMISSION
        )
        # PERMISSION is in the HIGH category set
        self.assertEqual(self.state.last_error_severity, FailureSeverity.HIGH)
        self.assertEqual(self.state.attempt_count, 1)
        self.assertEqual(len(self.state.error_history), 1)

    def test_consecutive_add_error_increments_attempts(self) -> None:
        """Test consecutive add error increments attempts."""
        for i in range(4):
            self.state.add_error(
                create_error_record(f"err {i}", ErrorCategory.COMPILATION)
            )
        self.assertEqual(self.state.attempt_count, 4)
        self.assertEqual(len(self.state.error_history), 4)

    def test_log_api_call_accumulates_cost(self) -> None:
        """Test log api call accumulates cost."""
        self.state.log_api_call(cost=0.01)
        self.state.log_api_call(cost=0.005)
        self.state.log_api_call()  # default 0
        self.assertEqual(self.state.api_calls_made, 3)
        self.assertAlmostEqual(self.state.api_cost_usd, 0.015)

    def test_command_cache_stores_and_retrieves(self) -> None:
        """Test command cache stores and retrieves."""
        cr = CommandResult("ls -la", 0, "out", "", 0.1)
        self.state.cache_command_result("ls -la", cr)
        got = self.state.get_cached_command_result("ls -la")
        self.assertIs(got, cr)

    def test_command_cache_miss_returns_none(self) -> None:
        """Test command cache miss returns none."""
        self.assertIsNone(self.state.get_cached_command_result("never-run"))

    def test_command_cache_keyed_by_hash_avoids_collisions(self) -> None:
        """Test command cache keyed by hash avoids collisions."""
        a = CommandResult("a", 0, "A", "", 0.1)
        b = CommandResult("b", 0, "B", "", 0.1)
        self.state.cache_command_result("a", a)
        self.state.cache_command_result("b", b)
        self.assertEqual(self.state.get_cached_command_result("a").stdout, "A")
        self.assertEqual(self.state.get_cached_command_result("b").stdout, "B")

    def test_log_event_records_agent_and_timestamp(self) -> None:
        """Test log event records agent and timestamp."""
        self.state.current_agent = AgentRole.SCOUT
        self.state.log_event("test_event", {"foo": 1})
        events = self.state.get_last_audit_events(limit=1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "test_event")
        self.assertEqual(events[0]["agent"], "scout")
        self.assertEqual(events[0]["data"], {"foo": 1})
        # ISO timestamp present
        self.assertIn("T", events[0]["timestamp"])

    def test_get_last_audit_events_limit(self) -> None:
        """Test get last audit events limit."""
        for i in range(10):
            self.state.log_event(f"e{i}", {})
        self.assertEqual(len(self.state.get_last_audit_events(limit=3)), 3)
        self.assertEqual(
            self.state.get_last_audit_events(limit=3)[-1]["event"], "e9"
        )

    def test_log_agent_decision_writes_decision_record(self) -> None:
        """Test log agent decision writes decision record."""
        self.state.log_agent_decision(AgentRole.SUPERVISOR, "SCOUT", "because")
        last = self.state.audit_trail[-1]
        self.assertEqual(last["event"], "decision")
        self.assertEqual(last["data"]["action"], "SCOUT")

    def test_add_build_artifact_records_metadata(self) -> None:
        """Test add build artifact records metadata."""
        self.state.add_build_artifact(
            "/p/binary", "binary", architecture="RISC-V"
        )
        self.assertEqual(len(self.state.build_artifacts), 1)
        art = self.state.build_artifacts[0]
        self.assertEqual(art["filepath"], "/p/binary")
        self.assertEqual(art["type"], "binary")
        self.assertEqual(art["architecture"], "RISC-V")


# ===========================================================================
# Error loop detection
# ===========================================================================


class TestErrorLoop(unittest.TestCase):
    """Tests for ErrorLoop."""

    def test_not_a_loop_with_fewer_than_three(self) -> None:
        """Test not a loop with fewer than three."""
        state = create_initial_state("https://github.com/a/b")
        for _ in range(2):
            state.add_error(
                create_error_record("e", ErrorCategory.COMPILATION)
            )
        self.assertFalse(state.is_in_error_loop())

    def test_loop_when_three_consecutive_same_category(self) -> None:
        """Test loop when three consecutive same category."""
        state = create_initial_state("https://github.com/a/b")
        for _ in range(3):
            state.add_error(
                create_error_record("e", ErrorCategory.COMPILATION)
            )
        self.assertTrue(state.is_in_error_loop())

    def test_mixed_categories_not_a_loop(self) -> None:
        """Test mixed categories not a loop."""
        state = create_initial_state("https://github.com/a/b")
        state.add_error(create_error_record("a", ErrorCategory.COMPILATION))
        state.add_error(create_error_record("b", ErrorCategory.LINKING))
        state.add_error(create_error_record("c", ErrorCategory.COMPILATION))
        self.assertFalse(state.is_in_error_loop())

    def test_loop_examines_only_last_three(self) -> None:
        """Older errors must not prevent loop detection on the last 3."""
        state = create_initial_state("https://github.com/a/b")
        state.add_error(create_error_record("old", ErrorCategory.NETWORK))
        for _ in range(3):
            state.add_error(
                create_error_record("e", ErrorCategory.COMPILATION)
            )
        self.assertTrue(state.is_in_error_loop())


# ===========================================================================
# classify_error — exhaustive parametrized tests
# ===========================================================================


class TestClassifyError(unittest.TestCase):
    """Tests for ClassifyError."""

    # (message, expected category) tuples — covers every branch in
    # state.classify_error.
    CASES = [
        # Rate limiting (must beat network "timeout" check)
        ("HTTP 429 Too Many Requests", ErrorCategory.RATE_LIMIT),
        ("rate limit exceeded for project", ErrorCategory.RATE_LIMIT),
        ("quota exceeded", ErrorCategory.RATE_LIMIT),
        # Network
        ("network unreachable", ErrorCategory.NETWORK),
        ("connection reset by peer", ErrorCategory.NETWORK),
        ("operation timeout", ErrorCategory.NETWORK),
        # Linking
        ("cannot find -lssl", ErrorCategory.LINKING),
        ("undefined symbol foo", ErrorCategory.LINKING),
        ("collect2: error: ld returned 1 exit status", ErrorCategory.LINKING),
        # Autotools/configure before COMPILATION (syntax error inside
        # configure).
        (
            "possibly undefined macro AM_GNU_GETTEXT",
            ErrorCategory.CONFIGURATION,
        ),
        (
            "configure: syntax error near unexpected token",
            ErrorCategory.CONFIGURATION,
        ),
        ("autoreconf: error: aclocal failed", ErrorCategory.CONFIGURATION),
        # Compilation
        ("undefined reference to 'main'", ErrorCategory.COMPILATION),
        ("error: 'foo' undeclared", ErrorCategory.COMPILATION),
        ("implicit declaration of function", ErrorCategory.COMPILATION),
        ("PATH_MAX unset, refusing to compile", ErrorCategory.COMPILATION),
        # Architecture
        ("illegal instruction", ErrorCategory.ARCHITECTURE),
        ("unsupported instruction vsetvli", ErrorCategory.ARCHITECTURE),
        ("AVX2 intrinsic not available", ErrorCategory.ARCHITECTURE),
        ("uses ARM NEON", ErrorCategory.ARCHITECTURE),
        # CONFIGURATION (Go errors + cmake)
        ("cmake error: missing variable", ErrorCategory.CONFIGURATION),
        ("no Go files in /workspace/repos/foo", ErrorCategory.CONFIGURATION),
        ("no buildable Go source files in .", ErrorCategory.CONFIGURATION),
        (
            "directory prefix . does not contain main module",
            ErrorCategory.CONFIGURATION,
        ),
        (
            'go: build output "cmd" already exists and is a directory',
            ErrorCategory.CONFIGURATION,
        ),
        (
            "go: inconsistent vendoring in /workspace/repos/foo",
            ErrorCategory.CONFIGURATION,
        ),
        (
            "CMake Error: The source directory "
            '"/workspace/repos/foo" does not appear to '
            "contain CMakeLists.txt.",
            ErrorCategory.CONFIGURATION,
        ),
        ("unrecognized option '--with-foo'", ErrorCategory.CONFIGURATION),
        ("No rule to make target 'all'", ErrorCategory.CONFIGURATION),
        # Missing tools
        ("cmake: not found", ErrorCategory.MISSING_TOOLS),
        ("/bin/sh: ninja: command not found", ErrorCategory.MISSING_TOOLS),
        ("go.mod requires go >= 1.22", ErrorCategory.MISSING_TOOLS),
        (
            "feature `edition2024` is required; "
            "not stabilized in this version of Cargo (1.75.0)",
            ErrorCategory.MISSING_TOOLS,
        ),
        (
            "error: this package requires rustc 1.85.0 or newer",
            ErrorCategory.MISSING_TOOLS,
        ),
        # Dependency (package not found)
        ("cannot find header 'zlib.h'", ErrorCategory.DEPENDENCY),
        ("package not found: foo", ErrorCategory.DEPENDENCY),
        ("module not found 'bar'", ErrorCategory.DEPENDENCY),
        ("unable to select packages", ErrorCategory.DEPENDENCY),
        (
            "E: Unable to locate package libgssapi-krb5-dev",
            ErrorCategory.DEPENDENCY,
        ),
        # Permission
        ("Permission denied: /usr/local/lib", ErrorCategory.PERMISSION),
        ("Operation forbidden", ErrorCategory.PERMISSION),
        # Disk space
        ("no space left on device", ErrorCategory.DISK_SPACE),
        ("disk full", ErrorCategory.DISK_SPACE),
        # Python exceptions → CONFIGURATION (code/config bug)
        ("KeyError: 'phases'", ErrorCategory.CONFIGURATION),
        (
            "AttributeError: 'NoneType' object has no attribute 'foo'",
            ErrorCategory.CONFIGURATION,
        ),
        # Unknown
        ("everything is fine", ErrorCategory.UNKNOWN),
        ("", ErrorCategory.UNKNOWN),
    ]

    def test_all_known_patterns(self) -> None:
        """Test all known patterns."""
        for msg, expected in self.CASES:
            with self.subTest(msg=msg):
                self.assertEqual(
                    classify_error(msg),
                    expected,
                    f"classify_error({msg!r}) -> wrong category",
                )

    def test_rate_limit_beats_timeout(self) -> None:
        """Regression: 'rate limit ... timeout' is RATE_LIMIT, not NETWORK."""
        self.assertEqual(
            classify_error("rate limit hit, request timeout"),
            ErrorCategory.RATE_LIMIT,
        )

    def test_classify_is_case_insensitive(self) -> None:
        """Test classify is case insensitive."""
        self.assertEqual(
            classify_error("CMAKE: NOT FOUND"), ErrorCategory.MISSING_TOOLS
        )
        self.assertEqual(
            classify_error("Permission Denied"), ErrorCategory.PERMISSION
        )


# ===========================================================================
# infer_failure_severity
# ===========================================================================


class TestInferFailureSeverity(unittest.TestCase):
    """Tests for InferFailureSeverity."""

    def test_which_probe_is_low(self) -> None:
        """Test which probe is low."""
        self.assertEqual(
            infer_failure_severity(
                ErrorCategory.MISSING_TOOLS, command="which ninja"
            ),
            FailureSeverity.LOW,
        )

    def test_high_categories_always_high(self) -> None:
        """Test high categories always high."""
        for cat in [
            ErrorCategory.LICENSE_INCOMPATIBLE,
            ErrorCategory.REQUIRES_HARDWARE,
            ErrorCategory.ARCHITECTURE_IMPOSSIBLE,
            ErrorCategory.PERMISSION,
            ErrorCategory.DISK_SPACE,
        ]:
            with self.subTest(cat=cat):
                self.assertEqual(
                    infer_failure_severity(cat), FailureSeverity.HIGH
                )

    def test_clone_pull_apk_high(self) -> None:
        """Test clone pull apk high."""
        for cmd in [
            "git clone --depth 1 https://x/y",
            "git pull origin main",
            "apk update",
            "apk add curl-dev",
            "apt-get update",
            "apt update",
        ]:
            with self.subTest(cmd=cmd):
                self.assertEqual(
                    infer_failure_severity(ErrorCategory.NETWORK, command=cmd),
                    FailureSeverity.HIGH,
                )

    def test_build_failure_medium(self) -> None:
        """Test build failure medium."""
        for cat in [
            ErrorCategory.CONFIGURATION,
            ErrorCategory.DEPENDENCY,
            ErrorCategory.COMPILATION,
            ErrorCategory.LINKING,
            ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.MISSING_TOOLS,
            ErrorCategory.ARCHITECTURE,
        ]:
            with self.subTest(cat=cat):
                self.assertEqual(
                    infer_failure_severity(cat, command="make -j4"),
                    FailureSeverity.MEDIUM,
                )

    def test_unknown_category_medium(self) -> None:
        """Test unknown category medium."""
        self.assertEqual(
            infer_failure_severity(ErrorCategory.UNKNOWN),
            FailureSeverity.MEDIUM,
        )


# ===========================================================================
# create_error_record auto-classify path
# ===========================================================================


class TestCreateErrorRecord(unittest.TestCase):
    """Tests for CreateErrorRecord."""

    def test_auto_classification(self) -> None:
        """Test auto classification."""
        rec = create_error_record("cannot find -lssl")
        self.assertEqual(rec.category, ErrorCategory.LINKING)
        # LINKING with non-clone command -> MEDIUM
        self.assertEqual(rec.severity, FailureSeverity.MEDIUM)

    def test_explicit_overrides_take_precedence(self) -> None:
        """Test explicit overrides take precedence."""
        rec = create_error_record(
            "anything",
            category=ErrorCategory.NETWORK,
            severity=FailureSeverity.HIGH,
            command="git clone http://x",
            attempt_number=4,
        )
        self.assertEqual(rec.category, ErrorCategory.NETWORK)
        self.assertEqual(rec.severity, FailureSeverity.HIGH)
        self.assertEqual(rec.command, "git clone http://x")
        self.assertEqual(rec.attempt_number, 4)


# ===========================================================================
# should_escalate
# ===========================================================================


class TestShouldEscalate(unittest.TestCase):
    """Tests for ShouldEscalate."""

    def _state(self, **kw) -> AgentState:
        """State."""
        s = create_initial_state("https://github.com/a/b")
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    def test_no_escalation_on_empty_state(self) -> None:
        """Test no escalation on empty state."""
        ok, reason = should_escalate(self._state())
        self.assertFalse(ok)
        self.assertEqual(reason, "")

    def test_escalate_when_max_attempts_reached(self) -> None:
        """Test escalate when max attempts reached."""
        s = self._state(max_attempts=3)
        for _ in range(3):
            s.add_error(create_error_record("x", ErrorCategory.COMPILATION))
        ok, reason = should_escalate(s)
        self.assertTrue(ok)
        self.assertIn("Maximum attempts", reason)

    def test_escalate_on_error_loop(self) -> None:
        """Test escalate on error loop."""
        s = self._state(max_attempts=99)  # ensure attempts aren't the trigger
        for _ in range(3):
            s.add_error(create_error_record("x", ErrorCategory.NETWORK))
        ok, reason = should_escalate(s)
        self.assertTrue(ok)
        self.assertIn("error loop", reason)

    def test_escalate_on_fundamental_blocker(self) -> None:
        """Test escalate on fundamental blocker."""
        s = self._state()
        s.last_error_category = ErrorCategory.LICENSE_INCOMPATIBLE
        ok, reason = should_escalate(s)
        self.assertTrue(ok)
        self.assertIn("Fundamental blocker", reason)

    def test_escalate_when_cost_exceeded(self) -> None:
        """Test escalate when cost exceeded."""
        s = self._state()
        s.api_cost_usd = 1.5
        ok, reason = should_escalate(s)
        self.assertTrue(ok)
        self.assertIn("cost limit", reason)


# ===========================================================================
# get_next_action_recommendation — decision tree
# ===========================================================================


class TestNextAction(unittest.TestCase):
    """Tests for NextAction."""

    def _state(self) -> AgentState:
        """State."""
        return create_initial_state("https://github.com/a/b")

    def test_plan_first(self) -> None:
        """Test plan first."""
        self.assertEqual(
            get_next_action_recommendation(self._state()), Action.PLAN
        )

    def test_scout_when_task_plan_but_no_build_plan(self) -> None:
        """Test scout when task plan but no build plan."""
        s = self._state()
        s.task_plan = TaskPlan(phases=[])
        self.assertEqual(get_next_action_recommendation(s), Action.SCOUT)

    def test_builder_when_pending_with_plan(self) -> None:
        """Test builder when pending with plan."""
        s = self._state()
        s.task_plan = TaskPlan(phases=[])
        s.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[],
            total_estimated_duration="1m",
        )
        # default BuildStatus.PENDING
        self.assertEqual(get_next_action_recommendation(s), Action.BUILDER)

    def test_fixer_when_failed_non_dependency(self) -> None:
        """Test fixer when failed non dependency."""
        s = self._state()
        s.task_plan = TaskPlan(phases=[])
        s.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[],
            total_estimated_duration="1m",
        )
        s.build_status = BuildStatus.FAILED
        s.last_error_category = ErrorCategory.COMPILATION
        self.assertEqual(get_next_action_recommendation(s), Action.FIXER)

    def test_scout_when_failed_dependency(self) -> None:
        """Test scout when failed dependency."""
        s = self._state()
        s.task_plan = TaskPlan(phases=[])
        s.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[],
            total_estimated_duration="1m",
        )
        s.build_status = BuildStatus.FAILED
        s.last_error_category = ErrorCategory.DEPENDENCY
        self.assertEqual(get_next_action_recommendation(s), Action.SCOUT)

    def test_scout_when_failed_unknown(self) -> None:
        """UNKNOWN failures should trigger replanning first."""
        s = self._state()
        s.task_plan = TaskPlan(phases=[])
        s.build_plan = BuildPlan(
            build_system="go",
            build_system_confidence=0.9,
            phases=[],
            total_estimated_duration="1m",
        )
        s.build_status = BuildStatus.FAILED
        s.last_error_category = ErrorCategory.UNKNOWN
        self.assertEqual(get_next_action_recommendation(s), Action.SCOUT)

    def test_finish_after_success_with_tests(self) -> None:
        """Test finish after success with tests."""
        s = self._state()
        s.task_plan = TaskPlan(phases=[])
        s.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[],
            total_estimated_duration="1m",
        )
        s.build_status = BuildStatus.SUCCESS
        s.tests_run = True
        self.assertEqual(get_next_action_recommendation(s), Action.FINISH)

    def test_finish_after_success_even_without_tests(self) -> None:
        """Finish after success even when tests did not run.

        Regression: previously returned BUILDER on SUCCESS+!tests_run
        which created an infinite supervisor↔builder loop on the
        cost-optimized routing path (chisel/gum/csvtk/+20 batch
        timeouts on 2026-05-23). builder_node doesn't actually run
        tests today — it only re-verifies artifacts — so the loop
        burned the full 1h batch timeout.
        """
        s = self._state()
        s.task_plan = TaskPlan(phases=[])
        s.build_plan = BuildPlan(
            build_system="go",
            build_system_confidence=0.9,
            phases=[],
            total_estimated_duration="1m",
        )
        s.build_status = BuildStatus.SUCCESS
        s.tests_run = False
        self.assertEqual(get_next_action_recommendation(s), Action.FINISH)

    def test_escalate_short_circuits_decision_tree(self) -> None:
        """Test escalate short circuits decision tree."""
        s = self._state()
        s.api_cost_usd = 2.0
        self.assertEqual(get_next_action_recommendation(s), Action.ESCALATE)


# ===========================================================================
# CommandResult convenience props
# ===========================================================================


class TestCommandResult(unittest.TestCase):
    """Tests for CommandResult."""

    def test_success_property(self) -> None:
        """Test success property."""
        self.assertTrue(CommandResult("c", 0, "", "", 0.1).success)
        self.assertFalse(CommandResult("c", 1, "", "", 0.1).success)

    def test_failed_property(self) -> None:
        """Test failed property."""
        self.assertTrue(CommandResult("c", 1, "", "", 0.1).failed)
        self.assertFalse(CommandResult("c", 0, "", "", 0.1).failed)

    def test_negative_one_is_failed(self) -> None:
        # execute_command uses -1 for timeout/exception sentinels
        """Test negative one is failed."""
        self.assertTrue(CommandResult("c", -1, "", "", 0.1).failed)


# ===========================================================================
# Enum stability
# ===========================================================================


class TestEnumStability(unittest.TestCase):
    """Guard enum names that callers read via state.to_dict()."""

    def test_build_status_values(self) -> None:
        """Test build status values."""
        expected = {
            "PENDING",
            "PLANNING",
            "SCOUTING",
            "BUILDING",
            "FIXING",
            "SUCCESS",
            "FAILED",
            "ESCALATED",
        }
        self.assertEqual({s.value for s in BuildStatus}, expected)

    def test_error_category_values_complete(self) -> None:
        """Test error category values complete."""
        # These are referenced by classify_error + should_escalate +
        # supervisor routing.
        for cat_name in [
            "UNKNOWN",
            "DEPENDENCY",
            "COMPILATION",
            "LINKING",
            "ARCHITECTURE",
            "NETWORK",
            "RATE_LIMIT",
            "CONFIGURATION",
            "MISSING_TOOLS",
            "PERMISSION",
            "DISK_SPACE",
            "LICENSE_INCOMPATIBLE",
            "REQUIRES_HARDWARE",
            "ARCHITECTURE_IMPOSSIBLE",
        ]:
            self.assertTrue(
                hasattr(ErrorCategory, cat_name),
                f"Missing ErrorCategory.{cat_name}",
            )

    def test_agent_role_values(self) -> None:
        """Test agent role values."""
        expected = {
            "planner",
            "supervisor",
            "scout",
            "builder",
            "fixer",
            "summarizer",
        }
        self.assertEqual({r.value for r in AgentRole}, expected)

    def test_action_values(self) -> None:
        """Test action values."""
        self.assertEqual(
            {a.value for a in Action},
            {"PLAN", "SCOUT", "BUILDER", "FIXER", "ESCALATE", "FINISH"},
        )


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization(unittest.TestCase):
    """Tests for Serialization."""

    def test_to_dict_includes_core_fields(self) -> None:
        """Test to dict includes core fields."""
        s = create_initial_state("https://github.com/x/y")
        s.log_api_call(cost=0.01)
        d = s.to_dict()
        self.assertEqual(d["repo_name"], "y")
        self.assertAlmostEqual(d["api_cost_usd"], 0.01)
        # created_at must be serialized (datetime/enum become strings)
        self.assertIsInstance(d["created_at"], str)
        self.assertIsInstance(d["build_status"], str)

    def test_to_dict_serializes_nested_enums(self) -> None:
        """Test to dict serializes nested enums."""
        s = create_initial_state("https://github.com/x/y")
        s.add_error(create_error_record("permission denied"))
        d = s.to_dict()
        # error_history is a list of dicts and ErrorCategory inside
        # must serialize.
        self.assertEqual(len(d["error_history"]), 1)
        self.assertIn("PERMISSION", d["error_history"][0]["category"])

    def test_save_to_json_writes_valid_json(
        self,
    ) -> None:
        """Test save to json writes valid json."""
        s = create_initial_state("https://github.com/x/y")
        s.log_api_call(cost=0.02)
        s.add_error(create_error_record("network down"))

        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            s.save_to_json(path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["repo_name"], "y")
            self.assertEqual(len(data["error_history"]), 1)
        finally:
            os.unlink(path)


class TestUncoveredHelperBranches(unittest.TestCase):
    """Cover helper methods and classifier branches missed elsewhere."""

    def setUp(self) -> None:
        """Create a fresh state."""
        self.state = create_initial_state("https://github.com/o/r")

    def test_error_history_trimmed_to_cap(self) -> None:
        """Test error history trimmed to cap."""
        for i in range(MAX_ERROR_HISTORY + 5):
            self.state.add_error(create_error_record(f"boom {i}"))
        self.assertEqual(len(self.state.error_history), MAX_ERROR_HISTORY)
        self.assertIn("boom 5", self.state.error_history[0].message)

    def test_add_fix_attempt_records(self) -> None:
        """Test add fix attempt records."""
        fix = FixAttempt(
            error_category=ErrorCategory.COMPILATION,
            strategy="patch include",
            changes_made=["a.c"],
            success=True,
        )
        self.state.add_fix_attempt(fix)
        self.assertEqual(self.state.fixes_attempted, [fix])

    def test_log_scripted_op_increments_counter(self) -> None:
        """Test log scripted op increments counter."""
        before = self.state.scripted_ops_count
        self.state.log_scripted_op("clone")
        self.assertEqual(self.state.scripted_ops_count, before + 1)
        self.assertEqual(self.state.audit_trail[-1]["event"], "scripted_op")

    def test_audit_trail_trimmed_to_cap(self) -> None:
        """Test audit trail trimmed to cap."""
        for i in range(MAX_AUDIT_EVENTS + 3):
            self.state.log_event("tick", {"i": i})
        self.assertEqual(len(self.state.audit_trail), MAX_AUDIT_EVENTS)

    def test_cache_file_content(self) -> None:
        """Test cache file content."""
        self.state.cache_file_content("/workspace/repos/r/a.c", "int x;")
        self.assertEqual(
            self.state.file_content_cache["/workspace/repos/r/a.c"],
            "int x;",
        )

    def test_get_execution_duration_non_negative(self) -> None:
        """Test get execution duration non negative."""
        self.assertGreaterEqual(self.state.get_execution_duration(), 0.0)

    def test_to_dict_serializes_arbitrary_object(self) -> None:
        """Test to dict serializes arbitrary object."""

        class Marker:
            """Marker."""

            def __init__(self) -> None:
                self.x = 1

        self.state.context_cache["marker"] = Marker()
        data = self.state.to_dict()
        self.assertEqual(data["context_cache"]["marker"], {"x": 1})

    def test_derive_repo_name_ambiguous_prefixes_owner(self) -> None:
        """Test derive repo name ambiguous prefixes owner."""
        self.assertEqual(
            derive_repo_name("https://github.com/cli/cli"), "cli-cli"
        )
        self.assertEqual(
            derive_repo_name("https://github.com/smallstep/cli.git"),
            "smallstep-cli",
        )

    def test_derive_repo_name_dotted_owner_falls_back(self) -> None:
        """Test derive repo name dotted owner falls back."""
        self.assertEqual(
            derive_repo_name("https://gitlab.com/foo.bar/cli"), "cli"
        )

    def test_create_initial_state_rejects_bad_url(self) -> None:
        """Test create initial state rejects bad url."""
        with self.assertRaises(ValueError):
            create_initial_state("git@github.com:o/r.git")

    def test_classify_clone_auth_as_network(self) -> None:
        """Test classify clone auth as network."""
        msg = "fatal: could not read Username for 'https://github.com'"
        self.assertEqual(classify_error(msg), ErrorCategory.NETWORK)

    def test_classify_repo_not_found_as_configuration(self) -> None:
        """Test classify repo not found as configuration."""
        msg = "fatal: repository 'https://github.com/x/y/' not found"
        self.assertEqual(classify_error(msg), ErrorCategory.CONFIGURATION)

    def test_classify_empty_repository_as_configuration(self) -> None:
        """Test classify empty repository as configuration."""
        msg = "warning: remote HEAD refers to an empty repository"
        self.assertEqual(classify_error(msg), ErrorCategory.CONFIGURATION)

    def test_severity_which_probe_not_found_is_low(self) -> None:
        """Test severity which probe not found is low."""
        severity = infer_failure_severity(
            ErrorCategory.UNKNOWN,
            command=None,
            message="sh: which ninja: not found",
        )
        self.assertEqual(severity, FailureSeverity.LOW)

    def _plan_state(self) -> AgentState:
        """Return a state with plans set, ready for action routing."""
        state = create_initial_state("https://github.com/o/r")
        state.task_plan = TaskPlan(phases=[])
        state.build_plan = BuildPlan(
            build_system="make",
            build_system_confidence=0.9,
            phases=[],
            total_estimated_duration="5m",
        )
        return state

    def test_replan_failure_recommends_scout(self) -> None:
        """Test replan failure recommends scout."""
        state = self._plan_state()
        state.build_status = BuildStatus.FAILED
        state.last_error_category = ErrorCategory.COMPILATION
        state.last_error = "./configure: No such file or directory"
        self.assertEqual(get_next_action_recommendation(state), Action.SCOUT)

    def test_default_action_is_builder(self) -> None:
        """Test default action is builder."""
        state = self._plan_state()
        state.build_status = BuildStatus.FIXING
        self.assertEqual(
            get_next_action_recommendation(state), Action.BUILDER
        )


if __name__ == "__main__":
    unittest.main()
