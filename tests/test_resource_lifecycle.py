"""Resource-lifecycle regression tests.

Locks in the open/close contracts around external resources:

  * logging FileHandlers are closed when configure_logging replaces
    them (no FD leak per reconfiguration);
  * the docker-py client is closed on every setup/cleanup exit path;
  * apply_patch never leaves its temporary patch file behind on the
    host, on any path (invalid diff, executor exception, success);
  * execute_command's pkg-lock retry keeps extra_env on host retries;
  * the GitHub auth header for a clone travels only in that command's
    environment: never in a command string or the git config, and
    never to a non-GitHub host (token hygiene);
  * cleanup_workspace degrades gracefully without interactive stdin.

No test here touches the network, a real Docker daemon, or the LLM.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from src.state import CommandResult

# configure_logging — handler FD hygiene


class TestConfigureLoggingHandlerHygiene:
    """configure_logging must close the handlers it removes."""

    def test_reconfigure_closes_previous_file_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import main as main_mod

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        try:
            monkeypatch.setattr(main_mod, "LOGS_DIR", str(tmp_path))

            main_mod.configure_logging(False, repo_name="t1")
            file_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.FileHandler)
            ]
            assert file_handlers, "expected a FileHandler after configure"
            first = file_handlers[0]

            main_mod.configure_logging(False, repo_name="t2")

            assert first not in root.handlers
            assert first.stream is None or first.stream.closed
        finally:
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                handler.close()
            for handler in saved_handlers:
                root.addHandler(handler)
            root.setLevel(saved_level)


# Docker client lifecycle — main.py


class TestDockerClientLifecycle:
    """docker.from_env() clients must be closed on every exit path."""

    def test_setup_docker_environment_closes_client_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Preflight failure still closes the client (finally path)."""
        import main as main_mod

        fake_client = mock.MagicMock()
        monkeypatch.setattr(
            main_mod.docker, "from_env", lambda: fake_client
        )
        monkeypatch.setattr(
            main_mod, "_ensure_riscv64_binfmt", lambda: False
        )

        assert main_mod.setup_docker_environment() is False
        fake_client.close.assert_called_once()

    def test_cleanup_container_closes_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cleanup_container closes the client even when nothing exists."""
        import docker as docker_pkg

        import main as main_mod

        fake_client = mock.MagicMock()
        fake_client.containers.get.side_effect = docker_pkg.errors.NotFound(
            "gone"
        )
        monkeypatch.setattr(
            main_mod.docker, "from_env", lambda: fake_client
        )

        main_mod.cleanup_container(remove_image=False)
        fake_client.close.assert_called_once()


# apply_patch — host temp file hygiene


class TestApplyPatchTempFileHygiene:
    """The host-side patch temp file must never outlive apply_patch."""

    def _spy_tempfiles(self, monkeypatch) -> list:
        created: list = []
        real_ntf = tempfile.NamedTemporaryFile

        def spy(*args, **kwargs):
            handle = real_ntf(*args, **kwargs)
            created.append(handle.name)
            return handle

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", spy)
        return created

    def test_invalid_diff_removes_tempfile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src import tools

        created = self._spy_tempfiles(monkeypatch)
        monkeypatch.setattr(
            tools,
            "execute_command",
            mock.MagicMock(
                side_effect=AssertionError("must not execute")
            ),
        )

        ok = tools.apply_patch(
            "just some prose, not a diff",
            filepath="f.txt",
            use_docker=False,
        )

        assert ok is False
        assert created and not os.path.exists(created[0])

    def test_executor_exception_removes_tempfile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src import tools

        created = self._spy_tempfiles(monkeypatch)
        monkeypatch.setattr(
            tools,
            "execute_command",
            mock.MagicMock(side_effect=RuntimeError("boom")),
        )

        ok = tools.apply_patch(
            "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-a\n+b\n",
            filepath="f.txt",
            use_docker=False,
        )

        assert ok is False
        assert created and not os.path.exists(created[0])

    def test_success_removes_tempfile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src import tools

        created = self._spy_tempfiles(monkeypatch)
        monkeypatch.setattr(
            tools,
            "execute_command",
            lambda cmd, **kw: CommandResult(str(cmd), 0, "", "", 0.0),
        )

        ok = tools.apply_patch(
            "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-a\n+b\n",
            filepath="f.txt",
            use_docker=False,
        )

        assert ok is True
        assert created and not os.path.exists(created[0])


# execute_command — retry env preservation


class TestPkgLockRetryEnv:
    """Host-side pkg-lock retries must keep the caller's extra_env."""

    def test_host_retry_passes_extra_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retried subprocess.run receives the merged env dict."""
        from src import tools

        calls: list = []
        lock_fail = SimpleNamespace(
            returncode=1, stdout="", stderr="Unable to lock database"
        )
        ok = SimpleNamespace(returncode=0, stdout="done", stderr="")

        def fake_run(*args, **kwargs):
            calls.append(kwargs)
            return lock_fail if len(calls) == 1 else ok

        monkeypatch.setattr(tools.subprocess, "run", fake_run)
        monkeypatch.setattr(tools.time, "sleep", lambda _s: None)

        result = tools.execute_command(
            "apk add somepkg",
            use_docker=False,
            validate=False,
            extra_env={"ATESOR_TEST_MARKER": "1"},
        )

        assert result.exit_code == 0
        assert len(calls) == 2
        for kwargs in calls:
            env = kwargs.get("env")
            assert env is not None
            assert env.get("ATESOR_TEST_MARKER") == "1"


# Clone auth — GitHub token hygiene


class TestCloneAuthTokenHygiene:
    """GitHub auth must travel per-invocation via env, never persist.

    Rationale: the sandbox later executes LLM-generated commands
    (``cat`` is whitelisted); a token written to the container's git
    config would be readable by any of them, and a set→unset scheme
    races when concurrent workers share a container. The env-based
    design (`GIT_CONFIG_*`) is stateless: nothing to read, nothing to
    race, and the token never appears in a command string.
    """

    def _run_clone(self, stub_execute_command, monkeypatch, url: str):
        from src.scripted_ops import ScriptedOperations

        # Fresh-clone branch: pretend nothing exists on the host.
        monkeypatch.setattr("os.path.exists", lambda _p: False)
        monkeypatch.setattr(
            "src.scripted_ops.ScriptedOperations."
            "_ensure_container_healthy",
            lambda _self: None,
        )
        monkeypatch.setattr(
            "src.scripted_ops.ScriptedOperations."
            "_init_submodules_if_present",
            lambda _self, _p: None,
        )

        calls: list = []

        def fake(cmd, **kwargs):
            text = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
            calls.append((text, kwargs.get("extra_env") or {}))
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations().clone_or_update_repository(url, "bar")
        return calls

    def test_token_travels_via_env_not_command_or_config(
        self, stub_execute_command, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_TOKEN", "sekret-token")
        monkeypatch.delenv("GH_TOKEN", raising=False)

        calls = self._run_clone(
            stub_execute_command,
            monkeypatch,
            "https://github.com/foo/bar.git",
        )

        clone_calls = [
            (cmd, env) for cmd, env in calls if "git clone" in cmd
        ]
        assert clone_calls, "a clone command must have been issued"
        for _cmd, env in clone_calls:
            assert env.get("GIT_CONFIG_KEY_0") == (
                "http.https://github.com/.extraHeader"
            )
            assert env.get("GIT_CONFIG_VALUE_0", "").endswith("sekret-token")

        # Statelessness: no git-config write, and the token never
        # appears in any command string (so it cannot leak into
        # CommandResult.command, state JSON, or log snippets).
        assert not any("extraHeader" in cmd for cmd, _env in calls)
        assert not any("sekret-token" in cmd for cmd, _env in calls)

    def test_no_token_means_no_auth_env(
        self, stub_execute_command, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GIT_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)

        calls = self._run_clone(
            stub_execute_command,
            monkeypatch,
            "https://github.com/foo/bar.git",
        )

        assert not any("GIT_CONFIG_KEY_0" in env for _cmd, env in calls)
        assert not any("extraHeader" in cmd for cmd, _env in calls)

    def test_non_github_url_never_gets_token(
        self, stub_execute_command, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_TOKEN", "sekret-token")

        calls = self._run_clone(
            stub_execute_command,
            monkeypatch,
            "https://git.example.org/foo/bar.git",
        )

        assert not any("GIT_CONFIG_KEY_0" in env for _cmd, env in calls)
        assert not any("sekret-token" in cmd for cmd, _env in calls)


# cleanup_workspace — non-interactive stdin


class TestCleanupWorkspaceNonInteractive:
    """cleanup_workspace must not crash without interactive stdin."""

    def _make_workspace(self, tmp_path, monkeypatch) -> None:
        import main as main_mod

        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "somefile.txt").write_text("x")
        monkeypatch.setattr(main_mod, "WORKSPACE_ROOT", str(ws))

    def test_eof_on_stdin_is_graceful(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An EOFError from input() cancels cleanup, not a crash."""
        import main as main_mod

        self._make_workspace(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "builtins.input",
            mock.MagicMock(side_effect=EOFError),
        )

        # Must simply return; a raised EOFError would fail the test.
        main_mod.cleanup_workspace()

    def test_quit_choice_cancels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Choosing 'q' cancels without deleting anything."""
        import main as main_mod

        self._make_workspace(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "builtins.input", mock.MagicMock(return_value="q")
        )

        main_mod.cleanup_workspace()
        assert (
            tmp_path / "workspace" / "somefile.txt"
        ).exists(), "files must survive a cancelled cleanup"
