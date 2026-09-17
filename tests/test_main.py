"""Unit tests for the Atesor AI CLI entry point."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import runpy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import main
from src.state import (
    AgentRole,
    BuildPhase,
    BuildPlan,
    BuildStatus,
    DependencyInfo,
    ErrorCategory,
    ErrorRecord,
    FailureSeverity,
    FixAttempt,
    create_initial_state,
)

_REPO_URL = "https://github.com/example/project"
_SCRATCH_PARENT = Path(__file__).resolve().parent / ".test_main_tmp"


class _CloseableBuildStream:
    """Fake Docker build stream that records whether it was closed."""

    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        """Yield Docker build chunks."""
        return iter(self.chunks)

    def close(self) -> None:
        """Record stream closure."""
        self.closed = True


class _FakeContainer:
    """Docker container test double for provisioning branches."""

    def __init__(
        self,
        workspace_root: str,
        status: str = "running",
        os_release: bytes = b"ID=alpine\n",
        arch: bytes = b"riscv64\n",
        ready: bytes = b"Container ready!\n",
    ) -> None:
        self.status = status
        self.short_id = "abc123"
        self.os_release = os_release
        self.arch = arch
        self.ready = ready
        self.os_release_exit = 0
        self.arch_exit = 0
        self.ready_exit = 0
        self.raise_on_os_release = False
        self.raise_on_ready = False
        self.start_calls = 0
        self.stop_calls = 0
        self.remove_calls: list[dict[str, object]] = []
        self.attrs = {
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": os.path.realpath(workspace_root),
                }
            ]
        }

    def start(self) -> None:
        """Start the fake container."""
        self.start_calls += 1
        self.status = "running"

    def stop(self, *args, **kwargs) -> None:
        """Stop the fake container."""
        self.stop_calls += 1
        self.status = "exited"

    def remove(self, *args, **kwargs) -> None:
        """Remove the fake container."""
        self.remove_calls.append(dict(kwargs))

    def reload(self) -> None:
        """Reload the fake container."""

    def exec_run(self, cmd: str):
        """Return canned responses for health-check commands."""
        if "os-release" in cmd:
            if self.raise_on_os_release:
                raise RuntimeError("cannot read os-release")
            return SimpleNamespace(
                exit_code=self.os_release_exit,
                output=self.os_release,
            )
        if "uname" in cmd:
            return SimpleNamespace(exit_code=self.arch_exit, output=self.arch)
        if "Container ready" in cmd:
            if self.raise_on_ready:
                raise RuntimeError("health failed")
            return SimpleNamespace(
                exit_code=self.ready_exit,
                output=self.ready,
            )
        return SimpleNamespace(exit_code=0, output=b"ok\n")


class _FlippingVerbose:
    """Truthiness helper for the defensive verbose-content branch."""

    def __init__(self) -> None:
        self.calls = 0

    def __bool__(self) -> bool:
        """Return True once and False on later checks."""
        self.calls += 1
        return self.calls == 1


class _MainTestCase(unittest.TestCase):
    """Base class that keeps scratch files out of temporary directories."""

    def setUp(self) -> None:
        """Create a per-test scratch directory under tests/."""
        _SCRATCH_PARENT.mkdir(exist_ok=True)
        self._temp_dir = tempfile.TemporaryDirectory(
            dir=str(_SCRATCH_PARENT)
        )
        self.temp_path = Path(self._temp_dir.name)
        self.addCleanup(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self) -> None:
        """Remove the per-test scratch directory and empty parent."""
        self._temp_dir.cleanup()
        with contextlib.suppress(OSError):
            _SCRATCH_PARENT.rmdir()

    def _workspace_paths(self) -> dict[str, str]:
        """Return isolated path constants for main.py."""
        paths = {
            "WORKSPACE_ROOT": self.temp_path / "workspace",
            "LOGS_DIR": self.temp_path / "workspace" / "logs",
            "OUTPUT_DIR": self.temp_path / "workspace" / "output",
            "REPOS_DIR": self.temp_path / "workspace" / "repos",
            "PACKAGES_DIR": self.temp_path / "workspace" / "packages",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return {key: str(value) for key, value in paths.items()}

    @contextlib.contextmanager
    def _patched_main_paths(self):
        """Patch main.py filesystem constants into the scratch tree."""
        with mock.patch.multiple(main, **self._workspace_paths()):
            yield

    def _profile(self, name: str = "alpine") -> SimpleNamespace:
        """Create a minimal platform profile."""
        return SimpleNamespace(
            name=name,
            display_name=f"{name.title()} Linux (riscv64)",
            image_name=f"atesor-{name}:latest",
            container_name=f"atesor-{name}",
            dockerfile=f"Dockerfile.{name}",
        )

    def _call_cli(
        self,
        args: list[str],
        cache: dict[str, object] | None = None,
        recipe_path: str | None = "recipe.md",
        key_result: bool = True,
        setup_result: bool = True,
        rebuild_result: bool = True,
        agent_code: int = 0,
        setup_side_effect=None,
    ) -> tuple[int, dict[str, mock.MagicMock]]:
        """Run main.main() with all heavy operations mocked."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.dict(main.os.environ, {}, clear=True)
            )
            stack.enter_context(mock.patch("sys.argv", ["atesor-ai"] + args))
            configure = stack.enter_context(
                mock.patch.object(main, "configure_logging")
            )
            cleanup_container = stack.enter_context(
                mock.patch.object(main, "cleanup_container")
            )
            cleanup_workspace = stack.enter_context(
                mock.patch.object(main, "cleanup_workspace")
            )
            check_keys = stack.enter_context(
                mock.patch.object(main, "check_keys", return_value=key_result)
            )
            if setup_side_effect is None:
                setup = stack.enter_context(
                    mock.patch.object(
                        main,
                        "setup_docker_environment",
                        return_value=setup_result,
                    )
                )
            else:
                setup = stack.enter_context(
                    mock.patch.object(
                        main,
                        "setup_docker_environment",
                        side_effect=setup_side_effect,
                    )
                )
            rebuild = stack.enter_context(
                mock.patch.object(
                    main,
                    "rebuild_all_sandboxes",
                    return_value=rebuild_result,
                )
            )
            run_agent = stack.enter_context(
                mock.patch.object(main, "run_agent", return_value=agent_code)
            )
            get_cached = stack.enter_context(
                mock.patch("src.memory.get_cached_recipe", return_value=cache)
            )
            materialize = stack.enter_context(
                mock.patch(
                    "src.memory.materialize_cached_recipe",
                    return_value=recipe_path,
                )
            )
            set_log = stack.enter_context(
                mock.patch("src.llm_logger.set_llm_log_repo")
            )

            code = main.main()
            return code, {
                "check_keys": check_keys,
                "cleanup_container": cleanup_container,
                "cleanup_workspace": cleanup_workspace,
                "configure": configure,
                "get_cached": get_cached,
                "materialize": materialize,
                "rebuild": rebuild,
                "run_agent": run_agent,
                "set_log": set_log,
                "setup": setup,
            }

    @contextlib.contextmanager
    def _patched_graph(self, app) -> None:
        """Provide a fake src.graph module for run_agent tests."""
        fake_graph = SimpleNamespace(app=app)
        with mock.patch.dict(sys.modules, {"src.graph": fake_graph}):
            yield


class TestConfigureLogging(_MainTestCase):
    """Tests for configure_logging()."""

    def test_verbose_logging_uses_debug_console_and_repo_file(self) -> None:
        """Configure DEBUG console logging with a per-repo file."""
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        try:
            with self._patched_main_paths():
                main.configure_logging(True, repo_name="demo")
                handlers = logging.getLogger().handlers
                file_handlers = [
                    handler
                    for handler in handlers
                    if isinstance(handler, logging.FileHandler)
                ]
                stream_handlers = [
                    handler
                    for handler in handlers
                    if type(handler) is logging.StreamHandler
                ]
                self.assertEqual(logging.DEBUG, root.level)
                self.assertEqual(logging.DEBUG, file_handlers[0].level)
                self.assertEqual(logging.DEBUG, stream_handlers[0].level)
                self.assertTrue(
                    str(file_handlers[0].baseFilename).endswith(
                        "agent_demo.log"
                    )
                )
        finally:
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                handler.close()
            for handler in saved_handlers:
                root.addHandler(handler)
            root.setLevel(saved_level)

    def test_non_verbose_logging_uses_error_console(self) -> None:
        """Configure ERROR console logging without a repo name."""
        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level
        try:
            with self._patched_main_paths():
                main.configure_logging(False)
                handlers = logging.getLogger().handlers
                file_handlers = [
                    handler
                    for handler in handlers
                    if isinstance(handler, logging.FileHandler)
                ]
                stream_handlers = [
                    handler
                    for handler in handlers
                    if type(handler) is logging.StreamHandler
                ]
                self.assertEqual(logging.ERROR, stream_handlers[0].level)
                self.assertTrue(
                    str(file_handlers[0].baseFilename).endswith("agent.log")
                )
        finally:
            for handler in root.handlers[:]:
                root.removeHandler(handler)
                handler.close()
            for handler in saved_handlers:
                root.addHandler(handler)
            root.setLevel(saved_level)


class TestCheckKeys(_MainTestCase):
    """Tests for API-key validation wrappers."""

    def test_check_keys_reports_success_and_prints_model_info(self) -> None:
        """Return True when provider keys are valid."""
        with mock.patch.object(
            main,
            "check_api_keys",
            return_value=(True, "Gemini ready", "gemini"),
        ), mock.patch.object(main, "print_model_info") as print_info:
            self.assertTrue(main.check_keys())
            print_info.assert_called_once_with()

    def test_check_keys_reports_failure_without_model_info(self) -> None:
        """Return False when provider keys are missing."""
        with mock.patch.object(
            main,
            "check_api_keys",
            return_value=(False, "missing key", "gemini"),
        ), mock.patch.object(main, "print_model_info") as print_info:
            self.assertFalse(main.check_keys())
            print_info.assert_not_called()


class TestBinfmtSetup(_MainTestCase):
    """Tests for qemu-riscv64 binfmt preflight."""

    def test_existing_enabled_binfmt_returns_true(self) -> None:
        """Use the existing enabled binfmt handler."""
        with mock.patch.object(
            main.os.path,
            "exists",
            return_value=True,
        ), mock.patch(
            "builtins.open",
            mock.mock_open(read_data="enabled\n"),
        ):
            self.assertTrue(main._ensure_riscv64_binfmt())

    def test_existing_binfmt_read_error_attempts_registration(self) -> None:
        """Fall through to installer when the existing file is unreadable."""
        completed = SimpleNamespace(returncode=0, stderr="")
        with mock.patch.object(
            main.os.path,
            "exists",
            side_effect=[True, True],
        ), mock.patch("builtins.open", side_effect=OSError), mock.patch(
            "subprocess.run",
            return_value=completed,
        ) as run:
            self.assertTrue(main._ensure_riscv64_binfmt())
            run.assert_called_once()

    def test_successful_registration_rechecks_binfmt_path(self) -> None:
        """Return True after the installer creates the handler."""
        completed = SimpleNamespace(returncode=0, stderr="")
        with mock.patch.object(
            main.os.path,
            "exists",
            side_effect=[False, True],
        ), mock.patch("subprocess.run", return_value=completed):
            self.assertTrue(main._ensure_riscv64_binfmt())

    def test_failed_registration_returns_false(self) -> None:
        """Return False when tonistiigi/binfmt exits non-zero."""
        completed = SimpleNamespace(returncode=1, stderr="denied")
        with mock.patch.object(
            main.os.path,
            "exists",
            return_value=False,
        ), mock.patch("subprocess.run", return_value=completed):
            self.assertFalse(main._ensure_riscv64_binfmt())

    def test_registration_exception_returns_false(self) -> None:
        """Return False when the installer command cannot run."""
        with mock.patch.object(
            main.os.path,
            "exists",
            return_value=False,
        ), mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertFalse(main._ensure_riscv64_binfmt())

    def test_registration_success_without_recheck_returns_false(self) -> None:
        """Return False when the installer succeeds but the path is absent."""
        completed = SimpleNamespace(returncode=0, stderr="")
        with mock.patch.object(
            main.os.path,
            "exists",
            return_value=False,
        ), mock.patch("subprocess.run", return_value=completed):
            self.assertFalse(main._ensure_riscv64_binfmt())


class TestDockerEnvironment(_MainTestCase):
    """Tests for Docker sandbox setup and provisioning."""

    def test_setup_docker_environment_handles_docker_exception(self) -> None:
        """Return False when Docker is not reachable."""
        with mock.patch.object(
            main.docker,
            "from_env",
            side_effect=main.docker.errors.DockerException("down"),
        ):
            self.assertFalse(main.setup_docker_environment())

    def test_setup_docker_environment_closes_client(self) -> None:
        """Close the Docker client after delegated provisioning."""
        client = mock.MagicMock()
        with mock.patch.object(
            main.docker,
            "from_env",
            return_value=client,
        ), mock.patch.object(
            main,
            "_provision_sandbox",
            return_value=True,
        ) as provision:
            self.assertTrue(main.setup_docker_environment())
            provision.assert_called_once()
            client.close.assert_called_once_with()

    def test_provision_sandbox_waits_for_existing_lock(self) -> None:
        """Block on an existing flock before entering the locked section."""
        with self._patched_main_paths(), mock.patch.object(
            main.fcntl,
            "flock",
            side_effect=[OSError, None, None],
        ) as flock, mock.patch.object(
            main,
            "_provision_sandbox_locked",
            return_value=True,
        ) as locked:
            self.assertTrue(
                main._provision_sandbox(
                    mock.MagicMock(),
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertEqual(3, flock.call_count)
            locked.assert_called_once()

    def test_locked_provision_stops_when_binfmt_missing(self) -> None:
        """Return False before touching images when binfmt is missing."""
        client = mock.MagicMock()
        with mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=False,
        ):
            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            client.images.get.assert_not_called()

    def test_locked_provision_uses_existing_running_container(self) -> None:
        """Use an existing healthy riscv64 container."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(main.WORKSPACE_ROOT)
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            client.containers.run.assert_not_called()

    def test_locked_provision_builds_missing_image_and_container(self) -> None:
        """Build a missing image and create the sandbox container."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            stream = _CloseableBuildStream(
                [
                    {"stream": "Step 1/2 : FROM alpine\n"},
                    {"stream": "fetch packages\n"},
                    {"stream": "\n"},
                ]
            )
            container = _FakeContainer(main.WORKSPACE_ROOT)
            client = mock.MagicMock()
            client.images.get.side_effect = main.docker.errors.ImageNotFound(
                "missing"
            )
            client.api.build.return_value = stream
            client.containers.get.side_effect = main.docker.errors.NotFound(
                "missing"
            )
            client.containers.run.return_value = container

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertTrue(stream.closed)
            client.containers.run.assert_called_once()

    def test_locked_provision_force_rebuild_uses_build_stream(self) -> None:
        """Honor REBUILD_IMAGE by rebuilding even when image lookup exists."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"), mock.patch.dict(
            main.os.environ,
            {"REBUILD_IMAGE": "true"},
        ):
            client = mock.MagicMock()
            stream = _CloseableBuildStream([{"stream": "Step 1/1\n"}])
            client.api.build.return_value = stream
            client.containers.get.return_value = _FakeContainer(
                main.WORKSPACE_ROOT
            )

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            client.api.build.assert_called_once()

    def test_locked_provision_reports_build_error_detail(self) -> None:
        """Return False when Docker emits errorDetail during build."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ):
            client = mock.MagicMock()
            client.images.get.side_effect = main.docker.errors.ImageNotFound(
                "missing"
            )
            stream = _CloseableBuildStream(
                [{"errorDetail": {"message": "bad Dockerfile"}}]
            )
            client.api.build.return_value = stream

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertTrue(stream.closed)

    def test_locked_provision_reports_build_error_key(self) -> None:
        """Return False when Docker emits a top-level error."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ):
            client = mock.MagicMock()
            client.images.get.side_effect = main.docker.errors.ImageNotFound(
                "missing"
            )
            client.api.build.return_value = _CloseableBuildStream(
                [{"error": "pull failed"}]
            )

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_reports_generic_build_exception(self) -> None:
        """Return False when Docker build setup raises unexpectedly."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ):
            client = mock.MagicMock()
            client.images.get.side_effect = main.docker.errors.ImageNotFound(
                "missing"
            )
            client.api.build.side_effect = RuntimeError("boom")

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_recreates_stale_mount(self) -> None:
        """Recreate a container whose /workspace mount is stale."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            stale = _FakeContainer(str(self.temp_path / "old-workspace"))
            fresh = _FakeContainer(main.WORKSPACE_ROOT)
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = stale
            client.containers.run.return_value = fresh

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertEqual([{"force": True}], stale.remove_calls)
            client.containers.run.assert_called_once()

    def test_locked_provision_starts_stopped_container(self) -> None:
        """Start an existing but stopped container."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(main.WORKSPACE_ROOT, status="created")
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertEqual(1, container.start_calls)

    def test_locked_provision_fails_after_non_running_retries(self) -> None:
        """Return False after recreated containers never reach running."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            first = _FakeContainer(main.WORKSPACE_ROOT, status="exited")
            second = _FakeContainer(main.WORKSPACE_ROOT, status="exited")
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.side_effect = main.docker.errors.NotFound(
                "missing"
            )
            client.containers.run.side_effect = [first, second]

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertEqual(2, client.containers.run.call_count)

    def test_locked_provision_logs_cleanup_failure_on_retry(self) -> None:
        """Log and continue when stale container cleanup fails."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            first = _FakeContainer(main.WORKSPACE_ROOT)
            first.reload = mock.MagicMock(
                side_effect=lambda: setattr(first, "status", "exited")
            )
            first.stop = mock.MagicMock(side_effect=RuntimeError("busy"))
            second = _FakeContainer(main.WORKSPACE_ROOT)
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.side_effect = [first, second]

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            first.stop.assert_called_once_with()

    def test_locked_provision_retries_when_container_disappears(self) -> None:
        """Retry when Docker loses the container during reload."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            lost = _FakeContainer(main.WORKSPACE_ROOT)
            lost.reload = mock.MagicMock(
                side_effect=main.docker.errors.NotFound("gone")
            )
            good = _FakeContainer(main.WORKSPACE_ROOT)
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.side_effect = [lost, good]

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertEqual(2, client.containers.get.call_count)

    def test_locked_provision_rejects_distro_mismatch(self) -> None:
        """Reject a container whose distro does not match the profile."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(
                main.WORKSPACE_ROOT,
                os_release=b"ID=debian\n",
            )
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile("alpine"),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_skips_failed_distro_exception(self) -> None:
        """Continue when the distro self-check raises unexpectedly."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(main.WORKSPACE_ROOT)
            container.raise_on_os_release = True
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_allows_ubuntu_for_debian_profile(self) -> None:
        """Accept ubuntu containers for the apt-based Debian profile."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(
                main.WORKSPACE_ROOT,
                os_release=b"ID=ubuntu\n",
            )
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile("debian"),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_continues_when_os_release_fails(self) -> None:
        """Continue health checks when distro self-check cannot run."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(main.WORKSPACE_ROOT)
            container.os_release_exit = 1
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_warns_on_wrong_architecture(self) -> None:
        """Warn but continue when uname is not riscv64."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(main.WORKSPACE_ROOT, arch=b"x86_64\n")
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_fails_when_arch_check_fails(self) -> None:
        """Return False when uname cannot execute."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(main.WORKSPACE_ROOT)
            container.arch_exit = 1
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_fails_when_health_check_fails(self) -> None:
        """Return False when the echo health check exits non-zero."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            container = _FakeContainer(main.WORKSPACE_ROOT)
            container.ready_exit = 1
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = container

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )

    def test_locked_provision_recovers_from_health_exception(self) -> None:
        """Recreate the container after a health-check exception."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            broken = _FakeContainer(main.WORKSPACE_ROOT)
            broken.raise_on_ready = True
            fresh = _FakeContainer(main.WORKSPACE_ROOT)
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = broken
            client.containers.run.return_value = fresh

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertEqual(1, broken.stop_calls)
            client.containers.run.assert_called_once()

    def test_locked_provision_logs_failed_initial_recreate_cleanup(
        self,
    ) -> None:
        """Continue when cleanup before recreation raises."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            broken = _FakeContainer(main.WORKSPACE_ROOT)
            broken.raise_on_ready = True
            broken.stop = mock.MagicMock(side_effect=RuntimeError("busy"))
            fresh = _FakeContainer(main.WORKSPACE_ROOT)
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = broken
            client.containers.run.return_value = fresh

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            broken.stop.assert_called_once_with()

    def test_locked_provision_skips_non_running_recreation(self) -> None:
        """Try the next recreation when the first container exits."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            broken = _FakeContainer(main.WORKSPACE_ROOT)
            broken.raise_on_ready = True
            stopped = _FakeContainer(main.WORKSPACE_ROOT, status="exited")
            fresh = _FakeContainer(main.WORKSPACE_ROOT)
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = broken
            client.containers.run.side_effect = [stopped, fresh]

            self.assertTrue(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertEqual(2, client.containers.run.call_count)

    def test_locked_provision_logs_recreate_cleanup_failure(self) -> None:
        """Log cleanup failures inside failed recreation attempts."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            broken = _FakeContainer(main.WORKSPACE_ROOT)
            broken.raise_on_ready = True
            first = _FakeContainer(main.WORKSPACE_ROOT)
            first.raise_on_ready = True
            first.stop = mock.MagicMock()
            first.remove = mock.MagicMock(side_effect=RuntimeError("busy"))
            second = _FakeContainer(main.WORKSPACE_ROOT)
            second.raise_on_ready = True
            second.stop = mock.MagicMock()
            second.remove = mock.MagicMock(side_effect=RuntimeError("busy"))
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = broken
            client.containers.run.side_effect = [first, second]

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            first.stop.assert_called_once_with()
            second.stop.assert_called_once_with()
            first.remove.assert_called_once_with()
            second.remove.assert_called_once_with()

    def test_locked_provision_fails_when_recreation_fails(self) -> None:
        """Return False when both recreation attempts raise."""
        with self._patched_main_paths(), mock.patch.object(
            main,
            "_ensure_riscv64_binfmt",
            return_value=True,
        ), mock.patch.object(main.time, "sleep"):
            broken = _FakeContainer(main.WORKSPACE_ROOT)
            broken.raise_on_ready = True
            client = mock.MagicMock()
            client.images.get.return_value = object()
            client.containers.get.return_value = broken
            client.containers.run.side_effect = RuntimeError("no capacity")

            self.assertFalse(
                main._provision_sandbox_locked(
                    client,
                    self._profile(),
                    "image",
                    "container",
                    "Dockerfile",
                )
            )
            self.assertEqual(2, client.containers.run.call_count)


class TestOutputReports(_MainTestCase):
    """Tests for report generation and output persistence."""

    def test_generate_success_report_includes_all_sections(self) -> None:
        """Render success details, truncation notes, and recommendations."""
        patch_text = "diff --git a/a b/a\n" + ("+x\n" * 250)
        state = {
            "repo_name": "demo",
            "repo_url": _REPO_URL,
            "build_status": "SUCCESS",
            "execution_duration": 12.34,
            "attempt_count": 2,
            "api_calls_made": 3,
            "scripted_ops_count": 4,
            "api_tokens_in": 111,
            "api_tokens_out": 222,
            "api_cost_usd": 0.12345,
            "build_plan": {
                "build_system": "cmake",
                "phases": [
                    {
                        "name": "Configure",
                        "commands": ["cmake -S . -B build"],
                    }
                ],
            },
            "dependencies": {
                "build_tools": ["cmake", "make"],
                "system_packages": ["zlib-dev"],
            },
            "arch_specific_code": [
                {
                    "file": f"src/file{i}.c",
                    "line": i,
                    "code_snippet": "#ifdef __x86_64__",
                }
                for i in range(6)
            ],
            "patches_generated": [patch_text],
            "error_log": [
                SimpleNamespace(category="COMPILATION", message="gcc failed")
            ],
        }

        report = main.generate_detailed_report(state)

        self.assertIn("# RISC-V Porting Report: demo", report)
        self.assertIn("cmake -S . -B build", report)
        self.assertIn("Build Tools**: cmake, make", report)
        self.assertIn("*...and 1 more instances*", report)
        self.assertIn("Truncated", report)
        self.assertIn("Build completed successfully", report)

    def test_generate_escalated_report_includes_manual_reason(self) -> None:
        """Render escalation recommendations when no build plan exists."""
        report = main.generate_detailed_report(
            {
                "repo_name": "demo",
                "build_status": "ESCALATED",
                "escalation_reason": "cost cap",
            }
        )

        self.assertIn("*No build plan available*", report)
        self.assertIn("Manual intervention required", report)
        self.assertIn("Reason: cost cap", report)

    def test_generate_failed_report_uses_failure_recommendations(self) -> None:
        """Render generic remediation guidance for failed builds."""
        report = main.generate_detailed_report(
            {"repo_name": "demo", "build_status": "FAILED"}
        )

        self.assertIn("Build did not complete successfully", report)
        self.assertIn("Check dependencies", report)

    def test_save_porting_outputs_writes_state_recipe_report_and_patches(
        self,
    ) -> None:
        """Write every output artifact for a completed state."""
        state = create_initial_state(_REPO_URL)
        state.porting_recipe = "# recipe\n"
        state.patches_generated = ["diff --git a/a b/a\n"]

        output_dir = self.temp_path / "out"
        main.save_porting_outputs(state, str(output_dir))

        self.assertTrue((output_dir / "project_recipe.md").exists())
        self.assertEqual(
            "# recipe\n",
            (output_dir / "project_recipe.md").read_text(),
        )
        self.assertEqual(1, len(list(output_dir.glob("project_state_*.json"))))
        self.assertEqual(1, len(list(output_dir.glob("project_report_*.md"))))
        patch_dirs = list(output_dir.glob("project_patches_*"))
        self.assertEqual(1, len(patch_dirs))
        self.assertEqual(1, len(list(patch_dirs[0].glob("patch_1.patch"))))

    def test_save_porting_outputs_continues_after_state_json_error(
        self,
    ) -> None:
        """Still write the report if state serialization fails."""
        state = create_initial_state(_REPO_URL)
        state.save_to_json = mock.MagicMock(side_effect=RuntimeError("bad"))
        output_dir = self.temp_path / "out"

        main.save_porting_outputs(state, str(output_dir))

        self.assertEqual(1, len(list(output_dir.glob("project_report_*.md"))))


class _FakeApp:
    """LangGraph app double for run_agent()."""

    def __init__(self, stream_factory) -> None:
        self.stream_factory = stream_factory
        self.initial_state = None
        self.config = None

    def stream(self, initial_state, config=None):
        """Yield fake graph updates."""
        self.initial_state = initial_state
        self.config = config
        for item in self.stream_factory(initial_state):
            yield item


class TestRunAgent(_MainTestCase):
    """Tests for the top-level graph runner."""

    def test_run_agent_success_with_dict_and_state_updates(self) -> None:
        """Return zero after a successful graph stream."""
        def stream_factory(state):
            state.build_status = BuildStatus.SUCCESS
            state.porting_recipe = "# recipe"
            yield {"Supervisor": {"attempt_count": 1}}
            yield {"Finish": state}

        fake_app = _FakeApp(stream_factory)
        with (
            self._patched_main_paths(),
            self._patched_graph(fake_app),
            mock.patch(
                "src.platforms.get_active_profile",
                return_value=self._profile(),
            ),
            mock.patch.object(main, "save_porting_outputs") as save_outputs,
        ):
            code = main.run_agent(_REPO_URL, max_attempts=9, verbose=False)

        self.assertEqual(0, code)
        self.assertEqual({"recursion_limit": 120}, fake_app.config)
        self.assertEqual(9, fake_app.initial_state.max_attempts)
        save_outputs.assert_called_once()

    def test_run_agent_verbose_prints_agent_messages(self) -> None:
        """Exercise verbose message rendering for named agents."""
        def stream_factory(state):
            state.build_status = BuildStatus.SUCCESS
            state.messages = [SimpleNamespace(name="Scout", content="ready")]
            yield {"Scout": state}

        fake_app = _FakeApp(stream_factory)
        with (
            self._patched_main_paths(),
            self._patched_graph(fake_app),
            mock.patch(
                "src.platforms.get_active_profile",
                return_value=self._profile(),
            ),
            mock.patch.object(main, "save_porting_outputs"),
        ):
            self.assertEqual(0, main.run_agent(_REPO_URL, verbose=True))

    def test_run_agent_verbose_colors_builder_and_fixer_messages(self) -> None:
        """Exercise verbose role rendering for builder and fixer agents."""
        def stream_factory(state):
            state.build_status = BuildStatus.SUCCESS
            state.messages = [
                SimpleNamespace(name="Builder", content="built"),
                SimpleNamespace(name="Fixer", content="fixed"),
            ]
            yield {"Builder": state}

        fake_app = _FakeApp(stream_factory)
        with (
            self._patched_main_paths(),
            self._patched_graph(fake_app),
            mock.patch(
                "src.platforms.get_active_profile",
                return_value=self._profile(),
            ),
            mock.patch.object(main, "save_porting_outputs"),
        ):
            self.assertEqual(0, main.run_agent(_REPO_URL, verbose=True))

    def test_run_agent_truncates_long_verbose_content(self) -> None:
        """Exercise the defensive long-content truncation branch."""
        def stream_factory(state):
            state.build_status = BuildStatus.SUCCESS
            state.messages = [
                SimpleNamespace(name="Scout", content="x" * 600),
            ]
            yield {"Scout": state}

        fake_app = _FakeApp(stream_factory)
        with (
            self._patched_main_paths(),
            self._patched_graph(fake_app),
            mock.patch(
                "src.platforms.get_active_profile",
                return_value=self._profile(),
            ),
            mock.patch.object(main, "save_porting_outputs"),
        ):
            code = main.run_agent(_REPO_URL, verbose=_FlippingVerbose())

        self.assertEqual(0, code)

    def test_run_agent_failure_prints_error_and_audit_trail(self) -> None:
        """Return one for failed builds and summarize diagnostics."""
        def stream_factory(state):
            state.build_status = BuildStatus.FAILED
            state.add_error(
                ErrorRecord(
                    ErrorCategory.COMPILATION,
                    "compile failed",
                    FailureSeverity.HIGH,
                )
            )
            state.log_agent_decision(
                AgentRole.SUPERVISOR,
                "FIXER",
                "retry once",
            )
            state.log_scripted_op("clone")
            state.log_event("error", {"message": "boom"})
            yield {"Builder": state}

        fake_app = _FakeApp(stream_factory)
        with (
            self._patched_main_paths(),
            self._patched_graph(fake_app),
            mock.patch(
                "src.platforms.get_active_profile",
                return_value=self._profile(),
            ),
            mock.patch.object(main, "save_porting_outputs"),
        ):
            self.assertEqual(1, main.run_agent(_REPO_URL))

    def test_run_agent_failure_prints_scripted_and_error_audit(self) -> None:
        """Render non-decision audit events in failed runs."""
        def stream_factory(state):
            state.build_status = BuildStatus.FAILED
            state.audit_trail = [
                {
                    "timestamp": "2026-09-17T12:00:00",
                    "agent": "system",
                    "event": "scripted_op",
                    "data": {"operation": "scan"},
                },
                {
                    "timestamp": "2026-09-17T12:00:01",
                    "agent": "builder",
                    "event": "error",
                    "data": {"message": "compile failed"},
                },
            ]
            yield {"Builder": state}

        fake_app = _FakeApp(stream_factory)
        with (
            self._patched_main_paths(),
            self._patched_graph(fake_app),
            mock.patch(
                "src.platforms.get_active_profile",
                return_value=self._profile(),
            ),
            mock.patch.object(main, "save_porting_outputs"),
        ):
            self.assertEqual(1, main.run_agent(_REPO_URL))

    def test_run_agent_success_packages_artifacts(self) -> None:
        """Invoke package_build when packaging is requested."""
        package_build = mock.MagicMock(return_value="pkg.zip")

        def stream_factory(state):
            state.build_status = BuildStatus.SUCCESS
            yield {"Finish": state}

        fake_app = _FakeApp(stream_factory)
        fake_packager = SimpleNamespace(package_build=package_build)
        with (
            self._patched_main_paths(),
            self._patched_graph(fake_app),
            mock.patch.dict(sys.modules, {"src.packager": fake_packager}),
            mock.patch(
                "src.platforms.get_active_profile",
                return_value=self._profile("debian"),
            ),
            mock.patch.object(main, "save_porting_outputs"),
        ):
            code = main.run_agent(_REPO_URL, package=True)

        self.assertEqual(0, code)
        package_build.assert_called_once()
        self.assertEqual(
            "debian",
            package_build.call_args.kwargs["platform_name"],
        )

    def test_run_agent_packaging_failure_returns_two(self) -> None:
        """Treat explicit packaging failures as fatal."""
        package_build = mock.MagicMock(side_effect=RuntimeError("zip bad"))

        def stream_factory(state):
            state.build_status = BuildStatus.SUCCESS
            yield {"Finish": state}

        fake_app = _FakeApp(stream_factory)
        fake_packager = SimpleNamespace(package_build=package_build)
        with (
            self._patched_main_paths(),
            self._patched_graph(fake_app),
            mock.patch.dict(sys.modules, {"src.packager": fake_packager}),
            mock.patch(
                "src.platforms.get_active_profile",
                return_value=self._profile(),
            ),
            mock.patch.object(main, "save_porting_outputs"),
        ):
            self.assertEqual(2, main.run_agent(_REPO_URL, package=True))

    def test_run_agent_keyboard_interrupt_returns_130(self) -> None:
        """Map KeyboardInterrupt to shell exit code 130."""
        def stream_factory(_state):
            raise KeyboardInterrupt
            yield {}

        fake_app = _FakeApp(stream_factory)
        with self._patched_graph(fake_app), mock.patch(
            "src.platforms.get_active_profile",
            return_value=self._profile(),
        ):
            self.assertEqual(130, main.run_agent(_REPO_URL))

    def test_run_agent_unexpected_exception_returns_one(self) -> None:
        """Map unexpected graph exceptions to exit code one."""
        def stream_factory(_state):
            raise RuntimeError("graph bad")
            yield {}

        fake_app = _FakeApp(stream_factory)
        with self._patched_graph(fake_app), mock.patch(
            "src.platforms.get_active_profile",
            return_value=self._profile(),
        ):
            self.assertEqual(1, main.run_agent(_REPO_URL))


class TestCleanupWorkspace(_MainTestCase):
    """Tests for workspace cleanup flows."""

    def _make_workspace(self) -> Path:
        """Create a workspace tree with files in every cleanable dir."""
        workspace = self.temp_path / "workspace"
        for dirname in ["repos", "output", "logs", ".cache", "patches"]:
            path = workspace / dirname
            path.mkdir(parents=True, exist_ok=True)
            (path / "file.txt").write_text("x", encoding="utf-8")
        return workspace

    def test_cleanup_workspace_missing_directory_returns(self) -> None:
        """Handle a missing workspace directory without prompting."""
        with mock.patch.object(
            main,
            "WORKSPACE_ROOT",
            str(self.temp_path / "missing"),
        ), mock.patch("builtins.input") as input_mock:
            main.cleanup_workspace()
            input_mock.assert_not_called()

    def test_cleanup_workspace_dry_run_does_not_delete(self) -> None:
        """Dry-run mode reports sizes without deleting files."""
        workspace = self._make_workspace()
        with mock.patch.object(main, "WORKSPACE_ROOT", str(workspace)):
            main.cleanup_workspace(dry_run=True)
        self.assertTrue((workspace / "repos" / "file.txt").exists())

    def test_cleanup_workspace_handles_eof(self) -> None:
        """Cancel cleanup when stdin is non-interactive."""
        workspace = self._make_workspace()
        with mock.patch.object(
            main,
            "WORKSPACE_ROOT",
            str(workspace),
        ), mock.patch("builtins.input", side_effect=EOFError):
            main.cleanup_workspace()
        self.assertTrue((workspace / "repos" / "file.txt").exists())

    def test_cleanup_workspace_invalid_choice_returns(self) -> None:
        """Reject invalid menu input without deleting files."""
        workspace = self._make_workspace()
        with mock.patch.object(
            main,
            "WORKSPACE_ROOT",
            str(workspace),
        ), mock.patch("builtins.input", return_value="x"):
            main.cleanup_workspace()
        self.assertTrue((workspace / "repos" / "file.txt").exists())

    def test_cleanup_workspace_quit_choice_returns(self) -> None:
        """Respect the quit menu option."""
        workspace = self._make_workspace()
        with mock.patch.object(
            main,
            "WORKSPACE_ROOT",
            str(workspace),
        ), mock.patch("builtins.input", return_value="q"):
            main.cleanup_workspace()
        self.assertTrue((workspace / "repos" / "file.txt").exists())

    def test_cleanup_workspace_cleans_each_single_directory(self) -> None:
        """Clean each menu option that maps to one directory."""
        choices = [
            ("1", "repos"),
            ("2", "output"),
            ("3", "logs"),
            ("4", ".cache"),
        ]
        for choice, dirname in choices:
            with self.subTest(choice=choice):
                workspace = self._make_workspace()
                with mock.patch.object(
                    main,
                    "WORKSPACE_ROOT",
                    str(workspace),
                ), mock.patch("builtins.input", return_value=choice):
                    main.cleanup_workspace()
                self.assertTrue((workspace / dirname).exists())
                self.assertFalse((workspace / dirname / "file.txt").exists())

    def test_cleanup_workspace_cleans_all_directories(self) -> None:
        """Clean every managed workspace directory."""
        workspace = self._make_workspace()
        with mock.patch.object(
            main,
            "WORKSPACE_ROOT",
            str(workspace),
        ), mock.patch("builtins.input", return_value="5"):
            main.cleanup_workspace()
        for dirname in ["repos", "output", "logs", ".cache", "patches"]:
            self.assertTrue((workspace / dirname).exists())
            self.assertFalse((workspace / dirname / "file.txt").exists())

    def test_cleanup_workspace_reports_rmtree_errors(self) -> None:
        """Keep going when a directory cannot be removed."""
        workspace = self._make_workspace()
        with (
            mock.patch.object(main, "WORKSPACE_ROOT", str(workspace)),
            mock.patch("builtins.input", return_value="1"),
            mock.patch.object(shutil, "rmtree", side_effect=OSError("busy")),
        ):
            main.cleanup_workspace()
        self.assertTrue((workspace / "repos" / "file.txt").exists())


class TestCleanupContainer(_MainTestCase):
    """Tests for Docker container cleanup."""

    def test_cleanup_container_stops_running_container_and_image(self) -> None:
        """Stop and remove both container and image when requested."""
        client = mock.MagicMock()
        container = mock.MagicMock(status="running")
        client.containers.get.return_value = container
        with mock.patch.object(main.docker, "from_env", return_value=client):
            main.cleanup_container(remove_image=True)
        container.stop.assert_called_once_with(timeout=10)
        container.remove.assert_called_once_with()
        client.images.remove.assert_called_once()
        client.close.assert_called_once_with()

    def test_cleanup_container_handles_missing_resources(self) -> None:
        """Report missing Docker resources without raising."""
        client = mock.MagicMock()
        client.containers.get.side_effect = main.docker.errors.NotFound(
            "missing"
        )
        client.images.remove.side_effect = main.docker.errors.ImageNotFound(
            "missing"
        )
        with mock.patch.object(main.docker, "from_env", return_value=client):
            main.cleanup_container(remove_image=True)
        client.close.assert_called_once_with()

    def test_cleanup_container_reports_image_remove_error(self) -> None:
        """Catch unexpected image removal errors."""
        client = mock.MagicMock()
        client.containers.get.side_effect = main.docker.errors.NotFound(
            "missing"
        )
        client.images.remove.side_effect = RuntimeError("denied")
        with mock.patch.object(main.docker, "from_env", return_value=client):
            main.cleanup_container(remove_image=True)
        client.close.assert_called_once_with()

    def test_cleanup_container_handles_docker_client_error(self) -> None:
        """Catch Docker client creation failures."""
        with mock.patch.object(
            main.docker,
            "from_env",
            side_effect=RuntimeError("socket gone"),
        ):
            main.cleanup_container(remove_image=False)


class TestRebuildSandboxes(_MainTestCase):
    """Tests for rebuilding all sandbox profiles."""

    def test_rebuild_all_sandboxes_runs_both_profiles_and_restores_env(
        self,
    ) -> None:
        """Rebuild both profiles and restore caller environment."""
        calls: list[str | None] = []

        def fake_setup() -> bool:
            calls.append(main.os.environ.get("ATESOR_PLATFORM"))
            return True

        with mock.patch.dict(
            main.os.environ,
            {"ATESOR_PLATFORM": "original", "ATESOR_CONTAINER": "worker"},
            clear=True,
        ), mock.patch.object(
            main,
            "setup_docker_environment",
            side_effect=fake_setup,
        ):
            self.assertTrue(main.rebuild_all_sandboxes())
            self.assertEqual(["alpine", "debian"], calls)
            self.assertEqual("original", main.os.environ["ATESOR_PLATFORM"])
            self.assertEqual("worker", main.os.environ["ATESOR_CONTAINER"])

    def test_rebuild_restores_absent_env_on_failure(self) -> None:
        """Restore absent platform variables after a failed rebuild."""
        with (
            mock.patch.dict(main.os.environ, {}, clear=True),
            mock.patch.object(
                main,
                "setup_docker_environment",
                side_effect=[True, False],
            ),
        ):
            self.assertFalse(main.rebuild_all_sandboxes())
            self.assertNotIn("ATESOR_PLATFORM", main.os.environ)
            self.assertNotIn("ATESOR_CONTAINER", main.os.environ)


class TestHelpFormatter(_MainTestCase):
    """Tests for compact argparse help formatting."""

    def test_formatter_combines_short_and_long_option_metavar(self) -> None:
        """Render one metavar for short and long options."""
        parser = argparse.ArgumentParser(
            prog="atesor-ai",
            formatter_class=main._CleanHelpFormatter,
        )
        option = parser.add_argument("-r", "--repo", metavar="URL")
        flag = parser.add_argument("-v", "--verbose", action="store_true")
        positional = parser.add_argument("path")
        formatter = main._CleanHelpFormatter("atesor-ai")

        self.assertEqual(
            "-r, --repo URL",
            formatter._format_action_invocation(option),
        )
        self.assertEqual(
            "-v, --verbose",
            formatter._format_action_invocation(flag),
        )
        self.assertEqual(
            "path",
            formatter._format_action_invocation(positional),
        )


class TestMainCli(_MainTestCase):
    """Tests for the argparse-driven CLI dispatcher."""

    def test_cli_without_repo_or_infra_flag_returns_one(self) -> None:
        """Require a repo unless an infrastructure-only flag is present."""
        code, patches = self._call_cli([])

        self.assertEqual(1, code)
        patches["configure"].assert_called_once_with(False, repo_name="")
        patches["setup"].assert_not_called()

    def test_cli_rejects_invalid_repository_url(self) -> None:
        """Reject unsafe repository URLs before configuring logging."""
        code, patches = self._call_cli(["--repo", "ssh://example/repo"])

        self.assertEqual(1, code)
        patches["configure"].assert_not_called()
        patches["setup"].assert_not_called()

    def test_cli_setup_only_sets_up_sandbox(self) -> None:
        """Run sandbox setup without checking keys or running the agent."""
        code, patches = self._call_cli(["--setup-only"])

        self.assertEqual(0, code)
        patches["setup"].assert_called_once_with()
        patches["check_keys"].assert_not_called()
        patches["run_agent"].assert_not_called()

    def test_cli_setup_only_failure_returns_one(self) -> None:
        """Return one when setup-only provisioning fails."""
        code, patches = self._call_cli(["--setup-only"], setup_result=False)

        self.assertEqual(1, code)
        patches["setup"].assert_called_once_with()

    def test_cli_clean_workspace_dispatches_and_exits(self) -> None:
        """Run workspace cleanup without provisioning Docker."""
        code, patches = self._call_cli(["--clean-workspace"])

        self.assertEqual(0, code)
        patches["cleanup_workspace"].assert_called_once_with()
        patches["setup"].assert_not_called()

    def test_cli_cleanup_dispatches_and_exits(self) -> None:
        """Run container cleanup without repository work."""
        code, patches = self._call_cli(["--cleanup"])

        self.assertEqual(0, code)
        patches["cleanup_container"].assert_called_once_with(
            remove_image=False
        )
        patches["setup"].assert_not_called()

    def test_cli_clean_image_implies_cleanup(self) -> None:
        """Remove the image even when --cleanup is omitted."""
        code, patches = self._call_cli(["--clean-image"])

        self.assertEqual(0, code)
        patches["cleanup_container"].assert_called_once_with(
            remove_image=True
        )
        patches["setup"].assert_not_called()

    def test_cli_cleanup_with_repo_exits_before_porting(self) -> None:
        """Let cleanup flags win over repository execution."""
        code, patches = self._call_cli(["--cleanup", "--repo", _REPO_URL])

        self.assertEqual(0, code)
        patches["cleanup_container"].assert_called_once_with(
            remove_image=False
        )
        patches["run_agent"].assert_not_called()

    def test_cli_rebuild_without_repo_rebuilds_all_profiles(self) -> None:
        """Run dual-profile rebuild for --rebuild alone."""
        code, patches = self._call_cli(["--rebuild"])

        self.assertEqual(0, code)
        patches["rebuild"].assert_called_once_with()
        patches["setup"].assert_not_called()

    def test_cli_rebuild_failure_returns_one(self) -> None:
        """Propagate failure from dual-profile rebuild."""
        code, patches = self._call_cli(
            ["--rebuild"],
            rebuild_result=False,
        )

        self.assertEqual(1, code)
        patches["rebuild"].assert_called_once_with()

    def test_cli_rebuild_single_platform_uses_setup(self) -> None:
        """Use one active profile when --platform is explicit."""
        code, patches = self._call_cli(["--rebuild", "--platform", "alpine"])

        self.assertEqual(0, code)
        patches["rebuild"].assert_not_called()
        patches["setup"].assert_called_once_with()

    def test_cli_cache_hit_short_circuits_before_keys_and_docker(self) -> None:
        """Materialize a cached recipe without provisioning the sandbox."""
        cached = {"build_system": "cmake", "last_built": "today"}
        code, patches = self._call_cli(
            ["--repo", _REPO_URL, "--package"],
            cache=cached,
            recipe_path="/safe/project_recipe.md",
        )

        self.assertEqual(0, code)
        patches["get_cached"].assert_called_once_with("project")
        patches["materialize"].assert_called_once_with(
            "project",
            main.OUTPUT_DIR,
            cached,
        )
        patches["check_keys"].assert_not_called()
        patches["setup"].assert_not_called()
        patches["run_agent"].assert_not_called()

    def test_cli_cache_hit_reports_materialization_failure(self) -> None:
        """Return success even if writing the cached recipe fails."""
        cached = {"build_system": "make", "last_built": "yesterday"}
        code, patches = self._call_cli(
            ["--repo", _REPO_URL],
            cache=cached,
            recipe_path=None,
        )

        self.assertEqual(0, code)
        patches["materialize"].assert_called_once()
        patches["setup"].assert_not_called()

    def test_cli_force_bypasses_cache_and_runs_agent(self) -> None:
        """Run the full pipeline when --force is present."""
        code, patches = self._call_cli(
            [
                "--repo",
                _REPO_URL,
                "--force",
                "--max-attempts",
                "7",
                "--verbose",
                "--package",
            ],
            cache={"build_system": "cmake"},
        )

        self.assertEqual(0, code)
        patches["get_cached"].assert_not_called()
        patches["check_keys"].assert_called_once_with()
        patches["setup"].assert_called_once_with()
        patches["run_agent"].assert_called_once_with(
            repo_url=_REPO_URL,
            max_attempts=7,
            verbose=True,
            package=True,
        )

    def test_cli_key_failure_stops_before_setup(self) -> None:
        """Do not provision Docker when API keys are invalid."""
        code, patches = self._call_cli(
            ["--repo", _REPO_URL],
            key_result=False,
        )

        self.assertEqual(1, code)
        patches["check_keys"].assert_called_once_with()
        patches["setup"].assert_not_called()
        patches["run_agent"].assert_not_called()

    def test_cli_setup_failure_stops_before_agent(self) -> None:
        """Do not invoke the graph when Docker setup fails."""
        code, patches = self._call_cli(
            ["--repo", _REPO_URL],
            setup_result=False,
        )

        self.assertEqual(1, code)
        patches["setup"].assert_called_once_with()
        patches["run_agent"].assert_not_called()

    def test_cli_returns_run_agent_exit_code(self) -> None:
        """Propagate the graph runner's exit code."""
        code, patches = self._call_cli(
            ["--repo", _REPO_URL],
            agent_code=2,
        )

        self.assertEqual(2, code)
        patches["run_agent"].assert_called_once()

    def test_cli_platform_and_container_overrides_are_set_before_setup(
        self,
    ) -> None:
        """Set ATESOR overrides before sandbox setup runs."""
        captured: dict[str, str | None] = {}

        def fake_setup() -> bool:
            captured["platform"] = main.os.environ.get("ATESOR_PLATFORM")
            captured["container"] = main.os.environ.get("ATESOR_CONTAINER")
            return True

        code, _patches = self._call_cli(
            [
                "--repo",
                _REPO_URL,
                "--force",
                "--platform",
                "debian",
                "--container",
                "worker-1",
            ],
            setup_side_effect=fake_setup,
        )

        self.assertEqual(0, code)
        self.assertEqual("debian", captured["platform"])
        self.assertEqual("worker-1", captured["container"])

    def test_cli_rebuild_with_repo_sets_rebuild_image_and_runs_agent(
        self,
    ) -> None:
        """Set REBUILD_IMAGE for repository rebuilds."""
        captured: dict[str, str | None] = {}

        def fake_setup() -> bool:
            captured["rebuild"] = main.os.environ.get("REBUILD_IMAGE")
            return True

        code, patches = self._call_cli(
            ["--repo", _REPO_URL, "--force", "--rebuild"],
            setup_side_effect=fake_setup,
        )

        self.assertEqual(0, code)
        self.assertEqual("true", captured["rebuild"])
        patches["run_agent"].assert_called_once()

    def test_cli_cleanup_rebuild_with_repo_continues_to_port(self) -> None:
        """Allow cleanup plus rebuild to proceed into repository work."""
        code, patches = self._call_cli(
            ["--cleanup", "--rebuild", "--repo", _REPO_URL, "--force"]
        )

        self.assertEqual(0, code)
        patches["cleanup_container"].assert_called_once()
        patches["setup"].assert_called_once_with()
        patches["run_agent"].assert_called_once()

    def test_cli_sets_per_repo_llm_log(self) -> None:
        """Switch the LLM log to the derived repository name."""
        code, patches = self._call_cli(["--repo", _REPO_URL, "--force"])

        self.assertEqual(0, code)
        patches["set_log"].assert_called_once_with("project")
        patches["configure"].assert_called_once_with(
            False,
            repo_name="project",
        )

    def test_cli_help_exits_zero(self) -> None:
        """Let argparse handle --help with SystemExit(0)."""
        with mock.patch("sys.argv", ["atesor-ai", "--help"]):
            with self.assertRaises(SystemExit) as caught:
                main.main()
        self.assertEqual(0, caught.exception.code)

    def test_cli_version_exits_zero(self) -> None:
        """Let argparse handle --version with SystemExit(0)."""
        with mock.patch("sys.argv", ["atesor-ai", "--version"]):
            with self.assertRaises(SystemExit) as caught:
                main.main()
        self.assertEqual(0, caught.exception.code)

    def test_cli_bad_platform_exits_two(self) -> None:
        """Let argparse reject unsupported platform choices."""
        with mock.patch(
            "sys.argv",
            ["atesor-ai", "--setup-only", "--platform", "fedora"],
        ):
            with self.assertRaises(SystemExit) as caught:
                main.main()
        self.assertEqual(2, caught.exception.code)

    def test_cli_bad_max_attempts_exits_two(self) -> None:
        """Let argparse reject non-integer max attempts."""
        with mock.patch(
            "sys.argv",
            ["atesor-ai", "--repo", _REPO_URL, "--max-attempts", "x"],
        ):
            with self.assertRaises(SystemExit) as caught:
                main.main()
        self.assertEqual(2, caught.exception.code)

    def test_module_entrypoint_loads_atesor_home_env_and_exits(self) -> None:
        """Exercise top-level env loading and __main__ dispatch."""
        atesor_home = str(self.temp_path / "atesor-home")
        with (
            mock.patch.dict(
                os.environ,
                {"ATESOR_HOME": atesor_home},
                clear=True,
            ),
            mock.patch("dotenv.load_dotenv") as load_dotenv,
            mock.patch("sys.argv", ["main.py", "--help"]),
        ):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(Path(main.__file__)), run_name="__main__")

        self.assertEqual(0, caught.exception.code)
        env_paths = [call.args[0] for call in load_dotenv.call_args_list[1:]]
        self.assertEqual(os.path.join(atesor_home, ".env"), env_paths[0])


class TestReportDataclassShapes(_MainTestCase):
    """Tests report rendering with AgentState-derived shapes."""

    def test_report_handles_build_plan_dataclass_dict_shape(self) -> None:
        """Render a report from AgentState.to_dict()-style values."""
        state = create_initial_state(_REPO_URL)
        state.build_status = BuildStatus.SUCCESS
        state.build_plan = BuildPlan(
            build_system="make",
            build_system_confidence=0.9,
            phases=[
                BuildPhase(
                    id=1,
                    name="Build",
                    commands=["make"],
                )
            ],
            total_estimated_duration="1m",
        )
        state.dependencies = DependencyInfo(
            system_packages=["zlib-dev"],
            build_tools=["make"],
        )
        state.add_fix_attempt(
            FixAttempt(
                ErrorCategory.CONFIGURATION,
                "autoreconf",
                ["regenerated configure"],
                True,
            )
        )

        report = main.generate_detailed_report(state.to_dict())

        self.assertIn("Build System**: make", report)
        self.assertIn("make", report)
        self.assertIn("zlib-dev", report)


if __name__ == "__main__":
    unittest.main()
