"""Tests for src/config.py — workspace + docker detection."""

import os
import shutil
import unittest
import uuid
from unittest import mock

import src.config as config


class TestConfigPaths(unittest.TestCase):
    """Tests for ConfigPaths."""

    def test_workspace_root_exists(self) -> None:
        """Test workspace root exists."""
        self.assertTrue(os.path.isdir(config.WORKSPACE_ROOT))

    def test_subdirs_under_workspace(self) -> None:
        """Test subdirs under workspace."""
        for d in [
            config.OUTPUT_DIR,
            config.REPOS_DIR,
            config.CACHE_DIR,
            config.LOGS_DIR,
        ]:
            with self.subTest(d=d):
                self.assertTrue(
                    d.startswith(config.WORKSPACE_ROOT),
                    f"{d} not under workspace",
                )
                self.assertTrue(os.path.isdir(d))

    def test_get_output_dir_creates_directory(self) -> None:
        """Test get output dir creates directory."""
        out = config.get_output_dir()
        self.assertTrue(os.path.isdir(out))

    def test_docker_detection_false_on_normal_host(self) -> None:
        # Local Linux host should not be misdetected as docker
        # (skip if running in actual container)
        """Test docker detection false on normal host."""
        if os.path.exists("/.dockerenv"):
            self.skipTest("running inside docker")
        self.assertFalse(config.is_running_in_docker())

    def test_dockerenv_file_triggers_detection(self) -> None:
        """Test dockerenv file triggers detection."""
        with mock.patch(
            "os.path.exists", side_effect=lambda p: p == "/.dockerenv"
        ):
            self.assertTrue(config.is_running_in_docker())

    def test_cgroup_container_marker_triggers_detection(self) -> None:
        """Docker/containerd/kubepods cgroups trigger detection."""
        opener = mock.mock_open(read_data="0::/kubepods.slice\n")
        with (
            mock.patch("os.path.exists", return_value=False),
            mock.patch("builtins.open", opener),
        ):
            self.assertTrue(config.is_running_in_docker())

    def test_cgroup_read_errors_are_ignored(self) -> None:
        """Unreadable cgroup files fall through to other checks."""
        with (
            mock.patch("os.path.exists", return_value=False),
            mock.patch("builtins.open", side_effect=PermissionError),
        ):
            self.assertFalse(config.is_running_in_docker())

    def test_workspace_without_home_triggers_detection(self) -> None:
        """A bare sandbox-style root with /workspace is detected."""

        def exists(path: str) -> bool:
            """Pretend only the sandbox workspace exists."""
            return path == "/workspace"

        with (
            mock.patch("os.path.exists", side_effect=exists),
            mock.patch("builtins.open", side_effect=FileNotFoundError),
            mock.patch("os.path.abspath", return_value="/"),
        ):
            self.assertTrue(config.is_running_in_docker())


class TestAtesorHomeOverride(unittest.TestCase):
    """ATESOR_HOME must win everywhere, including inside a container.

    Regression: get_workspace_root() used to short-circuit on docker
    detection BEFORE consulting ATESOR_HOME, silently sending all
    workspace state to /workspace when the packaged CLI ran inside any
    container (caught by packaging/deb/validate_deb.sh step 6).
    """

    def test_atesor_home_wins_inside_docker(self) -> None:
        """The override beats the in-docker /workspace shortcut."""
        home = os.path.join(
            os.getcwd(),
            "workspace",
            "test-config",
            f"atesor-home-{uuid.uuid4().hex}",
        )
        os.makedirs(home, exist_ok=True)
        try:
            with mock.patch.dict(os.environ, {"ATESOR_HOME": home}):
                with mock.patch.object(
                    config, "is_running_in_docker", return_value=True
                ):
                    root = config.get_workspace_root()
        finally:
            shutil.rmtree(home, ignore_errors=True)
        self.assertEqual(root, os.path.join(home, "workspace"))

    def test_docker_default_without_override(self) -> None:
        """Without the override the sandbox default stays /workspace."""
        env = {
            k: v for k, v in os.environ.items() if k != "ATESOR_HOME"
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                config, "is_running_in_docker", return_value=True
            ):
                self.assertEqual(config.get_workspace_root(), "/workspace")

    def test_state_home_uses_docker_default(self) -> None:
        """Without overrides, in-docker state lives at /workspace."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"ATESOR_HOME", "XDG_DATA_HOME"}
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                config, "is_running_in_docker", return_value=True
            ),
            mock.patch("os.makedirs"),
        ):
            self.assertEqual(config.get_state_home(), "/workspace")

    def test_state_home_uses_xdg_fallback_outside_source(self) -> None:
        """Non-source installs use XDG data home when writable source fails."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"ATESOR_HOME", "XDG_DATA_HOME"}
        }
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                config, "is_running_in_docker", return_value=False
            ),
            mock.patch("os.access", return_value=False),
            mock.patch("os.path.expanduser", return_value="/home/tester"),
            mock.patch("os.makedirs"),
        ):
            state_home = config.get_state_home()

        self.assertEqual(state_home, "/home/tester/.local/share/atesor-ai")


class TestPathTranslation(unittest.TestCase):
    """Tests for container-to-host path translation."""

    def test_workspace_path_translates_when_host_lacks_workspace(self) -> None:
        """Container workspace paths map to configured host workspace."""
        with mock.patch("os.path.exists", return_value=False):
            got = config.to_host_path("/workspace/repos/pkg")
        self.assertEqual(
            got,
            os.path.join(config.WORKSPACE_ROOT, "repos/pkg"),
        )

    def test_workspace_path_stays_inside_container(self) -> None:
        """When /workspace exists locally, paths are left unchanged."""
        with mock.patch("os.path.exists", return_value=True):
            self.assertEqual(
                config.to_host_path("/workspace/repos/pkg"),
                "/workspace/repos/pkg",
            )

    def test_non_workspace_path_stays_unchanged(self) -> None:
        """Non-container paths are returned as-is."""
        with mock.patch("os.path.exists", return_value=False):
            self.assertEqual(config.to_host_path("/opt/pkg"), "/opt/pkg")


if __name__ == "__main__":
    unittest.main()
