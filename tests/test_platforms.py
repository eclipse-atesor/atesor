"""Tests for src/platforms.py — detection, override, helpers."""

from __future__ import annotations

import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from src import platforms
from src.platforms import (
    ALPINE_RISCV,
    DEBIAN_RISCV,
    PROFILES,
    detect_platform,
    get_active_profile,
    get_container_name,
    set_active_profile,
)


class TestProfileHelpers(unittest.TestCase):
    """Tests for ProfileHelpers."""

    def test_install_cmd_chains_update_and_install(self) -> None:
        """Test install cmd chains update and install."""
        cmd = ALPINE_RISCV.install_cmd(["zlib-dev", "openssl-dev"])
        self.assertIn("apk update", cmd)
        self.assertIn("apk add zlib-dev openssl-dev", cmd)
        self.assertIn("&&", cmd)

    def test_install_cmd_debian_uses_apt(self) -> None:
        """Test install cmd debian uses apt."""
        cmd = DEBIAN_RISCV.install_cmd(["libssl-dev"])
        self.assertIn("apt-get update", cmd)
        self.assertIn(
            "apt-get install -y --no-install-recommends libssl-dev", cmd
        )

    def test_resolve_known_canonical_maps_to_distro_name(self) -> None:
        """Test resolve known canonical maps to distro name."""
        self.assertEqual(ALPINE_RISCV.resolve("zlib"), "zlib-dev")
        self.assertEqual(DEBIAN_RISCV.resolve("zlib"), "zlib1g-dev")

    def test_resolve_unknown_falls_back_to_input(self) -> None:
        """Test resolve unknown falls back to input."""
        self.assertEqual(ALPINE_RISCV.resolve("foo-bar-zzz"), "foo-bar-zzz")

    def test_alpine_libc_is_musl_debian_is_glibc(self) -> None:
        """Test alpine libc is musl debian is glibc."""
        self.assertEqual(ALPINE_RISCV.libc, "musl")
        self.assertEqual(DEBIAN_RISCV.libc, "glibc")

    def test_ubuntu_aliases_to_debian(self) -> None:
        """Test ubuntu aliases to debian."""
        self.assertIs(PROFILES["ubuntu"], DEBIAN_RISCV)


class TestSetActiveProfile(unittest.TestCase):
    """Tests for SetActiveProfile."""

    def test_set_by_name(self) -> None:
        """Test set by name."""
        p = set_active_profile("debian")
        self.assertIs(p, DEBIAN_RISCV)
        self.assertIs(get_active_profile(), DEBIAN_RISCV)

    def test_set_by_profile_object(self) -> None:
        """Test set by profile object."""
        set_active_profile(ALPINE_RISCV)
        self.assertIs(get_active_profile(), ALPINE_RISCV)

    def test_unknown_name_raises_value_error(self) -> None:
        """Test unknown name raises value error."""
        with self.assertRaises(ValueError) as ctx:
            set_active_profile("solaris")
        self.assertIn("solaris", str(ctx.exception))


class TestDetectPlatform(unittest.TestCase):
    """Tests for DetectPlatform."""

    def setUp(self) -> None:
        # Clear cache and env
        """Set up test fixtures."""
        platforms._cached_profile = None
        self._old_env = os.environ.pop("ATESOR_PLATFORM", None)

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        platforms._cached_profile = None
        if self._old_env is not None:
            os.environ["ATESOR_PLATFORM"] = self._old_env
        else:
            os.environ.pop("ATESOR_PLATFORM", None)

    def test_env_override_takes_precedence(self) -> None:
        """Test env override takes precedence."""
        os.environ["ATESOR_PLATFORM"] = "debian"
        # Even if subprocess would say alpine, the override wins
        with mock.patch("src.platforms.subprocess.run") as mrun:
            mrun.return_value = SimpleNamespace(
                returncode=0, stdout="ID=alpine\n", stderr=""
            )
            self.assertIs(detect_platform("dummy"), DEBIAN_RISCV)
            mrun.assert_not_called()  # override short-circuits the docker call

    def test_invalid_env_override_ignored(self) -> None:
        """Test invalid env override ignored."""
        os.environ["ATESOR_PLATFORM"] = "windows"
        with mock.patch("src.platforms.subprocess.run") as mrun:
            mrun.return_value = SimpleNamespace(
                returncode=0, stdout="ID=alpine\n", stderr=""
            )
            p = detect_platform("dummy")
            self.assertIs(p, ALPINE_RISCV)

    def test_detects_alpine_from_os_release(self) -> None:
        """Test detects alpine from os release."""
        with mock.patch("src.platforms.subprocess.run") as mrun:
            mrun.return_value = SimpleNamespace(
                returncode=0, stdout="ID=alpine\nVERSION_ID=3.19\n", stderr=""
            )
            self.assertIs(detect_platform("c"), ALPINE_RISCV)

    def test_detects_debian_from_os_release(self) -> None:
        """Test detects debian from os release."""
        with mock.patch("src.platforms.subprocess.run") as mrun:
            mrun.return_value = SimpleNamespace(
                returncode=0, stdout="ID=debian\nVERSION_ID=12\n", stderr=""
            )
            self.assertIs(detect_platform("c"), DEBIAN_RISCV)

    def test_unknown_distro_falls_back_to_default(self) -> None:
        """Test unknown distro falls back to default."""
        with mock.patch("src.platforms.subprocess.run") as mrun:
            mrun.return_value = SimpleNamespace(
                returncode=0, stdout="ID=arch\n", stderr=""
            )
            self.assertIs(detect_platform("c"), ALPINE_RISCV)

    def test_docker_failure_falls_back_to_default(self) -> None:
        """Test docker failure falls back to default."""
        with mock.patch("src.platforms.subprocess.run") as mrun:
            mrun.return_value = SimpleNamespace(
                returncode=1, stdout="", stderr="no such container"
            )
            self.assertIs(detect_platform("c"), ALPINE_RISCV)

    def test_docker_timeout_falls_back_to_default(self) -> None:
        """Test docker timeout falls back to default."""
        with mock.patch(
            "src.platforms.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=10),
        ):
            self.assertIs(detect_platform("c"), ALPINE_RISCV)

    def test_docker_missing_falls_back_to_default(self) -> None:
        """Test docker missing falls back to default."""
        with mock.patch(
            "src.platforms.subprocess.run",
            side_effect=FileNotFoundError("no docker"),
        ):
            self.assertIs(detect_platform("c"), ALPINE_RISCV)

    def test_os_release_quoted_id_handled(self) -> None:
        """Test os release quoted id handled."""
        with mock.patch("src.platforms.subprocess.run") as mrun:
            mrun.return_value = SimpleNamespace(
                returncode=0, stdout='ID="alpine"\n', stderr=""
            )
            self.assertIs(detect_platform("c"), ALPINE_RISCV)


class TestGetContainerName(unittest.TestCase):
    """Tests for GetContainerName."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self._old = os.environ.pop("ATESOR_CONTAINER", None)
        platforms._cached_profile = ALPINE_RISCV

    def tearDown(self) -> None:
        """Tear down test fixtures."""
        if self._old is not None:
            os.environ["ATESOR_CONTAINER"] = self._old
        else:
            os.environ.pop("ATESOR_CONTAINER", None)
        platforms._cached_profile = None

    def test_env_override_used(self) -> None:
        """Test env override used."""
        os.environ["ATESOR_CONTAINER"] = "atesor-w3"
        self.assertEqual(get_container_name(), "atesor-w3")

    def test_env_override_is_stripped(self) -> None:
        """Test env override is stripped."""
        os.environ["ATESOR_CONTAINER"] = "  atesor-w4 \n"
        self.assertEqual(get_container_name(), "atesor-w4")

    def test_falls_back_to_active_profile_name(self) -> None:
        """Test falls back to active profile name."""
        self.assertEqual(get_container_name(), "atesor-ai-sandbox")
        set_active_profile("debian")
        self.assertEqual(get_container_name(), "atesor-ai-sandbox-debian")


class TestNameCorrections(unittest.TestCase):
    """Tests for NameCorrections."""

    def test_alpine_corrects_debian_lzma_name(self) -> None:
        """Test alpine corrects debian lzma name."""
        self.assertEqual(
            ALPINE_RISCV.name_corrections.get("liblzma-dev"), "xz-dev"
        )

    def test_debian_corrects_alpine_xz_name(self) -> None:
        """Test debian corrects alpine xz name."""
        self.assertEqual(
            DEBIAN_RISCV.name_corrections.get("xz-dev"), "liblzma-dev"
        )

    def test_both_profiles_have_corrections_for_common_libs(self) -> None:
        """Test both profiles have corrections for common libs."""
        for lib in ["zlib-dev", "openssl-dev"]:
            with self.subTest(lib=lib):
                # Debian should know how to translate this Alpine name
                self.assertIn(lib, DEBIAN_RISCV.name_corrections)


class TestDebianGoBundling(unittest.TestCase):
    """Guard that Debian/Ubuntu Go is not routed through apt.

    The Debian/Ubuntu sandbox bundles a modern Go toolchain in
    /usr/local/go. Ubuntu jammy's ``golang`` package is Go 1.18 and
    cannot parse modern go.mod files. Regression guard: planner/scout
    must not be able to derive an ``apt-get install golang`` from the
    package map.
    """

    def test_debian_pkgmap_does_not_route_go_to_apt(self) -> None:
        # Either the key is absent, or it explicitly opts out (empty string).
        """Test debian pkgmap does not route go to apt."""
        if "go" in DEBIAN_RISCV.package_map:
            self.assertFalse(
                DEBIAN_RISCV.package_map["go"],
                "Debian package_map['go'] must be empty/absent — "
                "Go is bundled at /usr/local/go in Dockerfile.debian.",
            )

    def test_debian_notes_warn_about_bundled_go(self) -> None:
        """Test debian notes warn about bundled go."""
        notes = " ".join(DEBIAN_RISCV.extra_notes).lower()
        self.assertIn(
            "go is pre-installed",
            notes,
            "Debian extra_notes must warn the LLM that Go is bundled "
            "(prevents `apt-get install golang`).",
        )

    def test_debian_corrects_invented_package_names(self) -> None:
        # Real failures observed in workspace/logs/ batch run.
        """Test debian corrects invented package names."""
        self.assertEqual(
            DEBIAN_RISCV.name_corrections.get("libcurl4-libssl-dev"),
            "libcurl4-openssl-dev",
        )
        self.assertEqual(
            DEBIAN_RISCV.name_corrections.get("libcares-dev"), "libc-ares-dev"
        )


class TestProfileCacheAndDefaults(unittest.TestCase):
    """Cover the cache-miss path and default container resolution."""

    def test_get_active_profile_detects_on_cache_miss(self) -> None:
        """Test get active profile detects on cache miss."""
        platforms._cached_profile = None
        with mock.patch.object(
            platforms, "detect_platform", return_value=DEBIAN_RISCV
        ) as detect:
            self.assertIs(get_active_profile(), DEBIAN_RISCV)
        detect.assert_called_once()
        # Second call must hit the cache, not re-detect.
        self.assertIs(get_active_profile(), DEBIAN_RISCV)

    def test_detect_platform_uses_default_container(self) -> None:
        """Test detect platform uses default container."""
        env = {k: v for k, v in os.environ.items()}
        env.pop("ATESOR_PLATFORM", None)
        env.pop("ATESOR_CONTAINER", None)
        completed = SimpleNamespace(returncode=1, stdout="", stderr="")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch(
            "src.platforms.subprocess.run", return_value=completed
        ) as run:
            profile = detect_platform()
        self.assertIs(profile, platforms._DEFAULT_PROFILE)
        cmd = run.call_args[0][0]
        self.assertIn(platforms._DEFAULT_PROFILE.container_name, cmd)


if __name__ == "__main__":
    unittest.main()
