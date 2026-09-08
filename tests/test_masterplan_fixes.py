"""Regression tests for the MASTER-PLAN framework fixes.

Each class maps to one bucket in ``review/MASTER-PLAN.md``:

  * rust-toolchain-old (23 pkgs) — Rust is baked into BOTH images and a
    runtime ``apk add rust cargo`` can never shadow it.
  * autotools-bootstrap (11 pkgs) — a repo shipping ``configure.ac`` but
    no committed ``configure`` gets bootstrapped instead of dying on
    ``./configure: No such file or directory``.
  * no-makefile-wrong-cmd (7 pkgs) — ``make`` runs where the makefile
    actually is, and nested build roots are detected at all.
  * go-toolchain-old (2 pkgs) — the image pin satisfies the go.mod
    requirements that previously failed under ``GOTOOLCHAIN=local``.

Plus the drift guard that makes all of it stick: batch preflight
minimums must never lag the Dockerfile pins, otherwise existing worker
containers are never rebuilt and the fixes never reach CI.

No test here runs Docker, the network, or an LLM.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import tempfile

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _dockerfile_pins(name: str) -> dict:
    """Extract the toolchain ARG pins from a Dockerfile."""
    text = (_REPO_ROOT / name).read_text()
    pins = {}
    go = re.search(r"^ARG GO_VERSION=(\S+)", text, re.M)
    rust = re.search(r"^ARG RUST_TOOLCHAIN=(\S+)", text, re.M)
    if go:
        pins["go"] = go.group(1)
    if rust:
        pins["rust"] = rust.group(1)
    return pins


def _load_batch_module():
    spec = importlib.util.spec_from_file_location(
        "batch_test_mp", str(_REPO_ROOT / ".github/scripts/batch_test.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mkrepo(paths) -> str:
    """Create a temp repo tree containing ``paths`` (relative files)."""
    root = tempfile.mkdtemp()
    for rel in paths:
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w").close()
    return root


# ===========================================================================
# rust-toolchain-old — 23 packages
# ===========================================================================


class TestRustToolchainBakedIn:
    """Both images ship a modern Rust; distro Rust can never shadow it."""

    def test_both_images_pin_modern_rust(self) -> None:
        """Alpine previously had no Rust at all; Debian was 1.85.0."""
        for name in ("Dockerfile", "Dockerfile.debian"):
            pins = _dockerfile_pins(name)
            assert "rust" in pins, f"{name} does not bake in a Rust toolchain"
            major, minor = (int(p) for p in pins["rust"].split(".")[:2])
            assert (major, minor) >= (1, 88), (
                f"{name} pins rustc {pins['rust']}; crates require >= 1.88"
            )

    def test_both_images_pin_same_rust(self) -> None:
        """Skew between images reintroduces per-distro failures."""
        assert (
            _dockerfile_pins("Dockerfile")["rust"]
            == _dockerfile_pins("Dockerfile.debian")["rust"]
        )

    def test_runtime_rust_install_is_stripped(self) -> None:
        """`apk add rust cargo` must not shadow the rustup toolchain."""
        from src.tools import _strip_bundled_toolchain_packages as strip

        assert strip("apk add rust cargo") == "true"
        assert strip("apt-get install -y rustc cargo") == "true"
        # Other packages in the same command survive.
        assert strip("apk add --no-cache rust cargo build-base") == (
            "apk add --no-cache build-base"
        )

    def test_rust_not_mapped_to_a_distro_package(self) -> None:
        """Neither profile may resolve 'rust' to an installable package."""
        from src import platforms

        for profile in (platforms.ALPINE_RISCV, platforms.DEBIAN_RISCV):
            assert "rust" not in profile.package_map, (
                f"{profile.name} still maps rust to a distro package"
            )


# ===========================================================================
# autotools-bootstrap — 11 packages
# ===========================================================================


class TestAutotoolsBootstrap:
    """A missing ./configure is generated rather than fatal."""

    def test_configure_ac_triggers_autoreconf(self) -> None:
        """The canonical case behind grep/less/make/tar/wget."""
        from src.graph import _autotools_bootstrap_prefix as boot

        assert boot(_mkrepo(["configure.ac"])) == "autoreconf -fi"
        assert boot(_mkrepo(["configure.in"])) == "autoreconf -fi"

    def test_upstream_script_preferred_over_autoreconf(self) -> None:
        """autogen.sh/buildconf wrap project-specific steps."""
        from src.graph import _autotools_bootstrap_prefix as boot

        assert boot(_mkrepo(["autogen.sh", "configure.ac"])) == (
            "sh autogen.sh"
        )
        assert boot(_mkrepo(["buildconf", "configure.ac"])) == "sh buildconf"

    def test_committed_configure_is_left_alone(self) -> None:
        """No bootstrap when upstream ships a real configure."""
        from src.graph import _autotools_bootstrap_prefix as boot

        assert boot(_mkrepo(["configure", "configure.ac"])) == ""

    def test_non_autotools_repo_is_left_alone(self) -> None:
        """Never inject autoreconf into a non-autotools project."""
        from src.graph import _autotools_bootstrap_prefix as boot

        assert boot(_mkrepo(["Makefile"])) == ""
        assert boot(_mkrepo(["go.mod"])) == ""

    def test_gettext_repo_avoids_autoreconf(self) -> None:
        """Running autoreconf calls autopoint, reverting gettext.m4."""
        from src.graph import _autotools_bootstrap_prefix as boot

        cmd = boot(_mkrepo(["configure.ac", "m4/gettext.m4"]))
        assert "autoreconf" not in cmd
        assert "aclocal" in cmd

    def test_bootstrapped_commands_pass_the_validator(self) -> None:
        """A rewritten command must survive CommandValidator."""
        from src.graph import _autotools_bootstrap_prefix as boot
        from src.tools import CommandValidator

        validator = CommandValidator()
        for repo in (
            _mkrepo(["configure.ac"]),
            _mkrepo(["autogen.sh", "configure.ac"]),
            _mkrepo(["configure.ac", "m4/gettext.m4"]),
        ):
            command = f"{boot(repo)} && ./configure"
            ok, reason = validator.is_safe(command)
            assert ok, f"validator rejected {command!r}: {reason}"


# ===========================================================================
# no-makefile-wrong-cmd — 7 packages
# ===========================================================================


class TestBuildRootThreading:
    """`make` is steered to the directory that owns the makefile."""

    def test_finds_nested_makefile(self) -> None:
        """The exact shape behind 'no makefile found'."""
        from src.graph import _find_makefile_dir as find

        assert find(_mkrepo(["src/Makefile"])) == "src"
        assert find(_mkrepo(["build/GNUmakefile"])) == "build"

    def test_root_makefile_is_not_redirected(self) -> None:
        """No rewrite when the root already builds."""
        from src.graph import _find_makefile_dir as find

        assert find(_mkrepo(["Makefile", "src/Makefile"])) == ""

    def test_shallowest_wins_and_noise_is_skipped(self) -> None:
        """Vendored/test trees must never become the build root."""
        from src.graph import _find_makefile_dir as find

        assert find(_mkrepo(["a/b/Makefile", "z/Makefile"])) == "z"
        assert find(_mkrepo(["vendor/Makefile", "testdata/Makefile"])) == ""
        assert find(_mkrepo([".git/Makefile"])) == ""

    def test_no_makefile_anywhere_preserves_real_error(self) -> None:
        """Returning '' keeps the truthful build failure intact."""
        from src.graph import _find_makefile_dir as find

        assert find(_mkrepo(["README.md"])) == ""

    def test_nested_build_root_is_detected(self) -> None:
        """Detection reported 'unknown' for nested roots before."""
        from src.scripted_ops import ScriptedOperations

        ops = ScriptedOperations()
        for paths, expected_type, expected_dir in (
            (["src/CMakeLists.txt"], "cmake", "src"),
            (["src/Makefile"], "make", "src"),
            (["build/meson.build"], "meson", "build"),
            (["lib/configure.ac"], "autotools", "lib"),
        ):
            info = ops.detect_build_system(_mkrepo(paths))
            assert info.type == expected_type, paths
            assert info.module_dir == expected_dir, paths

    def test_root_detection_still_wins_over_nested(self) -> None:
        """The nested search is a fallback, never an override."""
        from src.scripted_ops import ScriptedOperations

        info = ScriptedOperations().detect_build_system(
            _mkrepo(["CMakeLists.txt", "src/Makefile"])
        )
        assert info.type == "cmake"
        assert info.module_dir == ""
        assert info.confidence == 0.95


# ===========================================================================
# go-toolchain-old — 2 packages
# ===========================================================================


class TestGoToolchainPin:
    """The pin covers the go.mod minimums that previously failed."""

    def test_pin_satisfies_observed_requirements(self) -> None:
        """The berty repo needs >= 1.26.4 and witness needs >= 1.26.5."""
        for name in ("Dockerfile", "Dockerfile.debian"):
            pin = _dockerfile_pins(name)["go"]
            parts = tuple(int(p) for p in pin.split("."))
            assert parts >= (1, 26, 5), f"{name} pins go {pin}"

    def test_both_images_pin_same_go(self) -> None:
        """Skew makes a package build on one distro and not the other."""
        assert (
            _dockerfile_pins("Dockerfile")["go"]
            == _dockerfile_pins("Dockerfile.debian")["go"]
        )


# ===========================================================================
# DATA bucket — dead upstreams, malformed URLs, repo_name collisions
# ===========================================================================


class TestPackageCatalogHygiene:
    """full.json holds no unbuildable entries."""

    def _catalog(self) -> list:
        import json

        path = _REPO_ROOT / ".github/packages/full.json"
        return json.loads(path.read_text())["packages"]

    def test_dead_upstreams_removed(self) -> None:
        """All seven returned HTTP 404 while a control returned 200."""
        dead_urls = {
            "https://github.com/dcantrell/bsdutils",
            "https://github.com/fmnx/cftun",
            "https://github.com/HuntDownProject/HEDnsExtractor",
            "https://github.com/xtaci/kcptun",
            "https://github.com/yosebyte/nodepass",
            "https://github.com/apernet/OpenGFW",
            "https://github.com/kleimont0x00/ppmap",
        }
        present = {p["url"] for p in self._catalog()}
        assert not (dead_urls & present)

    def test_urls_are_clonable_shape(self) -> None:
        """No homepages or query strings masquerading as git URLs."""
        from src.state import is_valid_repo_url

        for pkg in self._catalog():
            url = pkg["url"]
            assert is_valid_repo_url(url), f"{pkg['name']}: {url}"
            assert "?" not in url, f"{pkg['name']}: query string in {url}"

    def test_repointed_entries(self) -> None:
        """findutils/ncdu now point at real git remotes."""
        by_name = {p["name"]: p["url"] for p in self._catalog()}
        assert by_name["findutils"] == (
            "https://git.savannah.gnu.org/git/findutils.git"
        )
        assert by_name["ncdu"] == "https://code.blicky.net/yorhel/ncdu.git"

    def test_no_repo_name_collisions(self) -> None:
        """Colliding keys silently overwrite each other's everything.

        repo_name keys the workspace dir, recipe cache, log files and
        release asset. Five packages all deriving 'cli' meant at most
        one of them was ever really built.
        """
        import collections

        from src.state import derive_repo_name

        keys = collections.defaultdict(list)
        for pkg in self._catalog():
            keys[derive_repo_name(pkg["url"])].append(pkg["name"])
        collisions = {k: v for k, v in keys.items() if len(v) > 1}
        assert not collisions, f"repo_name collisions: {collisions}"


class TestRepoNameDerivation:
    """derive_repo_name is the single source of truth for the key."""

    def test_generic_basenames_are_disambiguated(self) -> None:
        """The five /cli upstreams get distinct keys."""
        from src.state import derive_repo_name

        assert derive_repo_name("https://github.com/smallstep/cli") == (
            "smallstep-cli"
        )
        assert derive_repo_name("https://github.com/cli/cli") == "cli-cli"
        assert derive_repo_name("https://github.com/ipinfo/cli") == (
            "ipinfo-cli"
        )
        assert derive_repo_name("https://gitlab.com/gitlab-org/cli") == (
            "gitlab-org-cli"
        )

    def test_ordinary_names_are_unchanged(self) -> None:
        """Historical keys must stay byte-identical or the cache misses."""
        from src.state import derive_repo_name

        assert derive_repo_name("https://github.com/madler/zlib") == "zlib"
        assert derive_repo_name("https://github.com/foo/bar.git") == "bar"
        assert derive_repo_name("https://github.com/foo/bar/") == "bar"

    def test_matches_create_initial_state(self) -> None:
        """main.py and the state factory must never diverge."""
        from src.state import create_initial_state, derive_repo_name

        for url in (
            "https://github.com/madler/zlib",
            "https://github.com/smallstep/cli",
            "https://github.com/foo/bar.git",
        ):
            assert derive_repo_name(url) == create_initial_state(url).repo_name

    def test_result_is_always_shell_safe(self) -> None:
        """The key is interpolated into rm -rf / cd / git clone."""
        import re

        from src.state import derive_repo_name

        for url in (
            "https://github.com/a/cli",
            "https://example.org/x/core",
            "https://github.com/foo/bar",
        ):
            assert re.fullmatch(r"[A-Za-z0-9._-]+", derive_repo_name(url))


# ===========================================================================
# Drift guard — makes every toolchain fix above actually reach CI
# ===========================================================================


class TestPreflightGateTracksImages:
    """batch preflight minimums must not lag the Dockerfile pins.

    If they do, `_refresh_worker_pool_if_needed` considers an old
    container current, never rebuilds it, and the toolchain fix never
    takes effect on the worker pool.
    """

    def test_minimums_match_image_pins(self) -> None:
        """Every platform's minimum is satisfied by its image."""
        batch = _load_batch_module()
        for platform, minimums in batch._MIN_TOOLCHAIN.items():
            dockerfile = (
                "Dockerfile" if platform == "alpine" else "Dockerfile.debian"
            )
            pins = _dockerfile_pins(dockerfile)
            assert not batch._version_lt(pins["go"], minimums["go"]), (
                f"{platform}: image go {pins['go']} < gate {minimums['go']}"
            )
            assert not batch._version_lt(pins["rust"], minimums["cargo"]), (
                f"{platform}: image rust {pins['rust']} < "
                f"gate {minimums['cargo']}"
            )

    def test_alpine_is_gated(self) -> None:
        """Alpine had no gate, so its pool was never refreshed."""
        assert "alpine" in _load_batch_module()._MIN_TOOLCHAIN

    def test_stale_container_is_detected(self) -> None:
        """The old toolchains must be reported as needing a rebuild."""
        batch = _load_batch_module()
        minimums = batch._MIN_TOOLCHAIN["debian"]

        assert batch._stale_reasons({"go": "1.26.3", "cargo": "1.90.0"},
                                    minimums)
        assert batch._stale_reasons({"go": "1.26.5", "cargo": "1.85.0"},
                                    minimums)
        # Missing Rust entirely (the old Alpine image).
        assert batch._stale_reasons({"go": "1.26.5"},
                                    batch._MIN_TOOLCHAIN["alpine"])
        # A current container triggers no rebuild.
        assert not batch._stale_reasons(
            {"go": "1.26.5", "cargo": "1.90.0"}, minimums
        )
