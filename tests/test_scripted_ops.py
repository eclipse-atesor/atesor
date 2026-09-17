"""Tests for src/scripted_ops.py using real temporary directory trees.

These tests build small fake repos on disk (no LLM, no network, no Docker)
and verify that the deterministic analysis layer makes the right calls.
"""

from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from unittest import mock

import pytest

from src.scripted_ops import ScriptedOperations, quick_analysis
from src.state import CommandResult

# ---------- helpers ----------


def _write(root, rel, content=""):
    """Write."""
    p = os.path.join(root, rel)
    (
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.dirname(rel)
        else None
    )
    with open(p, "w") as f:
        f.write(content)
    return p


@pytest.fixture
def repo():
    """Create a repository-local temporary repo tree."""
    scratch_root = os.path.abspath(
        os.path.join("workspace", "test_scripted_ops")
    )
    os.makedirs(scratch_root, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="repo-",
        dir=scratch_root,
    ) as repo_dir:
        yield repo_dir
    try:
        os.rmdir(scratch_root)
    except OSError:
        pass


def _command_result(cmd, exit_code=0, stdout="", stderr=""):
    """Build a CommandResult for mocked shell calls."""
    return CommandResult(str(cmd), exit_code, stdout, stderr, 0.0)


def _empty_execute(cmd, **kwargs):
    """Return a harmless empty result for mocked execute_command calls."""
    return _command_result(cmd)


# ---------- Build system detection ----------


class TestDetectBuildSystem:
    """Tests for DetectBuildSystem."""

    def test_no_build_files_returns_unknown(self, repo) -> None:
        """Test no build files returns unknown."""
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "unknown"
        assert info.confidence == 0.0

    def test_cmake_wins_when_marker_present(self, repo) -> None:
        """Test cmake wins when marker present."""
        _write(repo, "CMakeLists.txt", "project(foo)\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cmake"
        assert info.confidence >= 0.9
        assert info.primary_file == "CMakeLists.txt"

    def test_cmake_wins_over_makefile_when_both_present(self, repo) -> None:
        # cmake gets 0.95, make gets ~0.3 → cmake wins
        """Test cmake wins over makefile when both present."""
        _write(repo, "CMakeLists.txt", "project(foo)\n")
        _write(repo, "Makefile", "all:\n\techo hi\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cmake"

    def test_cargo_detected(self, repo) -> None:
        """Test cargo detected."""
        _write(
            repo, "Cargo.toml", '[package]\nname = "foo"\nversion = "0.1.0"\n'
        )
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "cargo"
        assert info.confidence >= 0.9

    def test_go_detected(self, repo) -> None:
        """Test go detected."""
        _write(repo, "go.mod", "module foo\ngo 1.21\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "go"

    def test_go_in_subdirectory_falls_back(self, repo) -> None:
        # No top-level go.mod, but a subdir has one
        """Test go in subdirectory falls back."""
        os.makedirs(os.path.join(repo, "src/sub"))
        _write(repo, "src/sub/go.mod", "module foo\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "go"
        assert info.module_dir.endswith("src/sub")

    def test_gopath_style_go_without_mod_detected(self, repo) -> None:
        """Test GOPATH-style Go repo is detected as Go."""
        _write(repo, "cmd/tool/main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "go"
        assert info.confidence >= 0.6

    def test_autotools_detected(self, repo) -> None:
        """Test autotools detected."""
        _write(repo, "configure.ac", "AC_INIT(foo, 1.0)\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        assert info.type == "autotools"
        assert info.confidence >= 0.9

    def test_meson_detected(self, repo) -> None:
        """Test meson detected."""
        _write(repo, "meson.build", "project('foo', 'c')\n")
        info = ScriptedOperations(repo).detect_build_system(repo)
        # meson uses lower-confidence (+0.3) heuristic but should win alone
        assert info.type == "meson"


# ---------- Dependency extraction ----------


class TestExtractDependencies:
    """Tests for ExtractDependencies."""

    def test_cmake_finds_find_package_calls(self, repo) -> None:
        """Test cmake finds find package calls."""
        _write(
            repo,
            "CMakeLists.txt",
            textwrap.dedent("""
            project(foo)
            find_package(OpenSSL REQUIRED)
            find_package(ZLIB)
            find_package(Threads)
        """),
        )
        deps = ScriptedOperations(repo)._extract_cmake_dependencies(repo)
        assert "OpenSSL" in deps.libraries
        assert "ZLIB" in deps.libraries
        assert "Threads" in deps.libraries
        assert deps.install_method == "apk"
        assert "cmake" in deps.build_tools

    def test_cargo_extracts_dependencies(self, repo) -> None:
        """Test cargo extracts dependencies."""
        _write(
            repo,
            "Cargo.toml",
            textwrap.dedent("""
            [package]
            name = "foo"
            version = "0.1.0"

            [dependencies]
            serde = "1.0"
            tokio = "1.0"
        """),
        )
        deps = ScriptedOperations(repo)._extract_cargo_dependencies(repo)
        assert set(deps.libraries) == {"serde", "tokio"}
        assert "cargo" in deps.build_tools

    def test_python_extracts_requirements(self, repo) -> None:
        """Test python extracts requirements."""
        _write(
            repo,
            "requirements.txt",
            textwrap.dedent("""
            # a comment
            requests==2.28.0
            numpy>=1.20
            click
        """),
        )
        deps = ScriptedOperations(repo)._extract_python_dependencies(repo)
        assert set(deps.libraries) == {"requests", "numpy", "click"}

    def test_npm_extracts_both_dep_buckets(self, repo) -> None:
        """Test npm extracts both dep buckets."""
        _write(
            repo,
            "package.json",
            json.dumps(
                {
                    "name": "foo",
                    "dependencies": {"express": "^4.0.0"},
                    "devDependencies": {"jest": "^29.0.0"},
                }
            ),
        )
        deps = ScriptedOperations(repo)._extract_npm_dependencies(repo)
        assert set(deps.libraries) == {"express", "jest"}

    def test_go_mod_extracts_require_lines(self, repo) -> None:
        """Test go mod extracts require lines."""
        _write(
            repo,
            "go.mod",
            textwrap.dedent("""
            module foo
            go 1.21

            require github.com/spf13/cobra v1.0.0
            require golang.org/x/sys v0.5.0
        """),
        )
        deps = ScriptedOperations(repo)._extract_go_dependencies(repo)
        assert "github.com/spf13/cobra" in deps.libraries
        assert "golang.org/x/sys" in deps.libraries

    def test_missing_file_returns_empty_deps(self, repo) -> None:
        """Test missing file returns empty deps."""
        deps = ScriptedOperations(repo)._extract_cmake_dependencies(repo)
        assert deps.libraries == []


# ---------- find_go_main_package ----------


class TestFindGoMainPackage:
    """Tests for FindGoMainPackage."""

    def test_no_go_files_returns_empty(self, repo) -> None:
        """Test no go files returns empty."""
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["has_main"] is False
        assert info["has_go_mod"] is False
        assert info["needs_go_init"] is False

    def test_gopath_style_repo_needs_init(self, repo) -> None:
        # .go files but no go.mod
        """Test gopath style repo needs init."""
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["needs_go_init"] is True
        assert info["has_main"] is True

    def test_simple_root_main(self, repo) -> None:
        """Test simple root main."""
        _write(repo, "go.mod", "module foo\n")
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(repo)
        assert info["has_main"] is True
        assert info["has_go_mod"] is True
        assert info["main_path"] == "."
        assert info["build_command"] == "go build ."

    def test_cmd_reponame_beats_root_main(self, repo) -> None:
        # repo basename matters for scoring
        """Test cmd reponame beats root main."""
        app_repo = os.path.join(repo, "myapp")
        os.makedirs(app_repo)
        _write(app_repo, "go.mod", "module myapp\n")
        _write(app_repo, "main.go", "package main\nfunc main() {}\n")
        _write(
            app_repo,
            "cmd/myapp/main.go",
            "package main\nfunc main() {}\n",
        )
        info = ScriptedOperations(repo).find_go_main_package(app_repo)
        assert info["main_path"] == "cmd/myapp"
        assert info["build_command"] == "go build ./cmd/myapp"

    def test_cmd_test_dir_demoted(self, repo) -> None:
        """Test cmd test dir demoted."""
        app_repo = os.path.join(repo, "myapp")
        os.makedirs(app_repo)
        _write(app_repo, "go.mod", "module myapp\n")
        _write(
            app_repo,
            "cmd/test-runner/main.go",
            "package main\nfunc main() {}\n",
        )
        _write(app_repo, "main.go", "package main\nfunc main() {}\n")
        info = ScriptedOperations(repo).find_go_main_package(app_repo)
        # root main (score 3) > cmd/test-runner (score 1)
        assert info["main_path"] == "."


# ---------- quick_analysis integration ----------


class TestQuickAnalysis:
    """Tests for QuickAnalysis."""

    def test_runs_on_simple_cmake_repo(self, repo) -> None:
        """Test runs on simple cmake repo."""
        _write(repo, "CMakeLists.txt", "project(foo)\nfind_package(ZLIB)\n")
        _write(repo, "main.c", "int main(){return 0;}\n")
        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=_empty_execute,
        ):
            result = quick_analysis(repo)
        assert result["build_system"].type == "cmake"
        assert "ZLIB" in result["dependencies"].libraries
        assert isinstance(result["file_tree"], str)

    def test_gopath_style_go_exposes_go_main_info(self, repo) -> None:
        """Test quick analysis keeps Go context for repos without go.mod."""
        _write(repo, "main.go", "package main\nfunc main() {}\n")
        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=_empty_execute,
        ):
            result = quick_analysis(repo)
        assert result["build_system"].type == "go"
        assert "go_main_info" in result
        assert result["go_main_info"]["needs_go_init"] is True


# ---------- Path translation ----------


class TestPathTranslation:
    """Tests for PathTranslation."""

    def test_to_host_path_translates_workspace(self) -> None:
        """Translate a /workspace path to the host root outside Docker."""
        ops = ScriptedOperations()
        # /workspace/repos/foo -> {WORKSPACE_ROOT}/repos/foo on host
        out = ops._to_host_path("/workspace/repos/foo")
        # When not running inside Docker, /workspace becomes WORKSPACE_ROOT
        assert out.endswith("/repos/foo")

    def test_to_host_path_passes_through_other_paths(self) -> None:
        """Test to host path passes through other paths."""
        ops = ScriptedOperations()
        assert ops._to_host_path("/tmp/x") == "/tmp/x"


# ---------- Homepage → git URL resolution ----------


class TestResolveHomepageToGitUrls:
    """Tests for ScriptedOperations._resolve_homepage_to_git_urls."""

    def _ops(self) -> ScriptedOperations:
        """Helper: build a ScriptedOperations bound to a tmp workspace."""
        return ScriptedOperations()

    def test_gnu_homepage_maps_to_savannah(self) -> None:
        """`gnu.org/software/wget/` -> savannah `wget.git`."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "https://www.gnu.org/software/wget/", "wget"
        )
        assert got == ["https://git.savannah.gnu.org/git/wget.git"]

    def test_gnu_homepage_without_trailing_slash(self) -> None:
        """`gnu.org/software/grep` (no slash) also resolves."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "http://www.gnu.org/software/grep", "grep"
        )
        assert got == ["https://git.savannah.gnu.org/git/grep.git"]

    def test_savannah_cgit_url_maps_to_git_dir(self) -> None:
        """`git.savannah.gnu.org/cgit/gawk.git` -> git/gawk.git (no dupes)."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "https://git.savannah.gnu.org/cgit/gawk.git", "gawk"
        )
        assert got == ["https://git.savannah.gnu.org/git/gawk.git"]

    def test_yorhel_maps_to_blicky(self) -> None:
        """`dev.yorhel.nl` -> `code.blicky.net/yorhel/<name>.git`."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "https://dev.yorhel.nl", "ncdu"
        )
        assert got == ["https://code.blicky.net/yorhel/ncdu.git"]

    def test_generic_github_url_yields_no_rewrite(self) -> None:
        """Regular GitHub URLs have no homepage rewrite."""
        got = self._ops()._resolve_homepage_to_git_urls(
            "https://github.com/foo/bar", "bar"
        )
        assert got == []


# ---------- Container-health precheck ----------


class TestEnsureContainerHealthy:
    """Tests for ScriptedOperations._ensure_container_healthy."""

    def test_git_missing_triggers_install(self, stub_execute_command) -> None:
        """Missing git in container -> profile.install_cmd(['git']) runs."""
        from src.state import CommandResult

        calls: list = []

        def fake(cmd, **kwargs):
            """Fake."""
            calls.append(cmd)
            # First probe: git --version fails with 127
            if isinstance(cmd, list) and cmd == ["git", "--version"]:
                return CommandResult(
                    "git --version", 127, "", "git: not found", 0.0
                )
            # Install command succeeds
            return CommandResult(str(cmd), 0, "installed", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations()._ensure_container_healthy()

        # The install command should reference git and use the active
        # profile's package manager (alpine → apk).
        install_calls = [c for c in calls if isinstance(c, str) and "git" in c]
        assert any("apk" in c or "apt-get" in c for c in install_calls), calls

    def test_healthy_container_makes_only_probes(
        self, stub_execute_command
    ) -> None:
        """A working sandbox runs only the read-only probes, no install."""
        from src.state import CommandResult

        calls: list = []

        def fake(cmd, **kwargs):
            """Fake."""
            calls.append(cmd)
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations()._ensure_container_healthy()

        # Only the git --version probe should run (alpine skips the apt
        # probe entirely).
        assert any(
            isinstance(c, list) and c[:2] == ["git", "--version"]
            for c in calls
        )
        # No install command
        assert not any(isinstance(c, str) and "install" in c for c in calls)


# ---------- Submodule init after clone ----------


class TestInitSubmodulesIfPresent:
    """Tests for ScriptedOperations._init_submodules_if_present."""

    def test_no_gitmodules_is_no_op(self, stub_execute_command) -> None:
        """When ``.gitmodules`` is absent, no submodule call is made."""
        from src.state import CommandResult

        calls: list = []

        def fake(cmd, **kwargs):
            """Fake."""
            calls.append(cmd)
            # test -f returns non-zero when the file doesn't exist
            if isinstance(cmd, str) and cmd.startswith("test -f"):
                return CommandResult(cmd, 1, "", "", 0.0)
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations()._init_submodules_if_present(
            "/workspace/repos/foo"
        )

        # test -f check ran, but no submodule command should follow.
        assert any(isinstance(c, str) and "test -f" in c for c in calls)
        assert not any(
            isinstance(c, str) and "submodule update" in c for c in calls
        )

    def test_gitmodules_triggers_submodule_init(
        self, stub_execute_command
    ) -> None:
        """When ``.gitmodules`` exists, git submodule update runs."""
        from src.state import CommandResult

        calls: list = []

        def fake(cmd, **kwargs):
            """Fake."""
            calls.append(cmd)
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations()._init_submodules_if_present(
            "/workspace/repos/foo"
        )

        assert any(
            isinstance(c, str)
            and "git submodule update --init --recursive" in c
            for c in calls
        )


class TestCloneResetsExistingRepo:
    """Regression: reused clones must be reset to pristine upstream state.

    Rationale: a previous run's LLM may have authored broken files (e.g.
    a syntactically-invalid Makefile) or half-applied patches; ``git
    pull`` alone preserves them, causing every subsequent run to replay
    the same failure. See src/scripted_ops.py::clone_or_update_repository.
    """

    def test_existing_repo_runs_fetch_reset_and_clean(
        self, stub_execute_command, monkeypatch
    ) -> None:
        """When the clone directory exists, fetch+reset+clean must run."""
        from src.state import CommandResult

        # Pretend the .git directory exists so we hit the "exists" branch.
        monkeypatch.setattr("os.path.exists", lambda _p: True)
        # Suppress side effects from health check and submodule init.
        monkeypatch.setattr(
            "src.scripted_ops.ScriptedOperations." "_ensure_container_healthy",
            lambda _self: None,
        )
        monkeypatch.setattr(
            "src.scripted_ops.ScriptedOperations."
            "_init_submodules_if_present",
            lambda _self, _p: None,
        )

        commands: list = []

        def fake(cmd, **kwargs):
            """Fake execute_command that records every issued command."""
            commands.append(cmd)
            return CommandResult(str(cmd), 0, "", "", 0.0)

        stub_execute_command("src.scripted_ops", fake)
        ScriptedOperations().clone_or_update_repository(
            "https://github.com/foo/bar.git", "bar"
        )

        joined = " ".join(str(c) for c in commands)
        # Must reset to upstream HEAD and remove untracked files.
        assert "git fetch" in joined
        assert "git reset --hard" in joined
        assert "git clean -fdx" in joined
        # Must NOT do a bare `git pull` which would preserve the poison.
        assert not any(
            isinstance(c, str) and c.strip().endswith("git pull")
            for c in commands
        )


class _RepoBackedTestCase(unittest.TestCase):
    """Base TestCase with a repository-local scratch workspace."""

    def setUp(self) -> None:
        """Create an isolated workspace under the project directory."""
        self.scratch_root = os.path.abspath(
            os.path.join("workspace", "test_scripted_ops")
        )
        os.makedirs(self.scratch_root, exist_ok=True)
        self.tempdir = tempfile.TemporaryDirectory(
            prefix="case-",
            dir=self.scratch_root,
        )
        self.workspace = self.tempdir.name
        self.repo = os.path.join(self.workspace, "repo")
        os.makedirs(self.repo, exist_ok=True)

    def tearDown(self) -> None:
        """Remove the isolated workspace created for the test."""
        self.tempdir.cleanup()
        try:
            os.rmdir(self.scratch_root)
        except OSError:
            pass

    def _ops(self) -> ScriptedOperations:
        """Create ScriptedOperations bound to this test workspace."""
        return ScriptedOperations(self.workspace)


class TestScriptedOpsCloneHelpers(unittest.TestCase):
    """Tests for clone helper methods and auth environment handling."""

    def test_init_reraises_workspace_permission_errors(self) -> None:
        """Re-raise PermissionError when workspace dirs cannot be made."""
        with mock.patch(
            "src.scripted_ops.os.makedirs",
            side_effect=PermissionError("denied"),
        ):
            with self.assertRaises(PermissionError):
                ScriptedOperations("workspace/no-access")

    def test_github_auth_env_uses_token_only_for_github(self) -> None:
        """Inject bearer auth only for HTTPS GitHub clone URLs."""
        ops = ScriptedOperations()
        with mock.patch.dict(
            os.environ,
            {"GIT_TOKEN": "top-secret"},
            clear=False,
        ):
            env = ops._github_auth_env("https://github.com/acme/repo.git")
            other_env = ops._github_auth_env("https://example.com/repo.git")

        self.assertEqual(env["GIT_CONFIG_COUNT"], "1")
        self.assertEqual(
            env["GIT_CONFIG_KEY_0"],
            "http.https://github.com/.extraHeader",
        )
        self.assertIn("top-secret", env["GIT_CONFIG_VALUE_0"])
        self.assertEqual(other_env, {})

    def test_git_env_merges_thread_local_auth_overlay(self) -> None:
        """Merge fail-fast git settings with per-clone auth settings."""
        ops = ScriptedOperations()
        ops._auth_tls.env = {"GIT_CONFIG_COUNT": "1"}
        try:
            env = ops._git_env()
        finally:
            ops._auth_tls.env = {}

        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(env["GIT_CONFIG_COUNT"], "1")

    def test_build_clone_cmd_rejects_non_http_and_quotes_path(self) -> None:
        """Reject unsafe schemes and quote the destination path."""
        ops = ScriptedOperations()
        with self.assertRaises(ValueError):
            ops._build_clone_cmd("git@github.com:foo/bar.git", "/workspace/r")

        cmd = ops._build_clone_cmd(
            "https://github.com/foo/bar.git",
            "/workspace/repos/space name",
        )
        self.assertIn("git clone --depth 1", cmd)
        self.assertIn("https://github.com/foo/bar.git", cmd)
        self.assertIn("'/workspace/repos/space name'", cmd)

    def test_is_auth_error_matches_known_messages(self) -> None:
        """Classify common credential failures as auth errors."""
        ops = ScriptedOperations()
        auth_result = _command_result(
            "git clone",
            exit_code=128,
            stderr="fatal: could not read Username for host",
        )
        network_result = _command_result(
            "git clone",
            exit_code=128,
            stderr="fatal: repository not found",
        )

        self.assertTrue(ops._is_auth_error(auth_result))
        self.assertFalse(ops._is_auth_error(network_result))


class TestScriptedOpsUrlVariants(_RepoBackedTestCase):
    """Tests for URL variant clone fallback generation."""

    def test_try_url_variants_rejects_unsafe_inputs(self) -> None:
        """Reject unsafe repository names and non-HTTP URLs."""
        ops = self._ops()
        with self.assertRaises(ValueError):
            ops._try_url_variants("https://example.com/repo", "../repo")
        with self.assertRaises(ValueError):
            ops._try_url_variants("ssh://example.com/repo", "repo")

    def test_try_url_variants_appends_git_and_returns_success(self) -> None:
        """Clone from an appended .git URL when the variant works."""
        calls = []

        def fake_execute(cmd, **kwargs):
            """Return success for the appended .git clone variant."""
            calls.append((cmd, kwargs))
            return _command_result(cmd)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = self._ops()._try_url_variants(
                "https://example.com/project",
                "project",
            )

        clone_calls = [
            call for call, _kwargs in calls if isinstance(call, list)
        ]
        self.assertTrue(result.success)
        self.assertIn("https://example.com/project.git", clone_calls[1])

    def test_try_url_variants_tries_cgit_and_homepage_until_success(
        self,
    ) -> None:
        """Try cgit, appended .git, then homepage-derived git URLs."""
        clone_urls = []

        def fake_execute(cmd, **kwargs):
            """Fail early variants and succeed on the savannah git URL."""
            if isinstance(cmd, list) and cmd[:2] == ["git", "clone"]:
                clone_urls.append(cmd[4])
                if cmd[4] == "https://git.savannah.gnu.org/git/gawk.git":
                    return _command_result(cmd)
                return _command_result(cmd, exit_code=128, stderr="missing")
            return _command_result(cmd)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = self._ops()._try_url_variants(
                "https://git.savannah.gnu.org/cgit/gawk",
                "gawk",
            )

        self.assertTrue(result.success)
        self.assertEqual(
            clone_urls,
            [
                "https://git.savannah.gnu.org/git/gawk",
                "https://git.savannah.gnu.org/cgit/gawk.git",
                "https://git.savannah.gnu.org/git/gawk.git",
            ],
        )

    def test_try_url_variants_returns_none_when_all_variants_fail(
        self,
    ) -> None:
        """Return None after every generated clone variant fails."""

        def fake_execute(cmd, **kwargs):
            """Fail git clone calls and allow cleanup calls."""
            if isinstance(cmd, list) and cmd[:2] == ["git", "clone"]:
                return _command_result(cmd, exit_code=128, stderr="missing")
            return _command_result(cmd)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = self._ops()._try_url_variants(
                "https://example.com/missing",
                "missing",
            )

        self.assertIsNone(result)


class TestScriptedOpsContainerHealth(unittest.TestCase):
    """Tests for container health checks that are fully mocked."""

    def test_debian_precheck_cleans_broken_apt_sources(self) -> None:
        """Remove recent apt sources and retry when apt update is broken."""
        profile = mock.Mock()
        profile.name = "debian"
        calls = []

        def fake_execute(cmd, **kwargs):
            """Return a broken apt update followed by a successful retry."""
            calls.append(cmd)
            if cmd == ["git", "--version"]:
                return _command_result(cmd)
            if cmd == "apt-get update" and calls.count(cmd) == 1:
                return _command_result(
                    cmd,
                    exit_code=100,
                    stderr="Release file expired",
                )
            return _command_result(cmd)

        with mock.patch(
            "src.platforms.get_active_profile",
            return_value=profile,
        ), mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            ScriptedOperations()._ensure_container_healthy()

        self.assertIn("apt-get update", calls)
        self.assertTrue(
            any(
                isinstance(cmd, str)
                and "sources.list.d" in cmd
                and "-delete" in cmd
                for cmd in calls
            )
        )
        self.assertEqual(calls.count("apt-get update"), 2)

    def test_debian_precheck_logs_failed_apt_repair(self) -> None:
        """Keep going when apt source cleanup does not repair updates."""
        profile = mock.Mock()
        profile.name = "debian"

        def fake_execute(cmd, **kwargs):
            """Fail both apt update attempts."""
            if cmd == ["git", "--version"]:
                return _command_result(cmd)
            if cmd == "apt-get update":
                return _command_result(
                    cmd,
                    exit_code=100,
                    stderr="does not have a Release file",
                )
            return _command_result(cmd)

        with mock.patch(
            "src.platforms.get_active_profile",
            return_value=profile,
        ), mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            ScriptedOperations()._ensure_container_healthy()

    def test_submodule_init_warning_is_non_fatal(self) -> None:
        """Continue when submodule initialization fails."""
        calls = []

        def fake_execute(cmd, **kwargs):
            """Succeed at .gitmodules check and fail submodule update."""
            calls.append(cmd)
            if "submodule update" in str(cmd):
                return _command_result(cmd, exit_code=1, stderr="boom")
            return _command_result(cmd)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            ScriptedOperations()._init_submodules_if_present(
                "/workspace/repos/repo"
            )

        self.assertTrue(any("submodule update" in str(cmd) for cmd in calls))


class TestScriptedOpsCloneFlows(_RepoBackedTestCase):
    """Tests for clone, reset, auth retry, and variant fallback flows."""

    def test_clone_inner_rejects_unsafe_name_and_scheme(self) -> None:
        """Validate clone inputs before any shell command can run."""
        ops = self._ops()
        with self.assertRaises(ValueError):
            ops._clone_or_update_inner("https://example.com/repo", ".repo")
        with self.assertRaises(ValueError):
            ops._clone_or_update_inner("file:///repo", "repo")

    def test_existing_repo_reclones_when_reset_fails(self) -> None:
        """Delete and re-clone an existing repo when fetch/reset fails."""
        ops = self._ops()
        git_dir = os.path.join(ops.repos_dir, "bar", ".git")
        os.makedirs(git_dir)
        commands = []
        init_submodules = mock.Mock()

        def fake_execute(cmd, **kwargs):
            """Fail reset, succeed clone, and fail safe.directory."""
            commands.append(cmd)
            if isinstance(cmd, str) and "git fetch --depth 1" in cmd:
                return _command_result(cmd, exit_code=1, stderr="diverged")
            if isinstance(cmd, str) and cmd.startswith("git clone"):
                return _command_result(cmd)
            if isinstance(cmd, str) and cmd.startswith("git config"):
                return _command_result(cmd, exit_code=1, stderr="safe fail")
            return _command_result(cmd)

        with mock.patch.object(
            ops,
            "_ensure_container_healthy",
            return_value=None,
        ), mock.patch.object(
            ops,
            "_init_submodules_if_present",
            init_submodules,
        ), mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = ops._clone_or_update_inner(
                "https://github.com/foo/bar.git",
                "bar",
            )

        self.assertTrue(result.success)
        joined = " ".join(str(cmd) for cmd in commands)
        self.assertIn("git fetch --depth 1", joined)
        self.assertIn("rm', '-rf", joined)
        self.assertIn("git clone --depth 1", joined)
        init_submodules.assert_called_once_with("/workspace/repos/bar")

    def test_new_clone_retries_without_token_after_auth_error(self) -> None:
        """Retry a public clone without bearer auth after auth failure."""
        ops = self._ops()
        clone_envs = []

        def fake_execute(cmd, **kwargs):
            """Fail the first clone with auth error, then succeed."""
            if isinstance(cmd, str) and cmd.startswith("git clone"):
                clone_envs.append(kwargs.get("extra_env", {}))
                if len(clone_envs) == 1:
                    return _command_result(
                        cmd,
                        exit_code=128,
                        stderr="could not read Username",
                    )
                return _command_result(cmd)
            return _command_result(cmd)

        with mock.patch.dict(
            os.environ,
            {"GIT_TOKEN": "secret"},
            clear=False,
        ), mock.patch.object(
            ops,
            "_ensure_container_healthy",
            return_value=None,
        ), mock.patch.object(
            ops,
            "_init_submodules_if_present",
            return_value=None,
        ), mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = ops.clone_or_update_repository(
                "https://github.com/foo/bar.git",
                "bar",
            )

        self.assertTrue(result.success)
        self.assertIn("GIT_CONFIG_COUNT", clone_envs[0])
        self.assertNotIn("GIT_CONFIG_COUNT", clone_envs[1])
        self.assertEqual(getattr(ops._auth_tls, "env", {}), {})

    def test_new_clone_uses_successful_url_variant(self) -> None:
        """Use a clone result from URL variants after the raw URL fails."""
        ops = self._ops()
        variant = _command_result("variant clone")

        def fake_execute(cmd, **kwargs):
            """Fail the direct clone but allow cleanup and safe.directory."""
            if isinstance(cmd, str) and cmd.startswith("git clone"):
                return _command_result(cmd, exit_code=128, stderr="missing")
            return _command_result(cmd)

        with mock.patch.object(
            ops,
            "_ensure_container_healthy",
            return_value=None,
        ), mock.patch.object(
            ops,
            "_init_submodules_if_present",
            return_value=None,
        ), mock.patch.object(
            ops,
            "_try_url_variants",
            return_value=variant,
        ) as try_variants, mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = ops._clone_or_update_inner(
                "https://example.com/project",
                "project",
            )

        self.assertIs(result, variant)
        try_variants.assert_called_once_with(
            "https://example.com/project",
            "project",
        )

    def test_new_clone_returns_original_failure_when_variants_fail(
        self,
    ) -> None:
        """Return the direct clone failure if no URL variant succeeds."""
        ops = self._ops()

        def fake_execute(cmd, **kwargs):
            """Fail direct clone and allow cleanup calls."""
            if isinstance(cmd, str) and cmd.startswith("git clone"):
                return _command_result(cmd, exit_code=128, stderr="missing")
            return _command_result(cmd)

        with mock.patch.object(
            ops,
            "_ensure_container_healthy",
            return_value=None,
        ), mock.patch.object(
            ops,
            "_try_url_variants",
            return_value=None,
        ), mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = ops._clone_or_update_inner(
                "https://example.com/project",
                "project",
            )

        self.assertFalse(result.success)


class TestScriptedOpsDetectionBranches(_RepoBackedTestCase):
    """Tests for build-system detection branches not covered above."""

    def test_to_container_path_translates_workspace_paths(self) -> None:
        """Translate this workspace root to the in-container prefix."""
        ops = self._ops()
        host_path = os.path.join(self.workspace, "repos", "pkg")

        self.assertEqual(
            ops._to_container_path(host_path),
            "/workspace/repos/pkg",
        )
        self.assertEqual(
            ops._to_container_path("/outside/repo"),
            "/outside/repo",
        )

    def test_score_go_subdir_counts_cmd_dir_and_ignores_read_errors(
        self,
    ) -> None:
        """Score cmd markers and tolerate unreadable Go files."""
        _write(self.repo, "source.go", "package main\n")
        os.makedirs(os.path.join(self.repo, "cmd"))
        ops = self._ops()

        with mock.patch("builtins.open", side_effect=OSError("boom")):
            score = ops._score_go_subdir(self.repo, self.repo)

        self.assertEqual(score, 8)

    def test_repository_info_collects_successful_git_values(self) -> None:
        """Collect commit, branch, and file count from mocked commands."""

        def fake_execute(cmd, **kwargs):
            """Return command-specific repository metadata."""
            if "rev-parse HEAD" in cmd:
                return _command_result(cmd, stdout="abc123\n")
            if "branch --show-current" in cmd:
                return _command_result(cmd, stdout="dev\n")
            if "wc -l" in cmd:
                return _command_result(cmd, stdout="42\n")
            return _command_result(cmd, exit_code=1)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            info = self._ops().get_repository_info(self.repo)

        self.assertEqual(
            info,
            {"commit": "abc123", "branch": "dev", "file_count": "42"},
        )

    def test_has_npm_build_script_handles_valid_and_invalid_json(self) -> None:
        """Detect npm build scripts and tolerate malformed package JSON."""
        ops = self._ops()
        missing = ops._has_npm_build_script(self.repo)
        _write(
            self.repo,
            "package.json",
            json.dumps({"scripts": {"build": "npm run compile"}}),
        )
        with_build = ops._has_npm_build_script(self.repo)
        _write(self.repo, "package.json", "{not-json")
        malformed = ops._has_npm_build_script(self.repo)
        _write(self.repo, "package.json", json.dumps({"scripts": []}))
        wrong_shape = ops._has_npm_build_script(self.repo)

        self.assertFalse(missing)
        self.assertTrue(with_build)
        self.assertFalse(malformed)
        self.assertFalse(wrong_shape)

    def test_npm_detection_requires_lockfile_or_build_script(self) -> None:
        """Ignore package.json without npm build evidence."""
        plain_repo = os.path.join(self.workspace, "plain")
        locked_repo = os.path.join(self.workspace, "locked")
        scripted_repo = os.path.join(self.workspace, "scripted")
        os.makedirs(plain_repo)
        os.makedirs(locked_repo)
        os.makedirs(scripted_repo)
        _write(plain_repo, "package.json", json.dumps({"name": "plain"}))
        _write(locked_repo, "package.json", json.dumps({"name": "locked"}))
        _write(locked_repo, "package-lock.json", "{}")
        _write(
            scripted_repo,
            "package.json",
            json.dumps({"scripts": {"build": "vite build"}}),
        )
        ops = self._ops()

        self.assertEqual(ops.detect_build_system(plain_repo).type, "unknown")
        self.assertEqual(ops.detect_build_system(locked_repo).type, "npm")
        self.assertEqual(ops.detect_build_system(scripted_repo).type, "npm")

    def test_detects_subdirectory_cargo_module(self) -> None:
        """Detect Cargo.toml below the repository root."""
        _write(self.repo, "crates/tool/Cargo.toml", "[package]\nname='tool'\n")

        info = self._ops().detect_build_system(self.repo)

        self.assertEqual(info.type, "cargo")
        self.assertEqual(info.primary_file, "crates/tool/Cargo.toml")
        self.assertEqual(info.module_dir, "crates/tool")

    def test_detects_nested_build_root_and_sets_module_dir(self) -> None:
        """Use a nested CMakeLists.txt as the primary build root."""
        _write(self.repo, "tests/Makefile", "all:\n\ttrue\n")
        _write(self.repo, "src/CMakeLists.txt", "project(nested)\n")

        info = self._ops().detect_build_system(self.repo)

        self.assertEqual(info.type, "cmake")
        self.assertEqual(info.confidence, 0.75)
        self.assertEqual(info.module_dir, "src")

    def test_detects_configure_with_lower_autotools_confidence(self) -> None:
        """Treat an executable configure script as autotools evidence."""
        _write(self.repo, "configure", "#!/bin/sh\n")

        info = self._ops().detect_build_system(self.repo)

        self.assertEqual(info.type, "autotools")
        self.assertEqual(info.confidence, 0.85)

    def test_detect_build_system_skips_git_directory_contents(self) -> None:
        """Ignore build markers that live inside .git directories."""
        _write(self.repo, ".git/go.mod", "module hidden\n")

        info = self._ops().detect_build_system(self.repo)

        self.assertEqual(info.type, "unknown")

    def test_make_fallback_handles_late_makefile_visibility(self) -> None:
        """Cover the defensive Makefile fallback branch."""
        makefile = _write(self.repo, "Makefile", "all:\n\ttrue\n")
        real_exists = os.path.exists
        makefile_checks = {"count": 0}

        def fake_exists(path):
            """Hide Makefile during the main scan, then reveal it."""
            if path == makefile:
                makefile_checks["count"] += 1
                return makefile_checks["count"] > 1
            return real_exists(path)

        with mock.patch("src.scripted_ops.os.path.exists", fake_exists):
            info = self._ops().detect_build_system(self.repo)

        self.assertEqual(info.type, "make")
        self.assertEqual(info.confidence, 0.5)

    def test_gopath_library_repo_uses_go_wildcard_primary_file(self) -> None:
        """Detect Go files without package main as GOPATH-style Go."""
        _write(self.repo, "lib.go", "package lib\n")

        info = self._ops().detect_build_system(self.repo)

        self.assertEqual(info.type, "go")
        self.assertEqual(info.primary_file, "*.go")
        self.assertEqual(info.confidence, 0.65)

    def test_find_nested_build_root_handles_non_dirs_and_depth(self) -> None:
        """Return None for non-directories and respect max_depth."""
        ops = self._ops()
        _write(self.repo, "too/deep/CMakeLists.txt", "project(deep)\n")

        self.assertIsNone(ops._find_nested_build_root("workspace/missing"))
        self.assertIsNone(ops._find_nested_build_root(self.repo, max_depth=1))

    def test_find_nested_build_root_prefers_shallow_authoritative_file(
        self,
    ) -> None:
        """Prefer the shallowest and strongest nested build marker."""
        _write(self.repo, "b/Makefile", "all:\n\ttrue\n")
        _write(self.repo, "a/CMakeLists.txt", "project(a)\n")
        _write(self.repo, "a/plugin/meson.build", "project('plugin')\n")

        nested = self._ops()._find_nested_build_root(self.repo)

        self.assertEqual(nested, ("cmake", "a/CMakeLists.txt"))


class TestScriptedOpsDependencyBranches(_RepoBackedTestCase):
    """Tests for dependency extraction dispatch and parser edge cases."""

    def test_extract_dependencies_dispatches_supported_build_systems(
        self,
    ) -> None:
        """Dispatch public extraction to every supported private parser."""
        _write(self.repo, "CMakeLists.txt", "find_package(PNG)\n")
        _write(self.repo, "Cargo.toml", "[dependencies]\nserde = '1'\n")
        _write(self.repo, "requirements.txt", "requests==2\n")
        _write(
            self.repo,
            "package.json",
            json.dumps({"dependencies": {"express": "4"}}),
        )
        _write(self.repo, "go.mod", "module m\nrequire example.com/a v1\n")
        _write(self.repo, "Makefile", "LDLIBS=-lssl -lcustom\n")
        ops = self._ops()

        expected = {
            "cmake": "PNG",
            "cargo": "serde",
            "pip": "requests",
            "npm": "express",
            "go": "example.com/a",
            "make": "ssl",
        }
        for build_system, library in expected.items():
            with self.subTest(build_system=build_system):
                deps = ops.extract_dependencies(self.repo, build_system)
                self.assertIn(library, deps.libraries)

        unknown_deps = ops.extract_dependencies(self.repo, "unknown")
        self.assertEqual(unknown_deps.libraries, [])

    def test_extractors_handle_missing_and_malformed_files(self) -> None:
        """Return defaults or warnings for missing and malformed manifests."""
        ops = self._ops()
        cargo_deps = ops._extract_cargo_dependencies(self.repo)
        npm_deps = ops._extract_npm_dependencies(self.repo)
        self.assertEqual(cargo_deps.libraries, [])
        self.assertEqual(npm_deps.libraries, [])

        _write(self.repo, "Cargo.toml", "[dependencies\n")
        _write(self.repo, "package.json", "{not-json")

        cargo_deps = ops._extract_cargo_dependencies(self.repo)
        npm_deps = ops._extract_npm_dependencies(self.repo)
        self.assertEqual(cargo_deps.libraries, [])
        self.assertEqual(npm_deps.libraries, [])

    def test_go_dependencies_parse_block_requirements_and_deduplicate(
        self,
    ) -> None:
        """Parse go.mod block and single-line require forms."""
        _write(
            self.repo,
            "go.mod",
            textwrap.dedent("""
            module example.com/app

            require (
                example.com/a v1.0.0
                // example.com/commented v1.0.0

                example.com/b v1.0.0
            )
            require example.com/a v1.0.0
            """),
        )

        deps = self._ops()._extract_go_dependencies(self.repo)

        self.assertEqual(deps.libraries, ["example.com/a", "example.com/b"])

    def test_make_dependencies_map_common_library_flags(self) -> None:
        """Extract -l flags and resolve common system packages."""
        _write(self.repo, "Makefile", "LDLIBS=-lssl -lcrypto -lz -lm -lfoo\n")

        deps = self._ops()._extract_make_dependencies(self.repo)

        self.assertEqual(deps.install_method, "apk")
        self.assertEqual(
            deps.libraries,
            ["ssl", "crypto", "z", "m", "foo"],
        )
        self.assertIn("openssl-dev", deps.system_packages)
        self.assertIn("zlib-dev", deps.system_packages)
        self.assertIn("musl-dev", deps.system_packages)
        self.assertEqual(deps.build_tools, ["make", "gcc", "g++"])

    def test_make_dependencies_missing_makefile_returns_defaults(self) -> None:
        """Return default make dependency info when Makefile is absent."""
        deps = self._ops()._extract_make_dependencies(self.repo)

        self.assertEqual(deps.libraries, [])
        self.assertEqual(deps.install_method, "apk")


class TestScriptedOpsGoMainBranches(_RepoBackedTestCase):
    """Tests for Go main-package scoring and exclusion branches."""

    def test_find_go_main_ignores_git_and_vendor_without_mod(self) -> None:
        """Skip .git and vendor while searching GOPATH-style main files."""
        _write(self.repo, ".git/hidden/main.go", "package main\n")
        _write(self.repo, "vendor/main.go", "package main\n")
        _write(self.repo, "lib.go", "package lib\n")

        info = self._ops().find_go_main_package(self.repo)

        self.assertTrue(info["needs_go_init"])
        self.assertTrue(info["has_go_files"])
        self.assertFalse(info["has_main"])

    def test_find_go_main_ignores_unreadable_gopath_files(self) -> None:
        """Tolerate read errors while checking GOPATH-style Go files."""
        _write(self.repo, "bad.go", "package main\n")

        with mock.patch("builtins.open", side_effect=OSError("boom")):
            info = self._ops().find_go_main_package(self.repo)

        self.assertTrue(info["needs_go_init"])
        self.assertFalse(info["has_main"])

    def test_find_go_main_selects_best_of_multiple_modules(self) -> None:
        """Choose the Go module whose root has main-package markers."""
        _write(self.repo, "library/go.mod", "module library\n")
        _write(self.repo, "cmd/app/go.mod", "module app\n")
        _write(self.repo, "cmd/app/main.go", "package main\nfunc main() {}\n")

        info = self._ops().find_go_main_package(self.repo)

        self.assertTrue(info["has_go_mod"])
        self.assertEqual(info["module_dir"], "cmd/app")
        self.assertEqual(info["main_path"], ".")
        self.assertEqual(info["build_command"], "go build .")

    def test_find_go_main_scores_generic_cmd_above_other_dirs(self) -> None:
        """Prefer a non-test cmd directory over arbitrary subdirs."""
        _write(self.repo, "go.mod", "module app\n")
        _write(self.repo, "pkg/tool/main.go", "package main\n")
        _write(self.repo, "cmd/worker/main.go", "package main\n")

        info = self._ops().find_go_main_package(self.repo)

        self.assertEqual(info["main_path"], "cmd/worker")
        self.assertEqual(info["build_command"], "go build ./cmd/worker")

    def test_find_go_main_skips_vendor_with_go_mod(self) -> None:
        """Skip vendored Go files while scanning a module for main files."""
        _write(self.repo, "go.mod", "module app\n")
        _write(self.repo, "vendor/tool/main.go", "package main\n")
        _write(self.repo, "lib.go", "package lib\n")

        info = self._ops().find_go_main_package(self.repo)

        self.assertTrue(info["has_go_mod"])
        self.assertFalse(info["has_main"])

    def test_find_go_main_ignores_unreadable_module_files(self) -> None:
        """Tolerate read errors while checking module Go files."""
        _write(self.repo, "go.mod", "module app\n")
        _write(self.repo, "bad.go", "package main\n")

        with mock.patch("builtins.open", side_effect=OSError("boom")):
            info = self._ops().find_go_main_package(self.repo)

        self.assertTrue(info["has_go_mod"])
        self.assertFalse(info["has_main"])

    def test_find_go_main_penalizes_tooling_directories(self) -> None:
        """Still report a main in tooling dirs, but with low priority."""
        _write(self.repo, "go.mod", "module app\n")
        _write(self.repo, "tools/generator/main.go", "package main\n")

        info = self._ops().find_go_main_package(self.repo)

        self.assertEqual(info["main_path"], "tools/generator")
        self.assertEqual(info["all_main_paths"], ["tools/generator"])

    def test_find_go_main_scores_integration_subdir_low(self) -> None:
        """Cover low-priority non-cmd integration directory scoring."""
        _write(self.repo, "go.mod", "module app\n")
        _write(self.repo, "foo-integration/main.go", "package main\n")

        info = self._ops().find_go_main_package(self.repo)

        self.assertEqual(info["main_path"], "foo-integration")
        self.assertEqual(info["build_command"], "go build ./foo-integration")


class TestScriptedOpsArchitectureBranches(_RepoBackedTestCase):
    """Tests for architecture-specific source and build-file detection."""

    def test_find_architecture_specific_code_parses_grep_hits(self) -> None:
        """Parse grep output into ArchSpecificCode findings."""

        def fake_execute(cmd, **kwargs):
            """Return x86 and SIMD grep hits for selected patterns."""
            if "'__x86_64__'" in cmd:
                return _command_result(
                    cmd,
                    stdout=(
                        "/workspace/repos/repo/a.c:12:#ifdef __x86_64__\n"
                        "/workspace/repos/repo/b.c:notnum:#ifdef __x86_64__"
                    ),
                )
            if "'_mm\\w+'" in cmd:
                return _command_result(
                    cmd,
                    stdout="/workspace/repos/repo/simd.c:8:_mm_add_epi32(x)",
                )
            return _command_result(cmd, exit_code=1)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            findings = self._ops().find_architecture_specific_code(self.repo)

        self.assertEqual(len(findings), 3)
        self.assertEqual(findings[0].arch_type, "x86")
        self.assertEqual(findings[0].severity, "medium")
        self.assertEqual(findings[0].line, 12)
        self.assertEqual(findings[1].line, 0)
        self.assertEqual(findings[2].arch_type, "x86_simd")
        self.assertEqual(findings[2].severity, "high")

    def test_suggest_fix_for_each_arch_type_and_default(self) -> None:
        """Return tailored suggestions for every known arch category."""
        ops = self._ops()

        self.assertIn("__riscv", ops._suggest_fix_for_arch_code("x86"))
        self.assertIn("RVV", ops._suggest_fix_for_arch_code("x86_simd"))
        self.assertIn("__riscv", ops._suggest_fix_for_arch_code("arm"))
        self.assertIn("RVV", ops._suggest_fix_for_arch_code("arm_simd"))
        self.assertIn("assembly", ops._suggest_fix_for_arch_code("inline_asm"))
        self.assertEqual(
            ops._suggest_fix_for_arch_code("mips"),
            "Review and port to RISC-V",
        )

    def test_detect_arch_build_files_suggests_riscv_counterparts(
        self,
    ) -> None:
        """Suggest RISC-V makefiles from x64 and arm64 build files."""

        def fake_execute(cmd, **kwargs):
            """Return mocked grep hits by architecture pattern."""
            if "x64|x86_64|amd64" in cmd:
                return _command_result(
                    cmd,
                    stdout=(
                        "/workspace/repo/makefile.x64.mk\n"
                        "/workspace/repo/build_amd64.mk"
                    ),
                )
            if "arm64|aarch64" in cmd:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo/build_arm64.mk",
                )
            return _command_result(cmd, exit_code=1)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = self._ops().detect_arch_specific_build_files(self.repo)

        self.assertTrue(result["has_arch_specific"])
        self.assertEqual(result["archs_found"], ["x64", "arm64"])
        targets = [item["target"] for item in result["suggested_riscv_files"]]
        self.assertIn("/workspace/repo/makefile.riscv64.mk", targets)
        self.assertIn("/workspace/repo/build_riscv64.mk", targets)

    def test_detect_arch_build_files_marks_existing_riscv_support(
        self,
    ) -> None:
        """Set riscv_exists when a RISC-V-specific makefile is present."""

        def fake_execute(cmd, **kwargs):
            """Return a result only for the RISC-V pattern."""
            if "riscv|riscv64|rv64" in cmd:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo/build_riscv64.mk",
                )
            return _command_result(cmd, exit_code=1)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            result = self._ops().detect_arch_specific_build_files(self.repo)

        self.assertTrue(result["riscv_exists"])
        self.assertEqual(result["arch_files"]["riscv"], [
            "/workspace/repo/build_riscv64.mk",
        ])
        self.assertEqual(result["suggested_riscv_files"], [])


class TestScriptedOpsFileAndSystemBranches(_RepoBackedTestCase):
    """Tests for file tree, documentation, and system-info helpers."""

    def test_get_file_tree_uses_tree_output_when_available(self) -> None:
        """Return tree output directly when the tree command succeeds."""

        def fake_execute(cmd, **kwargs):
            """Return a successful tree response."""
            if str(cmd).startswith("tree -L"):
                return _command_result(cmd, stdout="repo\n└── main.c\n")
            return _command_result(cmd, exit_code=1)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            output = self._ops().get_file_tree(self.repo, max_depth=1)

        self.assertIn("main.c", output)

    def test_get_file_tree_falls_back_to_find(self) -> None:
        """Use find output when tree is unavailable."""

        def fake_execute(cmd, **kwargs):
            """Fail tree and return a fallback find listing."""
            if str(cmd).startswith("tree -L"):
                return _command_result(cmd, exit_code=127, stderr="missing")
            return _command_result(cmd, stdout="/workspace/repo/main.c\n")

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            output = self._ops().get_file_tree(self.repo, max_depth=2)

        self.assertIn("/workspace/repo/main.c", output)

    def test_get_optimized_tree_includes_sections_and_counts(self) -> None:
        """Build the compact tree from mocked command output."""

        def fake_execute(cmd, **kwargs):
            """Return targeted output for each optimized-tree command."""
            cmd_text = str(cmd)
            if cmd_text.startswith("ls -la"):
                return _command_result(cmd, stdout="total 8\n-rw README.md")
            if "maxdepth 2 -type d" in cmd_text:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo\n/workspace/repo/src\n",
                )
            if "-name 'CMakeLists.txt'" in cmd_text:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo/CMakeLists.txt\n",
                )
            if "-name 'README*'" in cmd_text:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo/README.md\n",
                )
            if "-name '*.json'" in cmd_text:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo/package.json\n",
                )
            if "maxdepth 1 -type d" in cmd_text:
                return _command_result(cmd, stdout="2\n")
            if "-name '*.c'" in cmd_text:
                return _command_result(cmd, stdout="3\n")
            if "-name '*.go'" in cmd_text:
                return _command_result(cmd, stdout="1\n")
            if "-name '*.rs'" in cmd_text:
                return _command_result(cmd, stdout="2\n")
            if "-name '*.py'" in cmd_text:
                return _command_result(cmd, stdout="4\n")
            return _command_result(cmd)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            output = self._ops().get_optimized_tree(self.repo)

        self.assertIn("Repository Structure Overview", output)
        self.assertIn("Subdirectories", output)
        self.assertIn("Build Config", output)
        self.assertIn("Documentation", output)
        self.assertIn("Config", output)
        self.assertIn("Root subdirectories: 2", output)
        self.assertIn("C/C++ source files: 3", output)
        self.assertIn("Go source files: 1", output)
        self.assertIn("Rust source files: 2", output)
        self.assertIn("Python files: 4", output)

    def test_read_file_handles_missing_truncated_and_read_errors(self) -> None:
        """Read existing files, truncate long files, and report errors."""
        missing = self._ops().read_file(os.path.join(self.repo, "missing.c"))
        _write(self.repo, "long.txt", "one\ntwo\nthree\n")
        truncated = self._ops().read_file(
            os.path.join(self.repo, "long.txt"),
            max_lines=2,
        )

        with mock.patch(
            "src.scripted_ops.os.path.exists",
            return_value=True,
        ), mock.patch(
            "builtins.open",
            side_effect=OSError("boom"),
        ):
            errored = self._ops().read_file(os.path.join(self.repo, "bad.txt"))

        self.assertIn("File not found", missing)
        self.assertIn("one\ntwo\n", truncated)
        self.assertIn("truncated after 2 lines", truncated)
        self.assertIn("Error reading file: boom", errored)

    def test_find_documentation_prioritizes_common_files(self) -> None:
        """Return prioritized documentation files before other docs."""

        def fake_execute(cmd, **kwargs):
            """Return docs for selected find patterns."""
            cmd_text = str(cmd)
            if "README*" in cmd_text:
                return _command_result(cmd, stdout="/workspace/repo/README.md")
            if "INSTALL*" in cmd_text:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo/INSTALL.md",
                )
            if "BUILDING*" in cmd_text:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo/BUILDING.md",
                )
            if "BUILD*" in cmd_text:
                return _command_result(
                    cmd,
                    stdout="/workspace/repo/BUILD.md",
                )
            if "*.md" in cmd_text:
                return _command_result(
                    cmd,
                    stdout=(
                        "/workspace/repo/README.md\n"
                        "/workspace/repo/notes.md"
                    ),
                )
            return _command_result(cmd)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            docs = self._ops().find_documentation(self.repo)

        self.assertLessEqual(len(docs), 10)
        self.assertEqual(docs[0], "/workspace/repo/README.md")
        self.assertIn("/workspace/repo/notes.md", docs)

    def test_get_system_info_reports_available_missing_and_arch(self) -> None:
        """Report tool availability and detected container architecture."""

        def fake_execute(cmd, **kwargs):
            """Return availability for gcc and absence for make."""
            if cmd == "which gcc":
                return _command_result(cmd, stdout="/usr/bin/gcc\n")
            if cmd == "uname -m":
                return _command_result(cmd, stdout="riscv64\n")
            return _command_result(cmd, exit_code=1)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            info = self._ops().get_system_info(["gcc", "make"])

        self.assertEqual(info["gcc"], "Available in PATH")
        self.assertEqual(info["make"], "Not installed")
        self.assertEqual(info["architecture"], "riscv64")

    def test_get_system_info_uses_default_tool_list_and_unknown_arch(
        self,
    ) -> None:
        """Probe the default tool list and report unknown architecture."""
        calls = []

        def fake_execute(cmd, **kwargs):
            """Fail every mocked system-info command."""
            calls.append(cmd)
            return _command_result(cmd, exit_code=1)

        with mock.patch(
            "src.scripted_ops.execute_command",
            side_effect=fake_execute,
        ):
            info = self._ops().get_system_info()

        self.assertIn("which gcc", calls)
        self.assertIn("which cargo", calls)
        self.assertEqual(info["gcc"], "Not installed")
        self.assertEqual(info["architecture"], "unknown")


if __name__ == "__main__":
    unittest.main()
