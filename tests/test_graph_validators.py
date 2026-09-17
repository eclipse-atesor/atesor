"""Tests for graph.py validators.

Covers build plan, fix command, fixer response, and predictions.
"""

import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

from src.graph import (
    _autotools_bootstrap_prefix,
    _build_command_error_message,
    _builder_retry_allowed,
    _classify_clone_failure,
    _download_go_toolchain_cmd,
    _extract_cd_prefix,
    _fallback_analysis,
    _find_makefile_dir,
    _fixup_top_builddir_in_submakefiles,
    _format_analysis_for_prompt,
    _inject_go_flag,
    _inject_go_output,
    _is_go_build_command,
    _is_suspected_oom,
    _is_within_repo,
    _npm_scripts,
    _probe_config_files,
    _replan_signature,
    _repo_has_gitmodules,
    _resolve_header_to_packages,
    _resolve_missing_python_modules,
    _serialize_build_command,
    _setup_packages_for_plan,
    _should_force_replan,
    _try_clone_recovery,
    create_default_plan,
    create_fallback_build_plan,
    extract_content,
    extract_json_block,
    get_model_for_role,
    get_model_pool_for_role,
    is_toolchain_version_mismatch,
    predict_build_issues,
    validate_build_plan,
    validate_fix_command,
    validate_fixer_response,
)
from src.state import (
    AgentRole,
    ArchSpecificCode,
    BuildPhase,
    BuildPlan,
    BuildSystemInfo,
    CommandResult,
    DependencyInfo,
    ErrorCategory,
    PackageAnalysis,
    create_initial_state,
)


def _plan(*commands) -> BuildPlan:
    """Plan."""
    return BuildPlan(
        build_system="cmake",
        build_system_confidence=0.95,
        phases=[BuildPhase(id=1, name="build", commands=list(commands))],
        total_estimated_duration="5m",
    )


class _WorkspaceFixtureMixin:
    """Create ignored, project-local test directories without using /tmp."""

    root = Path("workspace") / "test_graph_validators"

    def setUp(self) -> None:
        """Create a clean workspace directory for each test."""
        test_name = self.id().replace(".", "_")
        self.workdir = self.root / test_name
        shutil.rmtree(self.workdir, ignore_errors=True)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Remove the workspace directory after each test."""
        shutil.rmtree(self.workdir, ignore_errors=True)


class TestBasicGraphHelpers(unittest.TestCase):
    """Tests for small, pure graph helper functions."""

    def test_model_factories_delegate_to_models_layer(self) -> None:
        """Role helpers delegate to the model factories."""
        model = mock.MagicMock(name="model")
        pool = [mock.MagicMock(name="primary")]

        with mock.patch("src.graph.create_llm", return_value=model) as maker:
            self.assertIs(get_model_for_role(AgentRole.SCOUT), model)
        maker.assert_called_once_with(AgentRole.SCOUT)

        with mock.patch(
            "src.models.create_llm_pool", return_value=pool
        ) as pool_maker:
            self.assertIs(get_model_pool_for_role(AgentRole.FIXER), pool)
        pool_maker.assert_called_once_with(AgentRole.FIXER)

    def test_extract_content_handles_lists_and_scalars(self) -> None:
        """LLM content blocks are flattened into plain text."""
        content = [{"text": "alpha"}, {"other": "beta"}, "gamma"]
        self.assertEqual(
            extract_content(content),
            "alpha\n{'other': 'beta'}\ngamma",
        )
        self.assertEqual(extract_content(123), "123")

    def test_extract_json_block_slices_outer_object(self) -> None:
        """JSON extraction returns the first outer object when present."""
        text = "prefix {\"ok\": true, \"nested\": {\"x\": 1}} suffix"
        self.assertEqual(
            extract_json_block(text),
            "{\"ok\": true, \"nested\": {\"x\": 1}}",
        )
        self.assertEqual(extract_json_block("no json here"), "no json here")

    def test_command_error_message_prefers_stderr_then_stdout(self) -> None:
        """Command errors use available output in priority order."""
        stderr_result = CommandResult("make", 2, "ignored", "boom", 0.1)
        stdout_result = CommandResult("make", 2, "hello", "", 0.1)
        empty_result = CommandResult("make", 2, "", "", 0.1)

        self.assertIn(
            "fallback (exit 2) - boom",
            _build_command_error_message(stderr_result, "fallback"),
        )
        self.assertIn(
            "fallback (exit 2) - hello",
            _build_command_error_message(stdout_result, "fallback"),
        )
        self.assertIn(
            "No stderr/stdout output captured",
            _build_command_error_message(empty_result, "fallback"),
        )

    def test_clone_failure_classification(self) -> None:
        """Clone failures are classified without falling to UNKNOWN."""
        cases = [
            ("could not read Username for https://x", ErrorCategory.NETWORK),
            ("remote: Repository not found.", ErrorCategory.CONFIGURATION),
            ("Could not resolve host: github.com", ErrorCategory.NETWORK),
            ("fatal: bad revision", ErrorCategory.CONFIGURATION),
        ]
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(_classify_clone_failure(message), expected)

    def test_force_replan_is_once_per_signature(self) -> None:
        """Only the first failure with a replan signature forces scout."""
        state = create_initial_state("https://x/y.git")
        state.last_error = 'go: build output "cmd" already exists'
        state.last_error += " and is a directory"

        self.assertTrue(_should_force_replan(state))
        self.assertFalse(_should_force_replan(state))
        self.assertEqual(
            state.context_cache["replan_attempts_by_signature"],
            {"go_output_dir_collision": 1},
        )

        state.last_error = "plain compiler error"
        self.assertFalse(_should_force_replan(state))

    def test_clone_recovery_records_success_and_attempts(self) -> None:
        """URL clone recovery mirrors deterministic variants into state."""
        from src import graph

        state = create_initial_state("https://git.example/cgit/foo")
        result = CommandResult(
            "git clone https://git.example/git/foo",
            0,
            "",
            "",
            1,
        )

        with (
            mock.patch.object(
                graph.scripted_ops,
                "_try_url_variants",
                return_value=result,
            ),
            mock.patch.object(
                graph.scripted_ops,
                "_resolve_homepage_to_git_urls",
                return_value=["https://git.example/foo.git"],
            ),
        ):
            self.assertIs(
                _try_clone_recovery(state, state.repo_url, state.repo_name),
                result,
            )

        recovery = state.context_cache["clone_recovery"]
        self.assertEqual(recovery["success_url"], result.command)
        self.assertIn("https://git.example/git/foo", recovery["attempted"])
        self.assertIn(
            "https://git.example/cgit/foo.git",
            recovery["attempted"],
        )
        self.assertIn("https://git.example/foo.git", recovery["attempted"])

    def test_clone_recovery_records_failure(self) -> None:
        """Failed clone recovery still records attempted variants."""
        from src import graph

        state = create_initial_state("https://git.example/foo.git")
        with (
            mock.patch.object(
                graph.scripted_ops,
                "_try_url_variants",
                return_value=None,
            ),
            mock.patch.object(
                graph.scripted_ops,
                "_resolve_homepage_to_git_urls",
                return_value=[],
            ),
        ):
            self.assertIsNone(
                _try_clone_recovery(state, state.repo_url, state.repo_name)
            )

        recovery = state.context_cache["clone_recovery"]
        self.assertIsNone(recovery["success_url"])
        self.assertEqual(recovery["attempted"], [state.repo_url])

    def test_default_task_plan_has_scout_then_build(self) -> None:
        """The structural task plan preserves the scout/build sequence."""
        plan = create_default_plan()
        self.assertEqual(
            [phase.name for phase in plan.phases],
            ["scout", "build"],
        )
        self.assertEqual(plan.phases[1].depends_on, [1])


class TestPromptFormattingHelpers(unittest.TestCase):
    """Tests for analysis fallback and prompt rendering helpers."""

    def test_fallback_analysis_uses_scripted_detection(self) -> None:
        """Fallback analysis summarizes deps, arch risks, and confidence."""
        state = create_initial_state("https://github.com/acme/tool.git")
        state.build_system_info = BuildSystemInfo(
            type="cmake",
            confidence=0.4,
            primary_file="CMakeLists.txt",
        )
        state.dependencies = DependencyInfo(
            libraries=[f"lib{i}" for i in range(12)]
        )
        state.arch_specific_code = [
            ArchSpecificCode(f"src/{i}.c", i, "__x86_64__", "x86", "high")
            for i in range(7)
        ]

        analysis = _fallback_analysis(state)

        self.assertFalse(analysis.llm_grounded)
        self.assertEqual(analysis.build_system, "cmake")
        self.assertTrue(analysis.needs_custom_plan)
        self.assertEqual(len(analysis.dependencies), 10)
        self.assertEqual(len(analysis.riscv_risks), 5)

    def test_format_analysis_handles_none(self) -> None:
        """Missing analysis renders as an explicit prompt placeholder."""
        self.assertEqual(
            _format_analysis_for_prompt(None),
            "(no package analysis available)",
        )

    def test_format_analysis_renders_compact_context(self) -> None:
        """Package analysis is rendered with deps, risks, and grounding."""
        analysis = PackageAnalysis(
            purpose="compresses data",
            language="C",
            build_system="autotools",
            build_system_confidence=0.93,
            build_system_reasoning="configure.ac declares AC_INIT",
            dependencies=[
                {"name": "zlib", "reason": "configure.ac checks zlib"},
            ],
            riscv_risks=["x86 asm directory is optional"],
            build_strategy="Run configure then make.",
            expected_artifacts=["libfoo.so"],
            llm_grounded=True,
        )

        rendered = _format_analysis_for_prompt(analysis)

        self.assertIn("Purpose: compresses data", rendered)
        self.assertIn("zlib (configure.ac checks zlib)", rendered)
        self.assertIn("libfoo.so", rendered)
        self.assertIn("LLM-read from repo files", rendered)


class TestRepositoryPathHelpers(_WorkspaceFixtureMixin, unittest.TestCase):
    """Tests for helpers that inspect project-local fixture trees."""

    def test_is_within_repo_accepts_child_and_rejects_escape(self) -> None:
        """Path containment uses real paths rather than string prefixes."""
        repo = self.workdir / "repo"
        repo.mkdir()

        self.assertTrue(_is_within_repo(str(repo / "src" / "x.c"), str(repo)))
        self.assertFalse(_is_within_repo(str(repo / ".." / "evil"), str(repo)))

    def test_probe_config_files_reports_existing_build_files(self) -> None:
        """Build config probing returns only files present at the repo root."""
        repo = self.workdir / "repo"
        repo.mkdir()
        (repo / "CMakeLists.txt").write_text("project(x)\n")
        (repo / "go.mod").write_text("module x\n")
        state = create_initial_state("https://x/y.git")
        state.repo_path = str(repo)

        self.assertEqual(
            _probe_config_files(state),
            ["CMakeLists.txt", "go.mod"],
        )

    def test_setup_packages_merges_detected_and_analyst_deps(self) -> None:
        """Setup package merging resolves known canonicals and dedupes."""
        from src.platforms import ALPINE_RISCV

        state = create_initial_state("https://x/y.git")
        state.scout_deps_result = {
            "build_tools": ["cmake"],
            "libraries": ["zlib", "openssl"],
            "system_packages": ["unknown-lib"],
        }
        state.package_analysis = PackageAnalysis(
            dependencies=[
                {"name": "openssl", "reason": "CMakeLists.txt"},
                {"name": "not-in-map", "reason": "ignored"},
            ]
        )

        packages = _setup_packages_for_plan(
            state,
            ["gcc"],
            ALPINE_RISCV,
        )

        self.assertEqual(
            packages,
            ["build-base", "openssl-dev", "cmake", "zlib-dev"],
        )

    def test_find_makefile_dir_uses_shallowest_non_skipped_dir(self) -> None:
        """Makefile search ignores test/examples and returns shallowest."""
        repo = self.workdir / "repo"
        (repo / "tests").mkdir(parents=True)
        (repo / "src").mkdir()
        (repo / "tests" / "Makefile").write_text("all:\n")
        (repo / "src" / "Makefile").write_text("all:\n")

        self.assertEqual(_find_makefile_dir(str(repo)), "src")

    def test_find_makefile_dir_root_or_missing_returns_empty(self) -> None:
        """Root makefiles, missing dirs, and absent makefiles need no cd."""
        missing = self.workdir / "missing"
        repo = self.workdir / "repo"
        repo.mkdir()

        self.assertEqual(_find_makefile_dir(str(missing)), "")
        self.assertEqual(_find_makefile_dir(str(repo)), "")

        (repo / "GNUmakefile").write_text("all:\n")
        self.assertEqual(_find_makefile_dir(str(repo)), "")

    def test_autotools_bootstrap_selects_specific_script(self) -> None:
        """Autotools bootstrap prefers upstream helper scripts."""
        repo = self.workdir / "repo"
        repo.mkdir()
        (repo / "autogen.sh").write_text("#!/bin/sh\n")

        self.assertEqual(
            _autotools_bootstrap_prefix(str(repo)),
            "sh autogen.sh",
        )

    def test_autotools_bootstrap_handles_configure_ac(self) -> None:
        """Autotools bootstrap uses autoreconf for configure.ac."""
        repo = self.workdir / "repo"
        repo.mkdir()
        (repo / "configure.ac").write_text("AC_INIT([x], [1])\n")

        self.assertEqual(
            _autotools_bootstrap_prefix(str(repo)),
            "autoreconf -fi",
        )

    def test_autotools_bootstrap_uses_gettext_copy_first(self) -> None:
        """Gettext projects use copy-first regeneration."""
        repo = self.workdir / "repo"
        (repo / "m4").mkdir(parents=True)
        (repo / "configure.in").write_text("AC_INIT(x)\n")
        (repo / "m4" / "gettext.m4").write_text("old\n")

        command = _autotools_bootstrap_prefix(str(repo))

        self.assertIn("cp /usr/share/gettext/m4/*.m4", command)
        self.assertIn("aclocal -I m4", command)
        self.assertNotIn("autoreconf", command)

    def test_autotools_bootstrap_not_needed_when_configure_exists(
        self,
    ) -> None:
        """A committed configure script disables bootstrap injection."""
        repo = self.workdir / "repo"
        repo.mkdir()
        (repo / "configure").write_text("#!/bin/sh\n")

        self.assertEqual(_autotools_bootstrap_prefix(str(repo)), "")

    def test_extract_cd_prefix(self) -> None:
        """Leading cd prefixes are preserved for retry companion commands."""
        self.assertEqual(
            _extract_cd_prefix("  cd build && make"),
            "cd build && ",
        )
        self.assertEqual(_extract_cd_prefix("make"), "")


class TestFallbackBuildPlan(_WorkspaceFixtureMixin, unittest.TestCase):
    """Tests for deterministic fallback BuildPlan generation."""

    def _state_for(self, build_type: str) -> object:
        """Create a state with a detected build system."""
        state = create_initial_state("https://github.com/acme/tool.git")
        state.repo_path = str(self.workdir / "repo")
        Path(state.repo_path).mkdir(parents=True, exist_ok=True)
        state.build_system_info = BuildSystemInfo(
            type=build_type,
            confidence=0.7,
            primary_file="build.conf",
        )
        return state

    def test_go_fallback_initializes_missing_module(self) -> None:
        """Go fallback initializes GOPATH-style repos and adds buildvcs."""
        state = self._state_for("go")
        state.context_cache["go_main_info"] = {
            "needs_go_init": True,
            "build_command": "go build .",
        }

        plan = create_fallback_build_plan(state)
        commands = plan.phases[1].commands

        self.assertEqual(plan.build_system, "go")
        self.assertIn("go mod init github.com/acme/tool.git", commands)
        self.assertIn("go build -buildvcs=false ./...", commands)

    def test_standard_fallbacks_cover_known_build_systems(self) -> None:
        """CMake, make, cargo, and meson each get deterministic recipes."""
        expected = {
            "cmake": "cmake .. -DCMAKE_BUILD_TYPE=Release",
            "make": "make -j$(nproc)",
            "cargo": "cargo build --release",
            "meson": "meson setup builddir",
        }

        for build_type, command in expected.items():
            with self.subTest(build_type=build_type):
                plan = create_fallback_build_plan(self._state_for(build_type))
                all_commands = [
                    cmd for phase in plan.phases for cmd in phase.commands
                ]
                self.assertEqual(plan.build_system, build_type)
                self.assertTrue(any(command in cmd for cmd in all_commands))

    def test_autotools_fallback_bootstraps_when_configure_missing(
        self,
    ) -> None:
        """Autotools fallback regenerates configure when it is absent."""
        state = self._state_for("autotools")

        plan = create_fallback_build_plan(state)

        self.assertIn("autoreconf -fi", plan.phases[1].commands)
        self.assertEqual(plan.build_system, "autotools")

    def test_unknown_fallback_discovers_existing_config_files(self) -> None:
        """Unknown fallback probes real build files before trying make."""
        state = self._state_for("unknown")
        Path(state.repo_path, "CMakeLists.txt").write_text("project(x)\n")

        plan = create_fallback_build_plan(state)

        self.assertEqual(plan.build_system, "unknown")
        self.assertTrue(
            any(
                cmd.startswith("ls -la CMakeLists.txt")
                for cmd in plan.phases[0].commands
            )
        )

    def test_unknown_fallback_uses_generic_find_when_no_files(self) -> None:
        """Unknown fallback uses a bounded find when no probes exist."""
        state = self._state_for("unknown")

        plan = create_fallback_build_plan(state)

        self.assertTrue(
            any(
                cmd.startswith("find . -maxdepth 2")
                for cmd in plan.phases[0].commands
            )
        )


class TestDockerShellHelpers(unittest.TestCase):
    """Tests for docker-shell helpers without touching Docker."""

    def test_fixup_top_builddir_logs_stdout_and_stderr(self) -> None:
        """top_builddir fixup shells out through a mocked docker call."""
        result = mock.MagicMock(
            stdout="Injected top_builddir=.. into ./po/Makefile\n",
            stderr="warning\n",
            returncode=1,
        )

        with (
            mock.patch("src.platforms.get_container_name", return_value="box"),
            mock.patch("src.graph.subprocess.run", return_value=result) as run,
        ):
            _fixup_top_builddir_in_submakefiles("/workspace/repos/pkg")

        run.assert_called_once()
        args = run.call_args.args[0]
        self.assertEqual(args[:3], ["docker", "exec", "box"])
        self.assertIn("top_builddir", args[-1])

    def test_fixup_top_builddir_swallows_exceptions(self) -> None:
        """Docker fixup is non-fatal when platform lookup fails."""
        with mock.patch(
            "src.platforms.get_container_name",
            side_effect=RuntimeError("no platform"),
        ):
            _fixup_top_builddir_in_submakefiles("/workspace/repos/pkg")


# ---------- validate_build_plan ----------


class TestValidateBuildPlan(unittest.TestCase):
    """Tests for ValidateBuildPlan."""

    def test_clean_plan_passes(self) -> None:
        """Test clean plan passes."""
        ok, msg = validate_build_plan(_plan("cmake -B build -S .", "make -j4"))
        self.assertTrue(ok, msg)

    def test_placeholder_path_rejected(self) -> None:
        """Test placeholder path rejected."""
        ok, msg = validate_build_plan(_plan("cp /path/to/foo bar"))
        self.assertFalse(ok)
        self.assertIn("Hallucination", msg)

    def test_your_username_placeholder_rejected(self) -> None:
        """Test your username placeholder rejected."""
        ok, msg = validate_build_plan(_plan("cd /home/your_username/foo"))
        self.assertFalse(ok)

    def test_example_com_placeholder_rejected(self) -> None:
        """Test example com placeholder rejected."""
        ok, _ = validate_build_plan(
            _plan("wget https://example.com/foo.tar.gz")
        )
        self.assertFalse(ok)

    def test_riscv_unknown_linux_gnu_gcc_rejected(self) -> None:
        """Test riscv unknown linux gnu gcc rejected."""
        ok, _ = validate_build_plan(
            _plan("riscv64-unknown-linux-gnu-gcc -O2 main.c")
        )
        self.assertFalse(ok)

    def test_home_path_outside_workspace_rejected(self) -> None:
        """Test home path outside workspace rejected."""
        ok, _ = validate_build_plan(_plan("cp /home/akif/foo bar"))
        self.assertFalse(ok)

    def test_home_inside_workspace_path_passes(self) -> None:
        # /home/.../workspace/... is fine
        """Test home inside workspace path passes."""
        ok, _ = validate_build_plan(_plan("ls /home/akif/workspace/foo"))
        self.assertTrue(ok)

    def test_go_subcommand_as_apk_package_rejected(self) -> None:
        """Test go subcommand as apk package rejected."""
        ok, msg = validate_build_plan(_plan("apk add go mod"))
        self.assertFalse(ok)
        self.assertIn("Go subcommand", msg)

    def test_go_subcommand_as_apt_package_rejected(self) -> None:
        """Test go subcommand as apt package rejected."""
        ok, _ = validate_build_plan(_plan("apt-get install build"))
        self.assertFalse(ok)

    def test_apt_install_with_flags_handles_pkg_list(self) -> None:
        """Test apt install with flags handles pkg list."""
        ok, _ = validate_build_plan(
            _plan("apt-get install -y libssl-dev curl")
        )
        self.assertTrue(ok)

    def test_nested_cmake_build_dir_rejected(self) -> None:
        """Test nested cmake build dir rejected."""
        ok, msg = validate_build_plan(
            _plan("cd build && cmake -S . -B build ..")
        )
        self.assertFalse(ok)
        self.assertIn("nested build dir", msg)


# ---------- validate_fix_command ----------


class TestValidateFixCommand(unittest.TestCase):
    """Tests for ValidateFixCommand."""

    @classmethod
    def setUpClass(cls) -> None:
        """SetUpClass."""
        cls.dangerous = [
            ("touch src/foo.go", "Go source"),
            ("touch include/x.h", "header"),
            ("touch foo.py", "Python"),
            ("touch app.cpp", "C++"),
            ("rm -rf /", "root rm"),
            ("rm -rf *", "wildcard rm"),
            ("git push origin main", "git push"),
            ("git reset --hard HEAD~1", "hard reset"),
            ('echo "" > foo.c', "empty content overwrite"),
            ("apk add go mod", "go subcommand as package"),
            ("apt install build", "go subcommand as package"),
        ]
        cls.safe = [
            "make -j4",
            "cmake -B build -S .",
            "sed -i 's/a/b/' foo.c",
            "cp src.c src.c.bak",
            "rm -f build/CMakeCache.txt",
            "apk add zlib-dev",
        ]

    def test_dangerous_commands_rejected(self) -> None:
        """Test dangerous commands rejected."""
        for cmd, label in self.dangerous:
            with self.subTest(cmd=cmd, label=label):
                ok, _ = validate_fix_command(cmd)
                self.assertFalse(ok, f"SHOULD REJECT [{label}]: {cmd}")

    def test_safe_commands_pass(self) -> None:
        """Test safe commands pass."""
        for cmd in self.safe:
            with self.subTest(cmd=cmd):
                ok, reason = validate_fix_command(cmd)
                self.assertTrue(ok, f"SHOULD PASS: {cmd} -> {reason}")


# ---------- validate_fixer_response ----------


class TestValidateFixerResponse(unittest.TestCase):
    """Tests for ValidateFixerResponse."""

    def _base(self, actions):
        """Build a base strategy payload for tests."""
        return {
            "strategies": [{"id": 1, "actions": actions}],
            "recommended_strategy_id": 1,
            "reflection": {
                "root_cause": "x",
                "this_fix_will_work_because": "y",
            },
        }

    def test_non_dict_rejected(self) -> None:
        """Test non dict rejected."""
        ok, _ = validate_fixer_response([])  # type: ignore[arg-type]
        self.assertFalse(ok)

    def test_no_strategies_rejected(self) -> None:
        """Test no strategies rejected."""
        ok, msg = validate_fixer_response({})
        self.assertFalse(ok)
        self.assertIn("strategies", msg.lower())

    def test_recommended_id_not_found_rejected(self) -> None:
        """Test recommended id not found rejected."""
        data = self._base([{"type": "command", "command": "ls"}])
        data["recommended_strategy_id"] = 99
        ok, msg = validate_fixer_response(data)
        self.assertFalse(ok)
        self.assertIn("not found", msg)

    def test_empty_actions_rejected(self) -> None:
        """Test empty actions rejected."""
        ok, msg = validate_fixer_response(self._base([]))
        self.assertFalse(ok)
        self.assertIn("no actions", msg.lower())

    def test_create_file_with_empty_content_rejected(self) -> None:
        """Test create file with empty content rejected."""
        ok, msg = validate_fixer_response(
            self._base(
                [{"type": "create_file", "path": "foo.c", "content": "   "}]
            )
        )
        self.assertFalse(ok)
        self.assertIn("empty content", msg)

    def test_create_file_with_absolute_home_path_rejected(self) -> None:
        """Test create file with absolute home path rejected."""
        ok, _ = validate_fixer_response(
            self._base(
                [
                    {
                        "type": "create_file",
                        "path": "/home/x/foo.c",
                        "content": "int main(){}",
                    }
                ]
            )
        )
        self.assertFalse(ok)

    def test_create_file_missing_path_rejected(self) -> None:
        """Test create file missing path rejected."""
        ok, _ = validate_fixer_response(
            self._base([{"type": "create_file", "path": "", "content": "x"}])
        )
        self.assertFalse(ok)

    def test_create_file_long_path_rejected(self) -> None:
        """Suspiciously long create_file paths are rejected."""
        ok, msg = validate_fixer_response(
            self._base(
                [
                    {
                        "type": "create_file",
                        "path": "a" * 101,
                        "content": "int main(void) { return 0; }",
                    }
                ]
            )
        )
        self.assertFalse(ok)
        self.assertIn("Suspiciously long filepath", msg)

    def test_create_file_parent_traversal_rejected(self) -> None:
        """create_file paths cannot escape with parent traversal."""
        ok, msg = validate_fixer_response(
            self._base(
                [
                    {
                        "type": "create_file",
                        "path": "../evil.c",
                        "content": "int main(void) { return 0; }",
                    }
                ]
            )
        )
        self.assertFalse(ok)
        self.assertIn("escapes the repository", msg)

    def test_patch_parent_traversal_rejected(self) -> None:
        """Patch file paths cannot escape the repository."""
        ok, msg = validate_fixer_response(
            self._base(
                [
                    {
                        "type": "patch",
                        "file": "../src/x.c",
                        "content": "--- a\n+++ b\n",
                    }
                ]
            )
        )
        self.assertFalse(ok)
        self.assertIn("Patch path escapes", msg)

    def test_empty_command_rejected(self) -> None:
        """Test empty command rejected."""
        ok, msg = validate_fixer_response(
            self._base([{"type": "command", "command": "   "}])
        )
        self.assertFalse(ok)
        self.assertIn("empty", msg.lower())

    def test_self_copy_command_rejected(self) -> None:
        """Test self copy command rejected."""
        ok, msg = validate_fixer_response(
            self._base([{"type": "command", "command": "cp -r src/* src/"}])
        )
        self.assertFalse(ok)
        self.assertIn("copying to self", msg)

    def test_unsafe_embedded_command_rejected(self) -> None:
        """Test unsafe embedded command rejected."""
        ok, msg = validate_fixer_response(
            self._base([{"type": "command", "command": "touch foo.go"}])
        )
        self.assertFalse(ok)
        self.assertIn("Unsafe", msg)

    def test_valid_response_accepted(self) -> None:
        """Test valid response accepted."""
        ok, msg = validate_fixer_response(
            self._base(
                [
                    {"type": "command", "command": "make clean"},
                    {
                        "type": "create_file",
                        "path": "patch.diff",
                        "content": "--- a\n+++ b\n",
                    },
                ]
            )
        )
        self.assertTrue(ok, msg)

    def test_missing_reflection_fields_only_warn(self) -> None:
        """A valid action can pass even with shallow reflection fields."""
        data = self._base([{"type": "command", "command": "make clean"}])
        data["reflection"] = {}

        ok, msg = validate_fixer_response(data)

        self.assertTrue(ok, msg)

    def test_malformed_strategy_is_reported_as_validation_error(self) -> None:
        """Unexpected strategy shapes are caught and returned as invalid."""
        ok, msg = validate_fixer_response(
            {"strategies": [{"actions": []}], "recommended_strategy_id": 1}
        )

        self.assertFalse(ok)
        self.assertIn("Validation error", msg)


# ---------- predict_build_issues ----------


class TestPredictBuildIssues(unittest.TestCase):
    """Tests for PredictBuildIssues."""

    def test_no_plan_returns_empty(self) -> None:
        """Test no plan returns empty."""
        state = create_initial_state("https://x/y.git")
        self.assertEqual(predict_build_issues(state), [])

    def test_go_build_without_buildvcs_flagged(self) -> None:
        """Test go build without buildvcs flagged."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("go build ./cmd/foo")
        preds = predict_build_issues(state)
        self.assertTrue(
            any(p["issue"] == "Go VCS ownership error" for p in preds)
        )

    def test_go_build_with_buildvcs_false_not_flagged(self) -> None:
        """Test go build with buildvcs false not flagged."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("go build -buildvcs=false ./cmd/foo")
        preds = predict_build_issues(state)
        self.assertFalse(
            any(p["issue"] == "Go VCS ownership error" for p in preds)
        )

    def test_cmake_with_high_severity_arch_code_flagged(self) -> None:
        """Test cmake with high severity arch code flagged."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("cmake -B build -S .")
        state.arch_specific_code = [
            ArchSpecificCode(
                file="x.c",
                line=10,
                code_snippet="_mm_add_ps",
                arch_type="x86",
                severity="high",
                suggested_fix="fallback",
            )
        ]
        preds = predict_build_issues(state)
        self.assertTrue(any(p["pattern"] == "arch_specific" for p in preds))

    def test_apk_install_with_missing_tools_flagged(self) -> None:
        """Test apk install with missing tools flagged."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("apk add zlib-dev")
        state.context_cache["missing_tools"] = ["protoc"]
        preds = predict_build_issues(state)
        self.assertTrue(any(p["pattern"] == "missing_tools" for p in preds))

    def test_cargo_build_not_flagged_as_go_vcs(self) -> None:
        """Regression: substring 'go build' must not match 'cargo build'."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("cargo build --release")
        preds = predict_build_issues(state)
        self.assertFalse(
            any(p["issue"] == "Go VCS ownership error" for p in preds)
        )

    def test_apk_add_cargo_build_base_not_flagged_as_go_vcs(self) -> None:
        """Regression: 'cargo build-base' (alpine pkg) is not a go build."""
        state = create_initial_state("https://x/y.git")
        state.build_plan = _plan("apk add --no-cache rust cargo build-base")
        preds = predict_build_issues(state)
        self.assertFalse(
            any(p["issue"] == "Go VCS ownership error" for p in preds)
        )


class TestIsGoBuildCommand(unittest.TestCase):
    """Regression tests pinning the word-boundary go-build detector.

    The previous implementation used substring checks like
    `"go build" in cmd`, which falsely matched `cargo build` (at index 3)
    and the alpine package list `apk add ... cargo build-base`. Downstream
    code then mangled those commands (e.g. injecting `-buildvcs=false`
    into apk arguments), which is what caused the dalfox build failures.
    """

    def test_matches_plain_go_build(self) -> None:
        """Test matches plain go build."""
        self.assertTrue(_is_go_build_command("go build ."))

    def test_matches_go_build_with_flags(self) -> None:
        """Test matches go build with flags."""
        self.assertTrue(_is_go_build_command("go build -v ./cmd/foo"))

    def test_matches_go_install(self) -> None:
        """Test matches go install."""
        self.assertTrue(_is_go_build_command("go install ./..."))

    def test_matches_go_build_after_env(self) -> None:
        """Test env-prefixed go build is still detected."""
        self.assertTrue(
            _is_go_build_command("env GOMAXPROCS=1 go build -p 1 ./...")
        )

    def test_does_not_match_cargo_build(self) -> None:
        """Test does not match cargo build."""
        self.assertFalse(_is_go_build_command("cargo build --release"))

    def test_does_not_match_apk_cargo_build_base(self) -> None:
        """Test does not match 'apk add ... cargo build-base'."""
        self.assertFalse(
            _is_go_build_command("apk add --no-cache rust cargo build-base")
        )

    def test_does_not_match_hyphenated_token(self) -> None:
        """Test does not match 'do go-build' (hyphenated, not a go command)."""
        self.assertFalse(_is_go_build_command("do go-build"))

    def test_does_not_match_empty(self) -> None:
        """Test empty / non-string inputs return False."""
        self.assertFalse(_is_go_build_command(""))


class TestToolchainMismatchDetection(unittest.TestCase):
    """Tests for toolchain-version mismatch detection helper."""

    def test_detects_go_version_mismatch(self) -> None:
        """Go `go.mod requires go >=` errors are treated as mismatches."""
        self.assertTrue(
            is_toolchain_version_mismatch(
                "go: go.mod requires go >= 1.26.0 (running go 1.22.5)"
            )
        )

    def test_detects_rust_edition_mismatch(self) -> None:
        """Cargo edition gating errors are treated as mismatches."""
        self.assertTrue(
            is_toolchain_version_mismatch(
                "feature `edition2024` is required and not stabilized "
                "in this version of Cargo"
            )
        )

    def test_ignores_regular_dependency_errors(self) -> None:
        """Ordinary missing package errors are not toolchain mismatches."""
        self.assertFalse(
            is_toolchain_version_mismatch("unable to select packages: libssl")
        )


class TestGoCommandRewriters(unittest.TestCase):
    """Tests for Go command rewrite helpers."""

    def test_inject_go_flag_once(self) -> None:
        """Do not duplicate flags when they already exist."""
        cmd = "go build -buildvcs=false ./cmd/foo"
        self.assertEqual(_inject_go_flag(cmd, "-buildvcs=false"), cmd)

    def test_inject_go_output(self) -> None:
        """Inject -o path into go build command."""
        cmd = "go build ./cmd"
        out = _inject_go_output(cmd, "./.atesor-bin/cmd")
        self.assertIn("-o ./.atesor-bin/cmd", out)
        self.assertIn("go build", out)

    def test_inject_go_output_keeps_existing_output(self) -> None:
        """Existing -o flags are not duplicated."""
        cmd = "go build -o ./bin/tool ./cmd/tool"
        self.assertEqual(_inject_go_output(cmd, "./other"), cmd)


class TestReplanSignature(unittest.TestCase):
    """Tests for plan-level error signature detection."""

    def test_go_output_collision_signature(self) -> None:
        """Output-dir collision should map to replan signature."""
        msg = 'go: build output "cmd" already exists and is a directory'
        self.assertEqual(_replan_signature(msg), "go_output_dir_collision")

    def test_unknown_signature(self) -> None:
        """Unrelated errors should return empty signature."""
        self.assertEqual(_replan_signature("some random error"), "")


class _Res:
    """Minimal CommandResult stand-in for retry-helper tests."""

    def __init__(self, exit_code, stdout, stderr, success):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.success = success


class TestSerializeBuildCommand(unittest.TestCase):
    """Tests for the OOM serialized-build rewriter."""

    def test_go_build_serialized(self) -> None:
        """Go build commands are serialized with env and -p limits."""
        out = _serialize_build_command("go build ./cmd/x")
        self.assertIn("GOMAXPROCS=1", out)
        self.assertIn("-p 1", out)

    def test_go_build_preserves_cd_prefix(self) -> None:
        """Serialized go builds keep the leading cd command."""
        out = _serialize_build_command("cd sub && go build ./cmd/x")
        self.assertTrue(out.startswith("cd sub && env GOMAXPROCS=1 go build"))

    def test_make_serialized(self) -> None:
        """Make commands replace nproc parallelism with -j1."""
        out = _serialize_build_command("cd build && make -j$(nproc)")
        self.assertIn("-j1", out)
        self.assertNotIn("nproc", out)
        self.assertTrue(out.startswith("cd build &&"))

    def test_plain_make_gets_j1(self) -> None:
        """Plain make commands gain a -j1 serialization flag."""
        out = _serialize_build_command("make")
        self.assertIn("make -j1", out)

    def test_cmake_not_mistaken_for_make(self) -> None:
        """Build commands using cmake are not rewritten as make."""
        out = _serialize_build_command("cmake --build build")
        self.assertEqual(out, "cmake --build build")

    def test_ninja_gets_j1(self) -> None:
        """Ninja commands gain a single-threaded jobs flag."""
        out = _serialize_build_command("cd build && ninja")
        self.assertEqual(out, "cd build && ninja -j1")

    def test_ninja_with_jobs_is_unchanged(self) -> None:
        """Ninja commands with an explicit jobs flag are unchanged."""
        cmd = "ninja -j2 -C build"
        self.assertEqual(_serialize_build_command(cmd), cmd)

    def test_cargo_fetch_gets_env_only(self) -> None:
        """Verify cargo fetch gets env only (it has no -j flag)."""
        out = _serialize_build_command("cargo fetch")
        self.assertIn("CARGO_BUILD_JOBS=1", out)
        # `cargo fetch` must not gain a spurious -j 1 flag.
        self.assertNotIn(" -j ", out)

    def test_cargo_build_gets_j_and_env(self) -> None:
        """Verify cargo build gets both -j 1 and CARGO_BUILD_JOBS."""
        out = _serialize_build_command("cargo build --release")
        self.assertIn("cargo build -j 1", out)
        self.assertIn("CARGO_BUILD_JOBS=1", out)

    def test_pip_install_gets_makeflags(self) -> None:
        """Verify pip install serializes via MAKEFLAGS."""
        out = _serialize_build_command("pip install .")
        self.assertIn("MAKEFLAGS=-j1", out)

    def test_npm_install_gets_jobs_flag(self) -> None:
        """Verify npm install gains --jobs 1."""
        out = _serialize_build_command("npm install")
        self.assertIn("npm install --jobs 1", out)


class TestBuilderRetryAllowed(unittest.TestCase):
    """Tests for single-use deterministic builder retry keys."""

    def test_retry_key_is_allowed_once(self) -> None:
        """The same retry signature is only admitted once."""
        state = create_initial_state("https://x/y.git")

        self.assertTrue(_builder_retry_allowed(state, "oom:make"))
        self.assertFalse(_builder_retry_allowed(state, "oom:make"))
        self.assertEqual(
            state.context_cache["builder_retry_keys"],
            ["oom:make"],
        )


class TestSuspectedOOM(unittest.TestCase):
    """Tests for the suspected-OOM detector."""

    def test_exit_137_go(self) -> None:
        """Exit 137 from a go build is treated as suspected OOM."""
        self.assertTrue(
            _is_suspected_oom(_Res(137, "", "", False), "go build ./x")
        )

    def test_empty_output_build(self) -> None:
        """Failed go builds with empty output are treated as OOM."""
        self.assertTrue(
            _is_suspected_oom(_Res(1, "", "", False), "go build ./x")
        )

    def test_real_error_not_oom(self) -> None:
        """Go build failures with stderr are not suspected OOM."""
        self.assertFalse(
            _is_suspected_oom(_Res(1, "", "boom", False), "go build ./x")
        )

    def test_non_parallel_command_with_empty_output_not_oom(self) -> None:
        """Silent failures from unrelated commands are not treated as OOM."""
        self.assertFalse(_is_suspected_oom(_Res(1, "", "", False), "ls"))

    def test_exit_137_any_command(self) -> None:
        """Exit 137 is treated as OOM even for non-build commands."""
        self.assertTrue(
            _is_suspected_oom(
                _Res(137, "", "", False), "git clone https://x /tmp/y"
            )
        )

    def test_cargo_fetch_exit137(self) -> None:
        """Verify cargo fetch OOM (exit 137) is caught."""
        self.assertTrue(
            _is_suspected_oom(_Res(137, "", "", False), "cargo fetch")
        )

    def test_npm_install_empty_output(self) -> None:
        """Verify npm install with no output is treated as OOM."""
        self.assertTrue(
            _is_suspected_oom(_Res(1, "", "", False), "npm install")
        )


class TestHeaderResolution(unittest.TestCase):
    """Tests for header->package resolution."""

    def test_png_header_resolves(self) -> None:
        """Missing png.h resolves to the Alpine libpng package."""
        from src.platforms import ALPINE_RISCV

        pkgs = _resolve_header_to_packages(
            "fatal error: png.h: No such file or directory", ALPINE_RISCV
        )
        self.assertEqual(pkgs, ["libpng-dev"])

    def test_nested_ogg_header_resolves(self) -> None:
        """Nested ogg headers resolve to the Debian libogg package."""
        from src.platforms import DEBIAN_RISCV

        pkgs = _resolve_header_to_packages(
            "fatal error: ogg/ogg.h: No such file", DEBIAN_RISCV
        )
        self.assertEqual(pkgs, ["libogg-dev"])

    def test_unknown_header_not_guessed(self) -> None:
        """Unknown headers do not produce guessed package names."""
        from src.platforms import ALPINE_RISCV

        pkgs = _resolve_header_to_packages(
            "fatal error: my_internal_thing.h: No such file", ALPINE_RISCV
        )
        self.assertEqual(pkgs, [])

    def test_empty_header_error_returns_empty(self) -> None:
        """Empty compiler output produces no package guesses."""
        from src.platforms import ALPINE_RISCV

        self.assertEqual(_resolve_header_to_packages("", ALPINE_RISCV), [])

    def test_clang_file_not_found_form_resolves(self) -> None:
        """Clang's quoted header diagnostic is also parsed."""
        from src.platforms import ALPINE_RISCV

        pkgs = _resolve_header_to_packages(
            "'webp/decode.h' file not found",
            ALPINE_RISCV,
        )
        self.assertEqual(pkgs, ["libwebp-dev"])


class TestPythonModuleResolution(unittest.TestCase):
    """Tests for missing-Python-module resolution."""

    def test_jinja2_module(self) -> None:
        """Missing jinja2 modules resolve to the jinja2 package."""
        pkgs = _resolve_missing_python_modules(
            "ModuleNotFoundError: No module named 'jinja2'"
        )
        self.assertEqual(pkgs, ["jinja2"])

    def test_yaml_maps_to_pyyaml(self) -> None:
        """Missing yaml modules resolve to the pyyaml package."""
        pkgs = _resolve_missing_python_modules(
            "ModuleNotFoundError: No module named 'yaml'"
        )
        self.assertEqual(pkgs, ["pyyaml"])

    def test_dotted_module_uses_top_level(self) -> None:
        """Dotted module errors resolve by top-level module name."""
        pkgs = _resolve_missing_python_modules(
            "No module named 'google.protobuf'"
        )
        self.assertEqual(pkgs, ["protobuf"])

    def test_no_module_error(self) -> None:
        """Errors without module names produce no Python packages."""
        self.assertEqual(_resolve_missing_python_modules("some error"), [])

    def test_empty_python_error(self) -> None:
        """Empty stderr produces no Python packages."""
        self.assertEqual(_resolve_missing_python_modules(""), [])


class TestGoToolchainDownloadCommand(unittest.TestCase):
    """Tests for the go.dev tarball installer shell command."""

    def test_version_appears_in_url(self) -> None:
        """The command references the requested Go version in the URL."""
        cmd = _download_go_toolchain_cmd("1.25.0")
        self.assertIn("go1.25.0.linux-riscv64.tar.gz", cmd)
        self.assertIn("https://go.dev/dl/", cmd)

    def test_command_replaces_usr_local_go(self) -> None:
        """The command wipes /usr/local/go before untar to avoid mixing."""
        cmd = _download_go_toolchain_cmd("1.26.3")
        self.assertIn("rm -rf /usr/local/go", cmd)
        self.assertIn("tar -xzf", cmd)
        self.assertIn("/usr/local/go/bin/go version", cmd)

    def test_command_uses_set_e(self) -> None:
        """The command uses ``set -e`` so any step failing aborts install."""
        cmd = _download_go_toolchain_cmd("1.25.0")
        self.assertTrue(cmd.startswith("set -e && "))


class TestRepoGitmodules(unittest.TestCase):
    """Tests for ``.gitmodules`` presence detection."""

    def test_present(self) -> None:
        """Returns True when the file exists."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, ".gitmodules"), "w") as f:
                f.write("x")
            self.assertTrue(_repo_has_gitmodules(d))

    def test_absent(self) -> None:
        """Returns False when the file does not exist."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_repo_has_gitmodules(d))

    def test_none_path_is_safe(self) -> None:
        """Passing ``None`` does not raise."""
        self.assertFalse(_repo_has_gitmodules(None))


class TestNpmScripts(unittest.TestCase):
    """Tests for reading npm scripts from package.json."""

    def _write_pkg(self, repo, payload) -> None:
        """Helper: write ``package.json`` inside ``repo`` from a dict."""
        with open(os.path.join(repo, "package.json"), "w") as f:
            import json as _json

            _json.dump(payload, f)

    def test_missing_package_json(self) -> None:
        """No package.json → empty list."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(_npm_scripts(d), [])

    def test_present_scripts(self) -> None:
        """Scripts dict is returned as an ordered list."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_pkg(d, {"scripts": {"test": "jest", "prod": "x"}})
            self.assertEqual(_npm_scripts(d), ["test", "prod"])

    def test_no_scripts_key(self) -> None:
        """Missing scripts key → empty list, not KeyError."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._write_pkg(d, {"name": "foo"})
            self.assertEqual(_npm_scripts(d), [])

    def test_unparseable_json(self) -> None:
        """Garbage in package.json → empty list, no crash."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "package.json"), "w") as f:
                f.write("{ not json")
            self.assertEqual(_npm_scripts(d), [])


if __name__ == "__main__":
    unittest.main()
