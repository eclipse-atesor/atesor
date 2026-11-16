"""How the agent workflow must use the learned data in data/.

These tests read the real, committed files under data/ (read-only; the
conftest guard fails the session if a test changes them). They run as
the Debian sandbox, because the real batch runs (CI and the Megrez) are
Debian runs. The other memory tests run as Alpine, which hides the
Debian behavior.

Each test states one rule of the intended workflow:

* data quality: every example can reach a prompt and holds a fix that
  the active distro can run;
* fixer retrieval: a real error from the run logs gets the example that
  fixes it, and never an example for an unrelated error;
* scout retrieval: the examples follow the package layout;
* recipe cache: a verified earlier plan of the same package reaches the
  planning and fixing prompts, also on a forced or native run;
* learning: a verified run writes data that a later run can match;
* audit: the call log names the examples each prompt got.

The error texts come from the native run logs of October 2026
(workspace-native/logs). A failing test is a workflow defect to fix,
not a test to loosen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

from src import memory, platforms
from src.evidence import has_error_line
from src.llm_helpers import LLMCallOutcome
from src.state import (
    BuildPhase,
    BuildPlan,
    BuildStatus,
    BuildSystemInfo,
    DependencyInfo,
    ErrorCategory,
    FixAttempt,
    PackageAnalysis,
    create_error_record,
    create_initial_state,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXAMPLES_DIR = DATA_DIR / "examples"
DEBIAN = "debian-riscv64"
ALPINE = "alpine-riscv64"
ANY = "any"
_OLD_STYLE_PREFIX = "Build\\ command\\ failed:"
_APK = re.compile(r"(^|[;&|]\s*)apk\s")
_APT = re.compile(r"(^|[;&|]\s*)(apt-get|apt)\s")
# A prefix of the go mod download noise that precedes most Go errors.
_GO_DOWNLOADS = (
    "go: downloading google.golang.org/protobuf v1.36.6\n"
    "go: downloading github.com/spf13/cobra v1.9.1\n"
    "go: downloading golang.org/x/sys v0.33.0\n"
)


def _load(role: str) -> List[Dict[str, Any]]:
    """Return the committed examples of one role."""
    path = EXAMPLES_DIR / f"{role}_examples.json"
    return json.loads(path.read_text())["examples"]


def _actions(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the fix actions of a fixer example."""
    fix = example.get("fix") or example.get("solution") or {}
    return list(fix.get("actions") or [])


def _commands(example: Dict[str, Any]) -> List[str]:
    """Return every command an example would make an agent run."""
    commands = [
        str(action.get("command") or "")
        for action in _actions(example)
        if isinstance(action, dict)
    ]
    plan = example.get("plan") or {}
    for phase in plan.get("phases") or example.get("phases") or []:
        commands.extend(str(c) for c in phase.get("commands") or [])
    return [c for c in commands if c.strip()]


FIXER = _load("fixer")
SCOUT = _load("scout")
DEBIAN_GO_SCOUT = [
    e
    for e in SCOUT
    if e.get("sandbox") == DEBIAN and e.get("build_system") == "go"
]


@pytest.fixture(autouse=True)
def _debian_run():
    """Run every test as the Debian sandbox with fresh memory."""
    saved = platforms._cached_profile
    platforms._cached_profile = platforms.DEBIAN_RISCV
    memory.MEMORY_INSTANCES.clear()
    try:
        yield
    finally:
        platforms._cached_profile = saved
        memory.MEMORY_INSTANCES.clear()


def _fixer_examples_for(error: str, build_system: str) -> list:
    """Return what the fixer prompt gets for this error on Debian."""
    agent_memory = memory.AgentMemory("fixer")
    return agent_memory.get_relevant_examples(
        {"build_system": build_system, "error_message": error},
        max_examples=2,
    )


def _ids(examples) -> List[str]:
    """Return the example IDs, for readable failure messages."""
    return [ex.id for ex in examples]


# ---------------------------------------------------------------------
# 1. Data quality: every example can reach a prompt and can be used.
# ---------------------------------------------------------------------


class TestFixerDataQuality:
    """Every committed fixer example must be usable by a real run."""

    @pytest.mark.parametrize("example", FIXER, ids=lambda e: e["id"])
    def test_example_names_its_sandbox(self, example) -> None:
        """An example names the sandbox its commands run on.

        An example with no sandbox counts as Alpine, so a Debian run
        never sees it. "any" marks commands that run on every distro.
        """
        assert example.get("sandbox") in (
            ALPINE,
            DEBIAN,
            ANY,
        ), f"{example['id']} has no sandbox; Debian runs never see it"

    @pytest.mark.parametrize("example", FIXER, ids=lambda e: e["id"])
    def test_example_has_actions(self, example) -> None:
        """Retrieval drops an example with no action, so it is dead."""
        assert _actions(example), f"{example['id']} holds no fix action"

    @pytest.mark.parametrize("example", FIXER, ids=lambda e: e["id"])
    def test_pattern_is_an_error_not_a_command_header(self, example) -> None:
        """The pattern must match the error, not the failed command.

        The old learner escaped the first 80 characters of "Build
        command failed: <command> (exit N) - ...", which keeps the
        command and cuts the error.
        """
        pattern = example.get("error_pattern") or ""
        assert pattern, f"{example['id']} has no error pattern"
        assert not pattern.startswith(
            _OLD_STYLE_PREFIX
        ), f"{example['id']} pattern is the command header: {pattern[:60]}"
        re.compile(pattern)

    @pytest.mark.parametrize(
        "example",
        [e for e in FIXER if e.get("source") == "auto"],
        ids=lambda e: e["id"],
    )
    def test_learned_example_keeps_the_error_line(self, example) -> None:
        """A learned example stores the decisive error line.

        A later run with the same error matches on this line.
        """
        context = example.get("error_context") or {}
        line = (context.get("error_message") or "").strip()
        assert (
            len(line) >= 24
        ), f"{example['id']} stores no usable error line: {line!r}"

    @pytest.mark.parametrize("example", FIXER, ids=lambda e: e["id"])
    def test_commands_fit_the_sandbox(self, example) -> None:
        """A Debian example must not run apk; an Alpine one, not apt.

        An example for any distro runs neither.
        """
        sandbox = example.get("sandbox") or ALPINE
        for command in _commands(example):
            if sandbox in (DEBIAN, ANY):
                assert not _APK.search(command), f"{example['id']}: {command}"
            if sandbox in (ALPINE, ANY):
                assert not _APT.search(command), f"{example['id']}: {command}"

    @pytest.mark.parametrize(
        "example",
        [e for e in FIXER if e.get("source") == "auto"],
        ids=lambda e: e["id"],
    )
    def test_learned_key_names_an_error(self, example) -> None:
        """A learned key is an error line, never Go's download noise.

        A "go: downloading" key matched every large Go build (the
        netmaker example reached 9 unrelated fixer prompts).
        """
        line = (example.get("error_context") or {}).get("error_message", "")
        assert has_error_line(line), f"{example['id']}: {line!r}"
        for text in (line, example.get("error_pattern") or ""):
            assert "go: downloading" not in text.replace(
                "\\", ""
            ), f"{example['id']} is keyed on download noise"


class TestScoutDataQuality:
    """Scout examples must carry the facts that tell layouts apart."""

    @pytest.mark.parametrize(
        "example", DEBIAN_GO_SCOUT, ids=lambda e: e["repo_name"]
    )
    def test_go_example_records_the_main_package(self, example) -> None:
        """A Go scout example records its main package.

        Without main_path, every Go example scores the same, and the
        first two in the file win every time.
        """
        trigger = example.get("trigger") or {}
        assert (
            "main_path" in trigger
        ), f"{example['repo_name']} trigger has no main_path: {trigger}"


# ---------------------------------------------------------------------
# 2. Fixer retrieval on Debian, with real errors from the run logs.
# ---------------------------------------------------------------------

# Errors that the committed data can fix. The fixer must get an example
# whose commands contain the needle, and the commands must run on Debian.
# Go is preinstalled in the sandbox image and the fixer must never install
# it with the package manager, so "go: command not found" has no data fix.
FIXABLE = [
    pytest.param(
        "Build command failed: go build -buildvcs=false . (exit 1) - "
        + _GO_DOWNLOADS
        + "# runtime/cgo\n"
        'cgo: C compiler "gcc" not found: exec: "gcc": executable file '
        "not found in $PATH",
        "go",
        "build-essential",
        id="cgo-gcc-missing",
    ),
    pytest.param(
        "Build command failed: cd build && make -j$(nproc) (exit 2) - "
        "In file included from /workspace/repos/massdns/src/main.c:7:\n"
        "/workspace/repos/massdns/src/net.h:12:10: fatal error: pcap.h: "
        "No such file or directory\ncompilation terminated.",
        "cmake",
        "libpcap-dev",
        id="pcap-header-missing",
    ),
    pytest.param(
        "Build command failed: make -j$(nproc) (exit 2) - "
        "/bin/sh: 1: bison: not found\n"
        "make[1]: *** [Makefile:40: emp_ematch.tab.c] Error 127",
        "make",
        "bison",
        id="bison-missing",
    ),
    pytest.param(
        "Build command failed: make -j$(nproc) (exit 2) - "
        "make[1]: xgettext: No such file or directory\n"
        "make[1]: *** [Makefile:70: po/whois.pot] Error 127",
        "make",
        "gettext",
        id="xgettext-missing",
    ),
    pytest.param(
        "Build command failed: meson setup builddir -DSKIP_TESTS=true "
        "(exit 1) - meson.build:120:2: ERROR: C header "
        "'sys/capability.h' not found",
        "meson",
        "libcap-dev",
        id="libcap-header-missing",
    ),
    pytest.param(
        "Build command failed: go build -buildvcs=false . (exit 1) - "
        + _GO_DOWNLOADS
        + "# github.com/anacrolix/mmsg/socket\n"
        "/root/go/pkg/mod/github.com/anacrolix/mmsg@v1.0.0/socket/"
        "socket.go:14:2: undefined: unix.Mmsghdr",
        "go",
        "anacrolix/mmsg",
        id="anacrolix-mmsg-riscv64",
    ),
]

# Real errors from the October 2026 native runs. No committed example
# fixes them, so the fixer must get no example at all.
UNRELATED = [
    pytest.param(
        "Build command failed: go build -buildvcs=false . (exit 1) - "
        + _GO_DOWNLOADS
        + "pkg/producers/kafka/kafkaProducer.go:47:2: no required module "
        "provides package github.com/jrnd-io/jr/pkg/types; to add it:\n"
        "\tgo get github.com/jrnd-io/jr/pkg/types",
        "go",
        id="jr-missing-module",
    ),
    pytest.param(
        "Build command failed: go build -buildvcs=false . (exit 1) - "
        + _GO_DOWNLOADS
        + "# github.com/turbot/steampipe/v2/pkg/filepaths\n"
        "pkg/filepaths/db_path.go:70:62: undefined: platform.Paths",
        "go",
        id="steampipe-no-riscv64-file",
    ),
    pytest.param(
        "Build command failed: go build -buildvcs=false ./cmd/vuls "
        "(exit 1) - " + _GO_DOWNLOADS + "package "
        "github.com/future-architect/vuls/cmd/vuls\n"
        "\timports encoding/json/v2: build constraints exclude all Go "
        "files in /usr/local/go/src/encoding/json/v2",
        "go",
        id="vuls-goexperiment",
    ),
    pytest.param(
        "Build command failed: go build -buildvcs=false . (exit 1) - "
        + _GO_DOWNLOADS
        + "# github.com/icodeface/tls/internal/cpu\n"
        "/root/go/pkg/mod/github.com/icodeface/tls@v0.0.0-20190904083142-"
        "17aec93c60e5/internal/cpu/cpu.go:15:30: undefined array length "
        "CacheLinePadSize or missing type constraint",
        "go",
        id="zombie-tls-fork",
    ),
    pytest.param(
        "Build command failed: go build -buildvcs=false -tags noweb -o "
        "gomuks ./cmd/gomuks (exit 1) - " + _GO_DOWNLOADS + "web/"
        "frontend.go:27:12: pattern all:dist: no matching files found",
        "go",
        id="gomuks-embed-dist",
    ),
    pytest.param(
        "Build command failed: cmake --build build (exit 2) - "
        "In file included from /workspace/repos/x/src/a.c:3:\n"
        "/workspace/repos/x/src/a.h:5:10: fatal error: zlib.h: No such "
        "file or directory",
        "cmake",
        id="c-zlib-header-is-not-pcap",
    ),
    pytest.param(
        "Build command failed: ./configure --without-libpsl (exit 1) - "
        "Package 'gnutls', required by 'virtual:world', not found",
        "autotools",
        id="wget-gnutls",
    ),
]


class TestFixerRetrieval:
    """The fixer gets the example that fixes the error, and only that."""

    @pytest.mark.parametrize("error, build_system, needle", FIXABLE)
    def test_fixable_error_gets_its_fix(
        self, error, build_system, needle
    ) -> None:
        examples = _fixer_examples_for(error, build_system)
        commands = [c for ex in examples for c in _commands(ex.raw)]
        assert any(
            needle in c for c in commands
        ), f"no example with {needle!r}; got {_ids(examples)}"
        for command in commands:
            assert not _APK.search(command), f"apk on Debian: {command}"

    @pytest.mark.parametrize("error, build_system", UNRELATED)
    def test_unrelated_error_gets_no_example(
        self, error, build_system
    ) -> None:
        examples = _fixer_examples_for(error, build_system)
        assert examples == [], f"unrelated examples: {_ids(examples)}"

    def test_go_download_lines_alone_match_nothing(self) -> None:
        """'go: downloading' lines are noise, never an error."""
        error = (
            "Build command failed: go build -buildvcs=false . (exit 1) - "
            + _GO_DOWNLOADS
        )
        examples = _fixer_examples_for(error, "go")
        assert examples == [], f"matched download noise: {_ids(examples)}"


# ---------------------------------------------------------------------
# 3. Scout retrieval follows the package layout.
# ---------------------------------------------------------------------


class TestScoutRetrieval:
    """Scout examples must depend on more than the build system."""

    def _scout_ids(self, context: Dict[str, Any]) -> List[str]:
        agent_memory = memory.AgentMemory("scout")
        return _ids(agent_memory.get_relevant_examples(context, 2))

    def test_go_layouts_get_different_examples(self) -> None:
        """Two different Go layouts get different examples.

        A root main package, and a cmd/ main in a module subdirectory
        with cgo, are different cases.
        """
        root = self._scout_ids(
            {
                "build_system": "go",
                "has_main": True,
                "main_path": ".",
                "module_dir": "",
            }
        )
        nested = self._scout_ids(
            {
                "build_system": "go",
                "has_main": True,
                "main_path": "./cmd/tool",
                "module_dir": "v1",
                "has_cgo": True,
            }
        )
        assert root and nested
        assert root != nested, f"both layouts got {root}"

    def test_same_package_plan_comes_first(self) -> None:
        """A verified plan of the same package is the best example."""
        agent_memory = memory.AgentMemory("scout")
        examples = agent_memory.get_relevant_examples(
            {"build_system": "go", "has_main": True, "repo_name": "restic"},
            2,
        )
        assert (
            examples and examples[0].repo_name == "restic"
        ), f"got {[ex.repo_name for ex in examples]}"


# ---------------------------------------------------------------------
# 4. The scout and fixer prompts as the workflow builds them.
# ---------------------------------------------------------------------


def _scout_state(repo_url: str, build_system: str = "go"):
    """Build a state that renders the scout prompt."""
    state = create_initial_state(repo_url)
    state.build_system_info = BuildSystemInfo(
        type=build_system,
        confidence=0.9,
        primary_file="go.mod",
        module_dir="",
    )
    state.dependencies = DependencyInfo(build_tools=[build_system])
    state.package_analysis = PackageAnalysis(
        purpose="test package",
        build_system=build_system,
        build_system_confidence=0.9,
        llm_grounded=True,
    )
    return state


def _scout_call(state) -> Dict[str, Any]:
    """Run scout_node with real retrieval; return the LLM call kwargs."""
    from src import graph

    seen: Dict[str, Any] = {}

    def fake_llm_call(**kwargs):
        seen.update(kwargs)
        return LLMCallOutcome(data=None, used_fallback=False, attempts=1)

    with (
        mock.patch.object(
            graph, "get_model_pool_for_role", return_value=[mock.MagicMock()]
        ),
        mock.patch.object(
            graph, "llm_call_with_validation", side_effect=fake_llm_call
        ),
        mock.patch.object(
            graph.scripted_ops,
            "get_system_info",
            return_value={"architecture": "riscv64", "gcc": "installed"},
        ),
        mock.patch.object(
            graph, "collect_build_evidence", return_value="build evidence"
        ),
        mock.patch.object(
            graph, "get_system_knowledge_summary", return_value="knowledge"
        ),
    ):
        graph.scout_node(state)
    return seen


def _fixer_state(repo_url: str, error: str, command: str, build_system: str):
    """Build a state that renders the fixer prompt for one failure."""
    state = create_initial_state(repo_url)
    state.build_plan = BuildPlan(
        build_system=build_system,
        build_system_confidence=0.9,
        phases=[BuildPhase(1, "build", [command])],
        total_estimated_duration="1m",
    )
    state.add_error(
        create_error_record(error, ErrorCategory.COMPILATION, command=command)
    )
    return state


def _fixer_call(state) -> Dict[str, Any]:
    """Run fixer_node with real retrieval; return the LLM call kwargs."""
    from src import graph

    seen: Dict[str, Any] = {}

    def fake_llm_call(**kwargs):
        seen.update(kwargs)
        return LLMCallOutcome(data=None, used_fallback=False, attempts=1)

    with (
        mock.patch.object(
            graph, "get_model_pool_for_role", return_value=[mock.MagicMock()]
        ),
        mock.patch.object(
            graph, "llm_call_with_validation", side_effect=fake_llm_call
        ),
        mock.patch.object(graph, "error_context_excerpts", return_value=""),
        mock.patch.object(graph, "top_level_listing", return_value="go.mod"),
    ):
        graph.fixer_node(state)
    return seen


class TestPromptsUseTheData:
    """The data reaches the real prompts, in the active distro."""

    def test_scout_prompt_has_debian_examples(self) -> None:
        """Wiring check: the scout prompt carries examples, none apk."""
        seen = _scout_call(_scout_state("https://github.com/a/tool.git"))
        prompt = seen["prompt"]
        assert "# Few-Shot Examples (scout)" in prompt
        block = prompt.split("# Few-Shot Examples (scout)", 1)[1]
        assert not _APK.search(block), "apk command in a Debian prompt"

    def test_fixer_prompt_has_the_fix_for_a_known_error(self) -> None:
        """Wiring check: a known error brings its fix into the prompt.

        The committed data holds a bison example (iproute2).
        """
        error = (
            "Build command failed: make -j$(nproc) (exit 2) - "
            "/bin/sh: 1: bison: not found\n"
            "make[1]: *** [Makefile:40: emp_ematch.tab.c] Error 127"
        )
        state = _fixer_state(
            "https://github.com/a/tool.git", error, "make -j$(nproc)", "make"
        )
        prompt = _fixer_call(state)["prompt"]
        assert "# Few-Shot Examples (fixer)" in prompt
        assert "bison" in prompt.split("# Few-Shot Examples (fixer)", 1)[1]

    def test_audit_log_names_the_scout_examples(self) -> None:
        """The call log names the examples that the prompt got.

        Without the names, the logs cannot measure the effect of data/.
        """
        seen = _scout_call(_scout_state("https://github.com/a/tool.git"))
        assert seen["audit_metadata"].get(
            "examples"
        ), f"audit metadata: {seen['audit_metadata']}"

    def test_audit_log_names_the_fixer_examples(self) -> None:
        error = (
            "Build command failed: make -j$(nproc) (exit 2) - "
            "/bin/sh: 1: bison: not found"
        )
        state = _fixer_state(
            "https://github.com/a/tool.git", error, "make -j$(nproc)", "make"
        )
        seen = _fixer_call(state)
        assert seen["audit_metadata"].get(
            "examples"
        ), f"audit metadata: {seen['audit_metadata']}"


# ---------------------------------------------------------------------
# 5. The recipe cache helps a forced (batch) or native run.
# ---------------------------------------------------------------------

# arc has a verified debian-riscv64 recipe: go.mod is in v1/ and the
# main package is ./cmd/arc. Only the cache knows this layout.
_ARC_URL = "https://github.com/jm33-m0/arc"
_ARC_STEP = "cd v1 && go mod tidy"


def test_arc_recipe_is_in_the_committed_cache() -> None:
    """Precondition for the recipe tests below."""
    cache = json.loads((DATA_DIR / "recipe_cache.json").read_text())
    recipe = cache["packages"]["arc"][DEBIAN]
    commands = [
        c for p in recipe["build_plan"]["phases"] for c in p["commands"]
    ]
    assert _ARC_STEP in commands


class TestRecipeCacheInTheWorkflow:
    """A batch run passes --force, so the cache must help the agents."""

    def test_scout_prompt_has_the_verified_plan(self) -> None:
        seen = _scout_call(_scout_state(_ARC_URL))
        assert (
            _ARC_STEP in seen["prompt"]
        ), "the scout re-plans arc without its verified plan"

    def test_fixer_prompt_has_the_verified_plan(self) -> None:
        error = (
            "Build command failed: go build -buildvcs=false . (exit 1) - "
            "go: cannot find main module, but found .git/config in "
            "/workspace/repos/arc"
        )
        state = _fixer_state(
            _ARC_URL, error, "go build -buildvcs=false .", "go"
        )
        prompt = _fixer_call(state)["prompt"]
        assert (
            _ARC_STEP in prompt
        ), "the fixer does not see the verified plan of arc"

    def test_native_run_uses_the_qemu_recipe_as_a_hint(
        self, monkeypatch
    ) -> None:
        """A native run sees the verified Debian QEMU plan.

        No native recipe exists for arc, but the source layout is the
        same on both targets.
        """
        from src import target

        monkeypatch.setenv("ATESOR_TARGET", "native")
        target.reset_target_cache()
        assert target.is_native()
        seen = _scout_call(_scout_state(_ARC_URL))
        assert _ARC_STEP in seen["prompt"]


# ---------------------------------------------------------------------
# 6. Learning writes data that a later run can match.
# ---------------------------------------------------------------------


def _learned(state) -> Dict[str, List[Dict[str, Any]]]:
    """Run _save_learning_data; return the payloads per role."""
    from src import graph

    with (
        mock.patch.object(graph, "save_learned_example") as save_example,
        mock.patch.object(graph, "save_to_recipe_cache"),
        mock.patch.object(
            graph, "get_model_for_role", return_value=mock.MagicMock()
        ),
        mock.patch(
            "src.artifact_curator.curate_artifacts",
            return_value=[{"filepath": "tool", "role": "primary"}],
        ),
    ):
        graph._save_learning_data(state)
    payloads: Dict[str, List[Dict[str, Any]]] = {}
    for call in save_example.call_args_list:
        payloads.setdefault(call.args[0], []).append(call.args[1])
    return payloads


def _verified_go_state():
    """A verified Go port with one successful fix."""
    state = create_initial_state("https://github.com/a/tool.git")
    state.build_plan = BuildPlan(
        build_system="go",
        build_system_confidence=0.9,
        phases=[
            BuildPhase(1, "build", ["go build -buildvcs=false ./cmd/tool"])
        ],
        total_estimated_duration="1m",
    )
    state.build_status = BuildStatus.SUCCESS
    state.artifact_verification = {"status": "verified"}
    state.curated_artifacts = [{"filepath": "tool", "role": "primary"}]
    state.context_cache["go_main_info"] = {
        "has_main": True,
        "main_path": "./cmd/tool",
        "module_dir": "v1",
    }
    state.add_fix_attempt(
        FixAttempt(
            error_category=ErrorCategory.DEPENDENCY,
            strategy="install libpcap headers",
            changes_made=[
                "Executed: apt-get install -y --no-install-recommends "
                "libpcap-dev"
            ],
            success=True,
            error_message=(
                "Build command failed: go build -buildvcs=false ./cmd/tool "
                "(exit 1) - " + _GO_DOWNLOADS + "# github.com/google/gopacket"
                "/pcap\npcap.go:30:10: fatal error: pcap/pcap.h: No such "
                "file or directory"
            ),
        )
    )
    return state


class TestLearningWritesMatchableData:
    """What a verified run saves must be found again by a later run."""

    def test_learned_fixer_example_keeps_the_real_error(self) -> None:
        fixer = _learned(_verified_go_state())["fixer"][0]
        line = fixer["error_context"]["error_message"]
        assert "pcap/pcap.h" in line, f"stored line: {line!r}"
        assert not fixer["error_pattern"].startswith(_OLD_STYLE_PREFIX)

    def test_learned_fixer_example_names_the_sandbox(self, tmp_path) -> None:
        payload = _learned(_verified_go_state())["fixer"][0]
        store = memory.AgentMemory("fixer", examples_dir=tmp_path)
        assert store.save_learned_example(payload)
        assert payload.get("sandbox") == DEBIAN

    def test_learned_scout_example_keeps_the_layout(self) -> None:
        scout = _learned(_verified_go_state())["scout"][0]
        trigger = scout["trigger"]
        assert trigger.get("main_path") == "./cmd/tool", trigger
        assert trigger.get("module_dir") == "v1", trigger

    def test_learned_fix_is_found_for_the_same_error(self, tmp_path) -> None:
        """A saved fix comes back for the same error.

        The new run has a different package and command.
        """
        payload = _learned(_verified_go_state())["fixer"][0]
        store = memory.AgentMemory("fixer", examples_dir=tmp_path)
        assert store.save_learned_example(payload)
        error = (
            "Build command failed: go build -buildvcs=false . (exit 1) - "
            + _GO_DOWNLOADS
            + "# github.com/google/gopacket/pcap\n"
            "pcap.go:30:10: fatal error: pcap/pcap.h: No such file or "
            "directory"
        )
        found = store.get_relevant_examples(
            {"build_system": "go", "error_message": error}, 2
        )
        assert [ex.repo_name for ex in found] == ["tool"]
