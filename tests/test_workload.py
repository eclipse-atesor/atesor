"""Tests for src/workload.py: package profile and workload choice."""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src import bench, workload
from src.state import (
    BuildPhase,
    BuildPlan,
    CommandResult,
    PackageAnalysis,
    create_initial_state,
)
from src.workload import (
    CHECK_PREFIX,
    PackageProfile,
    WorkloadContext,
    c_harness_build,
    check_program_proposal,
    choose_workload,
    profile_package,
    size_by_input,
    size_by_iterations,
    strategies,
)

REPO = "/workspace/repos/zlib"
ROOT = REPO + ".opt"
BASE = ROOT + "/base"


def _state(artifacts, build_system="cmake", tests=None, purpose="x"):
    state = create_initial_state("https://github.com/madler/zlib")
    state.build_plan = BuildPlan(
        build_system=build_system,
        build_system_confidence=0.9,
        phases=[BuildPhase(1, "build", ["cmake -S . -B build"])],
        total_estimated_duration="1m",
    )
    state.curated_artifacts = [
        {"filepath": f"{REPO}/{rel}", "type": kind, "role": "primary"}
        for rel, kind in artifacts
    ]
    state.package_tests = tests or {"status": "skipped"}
    state.package_analysis = PackageAnalysis(
        purpose=purpose,
        language="C",
        build_system=build_system,
        build_system_confidence=0.9,
        build_system_reasoning="",
        dependencies=[],
        riscv_risks=[],
        build_strategy="",
        expected_artifacts=[],
        needs_custom_plan=False,
        complexity=3,
    )
    return state


_LIBRARY = [("build/libz.so.1.3.2.1", "library_shared"),
            ("build/libz.a", "library_static")]


def _context(state, package=None, agent=None, kind="native", say=None):
    return WorkloadContext(
        state=state,
        package=package or profile_package(state, "c"),
        tree=BASE,
        env={"CC": f"{BASE}/.atesor/cc"},
        root=ROOT,
        kind=kind,
        agent=agent,
        say=say or (lambda text: None),
    )


def _runs(seconds, sha="s", check_lines=1, error=None, err=""):
    run = {"job": "B", "phase": "round", "wall_ns": int(seconds * 1e9),
           "sha": sha, "check_lines": check_lines,
           "rc": 1 if error else 0}
    if err:
        run["err"] = err
    return {"runs": [run, dict(run)], "error": error}


class TestPackageProfile(unittest.TestCase):
    """The artifacts of the verified build give the package type.

    The package type gives the order of the workload sources
    (``strategies``).
    """

    def test_library_with_test_programs(self) -> None:
        """Test programs are helpers; the libraries make a library."""
        state = _state(_LIBRARY + [
            ("build/test/minigzip", "binary"),
            ("build/examples/demo", "binary"),
        ])
        package = profile_package(state, "c")
        self.assertEqual(package.kind, "library")
        self.assertEqual(package.libraries,
                         ["build/libz.so.1.3.2.1", "build/libz.a"])
        self.assertEqual(package.programs, [])
        self.assertEqual(package.helpers,
                         ["build/test/minigzip", "build/examples/demo"])
        self.assertEqual(
            package.describe(),
            "C/C++ library (cmake): build/libz.so.1.3.2.1, build/libz.a",
        )

    def test_programs_and_cargo_output(self) -> None:
        """Cargo's build scripts and examples are not programs."""
        state = _state([
            ("target/release/rg", "binary"),
            ("target/release/build/x-1/build-script-build", "binary"),
            ("target/release/examples/demo", "binary"),
            ("target/release/deps/librg-1.so", "library_shared"),
        ], build_system="cargo")
        package = profile_package(state, "rust")
        self.assertEqual(package.programs, ["target/release/rg"])
        self.assertEqual(package.kind, "library+programs")
        self.assertEqual(len(package.helpers), 2)

    def test_purpose_without_analysis_is_empty(self) -> None:
        state = _state(_LIBRARY,
                       purpose="zlib (analysis unavailable — LLM starved)")
        self.assertEqual(profile_package(state, "c").purpose, "")
        empty = _state([])
        self.assertEqual(profile_package(empty, "c").kind, "unknown")

    def test_strategies_follow_the_type(self) -> None:
        cases = [
            (PackageProfile("c", "cmake", "library",
                            libraries=["a.so"], helpers=["t/x"]),
             ["harness", "helper-program", "tests"]),
            (PackageProfile("c", "make", "program", programs=["jq"]),
             ["program", "tests"]),
            (PackageProfile("go", "go", "program", programs=["gron"]),
             ["go-benchmarks", "program", "harness", "tests"]),
            (PackageProfile("rust", "cargo", "library+programs",
                            programs=["rg"], libraries=["x.so"]),
             ["program", "harness", "tests"]),
        ]
        for package, order in cases:
            with self.subTest(kind=package.kind, language=package.language):
                self.assertEqual(strategies(package), order)


class TestAgentReplies(unittest.TestCase):
    """The agent's answers are checked before anything runs."""

    def test_program_proposal(self) -> None:
        """The program must be built and the command must pass."""
        programs = ["build/zpipe", "build/example"]
        good = {"binary": "build/zpipe",
                "command": "{bin} -9 < {input} > {output}",
                "input_kind": "binary", "input_mb": 16,
                "description": "compress"}
        self.assertIsNone(check_program_proposal(good, programs))
        cases = {
            "not one of the built programs": dict(good, binary="/bin/sh"),
            "does not read {input}": dict(good, command="{bin} -9"),
            "input_kind must be one of": dict(good, input_kind="video"),
            "input_mb must be between": dict(good, input_mb=4096),
            "description is missing": dict(good, description=""),
            "';'": dict(good, command="{bin} < {input}; id"),
        }
        for reason, data in cases.items():
            with self.subTest(reason=reason):
                self.assertIn(reason, check_program_proposal(data, programs))

    def test_description_drops_the_requested_size(self) -> None:
        """The size the agent asked for goes; the real size is added."""
        cases = {
            ("Compress a 16 MB text file using xz.", 16):
                "Compress a text file using xz",
            ("Compresses 8 MiB of data in 4 KB blocks", 8):
                "Compresses data in 4 KB blocks",
            ("Encode 16MB of offsets", 16): "Encode offsets",
            ("Hash 4 MB of data", 8): "Hash 4 MB of data",
            ("Parse JSON records", None): "Parse JSON records",
        }
        for (text, size), expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(workload._description(
                    {"description": text, "input_mb": size}), expected)

    def test_source_rules(self) -> None:
        """No clocks, randomness, processes or network in a harness."""
        body = (f'int main(void) {{ printf("{CHECK_PREFIX} 1"); }}\n'
                "/* " + "padding " * 10 + "*/\n")
        self.assertIsNone(workload._source_problem(body, "c"))
        cases = {
            "c": ["srand(time(NULL));", "system(\"ls\");",
                  "open(\"/dev/urandom\")", "clock_gettime(0, &t);"],
            "go": ['import "math/rand"', "t := time.Now()"],
            "rust": ["std::process::Command::new(\"ls\")",
                     "let t = Instant::now();"],
        }
        for language, snippets in cases.items():
            for snippet in snippets:
                with self.subTest(language=language, snippet=snippet):
                    problem = workload._source_problem(body + snippet,
                                                       language)
                    self.assertIn("not allowed", problem)
        self.assertIn("never prints",
                      workload._source_problem("x" * 200, "c"))
        self.assertIn("complete file", workload._source_problem(None, "c"))
        self.assertIn("longer than", workload._source_problem(
            CHECK_PREFIX + "x" * workload.MAX_SOURCE_BYTES, "c"))
        # A name that only contains "time" is fine.
        self.assertIsNone(workload._source_problem(
            body + "runtime(x); timeout(y);", "c"))


class TestSizing(unittest.TestCase):
    """A candidate runs twice on the baseline and is sized there."""

    def test_iterations_scale_to_the_target(self) -> None:
        """0.02 s per run at N=1 becomes N=100 at about 2 s."""
        state = _state(_LIBRARY)
        harness = bench.Workload(mode="harness", command="{bin} {iters}",
                                 binary="h", check_prefix=CHECK_PREFIX)
        sessions = [_runs(0.02), _runs(1.9)]
        with mock.patch.object(bench, "run_jobs",
                               side_effect=sessions) as runner:
            self.assertIsNone(size_by_iterations(_context(state), harness))
        self.assertEqual(harness.iterations, 100)
        self.assertEqual(harness.run_seconds, 1.9)
        spec = runner.call_args_list[0].args[0]
        self.assertEqual(spec["check_prefix"], CHECK_PREFIX)
        self.assertEqual(spec["jobs"]["B"]["cmd"], f"{BASE}/h 1")

    def test_iteration_problems_go_back_to_the_agent(self) -> None:
        """Exit codes, missing check lines and drift are reported."""
        state = _state(_LIBRARY)
        harness = bench.Workload(mode="harness", command="{bin}",
                                 binary="h", check_prefix=CHECK_PREFIX)
        cases = [
            (_runs(1.0, error="B exited 1", err="inflate failed"),
             "inflate failed"),
            (_runs(1.0, check_lines=0), "printed no ATESOR-CHECK line"),
            ({"runs": [{"job": "B", "wall_ns": 10, "sha": "a",
                        "check_lines": 1},
                       {"job": "B", "wall_ns": 10, "sha": "b",
                        "check_lines": 1}], "error": None},
             "different results"),
            (_runs(9.0), "one iteration takes 9.0 s"),
        ]
        for session, reason in cases:
            with self.subTest(reason=reason), mock.patch.object(
                bench, "run_jobs", return_value=session
            ):
                self.assertIn(reason,
                              size_by_iterations(_context(state), harness))

    def test_a_baseline_crash_is_a_fault_of_the_port(self) -> None:
        """A signal in the baseline run names the verified build."""
        state = _state(_LIBRARY)
        harness = bench.Workload(mode="harness", command="{bin}",
                                 binary="h", check_prefix=CHECK_PREFIX)
        session = {
            "runs": [{"job": "B", "phase": "round", "wall_ns": 10,
                      "rc": 132, "err": "Illegal instruction"}],
            "error": "B exited 132",
        }
        ctx = _context(state)
        with mock.patch.object(bench, "run_jobs", return_value=session):
            problem = size_by_iterations(ctx, harness)
        self.assertIn("the verified build crashed with SIGILL", problem)
        self.assertIn("on this machine", problem)
        self.assertEqual(ctx.crashes, [problem])

    def test_a_run_that_passes_and_complains_is_a_harness_bug(self) -> None:
        """Exit 0 with stderr means the error path reports success."""
        state = _state(_LIBRARY)
        harness = bench.Workload(mode="harness", command="{bin}",
                                 binary="h", check_prefix=CHECK_PREFIX)
        run = {"job": "B", "phase": "round", "wall_ns": 10, "rc": 0,
               "sha": "a", "check_lines": 1,
               "err_tail": "opus_encode failed: invalid argument"}
        session = {"runs": [run, dict(run)], "error": None}
        ctx = _context(state)
        with mock.patch.object(bench, "run_jobs", return_value=session):
            problem = size_by_iterations(ctx, harness)
        self.assertIn("it exited 0 and still wrote an error to stderr",
                      problem)
        self.assertIn("opus_encode failed", problem)
        self.assertEqual(ctx.crashes, [])

    def test_the_workload_check_keeps_stderr(self) -> None:
        """The check asks the runner for the stderr of a passing run."""
        state = _state(_LIBRARY)
        harness = bench.Workload(mode="harness", command="{bin}",
                                 binary="h", check_prefix=CHECK_PREFIX)
        with mock.patch.object(bench, "run_jobs",
                               return_value=_runs(1.0)) as runner:
            size_by_iterations(_context(state), harness)
        self.assertTrue(runner.call_args.args[0]["keep_stderr"])

    def test_qemu_needs_one_iteration(self) -> None:
        """QEMU measures no time: two equal runs at N=1 are enough."""
        state = _state(_LIBRARY)
        harness = bench.Workload(mode="harness", command="{bin}",
                                 binary="h", check_prefix=CHECK_PREFIX)
        with mock.patch.object(bench, "run_jobs",
                               return_value=_runs(0.01)) as runner:
            self.assertIsNone(
                size_by_iterations(_context(state, kind="qemu"), harness)
            )
        self.assertEqual(harness.iterations, 1)
        self.assertEqual(runner.call_count, 1)

    def test_program_input_scales(self) -> None:
        """A 0.1 s run gets 20 times the input, then fits."""
        state = _state(_LIBRARY)
        program = bench.Workload(mode="program", command="{bin} < {input}",
                                 binary="build/zpipe", input_kind="binary",
                                 input_mb=16)
        with mock.patch.object(workload, "_sh", return_value=CommandResult(
                "x", 0, "", "", 0.1)), \
                mock.patch.object(bench, "write_input",
                                  return_value=None) as writer, \
                mock.patch.object(bench, "run_jobs",
                                  side_effect=[_runs(0.1), _runs(2.0)]):
            self.assertIsNone(size_by_input(_context(state), program))
        self.assertEqual(program.input_mb, 320)
        self.assertEqual(writer.call_count, 2)
        slow = bench.Workload(mode="program", command="{bin}",
                              binary="build/zpipe")
        with mock.patch.object(bench, "run_jobs",
                               return_value=_runs(30.0)):
            self.assertIn("one run takes 30.0 s",
                          size_by_input(_context(state), slow))

    def test_the_smallest_input_asks_for_less_work_per_byte(self) -> None:
        state = _state(_LIBRARY)
        program = bench.Workload(mode="program", command="{bin} -c {input}",
                                 binary="build/zpipe", input_kind="text",
                                 input_mb=16)
        with mock.patch.object(workload, "_sh", return_value=CommandResult(
                "x", 0, "", "", 0.1)), \
                mock.patch.object(bench, "write_input", return_value=None), \
                mock.patch.object(bench, "run_jobs",
                                  return_value=_runs(7.4)):
            problem = size_by_input(_context(state), program)
        self.assertEqual(program.input_mb, 1)
        self.assertEqual(
            problem,
            "one run takes 7.4 s with the smallest input (1 MiB); the limit "
            "is 5 s, so the command must do less work per byte (for example "
            "a faster mode or level of the program)",
        )


class TestChoice(unittest.TestCase):
    """The agent rounds, the prompts and the order of the workload sources.

    It also covers the error feedback and the report line (``describe``).
    """

    def test_harness_errors_go_back_to_the_agent(self) -> None:
        """A compile error reaches the next prompt; round 2 works."""
        state = _state(_LIBRARY)
        api = {"symbols": ["compress2", "uncompress"], "cxx": False,
               "headers": ["zlib.h"], "include_dirs": [".", "build"],
               "prototypes": ["int compress2(Bytef *d, uLongf *n, "
                              "const Bytef *s, uLong m, int l);"]}
        source = (
            '#include "zlib.h"\nint main(int c, char **v) { compress2(0, 0,'
            f' 0, 0, 9); printf("{CHECK_PREFIX} %d\\n", 1); return 0; }}\n'
        )
        answer = {"input_kind": "binary", "input_mb": 8,
                  "description": "compress and uncompress",
                  "source": source}
        prompts = []

        def agent(phase, prompt, check):
            prompts.append(prompt)
            self.assertEqual(phase, "harness-c")
            self.assertIsNone(check(answer))
            self.assertIn("calls no function", check(
                dict(answer, source=f'int main() {{ puts("{CHECK_PREFIX}");'
                                    ' }\n/* ' + "padding " * 10 + "*/")))
            return dict(answer), ""

        builds = iter([
            "`cc` exited 1:\nharness.c:3: error: 'Bytef' undeclared",
            None,
        ])
        said = []
        with mock.patch.object(workload, "_run_program",
                               return_value=(api, "")), \
                mock.patch.object(workload, "install_workload",
                                  side_effect=lambda *a: next(builds)), \
                mock.patch.object(workload, "size_by_iterations",
                                  return_value=None):
            chosen, notes = choose_workload(
                _context(state, agent=agent, say=said.append)
            )
        self.assertEqual(notes, [])
        self.assertEqual(chosen.mode, "harness")
        self.assertEqual(chosen.source_language, "c")
        self.assertEqual(chosen.files, {".atesor/bench/harness.c": source})
        self.assertEqual(chosen.check_prefix, CHECK_PREFIX)
        self.assertEqual(len(prompts), 2)
        self.assertIn("int compress2(", prompts[0])
        self.assertIn('#include "zlib.h"', prompts[0])
        self.assertIn("'Bytef' undeclared", prompts[1])
        self.assertIn("Your previous answer", prompts[1])
        self.assertTrue(any("round 1: the harness did not build" in line
                            for line in said))

    def test_rounds_end_and_the_tests_follow(self) -> None:
        """After three failed rounds the package tests are tried."""
        state = _state(_LIBRARY, tests={
            "status": "passed", "framework": "ctest",
            "command": f"ctest --test-dir {REPO}/build",
        })
        api = {"symbols": ["deflate"], "cxx": False, "headers": ["zlib.h"],
               "include_dirs": ["."], "prototypes": ["int deflate(z_streamp"
                                                     " s, int f);"]}
        answer = {"input_kind": None, "description": "d",
                  "source": f"int main() {{ deflate(0, 0); puts(\""
                            f"{CHECK_PREFIX}\"); }}\n/* "
                            + "padding " * 10 + "*/"}
        agent = mock.Mock(return_value=(answer, ""))
        timed = {"runs": [{"job": "B", "wall_ns": 3_000_000_000}],
                 "error": None}
        with mock.patch.object(workload, "_run_program",
                               return_value=(api, "")), \
                mock.patch.object(workload, "install_workload",
                                  return_value="`cc` exited 1: boom"), \
                mock.patch.object(bench, "run_jobs",
                                  return_value=timed) as runner:
            chosen, notes = choose_workload(_context(state, agent=agent))
        self.assertEqual(agent.call_count, workload.AGENT_ROUNDS)
        self.assertEqual(chosen.mode, "tests")
        self.assertTrue(chosen.weak)
        self.assertEqual(chosen.run_seconds, 3.0)
        self.assertIn("3 rounds failed", notes[0])
        # The port's test command names the verified tree; the run
        # uses the baseline copy instead.
        job = runner.call_args.args[0]["jobs"]["B"]
        self.assertEqual(job["cmd"], f"ctest --test-dir {BASE}/build")

    def test_no_fitting_program_skips_to_the_next_source(self) -> None:
        state = _state([("build/cjpeg", "binary")], tests={
            "status": "passed", "framework": "ctest",
            "command": f"ctest --test-dir {REPO}/build",
        })
        answer = {"binary": None, "reason": "each program needs an image"}
        phases = []

        def agent(phase, prompt, check):
            phases.append(phase)
            self.assertIn('"binary": null', prompt)
            self.assertIn("The smallest input is 1 MiB", prompt)
            self.assertIsNone(check(answer))
            self.assertIn("reason", check({"binary": None}))
            return dict(answer), ""

        timed = {"runs": [{"job": "B", "wall_ns": 3_000_000_000}],
                 "error": None}
        said = []
        with mock.patch.object(workload, "_sh", return_value=CommandResult(
                "x", 0, "usage: cjpeg [switches] [inputfile]\n", "", 0.1)), \
                mock.patch.object(bench, "run_jobs", return_value=timed):
            chosen, notes = choose_workload(
                _context(state, agent=agent, say=said.append))
        self.assertEqual(phases, ["workload-program"])
        self.assertIn("the optimizer agent finds no fitting run: each "
                      "program needs an image", said)
        self.assertTrue(any("no fitting run: each program needs an image"
                            in note for note in notes))
        self.assertEqual(chosen.mode, "tests")

    def _harness_rounds(self, errors):
        state = _state(_LIBRARY)
        api = {"symbols": ["compress2"], "cxx": False, "headers": ["zlib.h"],
               "include_dirs": ["."],
               "prototypes": ["int compress2(Bytef *d, uLongf *n, "
                              "const Bytef *s, uLong m, int l);"]}
        answer = {"input_kind": "binary", "input_mb": 8, "description": "d",
                  "source": '#include "zlib.h"\nint main(void) { return 0; }'}
        agent = mock.Mock(return_value=(answer, ""))
        with mock.patch.object(workload, "_run_program",
                               return_value=(api, "")), \
                mock.patch.object(workload, "install_workload",
                                  return_value=None), \
                mock.patch.object(workload, "size_by_iterations",
                                  side_effect=errors):
            chosen, notes = choose_workload(_context(state, agent=agent))
        return agent.call_count, chosen, notes

    def test_rounds_go_on_while_each_error_is_new(self) -> None:
        errors = [
            f"the baseline run exited 1 in the round phase: TJPARAM_{name} "
            "must be specified"
            for name in ("QUALITY", "SUBSAMP", "COLORSPACE", "DENSITY")
        ]
        calls, chosen, _ = self._harness_rounds(errors + [None])
        # One round per answer: the loop did not stop at AGENT_ROUNDS.
        self.assertEqual(calls, len(errors) + 1)
        self.assertEqual(chosen.mode, "harness")
        # An error that repeats after round 3 shows no progress.
        calls, chosen, notes = self._harness_rounds(
            ["one run takes 7.5 s", "it printed no line",
             "it printed no line", "x", "y"])
        self.assertEqual(calls, workload.AGENT_ROUNDS)
        self.assertIsNone(chosen)
        self.assertTrue(any("3 rounds failed" in note for note in notes))
        # The same error with other numbers (7.5 s, 7.4 s) is no
        # progress either.
        calls, _, _ = self._harness_rounds(
            ["one run takes 7.5 s", "one run takes 7.4 s",
             "one run takes 7.3 s", "c", "d"])
        self.assertEqual(calls, workload.AGENT_ROUNDS)

    def test_only_helpers_that_use_the_library_run(self) -> None:
        state = _state(_LIBRARY + [("build/tests/md5cmp", "binary"),
                                   ("build/tests/tjbench", "binary")])
        ctx = _context(state)
        self.assertEqual(ctx.package.helpers,
                         ["build/tests/md5cmp", "build/tests/tjbench"])
        uses = {"exported": 40, "uses": {"build/tests/md5cmp": 0,
                                         "build/tests/tjbench": 3}}
        said = []
        ctx.say = said.append
        with mock.patch.object(workload, "_run_program",
                               return_value=(uses, "")) as program, \
                mock.patch.object(workload, "_program_workload",
                                  return_value=(None, "x")) as runs:
            workload._helper_workload(ctx)
        self.assertEqual(program.call_args.args[1]["helpers"],
                         ["build/tests/md5cmp", "build/tests/tjbench"])
        runs.assert_called_once_with(ctx, ["build/tests/tjbench"])
        self.assertIn("1 of 2 test programs do not use the package's "
                      "library; they are left out", said)
        none = dict(uses, uses={"build/tests/md5cmp": 0,
                                "build/tests/tjbench": 0})
        with mock.patch.object(workload, "_run_program",
                               return_value=(none, "")):
            self.assertEqual(
                workload._helper_workload(ctx),
                (None, "no test or example program uses the package's "
                       "library"),
            )

    def test_harness_prompts_say_the_input_is_raw_bytes(self) -> None:
        """C, Go and Rust prompts: generated bytes, not the library format."""
        prompt = workload.c_harness_prompt(
            PackageProfile("c", "cmake", "library",
                           libraries=["build/libx.so"]),
            "x", {"headers": ["x.h"], "prototypes": ["int f(void);"]}, "c",
        )
        self.assertIn(workload._INPUT_RULE, prompt)
        self.assertIn("never a file in the library's own format",
                      workload._INPUT_RULE)
        self.assertIn("never require an exact size", workload._INPUT_RULE)
        prompts = []

        def capture(ctx, phase, prompt, check, attempt):
            prompts.append((phase, prompt))
            return None, "captured"

        ctx = _context(_state(_LIBRARY))
        crate = {"crate": "x", "package": "x", "src": "src/lib.rs",
                 "dir": ctx.tree + "/"}
        with mock.patch.object(workload, "_agent_rounds",
                               side_effect=capture), \
                mock.patch.object(workload, "_go_packages", return_value=[
                    {"import": "example.com/m/x", "name": "x",
                     "files": 3}]), \
                mock.patch.object(workload, "_rust_crate",
                                  return_value=(crate, "")), \
                mock.patch.object(workload, "_sh", return_value=CommandResult(
                    "x", 0, "", "", 0.1)):
            workload._go_harness(ctx)
            workload._rust_harness(ctx)
        self.assertEqual([phase for phase, _ in prompts],
                         ["harness-go", "harness-rust"])
        for phase, text in prompts:
            with self.subTest(phase=phase):
                self.assertEqual(text.count(workload._INPUT_RULE), 1)
        self.assertEqual(prompts[0][1].count("- Check every result and "
                                             "call b.Fatal"), 1)
        for phase, text in prompts:
            with self.subTest(phase=phase):
                self.assertEqual(text.count(workload._VALUES_RULE), 1)

    def test_the_c_prompt_forbids_a_check_line_on_an_error_path(self) -> None:
        """The C harness must leave with 1 and set every variable."""
        prompt = workload.c_harness_prompt(
            PackageProfile("c", "cmake", "library",
                           libraries=["build/libx.so"]),
            "x", {"headers": ["x.h"], "prototypes": ["int f(void);"]}, "c",
        )
        self.assertEqual(prompt.count(workload._ERROR_RULE), 1)
        self.assertEqual(prompt.count(workload._VALUES_RULE), 1)
        self.assertIn("leave at once with status 1", workload._ERROR_RULE)
        self.assertIn(f"Never print the {CHECK_PREFIX} line on an error "
                      "path", workload._ERROR_RULE)
        self.assertIn("A jump over a declaration leaves the variable unset",
                      workload._ERROR_RULE)
        self.assertIn("Pass only values that the API accepts",
                      workload._VALUES_RULE)

    def test_the_c_prompt_forbids_a_header_name_from_the_library(
        self,
    ) -> None:
        """A library libwebp.so does not give a header libwebp.h."""
        prompt = workload.c_harness_prompt(
            PackageProfile("c", "cmake", "library",
                           libraries=["build/libwebp.so.7"]),
            "libwebp",
            {"headers": ["webp/encode.h", "webp/decode.h"],
             "prototypes": ["int WebPGetEncoderVersion(void);"]},
            "c",
        )
        self.assertIn('#include "encode.h"', prompt)
        self.assertIn("Copy those #include lines as they stand", prompt)
        self.assertIn("a library libfoo.so does not give a header "
                      '"libfoo.h"', prompt)
        self.assertIn("Include no other header of this library", prompt)

    def test_the_prompts_forbid_an_equality_check_on_a_lossy_api(
        self,
    ) -> None:
        """A lossy codec never returns its input, so equality is wrong."""
        prompt = workload.c_harness_prompt(
            PackageProfile("c", "cmake", "library",
                           libraries=["build/libopus.a"]),
            "opus", {"headers": ["opus.h"], "prototypes": ["int f(void);"]},
            "c",
        )
        self.assertEqual(prompt.count(workload._LOSSY_RULE), 1)
        self.assertIn("never equal to it", workload._LOSSY_RULE)
        self.assertIn("Never compare the result with the input for "
                      "equality", workload._LOSSY_RULE)
        prompts = []

        def capture(ctx, phase, prompt, check, attempt):
            prompts.append((phase, prompt))
            return None, "captured"

        ctx = _context(_state(_LIBRARY))
        crate = {"crate": "x", "package": "x", "src": "src/lib.rs",
                 "dir": ctx.tree + "/"}
        with mock.patch.object(workload, "_agent_rounds",
                               side_effect=capture), \
                mock.patch.object(workload, "_go_packages", return_value=[
                    {"import": "example.com/m/x", "name": "x",
                     "files": 3}]), \
                mock.patch.object(workload, "_rust_crate",
                                  return_value=(crate, "")), \
                mock.patch.object(workload, "_sh", return_value=CommandResult(
                    "x", 0, "", "", 0.1)):
            workload._go_harness(ctx)
            workload._rust_harness(ctx)
        for phase, text in prompts:
            with self.subTest(phase=phase):
                self.assertEqual(text.count(workload._LOSSY_RULE), 1)

    def test_an_error_that_came_back_stops_the_rounds(self) -> None:
        """A repeat of any earlier error is no progress, not only the last."""
        # A, B, A: round 3 brings back round 1's error, so the loop
        # stops there instead of paying for a fourth round.
        calls, chosen, notes = self._harness_rounds(
            ["Decoded PCM does not match", "too many arguments to f",
             "Decoded PCM does not match", "x", "y", "z"])
        self.assertEqual(calls, 3)
        self.assertIsNone(chosen)
        self.assertTrue(any("3 rounds failed" in note for note in notes))
        # Before AGENT_ROUNDS a repeat still gets its round.
        calls, _, _ = self._harness_rounds(
            ["same error", "same error", None])
        self.assertEqual(calls, 3)

    def test_every_round_sees_the_errors_of_the_rounds_before(self) -> None:
        """A later answer must not bring an earlier error back."""
        state = _state(_LIBRARY)
        api = {"symbols": ["compress2"], "cxx": False, "headers": ["zlib.h"],
               "include_dirs": ["."],
               "prototypes": ["int compress2(Bytef *d, uLongf *n);"]}
        answer = {"input_kind": "binary", "input_mb": 8, "description": "d",
                  "source": '#include "zlib.h"\nint main(void) '
                            '{ compress2(0, 0); return 0; }'}
        agent = mock.Mock(return_value=(answer, ""))
        errors = ["TJPARAM_QUALITY must be specified",
                  "TJPARAM_SUBSAMP must be specified",
                  "TJPARAM_COLORSPACE must be specified",
                  None]
        with mock.patch.object(workload, "_run_program",
                               return_value=(api, "")), \
                mock.patch.object(workload, "install_workload",
                                  return_value=None), \
                mock.patch.object(workload, "size_by_iterations",
                                  side_effect=errors):
            chosen, _ = choose_workload(_context(state, agent=agent))
        self.assertIsNotNone(chosen)
        self.assertEqual(agent.call_count, 4)
        prompts = [call.args[1] for call in agent.call_args_list]
        # Round 1 gets the plain prompt; every later round gets the
        # errors of all the rounds before it.
        self.assertNotIn("must not bring any of them back", prompts[0])
        self.assertIn("TJPARAM_QUALITY", prompts[1])
        for name in ("TJPARAM_QUALITY", "TJPARAM_SUBSAMP"):
            with self.subTest(name=name):
                self.assertIn(name, prompts[3])
        self.assertIn("must not bring any of them back", prompts[3])

    def test_a_converging_harness_gets_more_than_five_rounds(self) -> None:
        """An API with many required parameters needs a round for each."""
        errors = [f"TJPARAM_{name} must be specified" for name in
                  ("A", "B", "C", "D", "E", "F", "G")]
        calls, chosen, notes = self._harness_rounds(errors + [None])
        self.assertEqual(calls, 8)
        self.assertEqual(calls, workload.AGENT_MAX_ROUNDS)
        self.assertIsNotNone(chosen)
        # One more new error than the cap allows still stops the rounds.
        # The names differ by a letter: _error_key strips numbers, so
        # digits here would read as the same error twice.
        calls, chosen, notes = self._harness_rounds(
            [f"TJPARAM_{name} must be specified" for name in "ABCDEFGHI"]
        )
        self.assertEqual(calls, workload.AGENT_MAX_ROUNDS)
        self.assertIsNone(chosen)
        self.assertTrue(any("8 rounds failed" in note for note in notes))

    def test_without_an_llm_only_the_package_sources_remain(self) -> None:
        """No agent: the notes say so for each agent source."""
        state = _state(_LIBRARY + [("build/test/minigzip", "binary")])
        with mock.patch.object(workload, "_run_program",
                               return_value=({"symbols": ["a"],
                                              "prototypes": ["int a();"],
                                              "headers": ["a.h"],
                                              "include_dirs": ["."]}, "")):
            chosen, notes = choose_workload(_context(state, agent=None))
        self.assertIsNone(chosen)
        self.assertEqual([note.split(":")[0] for note in notes],
                         ["harness", "helper-program", "tests"])
        self.assertIn("no LLM", notes[0])

    def test_a_crash_in_one_source_does_not_stop_the_rest(self) -> None:
        """An exception in one source becomes a note."""
        state = _state([("build/jq", "binary")], tests={
            "status": "passed", "framework": "make",
            "command": "make check",
        })
        timed = {"runs": [{"job": "B", "wall_ns": 1_000_000_000}],
                 "error": None}
        with mock.patch.object(workload, "_program_workload",
                               side_effect=RuntimeError("boom")), \
                mock.patch.object(bench, "run_jobs", return_value=timed):
            chosen, notes = choose_workload(_context(state, agent=None))
        self.assertEqual(chosen.mode, "tests")
        self.assertIn("RuntimeError: boom", notes[0])

    def test_program_run_with_help_text(self) -> None:
        """The prompt shows --help; a good answer is sized and kept."""
        state = _state([("build/jq", "binary")])
        answer = {"binary": "build/jq", "command": "{bin} . < {input}",
                  "input_kind": "json", "input_mb": 16,
                  "description": "parse and print JSON"}
        prompts = []

        def agent(phase, prompt, check):
            prompts.append((phase, prompt))
            self.assertIsNone(check(answer))
            return dict(answer), ""

        with mock.patch.object(workload, "_sh", return_value=CommandResult(
                "x", 0, "Usage: jq [OPTIONS] FILTER [FILES...]\n", "", 0.1)), \
                mock.patch.object(workload, "size_by_input",
                                  return_value=None):
            chosen, notes = choose_workload(_context(state, agent=agent))
        self.assertEqual(chosen.mode, "program")
        self.assertEqual(chosen.binary, "build/jq")
        self.assertEqual(prompts[0][0], "workload-program")
        self.assertIn("Usage: jq [OPTIONS]", prompts[0][1])

    def test_go_benchmarks_come_first(self) -> None:
        """The package with most benchmarks runs in its own directory."""
        state = _state([("bin/gron", "binary")], build_system="go")
        package = profile_package(state, "go")
        replies = {
            "-list": CommandResult("x", 0, (
                "BenchmarkA\nBenchmarkB\nok  \texample.com/m/pkg\t0.1s\n"
                "?   \texample.com/m/cmd\t[no test files]\n"
                "BenchmarkC\nok  \texample.com/m/other\t0.1s\n"
            ), "", 0.1),
            "list": CommandResult("x", 0, f"{BASE}/pkg\n", "", 0.1),
        }

        def fake_sh(command, cwd, timeout=300, env=None):
            return replies["-list" if "-list" in command else "list"]

        with mock.patch.object(workload, "_sh", side_effect=fake_sh), \
                mock.patch.object(workload, "install_workload",
                                  return_value=None) as install, \
                mock.patch.object(bench, "run_jobs",
                                  side_effect=[_runs(0.01, check_lines=None),
                                               _runs(2.0, check_lines=None)]):
            chosen, notes = choose_workload(
                _context(state, package=package)
            )
        self.assertEqual(notes, [])
        self.assertEqual(chosen.mode, "gobench")
        self.assertEqual(chosen.origin, "upstream")
        self.assertEqual(chosen.package, "example.com/m/pkg")
        self.assertEqual(chosen.workdir, "pkg")
        self.assertEqual(chosen.iterations, 200)
        self.assertIn("-test.benchtime {iters}x", chosen.command)
        self.assertFalse(chosen.deterministic)
        self.assertEqual(chosen.build,
                         ["go test -c -o atesor-bench.test "
                          "example.com/m/pkg"])
        self.assertEqual(install.call_args.args[1], BASE)

    def test_feedback_puts_the_error_first(self) -> None:
        """The agent reads the error before its own old answer."""
        prompt = workload._feedback(
            "TASK", {"source": "int main"}, "harness.c:3: error: x"
        )
        self.assertLess(prompt.index("did not work:\nharness.c:3: error"),
                        prompt.index("Your previous answer was:"))
        problem = (
            "the harness did not build: `mkdir -p x && \"$CC\" -O2` "
            "exited 1:\nIn function 'main':\n"
            "harness.c:12:5: error: too few arguments to function "
            "'deflate'\n"
        )
        self.assertEqual(
            workload._headline(problem),
            "harness.c:12:5: error: too few arguments to function 'deflate'",
        )
        self.assertEqual(workload._headline("one line"), "one line")
        self.assertEqual(workload._headline(""), "")

    def test_describe(self) -> None:
        """One line names the source, the work and the run time."""
        harness = bench.Workload(mode="harness", command="x",
                                 description="deflate and inflate",
                                 input_kind="binary", input_mb=8,
                                 iterations=14, run_seconds=1.92,
                                 source_language="c", origin="agent")
        self.assertEqual(
            workload.describe(harness),
            "C harness by the optimizer agent: deflate and inflate, 8 MiB "
            "binary input, N=14, 1.92 s per run",
        )
        tests = bench.Workload(mode="tests", command="ctest",
                               description="the package tests (ctest)",
                               origin="tests", run_seconds=3.0)
        self.assertEqual(
            workload.describe(tests),
            "package tests (weak signal) from the package: the package "
            "tests (ctest), 3.00 s per run",
        )


@unittest.skipUnless(shutil.which("gcc") and shutil.which("nm")
                     and shutil.which("ldd"), "needs gcc, nm and ldd")
class TestTreePaths(unittest.TestCase):
    """Artifacts in hash directories: every step uses the tree's path."""

    VERIFIED = "programs/cachedObjs/3a297cdcdda21e64139b4c81136d85cf/lz4"
    BASE_REL = "programs/cachedObjs/c3d2b3f18a5d80ee805db8ec9fe59945/lz4"
    LIB = "lib/cachedObjs/c325c18bd8879e9a60f926c1c986eb23/liblz4.so.1.10.0"
    LIB_BASE = "lib/cachedObjs/0b1e2c3d4f5a6b7c8d9e0f1a2b3c4d5e/liblz4.so.1"

    def _program(self):
        return bench.Workload(mode="program",
                              command="{bin} -q -f {input} {output}",
                              binary=self.VERIFIED, input_kind="text",
                              input_mb=4)

    def test_jobs_name_the_tree_path_and_the_input(self) -> None:
        job = workload.baseline_job(self._program(), "/w/t", "/w/in.text",
                                    "/w/out", None,
                                    {self.VERIFIED: self.BASE_REL})
        self.assertEqual(job["cmd"],
                         f"/w/t/{self.BASE_REL} -q -f /w/in.text /w/out")
        self.assertEqual(job["input"], "/w/in.text")
        tests = bench.Workload(mode="tests", command="make check")
        self.assertNotIn("input", workload.baseline_job(tests, "/w/t", None,
                                                        None))

    def test_base_tree_runs_use_its_paths(self) -> None:
        """Sizing runs, the API scan and --help use the base paths."""
        state = _state([(self.VERIFIED, "binary"),
                        (self.LIB, "library_shared")])
        ctx = _context(state)
        ctx.paths = {self.VERIFIED: self.BASE_REL, self.LIB: self.LIB_BASE}
        with mock.patch.object(bench, "run_jobs",
                               return_value=_runs(1.0)) as runner:
            seconds, problem = workload._two_runs(ctx, self._program(),
                                                  "/w/in.text")
        self.assertEqual(problem, "")
        job = runner.call_args.args[0]["jobs"]["B"]
        self.assertTrue(job["cmd"].startswith(f"{BASE}/{self.BASE_REL} "))
        self.assertEqual(job["input"], "/w/in.text")
        with mock.patch.object(workload, "_run_program", return_value=(
                {"symbols": ["f"], "prototypes": ["int f(void);"]}, "",
        )) as program:
            workload.library_api(ctx)
        self.assertEqual(program.call_args.args[1]["libraries"],
                         [self.LIB_BASE])
        with mock.patch.object(workload, "_sh", return_value=CommandResult(
                "x", 0, "usage: lz4\n", "", 0.1)) as shell:
            workload._help_texts(ctx, [self.VERIFIED])
        self.assertIn(f"{BASE}/{self.BASE_REL} --help",
                      shell.call_args.args[0])

    def test_install_builds_with_the_tree_paths(self) -> None:
        """Each {path:REL} in a harness build becomes the tree's file."""
        harness = bench.Workload(
            mode="harness", command="{bin}", binary=".atesor/bench/h",
            build=[f"cc h.c {bench.path_word(self.LIB)} -o .atesor/bench/h"],
        )
        with mock.patch.object(workload, "write_files", return_value=None), \
                mock.patch.object(workload, "_sh", return_value=CommandResult(
                    "x", 0, "", "", 0.1)) as shell:
            self.assertIsNone(workload.install_workload(
                harness, "/w/t", {}, 60, {self.LIB: self.LIB_BASE}
            ))
        self.assertEqual(shell.call_args.args[0],
                         f"cc h.c {self.LIB_BASE} -o .atesor/bench/h")
        self.assertEqual(shell.call_args.args[1], "/w/t")


class TestHarnessKitOnThisHost(unittest.TestCase):
    """The API scan, the helper scan and the harness build run on this host.

    They need gcc, ar, nm, ldd and bash on the host.
    """

    def setUp(self) -> None:
        """Build a small shared and static library with a header."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tree = os.path.join(tmp.name, "demo.opt", "base")
        os.makedirs(os.path.join(self.tree, "include"))
        os.makedirs(os.path.join(self.tree, "build"))
        with open(os.path.join(self.tree, "include", "demo.h"), "w") as fh:
            fh.write(
                "#ifndef DEMO_H\n#define DEMO_H\n#include \"demo_conf.h\"\n"
                "/* Sums the bytes of a buffer. */\n"
                "unsigned long demo_sum(const unsigned char *data,\n"
                "                       unsigned long size);\n"
                "int demo_version(void);\n#endif\n"
            )
        with open(os.path.join(self.tree, "build", "demo_conf.h"),
                  "w") as fh:
            fh.write("#define DEMO_CONF 1\n")
        with open(os.path.join(self.tree, "demo.c"), "w") as fh:
            fh.write(
                '#include "demo.h"\n'
                "unsigned long demo_sum(const unsigned char *d, "
                "unsigned long n) { unsigned long s = 0; while (n--) "
                "s += *d++; return s; }\n"
                "int demo_version(void) { return DEMO_CONF; }\n"
                "static int hidden(void) { return 0; }\n"
            )
        build = os.path.join(self.tree, "build")
        for command in (
            ["gcc", "-O2", "-fPIC", "-Iinclude", "-Ibuild", "-c", "demo.c",
             "-o", "build/demo.o"],
            ["gcc", "-shared", "-Wl,-soname,libdemo.so.1", "-o",
             "build/libdemo.so.1.0", "build/demo.o"],
            ["ar", "rcs", "build/libdemo.a", "build/demo.o"],
        ):
            subprocess.run(command, cwd=self.tree, check=True,
                           capture_output=True)
        os.symlink("libdemo.so.1.0", os.path.join(build, "libdemo.so.1"))

    def _program(self, program, spec):
        done = subprocess.run(
            [sys.executable, "-c", program, json.dumps(spec)],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        line = done.stdout.strip().splitlines()[-1]
        return json.loads(line[len(workload._MARKER):])

    def test_programs_parse(self) -> None:
        ast.parse(workload._API_PROGRAM)
        ast.parse(workload._WRITE_PROGRAM)

    def test_api_scan_finds_exports_headers_and_includes(self) -> None:
        api = self._program(workload._API_PROGRAM, {
            "tree": self.tree,
            "libraries": ["build/libdemo.so.1.0", "build/libdemo.a"],
            "headers": 3,
            "prototypes": 80,
        })
        self.assertEqual(api["symbols"], ["demo_sum", "demo_version"])
        self.assertFalse(api["cxx"])
        self.assertEqual(api["headers"], ["include/demo.h"])
        self.assertEqual(api["include_dirs"], ["include", "build"])
        self.assertEqual(api["prototypes"], [
            "unsigned long demo_sum(const unsigned char *data, unsigned "
            "long size);",
            "int demo_version(void);",
        ])

    def test_api_scan_adds_the_parent_of_a_header_folder(self) -> None:
        """<pkg/x.h> includes work: the header's parent folder is in."""
        inc = os.path.join(self.tree, "c", "include", "br")
        os.makedirs(inc)
        with open(os.path.join(inc, "port.h"), "w") as fh:
            fh.write("#define BR_API extern\n")
        with open(os.path.join(inc, "encode.h"), "w") as fh:
            fh.write("#include <br/port.h>\n"
                     "BR_API unsigned long br_sum(const unsigned char *d, "
                     "unsigned long n);\n")
        with open(os.path.join(self.tree, "br.c"), "w") as fh:
            fh.write("unsigned long br_sum(const unsigned char *d, unsigned "
                     "long n) { unsigned long s = 0; while (n--) s += *d++; "
                     "return s; }\n")
        for command in (["gcc", "-O2", "-c", "br.c", "-o", "build/br.o"],
                        ["ar", "rcs", "build/libbr.a", "build/br.o"]):
            subprocess.run(command, cwd=self.tree, check=True,
                           capture_output=True)
        api = self._program(workload._API_PROGRAM, {
            "tree": self.tree, "libraries": ["build/libbr.a"],
            "headers": 3, "prototypes": 80,
        })
        self.assertEqual(api["headers"], ["c/include/br/encode.h"])
        self.assertEqual(api["include_dirs"][:2],
                         ["c/include/br", "c/include"])
        with open(os.path.join(self.tree, "h.c"), "w") as fh:
            fh.write('#include <br/encode.h>\n#include "encode.h"\n'
                     "int main(void) { const unsigned char b[2] = {1, 2}; "
                     "return br_sum(b, 2) == 3 ? 0 : 1; }\n")
        flags = [f"-I{d}" for d in api["include_dirs"]]
        done = subprocess.run(["gcc"] + flags + ["h.c", "build/libbr.a",
                                                 "-o", "h"],
                              cwd=self.tree, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(subprocess.run([os.path.join(self.tree, "h")])
                         .returncode, 0)

    def test_library_use_of_helper_programs(self) -> None:
        """The nm scan finds library calls of a dynamic and a static helper."""
        sources = {
            "uses.c": '#include "demo.h"\nint main(void) { const unsigned '
                      'char b[1] = {1}; return (int)demo_sum(b, 1) - 1; }\n',
            "md5.c": "unsigned md5_step(unsigned x) { return x * 33; }\n"
                     "int main(void) { return (int)md5_step(0); }\n",
        }
        for name, body in sources.items():
            with open(os.path.join(self.tree, name), "w") as fh:
                fh.write(body)
        for command in (
            ["gcc", "-Iinclude", "-Ibuild", "uses.c", "-o", "build/uses",
             "build/libdemo.so.1.0"],
            ["gcc", "-Iinclude", "-Ibuild", "uses.c", "-o",
             "build/uses_static", "build/libdemo.a"],
            ["gcc", "md5.c", "-o", "build/md5"],
        ):
            subprocess.run(command, cwd=self.tree, check=True,
                           capture_output=True)
        found = self._program(workload._USES_PROGRAM, {
            "tree": self.tree,
            "libraries": ["build/libdemo.so.1.0", "build/libdemo.a"],
            "helpers": ["build/uses", "build/uses_static", "build/md5"],
        })
        self.assertEqual(found["exported"], 2)
        self.assertGreaterEqual(found["uses"]["build/uses"], 1)
        self.assertGreaterEqual(found["uses"]["build/uses_static"], 1)
        self.assertEqual(found["uses"]["build/md5"], 0)

    def test_api_scan_skips_side_and_cxx_headers(self) -> None:
        """Extras and C++ wrappers are not the C API; minor headers rank lower.

        ``internal.h`` declares one export, so it stays in the list after
        ``include/demo.h``.
        """
        contrib = os.path.join(self.tree, "contrib", "cxx")
        os.makedirs(contrib)
        with open(os.path.join(contrib, "demo_stream.h"), "w") as fh:
            fh.write("#include <iostream>\nclass Stream {\n"
                     "  unsigned long demo_sum(const unsigned char *d);\n"
                     "  int demo_version();\n};\n")
        with open(os.path.join(self.tree, "internal.h"), "w") as fh:
            fh.write('#include "demo.h"\nint demo_version(void);\n')
        with open(os.path.join(self.tree, "wrap.hpp"), "w") as fh:
            fh.write("namespace demo {\nint demo_version();\n"
                     "unsigned long demo_sum(const unsigned char *d);\n}\n")
        api = self._program(workload._API_PROGRAM, {
            "tree": self.tree,
            "libraries": ["build/libdemo.so.1.0"],
            "headers": 3,
            "prototypes": 80,
        })
        self.assertEqual(api["headers"][0], "include/demo.h")
        self.assertNotIn("contrib/cxx/demo_stream.h", api["headers"])
        self.assertNotIn("wrap.hpp", api["headers"])
        self.assertNotIn("contrib/cxx", api["include_dirs"])

    def test_harness_builds_and_loads_the_tree_library(self) -> None:
        """The build commands link the tree's .so and pass the ldd check."""
        api = {"include_dirs": ["include", "build"]}
        source_rel, commands = c_harness_build(
            api, ["build/libdemo.so.1.0", "build/libdemo.a"], "c"
        )
        self.assertEqual(source_rel, ".atesor/bench/harness.c")
        os.makedirs(os.path.join(self.tree, ".atesor", "bench"))
        with open(os.path.join(self.tree, source_rel), "w") as fh:
            fh.write(
                '#include <stdio.h>\n#include "demo.h"\n'
                "int main(int argc, char **argv) { unsigned char b[3] = "
                "{1, 2, 3}; printf(\"ATESOR-CHECK %lu %d\\n\", "
                "demo_sum(b, 3), demo_version()); return 0; }\n"
            )
        env = dict(os.environ, CC="gcc")
        for command in commands:
            done = subprocess.run(["bash", "-c", bench.render_paths(command)],
                                  cwd=self.tree, env=env,
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
        run = subprocess.run([os.path.join(self.tree, ".atesor", "bench",
                                           "harness")],
                             capture_output=True, text=True)
        self.assertEqual(run.stdout, "ATESOR-CHECK 6 1\n")
        static_source, static_commands = c_harness_build(
            api, ["build/libdemo.a"], "c"
        )
        self.assertEqual(len(static_commands), 1)
        self.assertIn(
            "-Wl,--start-group {path:build/libdemo.a} -Wl,--end-group",
            static_commands[0],
        )
        rendered = bench.render_paths(static_commands[0])
        self.assertIn("-Wl,--start-group build/libdemo.a -Wl,--end-group",
                      rendered)
        done = subprocess.run(["bash", "-c", rendered], cwd=self.tree,
                              env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_library_in_a_hash_directory_comes_from_the_inspection(
        self,
    ) -> None:
        """The harness links the inspected library, not the newest one.

        lz4 names its output directory after a hash of the flags. The
        package tests build the library again into another such
        directory; that newer copy must not replace the tune's build.
        """
        verified = "build/cachedObjs/114b1ec24f5c1d70f6731d54b5e28dc1"
        tune_dir = "build/cachedObjs/c325c18bd8879e9a60f926c1c986eb23"
        tests_dir = "build/cachedObjs/4886268cf478831edc7f85abb8cf3b85"
        here = os.path.join(self.tree, tune_dir)
        os.makedirs(here)
        os.rename(os.path.join(self.tree, "build", "libdemo.so.1.0"),
                  os.path.join(here, "libdemo.so.1.0"))
        os.remove(os.path.join(self.tree, "build", "libdemo.so.1"))
        os.symlink("libdemo.so.1.0", os.path.join(here, "libdemo.so.1"))
        # The test build: newer, and its demo_version() returns 2.
        with open(os.path.join(self.tree, "demo2.c"), "w") as fh:
            fh.write('#include "demo.h"\n'
                     "int demo_version(void) { return 2; }\n")
        os.makedirs(os.path.join(self.tree, tests_dir))
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-Iinclude", "-Ibuild",
             "-Wl,-soname,libdemo.so.1", "-o",
             f"{tests_dir}/libdemo.so.1.0", "demo2.c"],
            cwd=self.tree, check=True, capture_output=True,
        )
        os.symlink("libdemo.so.1.0",
                   os.path.join(self.tree, tests_dir, "libdemo.so.1"))
        later = os.path.getmtime(os.path.join(here, "libdemo.so.1.0")) + 60
        os.utime(os.path.join(self.tree, tests_dir, "libdemo.so.1.0"),
                 (later, later))
        api = {"include_dirs": ["include", "build"]}
        source_rel, commands = c_harness_build(
            api, [f"{verified}/libdemo.so.1.0"], "c"
        )
        self.assertNotIn("ls -t", commands[0])
        self.assertIn(f"{{path:{verified}/libdemo.so.1.0}}", commands[0])
        os.makedirs(os.path.join(self.tree, ".atesor", "bench"))
        with open(os.path.join(self.tree, source_rel), "w") as fh:
            fh.write('#include <stdio.h>\n#include "demo.h"\n'
                     "int main(void) { printf(\"ATESOR-CHECK %d\\n\", "
                     "demo_version()); return 0; }\n")
        paths = {f"{verified}/libdemo.so.1.0": f"{tune_dir}/libdemo.so.1.0"}
        env = dict(os.environ, CC="gcc")
        for command in commands:
            done = subprocess.run(
                ["bash", "-c", bench.render_paths(command, paths)],
                cwd=self.tree, env=env, capture_output=True, text=True,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
        run = subprocess.run([os.path.join(self.tree, ".atesor", "bench",
                                           "harness")],
                             capture_output=True, text=True)
        self.assertEqual(run.stdout, "ATESOR-CHECK 1\n")

    def test_ldd_check_refuses_a_foreign_library(self) -> None:
        """A harness that loads no library of this tree fails the check."""
        api = {"include_dirs": ["include", "build"]}
        _, commands = c_harness_build(api, ["build/libdemo.so.1.0"], "c")
        with open(os.path.join(self.tree, "x.c"), "w") as fh:
            fh.write("int main(void) { return 0; }\n")
        os.makedirs(os.path.join(self.tree, ".atesor", "bench"))
        subprocess.run(["gcc", "x.c", "-o", ".atesor/bench/harness"],
                       cwd=self.tree, check=True)
        done = subprocess.run(["bash", "-c", commands[-1]], cwd=self.tree,
                              capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("does not load the library of this tree", done.stderr)


if __name__ == "__main__":
    unittest.main()
