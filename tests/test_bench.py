"""Tests for src/bench.py: workloads, the runner and the statistics."""

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src import bench
from src.bench import (
    Workload,
    bootstrap_ratio_interval,
    check_workload_command,
    coefficient_of_variation,
    compare,
    go_benchtime,
    mann_whitney_p,
    measure,
    median,
    minimum_detectable_effect,
    parse_go_benchmarks,
    relative_spread,
    render,
    render_env,
    rounds_for,
)


class TestStatistics(unittest.TestCase):
    """The statistics behave as the protocol describes."""

    def test_median_and_spread(self) -> None:
        """Median, MAD/median and CV on small fixed samples."""
        self.assertEqual(median([3, 1, 2]), 2.0)
        self.assertEqual(median([4, 1, 2, 3]), 2.5)
        self.assertAlmostEqual(relative_spread([1.0, 1.0, 2.0]), 0.0)
        self.assertAlmostEqual(
            relative_spread([9.0, 10.0, 11.0]), 0.1, places=6
        )
        self.assertAlmostEqual(
            coefficient_of_variation([9.0, 10.0, 11.0]), 0.1, places=6
        )
        self.assertEqual(coefficient_of_variation([5.0]), 0.0)

    def test_mde_and_rounds(self) -> None:
        """A 1% CV gives about a 0.9% MDE at 20 rounds."""
        self.assertAlmostEqual(
            minimum_detectable_effect(0.01, 20), 0.00885, places=4
        )
        self.assertEqual(rounds_for(0.01), 20)
        self.assertEqual(rounds_for(0.05), 40)
        self.assertEqual(minimum_detectable_effect(0.01, 0), float("inf"))

    def test_mann_whitney_matches_a_known_case(self) -> None:
        """Fully separated groups of 10 give p near 0.0002."""
        low = [float(value) for value in range(10)]
        high = [float(value) for value in range(10, 20)]
        self.assertLess(mann_whitney_p(low, high), 0.001)
        self.assertAlmostEqual(
            mann_whitney_p(low, high), mann_whitney_p(high, low)
        )
        same = [1.0] * 10
        self.assertEqual(mann_whitney_p(same, same), 1.0)
        self.assertEqual(mann_whitney_p([], [1.0]), 1.0)

    def test_bootstrap_interval_is_reproducible(self) -> None:
        baseline = [1.0 + 0.01 * (i % 3) for i in range(20)]
        variant = [0.9 + 0.01 * (i % 3) for i in range(20)]
        first = bootstrap_ratio_interval(baseline, variant)
        second = bootstrap_ratio_interval(baseline, variant)
        self.assertEqual(first, second)
        ratio, low, high = first
        self.assertLessEqual(low, ratio)
        self.assertLessEqual(ratio, high)
        self.assertAlmostEqual(ratio, 0.9, places=1)
        with self.assertRaises(ValueError):
            bootstrap_ratio_interval([0.0], [1.0])

    def test_verdicts(self) -> None:
        """10% faster, 10% slower and equal give the three verdicts."""
        baseline = [1.0 + 0.002 * (i % 5) for i in range(20)]
        faster = [value * 0.9 for value in baseline]
        slower = [value * 1.1 for value in baseline]
        self.assertEqual(compare(baseline, faster).verdict, "faster")
        self.assertEqual(compare(baseline, slower).verdict, "slower")
        self.assertEqual(compare(baseline, baseline).verdict, "no change")
        result = compare(baseline, faster).to_dict()
        self.assertEqual(result["rounds"], 20)
        self.assertLess(result["ci_high"], bench.FASTER_BELOW)


class TestWorkloadGrammar(unittest.TestCase):
    """LLM workload commands must pass a strict grammar.

    It also covers the rendering of placeholders and tree paths, the
    Workload round trip and the Go benchmark parser.
    """

    def test_accepted_shapes(self) -> None:
        """Pipes of the program and helpers, with fixed redirections."""
        for command in (
            "{bin} -9 < {input} > {output}",
            "{bin} {input} | {bin} -u > {output}",
            "cat {input} | {bin} -c 2>/dev/null | sha256sum",
            "{bin} '.a.b' < {input}",
        ):
            with self.subTest(command=command):
                self.assertIsNone(check_workload_command(command))

    def test_rejected_shapes(self) -> None:
        """Chains, expansions, foreign paths and stray stages fail."""
        cases = {
            "{bin} < {input}; rm -rf /": "';'",
            "{bin} $(id)": "'$'",
            "{bin} `id`": "'`'",
            "{bin} < /etc/passwd": "may only use",
            "{bin} > {input}": "may only use",
            "{bin} >> {output}": "may only use",
            "curl -s {bin}": "may not start",
            "{bin} ../secret": "not allowed",
            "{bin} /etc/shadow": "not allowed",
            "{bin} |": "ends early",
            "{bin} &": "'&'",
            "cat {input}": "does not run {bin}",
        }
        for command, reason in cases.items():
            with self.subTest(command=command):
                self.assertIn(reason, check_workload_command(command))

    def test_render_quotes_paths(self) -> None:
        """Placeholders become quoted sandbox paths and the count."""
        workload = Workload(
            mode="program",
            command="{bin} -9 < {input} > {output}",
            binary="build/mini gzip",
        )
        command = render(workload, "/w/r", "/w/in.bin", "/w/out B")
        self.assertEqual(
            command, "/w/r/build/'mini gzip' -9 < /w/in.bin > '/w/out B'"
        )
        # A hex-hash directory (lz4's cachedObjs) has another name in
        # every tree: the tree's inspection gives the exact path. A
        # wildcard would also match the hash directories of the package
        # tests, and the shell would pass two programs.
        verified = "programs/cachedObjs/3a297cdcdda21e64139b4c81136d85cf/lz4"
        tree_rel = "programs/cachedObjs/040a18f2705e92c766de69f149358e8a/lz4"
        hashed = Workload(mode="program", command="{bin} -9", binary=verified)
        self.assertEqual(
            render(hashed, "/w/t", None, None, {verified: tree_rel}),
            f"/w/t/{tree_rel} -9",
        )
        self.assertEqual(
            render(hashed, "/w/t", None, None), f"/w/t/{verified} -9"
        )
        self.assertNotIn("*", render(hashed, "/w/t", None, None))
        harness = Workload(
            mode="harness",
            command="{bin} {input} {iters}",
            binary=".atesor/bench/harness",
            iterations=12,
            env={"ATESOR_INPUT": "{input}"},
        )
        self.assertEqual(
            render(harness, "/w/t", "/w/in.bin", None),
            "/w/t/.atesor/bench/harness /w/in.bin 12",
        )
        self.assertEqual(
            render_env(harness, "/w/in.bin"), {"ATESOR_INPUT": "/w/in.bin"}
        )
        self.assertEqual(
            render_env(harness, None), {"ATESOR_INPUT": "/dev/null"}
        )

    def test_build_paths_are_the_tree_paths(self) -> None:
        """Each {path:REL} becomes the tree's quoted path of REL."""
        verified = "lib/cachedObjs/c325c18bd8879e9a60f926c1c986eb23/a.so"
        tree_rel = "lib/cachedObjs/0b1e2c3d4f5a6b7c8d9e0f1a2b3c4d5e/a.so"
        command = (
            f"cc x.c {bench.path_word(verified)} "
            f"{bench.path_word('my lib/b.a')} && "
            "{ echo kept; }"
        )
        self.assertEqual(
            bench.render_paths(command, {verified: tree_rel}),
            f"cc x.c {tree_rel} 'my lib/b.a' && {{ echo kept; }}",
        )
        self.assertEqual(
            bench.render_paths(command),
            f"cc x.c {verified} 'my lib/b.a' && {{ echo kept; }}",
        )
        self.assertEqual(
            bench.shell_path("/w/t/", verified, {verified: tree_rel}),
            f"/w/t/{tree_rel}",
        )

    def test_workload_round_trip(self) -> None:
        """to_dict and from_dict keep every field; unknown keys drop."""
        workload = Workload(
            mode="harness",
            command="{bin} {input} {iters}",
            files={".atesor/bench/harness.c": "int main(void){}"},
            build=["cc x.c"],
            check_prefix="ATESOR-CHECK",
            source_language="c",
            origin="agent",
        )
        data = dict(workload.to_dict(), unknown=1)
        self.assertEqual(Workload.from_dict(data), workload)

    def test_go_benchmark_parsing(self) -> None:
        """ns/op lines give one iteration count for about 2 s in total."""
        output = (
            "goos: linux\n"
            "BenchmarkParse-4   \t  120000\t      9876 ns/op\n"
            "BenchmarkEmit-4    \t 1000000\t      1234.5 ns/op\n"
            "PASS\n"
        )
        parsed = parse_go_benchmarks(output)
        self.assertEqual(parsed[0], ("BenchmarkParse-4", 120000, 9876.0))
        # 2e9 ns / (9876 + 1234.5) ns per iteration of both benchmarks.
        self.assertEqual(go_benchtime([9876.0, 1234.5]), 180009)
        self.assertEqual(go_benchtime([]), 1)


class TestProtocol(unittest.TestCase):
    """The protocol length and the absence of package-specific tables."""

    def test_no_package_specific_workloads(self) -> None:
        """No table of program names chooses a workload."""
        self.assertFalse(hasattr(bench, "CLASS_WORKLOADS"))
        self.assertFalse(hasattr(bench, "class_workload"))

    def test_protocol_length(self) -> None:
        """56 runs at 20 rounds; 40 s runs need over half an hour."""
        self.assertEqual(bench.protocol_seconds(1.0), 56.0)
        self.assertEqual(bench.protocol_seconds(1.0, rounds=40), 96.0)
        self.assertGreater(bench.protocol_seconds(40.0), 30 * 60)


class TestPrograms(unittest.TestCase):
    """The in-sandbox programs run for real on this host."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def _run(self, program, *args):
        done = subprocess.run(
            [sys.executable, "-c", program, *args],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout

    def test_programs_parse(self) -> None:
        ast.parse(bench._RUNNER_PROGRAM)
        ast.parse(bench._INPUT_PROGRAM)

    def test_input_is_deterministic_per_seed(self) -> None:
        """The same seed gives the same bytes; another seed does not."""
        digests = {}
        for kind in bench.INPUT_KINDS:
            for seed in ("A", "A", "B"):
                path = os.path.join(self.dir, f"{kind}-{seed}")
                self._run(bench._INPUT_PROGRAM, kind, "1", seed, path)
                with open(path, "rb") as handle:
                    data = handle.read()
                self.assertGreaterEqual(len(data), 1024 * 1024)
                digests.setdefault((kind, seed), set()).add(
                    hashlib.sha256(data).hexdigest()
                )
            self.assertEqual(len(digests[(kind, "A")]), 1)
            self.assertNotEqual(digests[(kind, "A")], digests[(kind, "B")])

    def test_runner_times_interleaved_rounds(self) -> None:
        """Warm-ups, calibration and shuffled rounds all run."""
        spec = {
            "jobs": {
                "B": {"cmd": "echo same", "cwd": self.dir},
                "V": {"cmd": "echo same", "cwd": self.dir},
            },
            "warmups": 2,
            "calibrate": "B",
            "calibration": 3,
            "rounds": 4,
            "pin": None,
            "pad_max": 16,
            "seed": 3,
            "hash_output": True,
        }
        stdout = self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        payload = bench._parse_runner_output(stdout)
        self.assertIsNone(payload["error"])
        phases = [run["phase"] for run in payload["runs"]]
        self.assertEqual(phases.count("warmup"), 4)
        self.assertEqual(phases.count("calibration"), 3)
        self.assertEqual(phases.count("round"), 8)
        self.assertTrue(bench._outputs_match(payload))
        baseline, variant = bench._round_pairs(payload)
        self.assertEqual(len(baseline), 4)
        self.assertEqual(len(variant), 4)
        self.assertIn("loadavg_1m", payload["pre"])
        self.assertIsInstance(payload["pre"]["busy_cpus"], float)

    def test_runner_ignores_a_cpu_it_may_not_use(self) -> None:
        """A pin outside the affinity set gives unpinned runs."""
        allowed = sorted(os.sched_getaffinity(0))
        spec = {
            "jobs": {"B": {"cmd": "true"}},
            "rounds": 1,
            "pin": max(allowed) + 1000,
            "seed": 1,
        }
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        self.assertIsNone(payload["error"])
        self.assertIsNone(payload["pin"])
        spec["pin"] = allowed[-1]
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        self.assertEqual(payload["pin"], allowed[-1])

    def test_runner_stops_at_a_failing_job(self) -> None:
        spec = {
            "jobs": {"B": {"cmd": "exit 3"}, "V": {"cmd": "true"}},
            "rounds": 2,
            "seed": 1,
        }
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        self.assertEqual(payload["error"], "B exited 3")

    def test_runner_stops_a_run_that_changes_its_input(self) -> None:
        source = os.path.join(self.dir, "in.text")
        with open(source, "w") as fh:
            fh.write("data\n")
        spec = {
            "jobs": {
                "B": {"cmd": f"cat {source} > /dev/null", "input": source},
                # Like lz4 with a second program word: it compresses
                # that program into the input path.
                "V": {"cmd": f"echo other > {source}", "input": source},
            },
            "rounds": 3,
            "seed": 1,
        }
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        self.assertEqual(payload["error"], "V changed its input file")
        self.assertTrue(payload["input_changed"])
        self.assertTrue(payload["runs"][-1]["input_changed"])
        self.assertEqual(payload["runs"][-1]["rc"], 0)
        self.assertEqual(
            bench.failure_reason(payload),
            (
                "the tune run changed its input file in the round phase "
                "(the command must only read {input})",
                "V",
            ),
        )
        self.assertEqual(bench._failure(payload)["input_changed"], True)
        reading = {
            "jobs": {
                "B": {"cmd": f"cat {source} > /dev/null", "input": source}
            },
            "rounds": 2,
            "seed": 1,
        }
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(reading))
        )
        self.assertIsNone(payload["error"])
        self.assertNotIn("input_changed", payload)

    def test_runner_hashes_output_files(self) -> None:
        """Different output files make the outputs differ."""
        out_b = os.path.join(self.dir, "b.out")
        out_v = os.path.join(self.dir, "v.out")
        spec = {
            "jobs": {
                "B": {"cmd": f"echo one > {out_b}", "output": out_b},
                "V": {"cmd": f"echo two > {out_v}", "output": out_v},
            },
            "rounds": 1,
            "seed": 1,
            "hash_output": True,
        }
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        self.assertFalse(bench._outputs_match(payload))

    def test_runner_compares_check_lines_only(self) -> None:
        """With a check prefix, timing lines do not count."""
        spec = {
            "jobs": {
                "B": {"cmd": "echo 'took 11 ns'; echo 'ATESOR-CHECK 42'"},
                "V": {"cmd": "echo 'took 9 ns'; echo 'ATESOR-CHECK 42'"},
            },
            "rounds": 2,
            "seed": 1,
            "hash_output": True,
            "check_prefix": "ATESOR-CHECK",
        }
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        self.assertTrue(bench._outputs_match(payload))
        self.assertTrue(
            all(run["check_lines"] == 1 for run in payload["runs"])
        )
        self.assertEqual(payload["runs"][0]["check_tail"], "ATESOR-CHECK 42")
        spec["jobs"]["V"]["cmd"] = "echo 'ATESOR-CHECK 41'"
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        self.assertFalse(bench._outputs_match(payload))

    def test_runner_flags_a_missing_check_line(self) -> None:
        """A run without the check line fails the output comparison."""
        spec = {
            "jobs": {
                "B": {"cmd": "echo 'ATESOR-CHECK 1'"},
                "V": {"cmd": "echo 'nothing here'"},
            },
            "rounds": 1,
            "seed": 1,
            "hash_output": True,
            "check_prefix": "ATESOR-CHECK",
        }
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        missing = [run for run in payload["runs"] if run["job"] == "V"]
        self.assertEqual(missing[0]["check_lines"], 0)
        self.assertIn("nothing here", missing[0]["out_tail"])
        self.assertFalse(bench._outputs_match(payload))

    def test_a_crash_signal_is_named(self) -> None:
        """A run that a signal killed says which signal, not a code."""
        spec = {
            "jobs": {"B": {"cmd": "kill -ILL $$"}},
            "rounds": 1,
            "seed": 1,
        }
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        self.assertIsNotNone(payload["error"])
        reason, job = bench.failure_reason(payload)
        self.assertEqual(job, "B")
        self.assertIn("crashed with SIGILL", reason)
        self.assertNotIn("exited", reason)

    def test_stderr_of_a_passing_run_is_kept_on_request(self) -> None:
        """keep_stderr records what a run that exits 0 complained about."""
        cmd = "echo ok; echo 'boom: bad argument' >&2"
        spec = {"jobs": {"B": {"cmd": cmd}}, "rounds": 1, "seed": 1}
        payload = bench._parse_runner_output(
            self._run(bench._RUNNER_PROGRAM, json.dumps(spec))
        )
        run = payload["runs"][-1]
        self.assertEqual(run["rc"], 0)
        self.assertNotIn("err_tail", run)
        payload = bench._parse_runner_output(
            self._run(
                bench._RUNNER_PROGRAM, json.dumps(dict(spec, keep_stderr=True))
            )
        )
        run = payload["runs"][-1]
        self.assertEqual(run["rc"], 0)
        self.assertIn("boom: bad argument", run["err_tail"])


class TestCrashSignal(unittest.TestCase):
    """crash_signal() reads both shell and subprocess exit codes."""

    def test_codes(self) -> None:
        """128+N and -N name a crash; other codes do not."""
        cases = [
            (132, "SIGILL"),
            (139, "SIGSEGV"),
            (134, "SIGABRT"),
            (135, "SIGBUS"),
            (136, "SIGFPE"),
            (-4, "SIGILL"),
            (-11, "SIGSEGV"),
            # The runner's own timeout code, and plain failures.
            (-9, None),
            (137, None),
            (0, None),
            (1, None),
            (127, None),
            (200, None),
            (None, None),
        ]
        for rc, name in cases:
            with self.subTest(rc=rc):
                self.assertEqual(bench.crash_signal(rc), name)


def _session(
    base, variant, sha_b="x", sha_v="x", pre=None, post=None, calibration=None
):
    runs = []
    for value in calibration or [base[0]] * 10:
        runs.append(
            {
                "job": "B",
                "phase": "calibration",
                "round": 0,
                "wall_ns": int(value * 1e9),
                "rc": 0,
                "sha": sha_b,
            }
        )
    runs.append(
        {
            "job": "V",
            "phase": "warmup",
            "round": -1,
            "wall_ns": int(variant[0] * 1e9),
            "rc": 0,
            "sha": sha_v,
        }
    )
    for index, (b_time, v_time) in enumerate(zip(base, variant)):
        runs.append(
            {
                "job": "B",
                "phase": "round",
                "round": index,
                "wall_ns": int(b_time * 1e9),
                "rc": 0,
                "sha": sha_b,
            }
        )
        runs.append(
            {
                "job": "V",
                "phase": "round",
                "round": index,
                "wall_ns": int(v_time * 1e9),
                "rc": 0,
                "sha": sha_v,
            }
        )
    board = {
        "governor": "performance",
        "cur_freq_khz": "1800000",
        "temperatures_mc": [50000],
        "loadavg_1m": 0.1,
        "busy_cpus": 0.02,
    }
    return {
        "pre": pre or board,
        "post": post or board,
        "runs": runs,
        "error": None,
    }


class TestMeasure(unittest.TestCase):
    """measure() and check_once() follow the protocol with a mocked runner.

    It also covers the failure reasons (failure_reason()).
    """

    def setUp(self) -> None:
        """Prepare a deterministic program workload and two jobs."""
        self.workload = Workload(mode="program", command="{bin}", binary="x")
        self.jobs = ({"cmd": "b"}, {"cmd": "v"})

    def test_faster_variant(self) -> None:
        """A steady 10% gain is a measured, faster variant."""
        base = [1.0 + 0.001 * (i % 4) for i in range(20)]
        variant = [value * 0.9 for value in base]
        session = _session(base, variant)
        with mock.patch.object(bench, "run_jobs", return_value=session):
            outcome = measure(*self.jobs, self.workload, pin=3)
        self.assertEqual(outcome["status"], "measured")
        self.assertEqual(outcome["comparison"]["verdict"], "faster")
        self.assertEqual(outcome["rounds"], 20)
        self.assertTrue(outcome["outputs_match"])

    def test_the_timeout_covers_the_whole_measurement(self) -> None:
        """Two sessions share one budget, so a tune cannot spend it twice."""
        base = [1.0 + 0.001 * (i % 4) for i in range(20)]
        session = _session(base, base)
        seen = []
        clock = [0.0]

        def run(spec, timeout, extra_env=None):
            seen.append(timeout)
            clock[0] += 400.0  # each session burns 400 s
            return session

        with (
            mock.patch.object(bench, "run_jobs", side_effect=run),
            mock.patch.object(
                bench.time, "monotonic", side_effect=lambda: clock[0]
            ),
        ):
            outcome = measure(*self.jobs, self.workload, timeout=1000)
        self.assertEqual(outcome["status"], "measured")
        # First session gets the budget; the second gets what is left.
        self.assertEqual(seen, [1000, 600])

    def test_a_measurement_stops_when_its_budget_ends(self) -> None:
        """No timed rounds start with less than one session left."""
        base = [1.0 + 0.001 * (i % 4) for i in range(20)]
        session = _session(base, base)
        clock = [0.0]

        def run(spec, timeout, extra_env=None):
            clock[0] += 980.0
            return session

        with (
            mock.patch.object(bench, "run_jobs", side_effect=run) as runner,
            mock.patch.object(
                bench.time, "monotonic", side_effect=lambda: clock[0]
            ),
        ):
            outcome = measure(*self.jobs, self.workload, timeout=1000)
        self.assertEqual(outcome["status"], "failed")
        self.assertIn(
            "budget ended before the timed rounds", outcome["reason"]
        )
        self.assertEqual(runner.call_count, 1)

    def test_check_prefix_reaches_the_runner(self) -> None:
        """A harness compares its check lines in both sessions."""
        harness = Workload(
            mode="harness",
            command="{bin} {input} {iters}",
            check_prefix="ATESOR-CHECK",
        )
        base = [1.0 + 0.001 * (i % 4) for i in range(20)]
        session = _session(base, base)
        with mock.patch.object(
            bench, "run_jobs", return_value=session
        ) as runner:
            measure(*self.jobs, harness)
        for call in runner.call_args_list:
            self.assertEqual(call.args[0]["check_prefix"], "ATESOR-CHECK")

    def test_different_output_fails(self) -> None:
        """Gate 3: a different output ends the measurement."""
        base = [1.0] * 20
        session = _session(base, base, sha_b="a", sha_v="b")
        with mock.patch.object(bench, "run_jobs", return_value=session):
            outcome = measure(*self.jobs, self.workload)
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("differs", outcome["reason"])

    def test_frequency_drift_makes_the_session_unstable(self) -> None:
        """A frequency change between sessions or 80 C voids the verdict."""
        base = [1.0 + 0.001 * (i % 4) for i in range(20)]
        slow = {
            "governor": "schedutil",
            "cur_freq_khz": "1400000",
            "temperatures_mc": [80000],
            "loadavg_1m": 2.5,
        }
        session = _session(base, base, post=slow, pre=slow)
        first = _session(base, base)
        with mock.patch.object(
            bench, "run_jobs", side_effect=[first, session]
        ):
            outcome = measure(*self.jobs, self.workload)
        self.assertEqual(outcome["status"], "unstable")
        self.assertIn("frequency", outcome["reason"])
        self.assertIn("temperature", outcome["reason"])

    def test_board_warnings_and_noise(self) -> None:
        """A busy, non-performance board and a noisy workload warn."""
        busy = {
            "governor": "ondemand",
            "cur_freq_khz": "1400000",
            "temperatures_mc": [],
            "loadavg_1m": 3.0,
            "busy_cpus": 1.5,
        }
        noisy = [1.0, 1.3, 0.8, 1.2, 0.9, 1.1, 0.7, 1.4, 1.0, 0.95]
        base = [1.0 + 0.001 * (i % 4) for i in range(40)]
        session = _session(base, base, pre=busy, post=busy, calibration=noisy)
        with mock.patch.object(bench, "run_jobs", return_value=session):
            outcome = measure(*self.jobs, self.workload)
        text = " ".join(outcome["warnings"])
        self.assertIn("other work used 1.5 CPUs", text)
        self.assertIn("ondemand", text)
        self.assertIn("noisy", text)
        self.assertEqual(outcome["rounds"], 40)

    def test_runner_error_fails(self) -> None:
        """A failing first session reports the runner's error."""
        with mock.patch.object(
            bench, "run_jobs", return_value={"runs": [], "error": "boom"}
        ):
            outcome = measure(*self.jobs, self.workload)
        self.assertEqual(
            outcome,
            {
                "warnings": [],
                "status": "failed",
                "reason": "boom",
                "failed_job": None,
            },
        )

    def test_failure_names_the_build_and_its_error(self) -> None:
        """The reason says which build failed, when and why."""
        session = {
            "runs": [
                {"job": "B", "phase": "warmup", "rc": 0},
                {
                    "job": "B",
                    "phase": "calibration",
                    "rc": 1,
                    "err": "Round-trip data mismatch\nat iteration 0\n",
                },
            ],
            "error": "B exited 1",
        }
        with mock.patch.object(bench, "run_jobs", return_value=session):
            outcome = measure(*self.jobs, self.workload)
        self.assertEqual(outcome["failed_job"], "B")
        self.assertEqual(
            outcome["reason"],
            "the baseline run exited 1 in the calibration phase: "
            "Round-trip data mismatch at iteration 0",
        )
        tune = {
            "runs": [
                {"job": "V", "phase": "round", "rc": -9, "err": "timeout"}
            ],
            "error": "V exited -9",
        }
        self.assertEqual(
            bench.failure_reason(tune),
            ("the tune run exited -9 in the round phase: timeout", "V"),
        )

    def test_check_once_compares_outputs(self) -> None:
        """QEMU gate 3 runs each build once and compares outputs."""
        same = _session([1.0], [1.0])
        differ = _session([1.0], [1.0], sha_b="a", sha_v="b")
        with mock.patch.object(
            bench, "run_jobs", side_effect=[same, differ]
        ) as runner:
            first = bench.check_once(
                *self.jobs, self.workload, extra_env={"QEMU_CPU": "rv64i"}
            )
            second = bench.check_once(*self.jobs, self.workload)
        self.assertEqual(first["status"], "checked")
        self.assertEqual(second["status"], "failed")
        self.assertEqual(
            runner.call_args_list[0].kwargs["extra_env"],
            {"QEMU_CPU": "rv64i"},
        )


if __name__ == "__main__":
    unittest.main()
