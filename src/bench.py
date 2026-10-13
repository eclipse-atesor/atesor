#############################################################################
# Copyright (c) 2026 10xEngineers
#
# Author: Akif Ejaz <akif.ejaz@10xengineers.ai>
# This program and the accompanying materials are made available under the
# terms of the MIT License which is available at
# https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
#############################################################################

"""Measure an optimized variant against the baseline build.

Speed is measured on native hardware only; under QEMU a variant gets
correctness checks and no speed claim. One in-sandbox runner program
executes every timed run, so SSH and podman start-up never enter a
measurement.

Protocol (tab D of the research doc):

1. Warm up both builds; time 10 baseline runs to get the noise (CV)
   and the minimum detectable effect (MDE).
2. Run 20 rounds (40 when the MDE is above 2%). Each round runs the
   baseline B and the variant V in random order, pinned to one CPU,
   with a new random environment padding.
3. Compare the per-round ratios V/B (the median and a 95% bootstrap
   interval) and the two sets of times (the Mann-Whitney p-value). V is
   faster when the upper bound is below 0.98 and p < 0.05; slower when
   the lower bound is above 1.02 and p < 0.05; otherwise there is no
   change.
4. Discard the session when the CPU frequency moves or the temperature
   reaches 75 C.
"""

import json
import logging
import math
import random
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tools import execute_command

logger = logging.getLogger(__name__)

WARMUPS = 3
CALIBRATION_RUNS = 10
ROUNDS = 20
MAX_ROUNDS = 40
MDE_LIMIT = 0.02
FASTER_BELOW = 0.98
SLOWER_ABOVE = 1.02
P_LIMIT = 0.05
BOOTSTRAP_ITERATIONS = 2000
# Other work that uses more than this many CPUs when a session starts
# may disturb the numbers. The load average is no good for this check:
# it lags a build that ended a moment ago.
BUSY_CPUS_LIMIT = 0.5
TEMPERATURE_LIMIT_MC = 75000  # Millicelsius: 75 C.
# One workload run should take this long. With about 56 runs per
# measurement, 5 s gives sessions of about 5 min, so the ladder fits in
# the one-hour budget.
RUN_SECONDS = (0.5, 5.0)
RUN_TIMEOUT = 300  # Seconds per run; sizing keeps runs at 0.5-5 s.
# The least time that a runner session gets. A measurement with less
# time left stops before its timed rounds. The first session gets this
# much even when the budget is smaller.
MIN_SESSION_SECONDS = 60
INPUT_KINDS = ("binary", "text", "json", "csv")
MAX_INPUT_MB = 512
GOBENCH_BINARY = "atesor-bench.test"
_RUNNER_MARKER = "ATESOR_BENCH_JSON "
# "{path:REL}" in a harness build command stands for artifact REL in
# the tree that the command builds in.
_PATH_WORD = re.compile(r"\{path:([^{}\n]+)\}")


def shell_path(
    tree: str,
    rel: str,
    paths: Optional[Dict[str, str]] = None,
) -> str:
    """Return the quoted path of an artifact in one tree.

    Some builds put their outputs in a directory named after a hash of
    the compiler and the flags (lz4: programs/cachedObjs/<md5>/lz4), so
    the artifact has another path in every tree. ``paths`` maps the
    verified path to the path that the tree's inspection found right
    after its build. Atesor does not search again later: the package
    tests can add more hash directories to the tree.
    """
    actual = (paths or {}).get(rel, rel)
    return shlex.quote(tree.rstrip("/")) + "/" + "/".join(
        shlex.quote(part) for part in actual.split("/")
    )


def path_word(rel: str) -> str:
    """Return the placeholder of artifact ``rel`` in a build command."""
    return "{path:" + rel + "}"


def render_paths(
    command: str,
    paths: Optional[Dict[str, str]] = None,
) -> str:
    """Replace each ``{path:REL}`` with the tree's path of REL.

    The command runs in the tree, so each path stays relative to it.
    """
    return _PATH_WORD.sub(
        lambda match: shlex.quote(
            (paths or {}).get(match.group(1), match.group(1))
        ),
        command,
    )


# Writes deterministic test data. Arguments: kind, size in MiB, seed,
# output path. The same seed always gives the same bytes.
_INPUT_PROGRAM = r'''
import json
import random
import sys

kind, size_mb, seed, path = sys.argv[1:5]
size_mb = int(size_mb)
rng = random.Random(seed)
target = size_mb * 1024 * 1024
words = ["riscv", "vector", "atesor", "port", "build", "cache", "alpha",
         "beta", "gamma", "delta", "kernel", "board", "value", "index",
         "stream", "buffer", "record", "packet", "string", "number"]
written = 0
with open(path, "wb") as out:
    if kind == "csv":
        header = b"id,name,score,flag,comment\n"
        out.write(header)
        written += len(header)
    while written < target:
        if kind == "binary":
            if rng.random() < 0.5:
                chunk = bytes(rng.getrandbits(8) for _ in range(4096))
            else:
                chunk = (rng.choice(words) * 512).encode()[:4096]
        elif kind == "text":
            count = rng.randint(3, 16)
            line = " ".join(rng.choice(words) for _ in range(count))
            chunk = (line + "\n").encode()
        elif kind == "json":
            item = {
                "id": written,
                "name": rng.choice(words),
                "tags": [rng.choice(words) for _ in range(rng.randint(0, 4))],
                "score": rng.random(),
                "nested": {"ok": rng.random() < 0.5, "n": rng.randint(0, 99)},
            }
            chunk = (json.dumps(item) + "\n").encode()
        else:
            chunk = ("%d,%s,%.4f,%d,%s\n" % (
                written, rng.choice(words), rng.random(),
                rng.randint(0, 1),
                " ".join(rng.choice(words) for _ in range(4)),
            )).encode()
        out.write(chunk)
        written += len(chunk)
'''

# Times jobs inside the sandbox. Argument: one JSON spec with "jobs"
# (name -> cmd, cwd, env, output, input; a run that changes its input
# file stops the session), "warmups", "calibrate" (a job name),
# "calibration", "rounds", "pin", "pad_max", "seed", "timeout",
# "hash_output", "check_prefix" (hash only the stdout lines that
# start with it, so timing lines do not count) and "keep_stderr"
# (return the stderr of a run that exits 0).
_RUNNER_PROGRAM = r'''
import glob
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time

SPEC = json.loads(sys.argv[1])
PIN = SPEC.get("pin")
# A CPU outside this process's affinity set cannot be used; such a run
# is not pinned, and the result says so.
if PIN is not None and int(PIN) not in os.sched_getaffinity(0):
    PIN = None
CPUFREQ = "/sys/devices/system/cpu/cpu%d/cpufreq/" % int(PIN or 0)


def read(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def busy_cpus():
    # The busy CPUs right now, from two /proc/stat samples 0.5 s apart.
    def sample():
        with open("/proc/stat") as handle:
            values = [int(x) for x in handle.readline().split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    try:
        total1, idle1 = sample()
        time.sleep(0.5)
        total2, idle2 = sample()
    except (OSError, ValueError, IndexError):
        return None
    span = total2 - total1
    if span <= 0:
        return None
    return round((1.0 - (idle2 - idle1) / span) * (os.cpu_count() or 1), 2)


def board():
    # The readings that the drift check and the warnings compare.
    temps = []
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        value = read(zone)
        if value and value.lstrip("-").isdigit():
            temps.append(int(value))
    load = (read("/proc/loadavg") or "").split()
    return {
        "governor": read(CPUFREQ + "scaling_governor"),
        "cur_freq_khz": read(CPUFREQ + "scaling_cur_freq"),
        "temperatures_mc": temps,
        "loadavg_1m": float(load[0]) if load else None,
        "busy_cpus": busy_cpus(),
    }


def stamp(path):
    # (size, mtime_ns) of a file, so a later stamp detects a change.
    try:
        info = os.stat(path)
    except OSError:
        return None
    return [info.st_size, info.st_mtime_ns]


def run_job(job, rng):
    # Time one run of a job; record its output hash and input change.
    env = dict(os.environ)
    env.update(job.get("env") or {})
    # A random padding shifts each run's stack layout, so an address
    # accident cannot favour the same build in every round.
    env["ATESOR_PAD"] = "x" * rng.randint(0, int(SPEC.get("pad_max", 4096)))
    output = job.get("output")
    if output and os.path.exists(output):
        os.remove(output)
    source = job.get("input")
    source_stamp = stamp(source) if source else None

    def pin_cpu():
        if PIN is not None:
            os.sched_setaffinity(0, {int(PIN)})

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter_ns()
    try:
        proc = subprocess.run(
            ["/bin/sh", "-c", job["cmd"]],
            cwd=job.get("cwd") or None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(SPEC.get("timeout", 300)),
            preexec_fn=pin_cpu,
        )
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        code, out, err = -9, b"", b"timeout"
    wall = time.perf_counter_ns() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {
        "wall_ns": wall,
        "user_s": round(after.ru_utime - before.ru_utime, 6),
        "sys_s": round(after.ru_stime - before.ru_stime, 6),
        "rc": code,
    }
    if SPEC.get("hash_output"):
        prefix = SPEC.get("check_prefix")
        if prefix:
            lines = [
                line for line in out.splitlines()
                if line.startswith(prefix.encode())
            ]
            record["check_lines"] = len(lines)
            if lines:
                record["check_tail"] = lines[-1].decode("utf-8", "replace")[
                    -200:
                ]
            else:
                record["out_tail"] = out.decode("utf-8", "replace")[-300:]
            digest = hashlib.sha256(b"\n".join(lines))
        else:
            digest = hashlib.sha256(out)
        if output and os.path.exists(output):
            with open(output, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
        record["sha"] = digest.hexdigest()
    if code != 0:
        record["err"] = err.decode("utf-8", "replace")[-300:]
    elif SPEC.get("keep_stderr") and err:
        # A run that exits 0 and still complains hides a bug of its own:
        # the workload check reads this text.
        record["err_tail"] = err.decode("utf-8", "replace")[-300:]
    if source and stamp(source) != source_stamp:
        # A run that writes its input (an in-place command, or a file
        # name in the wrong place) changes the work of every later run.
        record["input_changed"] = True
    return record


def main():
    rng = random.Random(SPEC.get("seed", 1))
    jobs = SPEC["jobs"]
    plan = []
    for name in sorted(jobs):
        plan += [(name, "warmup", -1)] * int(SPEC.get("warmups", 0))
    if SPEC.get("calibrate"):
        count = int(SPEC.get("calibration", 0))
        plan += [(SPEC["calibrate"], "calibration", i) for i in range(count)]
    for index in range(int(SPEC.get("rounds", 0))):
        # Random order per round: an order effect (a heating board, a
        # warming cache) then hits B and V equally.
        order = sorted(jobs)
        rng.shuffle(order)
        plan += [(name, "round", index) for name in order]
    result = {"pre": board(), "runs": [], "error": None, "pin": PIN}
    for name, phase, index in plan:
        record = run_job(jobs[name], rng)
        record.update(job=name, phase=phase, round=index)
        result["runs"].append(record)
        if record.get("input_changed"):
            result["error"] = name + " changed its input file"
            result["input_changed"] = True
            break
        if record["rc"] != 0:
            result["error"] = name + " exited " + str(record["rc"])
            break
    result["post"] = board()
    sys.stdout.write("@@MARKER@@" + json.dumps(result) + "\n")


main()
'''.replace("@@MARKER@@", _RUNNER_MARKER)


@dataclass
class Workload:
    """A repeatable task that does a fixed amount of work per run.

    Attributes:
        mode: ``program`` (a program of the package on generated
            input), ``harness`` (a driver program that the optimizer
            agent wrote against the package's API), ``gobench`` (Go
            benchmarks with a fixed iteration count) or ``tests`` (the
            package test suite; a weak signal).
        command: The shell command. ``{bin}`` stands for the program
            in the tree under test, ``{input}`` for the input file,
            ``{output}`` for a per-run output file and ``{iters}`` for
            the iteration count.
        binary: The tree-relative path of the program, or None.
        input_kind: One of :data:`INPUT_KINDS`, or None.
        input_mb: The input size in MiB.
        description: One line for the report.
        package: The Go import path or the Cargo package, or None.
        weak: True when timings say little about real use.
        deterministic: True when B and V must give the same output.
        workdir: The tree-relative directory to run in, or None for
            the tree root. Go benchmarks run in their package directory,
            as ``go test`` runs them.
        run_seconds: The time of one baseline run, or 0 when unknown.
        iterations: The value of ``{iters}``, or 0.
        check_prefix: When set, only the stdout lines that start with
            it are compared between B and V.
        files: Tree-relative path to file content: the harness source,
            written into every tree before ``build`` runs.
        build: Commands that build the harness (or the Go test binary)
            in a tree; they run in the tree with that tree's
            environment.
        env: Extra environment of each run; ``{input}`` is replaced.
        source_language: ``c``, ``c++``, ``go`` or ``rust`` for a
            harness, else None.
        origin: Who chose it: ``upstream``, ``agent`` or ``tests``.
    """

    mode: str
    command: str
    binary: Optional[str] = None
    input_kind: Optional[str] = None
    input_mb: int = 0
    description: str = ""
    package: Optional[str] = None
    weak: bool = False
    deterministic: bool = True
    workdir: Optional[str] = None
    run_seconds: float = 0.0
    iterations: int = 0
    check_prefix: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)
    build: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    source_language: Optional[str] = None
    origin: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workload":
        """Build a workload from :meth:`to_dict` output."""
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Comparison:
    """The statistics of one B/V comparison.

    Attributes:
        ratio: The median of the per-round ratios V/B.
        ci_low: The lower bound of the 95% bootstrap interval.
        ci_high: The upper bound of the 95% bootstrap interval.
        p_value: The two-sided Mann-Whitney p-value.
        rounds: The number of paired rounds.
        b_median_s: The median baseline wall time in seconds.
        v_median_s: The median variant wall time in seconds.
        b_spread: MAD/median of the baseline times.
        v_spread: MAD/median of the variant times.
        verdict: ``faster``, ``slower`` or ``no change``.
    """

    ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    rounds: int
    b_median_s: float
    v_median_s: float
    b_spread: float
    v_spread: float
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def median(values: Sequence[float]) -> float:
    """Return the median of a non-empty sequence."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def relative_spread(values: Sequence[float]) -> float:
    """Return MAD/median, a robust relative spread."""
    center = median(values)
    if center == 0:
        return 0.0
    return median([abs(value - center) for value in values]) / center


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Return the sample standard deviation divided by the mean."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values)
    return math.sqrt(variance / (len(values) - 1)) / mean


def minimum_detectable_effect(cv: float, rounds: int) -> float:
    """Return the smallest relative change the rounds can detect.

    Two-sided alpha 0.05 and power 0.8 give 2.8 standard errors; the
    difference of two means of ``rounds`` runs each adds sqrt(2/n).
    """
    if rounds <= 0:
        return float("inf")
    return 2.8 * cv * math.sqrt(2.0 / rounds)


def rounds_for(cv: float) -> int:
    """Return 20 rounds, or 40 when 20 cannot detect a 2% change."""
    if minimum_detectable_effect(cv, ROUNDS) > MDE_LIMIT:
        return MAX_ROUNDS
    return ROUNDS


def bootstrap_ratio_interval(
    baseline: Sequence[float],
    variant: Sequence[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 7,
) -> Tuple[float, float, float]:
    """Return the median paired ratio V/B and its 95% interval.

    Rounds pair one baseline run with one variant run, so the
    bootstrap resamples whole rounds (percentile method).

    Args:
        baseline: Baseline times, one per round.
        variant: Variant times, one per round, in the same order.
        iterations: The number of bootstrap resamples.
        seed: The random seed, so a result can be reproduced.

    Returns:
        ``(median ratio, lower bound, upper bound)``.

    Raises:
        ValueError: When no round has a baseline time above 0.
    """
    ratios = [v / b for b, v in zip(baseline, variant) if b > 0]
    if not ratios:
        raise ValueError("no paired rounds to compare")
    rng = random.Random(seed)
    medians = sorted(
        median([rng.choice(ratios) for _ in ratios])
        for _ in range(iterations)
    )
    low = medians[int(0.025 * iterations)]
    high = medians[max(0, int(math.ceil(0.975 * iterations)) - 1)]
    return median(ratios), low, high


def mann_whitney_p(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the two-sided Mann-Whitney U p-value (normal approximation).

    Tied values get their average rank, the variance gets the tie
    correction, and the z-score gets a continuity correction.
    """
    n1, n2 = len(first), len(second)
    if n1 == 0 or n2 == 0:
        return 1.0
    combined = sorted(
        [(value, 0) for value in first] + [(value, 1) for value in second]
    )
    ranks: List[float] = [0.0] * len(combined)
    ties: List[int] = []
    index = 0
    while index < len(combined):
        end = index
        while (
            end + 1 < len(combined)
            and combined[end + 1][0] == combined[index][0]
        ):
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = rank
        if end > index:
            ties.append(end - index + 1)
        index = end + 1
    rank_sum = sum(
        rank for rank, (_, group) in zip(ranks, combined) if group == 0
    )
    u_value = rank_sum - n1 * (n1 + 1) / 2.0
    total = n1 + n2
    tie_term = sum(t ** 3 - t for t in ties) / (total * (total - 1))
    variance = n1 * n2 / 12.0 * ((total + 1) - tie_term)
    if variance <= 0:
        return 1.0
    mean = n1 * n2 / 2.0
    distance = max(0.0, abs(u_value - mean) - 0.5)
    z_score = distance / math.sqrt(variance)
    return math.erfc(z_score / math.sqrt(2.0))


def compare(
    baseline: Sequence[float],
    variant: Sequence[float],
) -> Comparison:
    """Compare paired baseline and variant times.

    Args:
        baseline: Baseline wall times in seconds, one per round.
        variant: Variant wall times in seconds, one per round.

    Returns:
        The :class:`Comparison`, with the verdict from the thresholds
        :data:`FASTER_BELOW`, :data:`SLOWER_ABOVE` and :data:`P_LIMIT`.
    """
    ratio, low, high = bootstrap_ratio_interval(baseline, variant)
    p_value = mann_whitney_p(baseline, variant)
    if high < FASTER_BELOW and p_value < P_LIMIT:
        verdict = "faster"
    elif low > SLOWER_ABOVE and p_value < P_LIMIT:
        verdict = "slower"
    else:
        verdict = "no change"
    return Comparison(
        ratio=round(ratio, 4),
        ci_low=round(low, 4),
        ci_high=round(high, 4),
        p_value=round(p_value, 6),
        rounds=min(len(baseline), len(variant)),
        b_median_s=round(median(baseline), 6),
        v_median_s=round(median(variant), 6),
        b_spread=round(relative_spread(baseline), 4),
        v_spread=round(relative_spread(variant), 4),
        verdict=verdict,
    )


def _parse_runner_output(stdout: str) -> Optional[Dict[str, Any]]:
    """Return the result of the last marker line, or None.

    The scan goes from the end: the runner prints its marker last,
    after job output that can contain anything.
    """
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_RUNNER_MARKER):
            try:
                payload = json.loads(line[len(_RUNNER_MARKER):])
            except ValueError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def run_jobs(
    spec: Dict[str, Any],
    timeout: int,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run the in-sandbox runner with one spec and return its result.

    Args:
        spec: The runner spec (see :data:`_RUNNER_PROGRAM`).
        timeout: The whole session's timeout in seconds.
        extra_env: Environment for the runner, for example QEMU_CPU.

    Returns:
        The runner result; ``error`` is set when the session failed.
    """
    # validate=False: the runner is a fixed program. Every workload
    # command it runs was built by Atesor or passed
    # check_workload_command (an agent's program run).
    result = execute_command(
        ["python3", "-c", _RUNNER_PROGRAM, json.dumps(spec)],
        timeout=timeout,
        validate=False,
        extra_env=extra_env,
    )
    payload = _parse_runner_output(result.stdout)
    if payload is None:
        detail = (result.stderr or result.stdout or "").strip()[-300:]
        return {"runs": [], "error": f"runner exit {result.exit_code}: "
                f"{detail}"}
    return payload


def write_input(
    kind: str,
    size_mb: int,
    seed: str,
    path: str,
) -> Optional[str]:
    """Write deterministic input data in the sandbox.

    Returns:
        None on success, or an error string.
    """
    if kind not in INPUT_KINDS:
        return f"unknown input kind {kind!r}"
    # validate=False: the program is a fixed constant, and each argument
    # is one shlex-quoted argv element.
    result = execute_command(
        [
            "python3",
            "-c",
            _INPUT_PROGRAM,
            kind,
            str(int(size_mb)),
            seed,
            path,
        ],
        timeout=900,
        validate=False,
    )
    if result.exit_code != 0:
        return (result.stderr or result.stdout or "").strip()[-300:]
    return None


def render(
    workload: Workload,
    tree: str,
    input_path: Optional[str],
    output_path: Optional[str],
    paths: Optional[Dict[str, str]] = None,
) -> str:
    """Return the workload command for one tree.

    Args:
        workload: The workload.
        tree: The sandbox path of the tree under test.
        input_path: The input file, or None.
        output_path: The per-run output file, or None.
        paths: The tree's path of each artifact that its build put in
            another directory (see :func:`shell_path`).
    """
    binary = workload.binary or GOBENCH_BINARY
    replacements = {
        "{bin}": shell_path(tree, binary, paths),
        "{input}": shlex.quote(input_path or "/dev/null"),
        "{output}": shlex.quote(output_path or "/dev/null"),
        "{iters}": str(max(1, int(workload.iterations or 1))),
    }
    command = workload.command
    for key, value in replacements.items():
        command = command.replace(key, value)
    return command


def render_env(
    workload: Workload,
    input_path: Optional[str],
) -> Dict[str, str]:
    """Return the run environment of a workload for one input file."""
    return {
        name: value.replace("{input}", input_path or "/dev/null")
        for name, value in (workload.env or {}).items()
    }


def _times(payload: Dict[str, Any], job: str, phase: str) -> List[float]:
    return [
        run["wall_ns"] / 1e9
        for run in payload.get("runs", [])
        if run.get("job") == job and run.get("phase") == phase
    ]


def _round_pairs(payload: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """Return the paired (baseline, variant) times of complete rounds.

    A round counts only when both B and V ran; the pairing is what the
    bootstrap resamples.
    """
    rounds: Dict[int, Dict[str, float]] = {}
    for run in payload.get("runs", []):
        if run.get("phase") == "round":
            rounds.setdefault(run["round"], {})[run["job"]] = (
                run["wall_ns"] / 1e9
            )
    baseline, variant = [], []
    for index in sorted(rounds):
        pair = rounds[index]
        if "B" in pair and "V" in pair:
            baseline.append(pair["B"])
            variant.append(pair["V"])
    return baseline, variant


def _board_drift(pre: Dict[str, Any], post: Dict[str, Any]) -> List[str]:
    """Return the drift between the board readings around a session.

    A moved CPU frequency or a temperature at the limit makes the
    early and late rounds incomparable, so the session is discarded.
    """
    problems = []
    if pre.get("cur_freq_khz") != post.get("cur_freq_khz"):
        problems.append(
            f"CPU frequency moved from {pre.get('cur_freq_khz')} to "
            f"{post.get('cur_freq_khz')} kHz"
        )
    hottest = max(
        (pre.get("temperatures_mc") or [0])
        + (post.get("temperatures_mc") or [0])
    )
    if hottest >= TEMPERATURE_LIMIT_MC:
        problems.append(f"temperature reached {hottest / 1000:.0f} C")
    return problems


# Crash signals by signal number N. The runner starts every job with
# "/bin/sh -c", and a shell reports a child that a signal killed as
# 128 + N. A build that dies this way runs code the machine
# cannot execute, or corrupts its own memory; it is not a slow build.
_CRASH_SIGNALS = {
    4: "SIGILL",
    6: "SIGABRT",
    7: "SIGBUS",
    8: "SIGFPE",
    11: "SIGSEGV",
}


def crash_signal(rc: Optional[int]) -> Optional[str]:
    """Return the signal name when a run died from a crash signal.

    Args:
        rc: The exit code of the run, as the shell reported it.

    Returns:
        ``SIGILL``, ``SIGSEGV``, ``SIGABRT``, ``SIGBUS``, ``SIGFPE``, or
        None when the code is not a crash signal. The runner's own
        timeout code (-9) is not a crash.
    """
    if rc is None:
        return None
    if rc < 0:
        # The shell itself died: subprocess reports -N. SIGKILL (-9,
        # which the runner also uses for its timeout) is not a crash.
        return _CRASH_SIGNALS.get(-rc)
    if 128 < rc <= 128 + 64:
        return _CRASH_SIGNALS.get(rc - 128)
    return None


def failure_reason(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Return why a runner session failed, and the job that failed.

    Returns:
        ``(reason, job)``: the reason names the build (``baseline`` for
        B, ``tune`` for V), the phase and the program's own error text;
        a run that a crash signal killed says so. The job is ``B``,
        ``V`` or None.
    """
    error = payload.get("error") or "the runner failed"
    failed = [
        run for run in payload.get("runs", [])
        if run.get("input_changed") or run.get("rc") not in (0, None)
    ]
    if not failed:
        return error, None
    run = failed[-1]
    job = run.get("job")
    who = {"B": "the baseline", "V": "the tune"}.get(job, f"job {job}")
    if run.get("input_changed"):
        return (
            f"{who} run changed its input file in the {run.get('phase')} "
            "phase (the command must only read {input})",
            job,
        )
    detail = " ".join(str(run.get("err") or "").split())[:240]
    signal = crash_signal(run.get("rc"))
    if signal:
        reason = (
            f"{who} run crashed with {signal} in the "
            f"{run.get('phase')} phase"
            + (f": {detail}" if detail else "")
        )
        return reason, job
    reason = (
        f"{who} run exited {run.get('rc')} in the {run.get('phase')} "
        f"phase" + (f": {detail}" if detail else "")
    )
    return reason, job


def _failure(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the failed outcome of a runner payload."""
    reason, job = failure_reason(payload)
    failure = {"status": "failed", "reason": reason, "failed_job": job}
    if payload.get("input_changed"):
        # The caller must write the input again before the next session.
        failure["input_changed"] = True
    return failure


def _outputs_match(payload: Dict[str, Any]) -> Optional[bool]:
    """Return whether the B and V runs produced identical output.

    True when each job gave exactly one digest and they match; False
    when a run printed no check line or the digests differ; None when
    nothing was hashed (a non-deterministic workload).
    """
    digests: Dict[str, set] = {}
    for run in payload.get("runs", []):
        if run.get("check_lines") == 0:
            # A harness that printed no check line proves nothing.
            return False
        if "sha" in run:
            digests.setdefault(run["job"], set()).add(run["sha"])
    if "B" not in digests or "V" not in digests:
        return None
    return digests["B"] == digests["V"] and len(digests["B"]) == 1


def measure(
    baseline_job: Dict[str, Any],
    variant_job: Dict[str, Any],
    workload: Workload,
    pin: Optional[int] = None,
    timeout: int = 3600,
) -> Dict[str, Any]:
    """Measure one variant against the baseline on native hardware.

    Args:
        baseline_job: The baseline job (cmd, cwd, env, output).
        variant_job: The variant job (cmd, cwd, env, output).
        workload: The workload, for output comparison.
        pin: The CPU to pin each run to, or None.
        timeout: The seconds that the whole measurement may take. A
            measurement runs two runner sessions, so each one gets
            what the one before it left. The caller's budget then
            holds: without this, one measurement could spend the
            budget twice and leave none for the tunes after it. The
            first session gets at least ``MIN_SESSION_SECONDS``, so a
            budget below that can run over.

    Returns:
        A dict with ``status`` (``measured``, ``failed`` or
        ``unstable``), the reason, the comparison, the noise figures
        and the board readings.
    """
    started = time.monotonic()

    def left() -> int:
        return int(timeout - (time.monotonic() - started))

    jobs = {"B": baseline_job, "V": variant_job}
    common = {
        "jobs": jobs,
        "pin": pin,
        "pad_max": 4096,
        "timeout": RUN_TIMEOUT,
        "hash_output": workload.deterministic,
        "check_prefix": workload.check_prefix,
    }
    first = run_jobs(
        dict(
            common,
            warmups=WARMUPS,
            calibrate="B",
            calibration=CALIBRATION_RUNS,
            seed=11,
        ),
        max(MIN_SESSION_SECONDS, left()),
    )
    outcome: Dict[str, Any] = {"warnings": []}
    if first.get("error"):
        outcome.update(_failure(first))
        return outcome
    pre = first.get("pre") or {}
    busy = pre.get("busy_cpus")
    if busy is not None and busy > BUSY_CPUS_LIMIT:
        outcome["warnings"].append(
            f"other work used {busy} CPUs when the session started: it "
            "may disturb the numbers"
        )
    if pre.get("governor") and pre["governor"] != "performance":
        outcome["warnings"].append(
            f"cpufreq governor is {pre['governor']}, not performance"
        )
    if workload.deterministic:
        match = _outputs_match(first)
        outcome["outputs_match"] = match
        if match is False:
            outcome.update(
                status="failed",
                reason="the variant output differs from the baseline",
            )
            return outcome
    calibration = _times(first, "B", "calibration")
    cv = coefficient_of_variation(calibration)
    rounds = rounds_for(cv)
    outcome.update(
        cv=round(cv, 4),
        mde=round(minimum_detectable_effect(cv, rounds), 4),
        rounds=rounds,
    )
    if minimum_detectable_effect(cv, rounds) > MDE_LIMIT:
        outcome["warnings"].append(
            f"the workload is noisy (CV {cv:.1%}): changes below "
            f"{minimum_detectable_effect(cv, rounds):.1%} stay hidden"
        )
    if left() < MIN_SESSION_SECONDS:
        outcome.update(
            status="failed",
            reason="the time budget ended before the timed rounds",
        )
        return outcome
    second = run_jobs(
        dict(common, warmups=1, rounds=rounds, seed=23),
        left(),
    )
    if second.get("error"):
        outcome.update(_failure(second))
        return outcome
    drift = _board_drift(first.get("pre") or {}, second.get("post") or {})
    baseline, variant = _round_pairs(second)
    outcome.update(
        board_pre=first.get("pre"),
        board_post=second.get("post"),
    )
    if len(baseline) < 5:
        outcome.update(status="failed", reason="too few complete rounds")
        return outcome
    comparison = compare(baseline, variant)
    outcome["comparison"] = comparison.to_dict()
    if drift:
        outcome.update(status="unstable", reason="; ".join(drift))
        return outcome
    outcome.update(status="measured", reason=comparison.verdict)
    return outcome


def check_once(
    baseline_job: Dict[str, Any],
    variant_job: Dict[str, Any],
    workload: Workload,
    extra_env: Optional[Dict[str, str]] = None,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Run B and V once each and compare their outputs (no timing).

    QEMU runs use this for gate 3: the speed of emulation says nothing
    about hardware, but a differing output is still a failure.

    Returns:
        A dict with ``status`` (``checked`` or ``failed``) and the
        reason. A failed runner session also gives ``failed_job``.
    """
    payload = run_jobs(
        {
            "jobs": {"B": baseline_job, "V": variant_job},
            "rounds": 1,
            "pin": None,
            "pad_max": 0,
            "timeout": RUN_TIMEOUT * 4,
            "hash_output": workload.deterministic,
            "check_prefix": workload.check_prefix,
            "seed": 5,
        },
        timeout,
        extra_env=extra_env,
    )
    if payload.get("error"):
        return _failure(payload)
    if workload.deterministic and _outputs_match(payload) is False:
        return {
            "status": "failed",
            "reason": "the variant output differs from the baseline",
        }
    return {"status": "checked", "reason": "outputs match"}


# Workload choice

_BENCH_LINE = re.compile(
    r"^(Benchmark\S+)\s+(\d+)\s+([\d.]+) ns/op", re.MULTILINE
)
# Workload commands are LLM-proposed, so they get a strict grammar of
# their own: the build CommandValidator does not know the programs a
# build just made (it rejects ./build/tool).
_WORKLOAD_TOKEN = re.compile(
    r"""(?P<placeholder>\{bin\}|\{input\}|\{output\})"""
    r"""|(?P<quoted>'[^'\n]*'|"[^"$`\\\n]*")"""
    r"""|(?P<redirect>2>/dev/null|2>&1|[<>])"""
    r"""|(?P<pipe>\|)"""
    r"""|(?P<word>[A-Za-z0-9_./=:,+%@^-]+)"""
    r"""|(?P<space>\s+)"""
)
WORKLOAD_HELPERS = (
    "base64",
    "cat",
    "cmp",
    "cut",
    "gunzip",
    "gzip",
    "head",
    "md5sum",
    "sha256sum",
    "sort",
    "tail",
    "tr",
    "uniq",
    "wc",
    "xz",
)


def parse_go_benchmarks(output: str) -> List[Tuple[str, int, float]]:
    """Return ``(name, iterations, ns per op)`` from ``go test -bench``."""
    return [
        (name, int(iterations), float(ns))
        for name, iterations, ns in _BENCH_LINE.findall(output or "")
    ]


def go_benchtime(ns_per_op: Sequence[float], seconds: float = 2.0) -> int:
    """Return a fixed iteration count for a set of Go benchmarks.

    Every benchmark runs the same number of iterations, so B and V do
    the same work and their wall times can be compared. All the
    benchmarks together take about ``seconds`` per run.
    """
    total = sum(ns_per_op) if ns_per_op else 0.0
    if total <= 0:
        return 1
    return max(1, int(seconds * 1e9 / total))


def check_workload_command(command: str) -> Optional[str]:
    """Check a workload command template against the strict grammar.

    Each pipe stage starts with ``{bin}`` or a helper from
    :data:`WORKLOAD_HELPERS`. ``<`` reads only ``{input}`` or
    ``/dev/null``, and ``>`` writes only ``{output}`` or ``/dev/null``.
    Outside quotes there is no ``;``, ``$``, backtick, parenthesis, or
    ``&`` other than in ``2>&1``, and no absolute or ``..`` path. A
    double-quoted word has no ``$``, backtick or backslash. The path
    rule does not apply to a quoted word.

    Args:
        command: The template, for example ``{bin} -9 < {input}``.

    Returns:
        None when the template is acceptable, else the reason.
    """
    tokens: List[Tuple[str, str]] = []
    position = 0
    while position < len(command):
        match = _WORKLOAD_TOKEN.match(command, position)
        if not match:
            return f"character {command[position]!r} is not allowed"
        if match.lastgroup != "space":
            tokens.append((match.lastgroup, match.group(0)))
        position = match.end()
    if not any(text == "{bin}" for _, text in tokens):
        return "the command does not run {bin}"
    stage_start = True
    expect_target: Optional[str] = None
    for kind, text in tokens:
        if expect_target:
            allowed = (
                ("{input}", "/dev/null")
                if expect_target == "<"
                else ("{output}", "/dev/null")
            )
            if text not in allowed:
                return f"{expect_target} may only use {' or '.join(allowed)}"
            expect_target = None
            continue
        if kind == "pipe":
            if stage_start:
                return "an empty pipe stage"
            stage_start = True
            continue
        if stage_start:
            if text != "{bin}" and text not in WORKLOAD_HELPERS:
                return f"a pipe stage may not start with {text!r}"
            stage_start = False
            continue
        if kind == "redirect" and text in ("<", ">"):
            expect_target = text
        elif kind == "word" and (text.startswith("/") or ".." in text):
            return f"the path {text!r} is not allowed"
    if expect_target or stage_start:
        return "the command ends early"
    return None


def protocol_seconds(run_seconds: float, rounds: int = ROUNDS) -> float:
    """Return the time one measurement takes for runs of this length.

    Warm-ups of both builds, the calibration runs of B, then ``rounds``
    rounds of B and V.
    """
    runs = 2 * WARMUPS + CALIBRATION_RUNS + 2 * rounds
    return runs * max(0.0, run_seconds)
