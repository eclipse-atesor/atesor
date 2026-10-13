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

"""Choose the workload that performance mode times (``--optimize``).

A workload must do the same, fixed amount of work in every run, so its
run time shows the speed of the build. Performance mode chooses it from
the type of the package, not from its name:

* Go: the module's own benchmarks, with a fixed iteration count.
* A package that builds its own programs: the optimizer agent (an LLM)
  proposes one command that runs a program on generated input.
* A library: the optimizer agent writes a small driver program, a
  harness, against the library's public API. C or C++ for a C/C++
  library, a Go benchmark function for a Go package, a Cargo example
  for a Rust crate.
* A test or example program of the package that calls its library
  (``nm`` check): the optimizer agent proposes a run of it.
* Last, the package tests: a weak signal, because tests check
  correctness and do not model real use.

Benchmark suites that run for a fixed time (Google Benchmark,
criterion, most CTest and Meson benchmarks) are not used: their run
time does not show the speed.

Every candidate runs on the baseline tree first. It must build, exit 0,
print the same check line in two runs, and take 0.5-5 s per run on
native hardware. A program run compares its whole output and can take
less than 0.5 s. The module's own Go benchmarks compare no output. The
package tests run once, on native hardware only, and only the 5 s limit
applies. When a check fails, the agent gets the error and tries
again, for :data:`AGENT_ROUNDS` rounds, and up to
:data:`AGENT_MAX_ROUNDS` while each round gives a new error.
"""

import base64
import json
import logging
import posixpath
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import bench
from .tools import execute_command

logger = logging.getLogger(__name__)

CHECK_PREFIX = "ATESOR-CHECK"
HARNESS_DIR = ".atesor/bench"
GO_HARNESS_FILE = "atesor_bench_test.go"
GO_BENCHMARK = "BenchmarkAtesor"
RUST_EXAMPLE = "atesor_bench"
AGENT_ROUNDS = 3
# A round whose error no earlier round met shows progress (a harness
# fixed one error and met the next one): the rounds go on, up to this
# many. An API that needs several required parameters costs one
# round per parameter, so a low limit stops a harness that still
# converges. Every round costs at least one LLM request.
AGENT_MAX_ROUNDS = 8
# The agent rounds stop when less than this much of the budget is left.
AGENT_MIN_SECONDS = 600
# A harness, Go benchmark or program run is sized to take about this
# many seconds.
TARGET_SECONDS = 2.0
MAX_ITERATIONS = 10_000_000
MAX_SOURCE_BYTES = 32_000
MAX_PROGRAMS = 6
HELP_LINES = 20
# A program in one of these directories is a test, example or benchmark
# program of the package, not one of its own programs.
HELPER_DIRS = (
    "test",
    "tests",
    "testing",
    "example",
    "examples",
    "bench",
    "benches",
    "benchmark",
    "benchmarks",
    "demo",
    "demos",
    "sample",
    "samples",
    "fuzz",
    "fuzzing",
    "cmakefiles",
)
# Cargo writes these under target/<profile>/; none is a package program.
_CARGO_HELPER_DIRS = ("build", "deps", "examples", "incremental")
# A harness may not use these: they make the work or the output change
# from run to run, or reach outside the sandbox tree.
_FORBIDDEN = {
    "c": re.compile(
        r"\b(system|popen|fork|vfork|execv\w*|execl\w*|socket|connect|"
        r"getrandom|gettimeofday|clock_gettime|clock|time|rand|srand|"
        r"pthread_create)\s*\(|/dev/u?random"
    ),
    "go": re.compile(
        r'"(os/exec|net|net/http|math/rand|crypto/rand)"|time\.Now\s*\('
    ),
    "rust": re.compile(
        r"std::process::Command|std::net::|SystemTime|Instant::now|"
        r"thread::spawn|\brand::"
    ),
}

AskAgent = Callable[
    [str, str, Callable[[Dict[str, Any]], Optional[str]]],
    Tuple[Optional[Dict[str, Any]], str],
]
Say = Callable[[str], None]


@dataclass
class PackageProfile:
    """What the verified build made, as performance mode sees it.

    Attributes:
        language: ``c`` (C and C++), ``go`` or ``rust``.
        build_system: The build system of the plan, for example
            ``cmake``.
        kind: ``library``, ``program``, ``library+programs`` or
            ``unknown``.
        programs: The package's own programs (tree-relative paths).
        helpers: Its test, example and benchmark programs.
        libraries: Its static and shared libraries.
        purpose: The analyst's one-line purpose, or "".
    """

    language: str
    build_system: str
    kind: str
    programs: List[str] = field(default_factory=list)
    helpers: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    purpose: str = ""

    def describe(self) -> str:
        """Return one line for the console.

        For example ``C/C++ library (cmake): build/libz.a``.
        """
        names = {"c": "C/C++", "go": "Go", "rust": "Rust"}
        kind = {
            "library": "library",
            "program": "program",
            "library+programs": "library and programs",
        }.get(self.kind, "package")
        shown = (self.libraries + self.programs)[:3]
        more = len(self.libraries) + len(self.programs) - len(shown)
        files = ", ".join(shown) + (f" and {more} more" if more > 0 else "")
        return (
            f"{names.get(self.language, self.language)} {kind} "
            f"({self.build_system})" + (f": {files}" if files else "")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


@dataclass
class WorkloadContext:
    """What the workload choice needs from the optimization run.

    Attributes:
        state: The AgentState of the port.
        package: The package profile.
        tree: The baseline tree in the sandbox (a copy of the verified
            build); every candidate runs there first.
        env: The build environment of the baseline tree.
        root: The ``<repo>.opt`` working directory.
        kind: ``native`` or ``qemu``.
        qemu_env: QEMU_CPU for QEMU runs, or None.
        agent: Asks the optimizer agent, or None without an LLM.
        say: Prints one progress line.
        time_left: Seconds left in the budget.
        build_timeout: The timeout of one harness build command, in
            seconds.
        paths: The baseline tree's path of each artifact that its build
            put in another directory (see :func:`bench.shell_path`).
        crashes: Every crash of the baseline build during the workload
            check. The baseline carries no tune flags, so a crash is a
            fault of the port; the caller reports it as one.
    """

    state: Any
    package: PackageProfile
    tree: str
    env: Dict[str, str]
    root: str
    kind: str
    qemu_env: Optional[Dict[str, str]] = None
    agent: Optional[AskAgent] = None
    say: Say = lambda text: None
    time_left: Callable[[], float] = lambda: float("inf")
    build_timeout: int = 1800
    paths: Dict[str, str] = field(default_factory=dict)
    crashes: List[str] = field(default_factory=list)


# Package profile


def _is_helper(rel: str) -> bool:
    """Return True when ``rel`` is a test, example or benchmark program."""
    parts = [part.lower() for part in rel.split("/")[:-1]]
    if any(part in HELPER_DIRS for part in parts):
        return True
    # target/<profile>/<dir>/... is Cargo's own output, not a program.
    return (
        len(parts) >= 3
        and parts[0] == "target"
        and parts[2] in _CARGO_HELPER_DIRS
    )


def profile_package(state: Any, language: str) -> PackageProfile:
    """Return the package profile from the verified build's artifacts.

    Args:
        state: The AgentState of the verified port.
        language: ``c``, ``go`` or ``rust``.

    Returns:
        The profile. A program under a test, example or benchmark
        directory is a helper, not one of the package's programs.
    """
    repo = state.repo_path.rstrip("/") + "/"
    plan = getattr(state, "build_plan", None)
    programs: List[str] = []
    helpers: List[str] = []
    libraries: List[str] = []
    for item in state.curated_artifacts or state.build_artifacts or []:
        path = item.get("filepath") or item.get("path") or ""
        if not path.startswith(repo):
            continue
        rel = posixpath.normpath(path[len(repo):])
        if not rel or rel.startswith(".."):
            continue
        kind = item.get("type")
        if kind in ("library_static", "library_shared"):
            target = libraries
        elif _is_helper(rel):
            target = helpers
        else:
            target = programs
        if rel not in target:
            target.append(rel)
    if programs and libraries:
        kind = "library+programs"
    elif programs:
        kind = "program"
    elif libraries:
        kind = "library"
    else:
        kind = "unknown"
    analysis = getattr(state, "package_analysis", None)
    purpose = str(getattr(analysis, "purpose", "") or "")
    # The analyst's fallback purpose is "<repo> (analysis unavailable —
    # LLM starved)"; it says nothing about the package.
    if "unavailable" in purpose:
        purpose = ""
    return PackageProfile(
        language=language,
        build_system=(plan.build_system if plan else None) or "unknown",
        kind=kind,
        programs=programs,
        helpers=helpers,
        libraries=libraries,
        purpose=purpose[:300],
    )


# Sandbox helpers


# Runs in the sandbox. Argument: JSON {"tree", "files": {rel: base64}}.
# Writes each file below the tree; a path that resolves outside the
# tree (through ".." or a symlink) stops the program.
_WRITE_PROGRAM = r'''
import base64
import json
import os
import sys

SPEC = json.loads(sys.argv[1])
tree = os.path.realpath(SPEC["tree"])
for rel, encoded in SPEC["files"].items():
    target = os.path.realpath(os.path.join(tree, rel))
    if not target.startswith(tree + os.sep):
        sys.exit("refusing to write outside the tree: " + rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(base64.b64decode(encoded))
sys.stdout.write("@@MARKER@@" + json.dumps({"written": len(SPEC["files"])}))
'''

# Runs in the sandbox. Argument: JSON {"tree", "libraries", "headers",
# "prototypes"}. Lists the functions that the libraries export (nm),
# finds the headers that declare most of them, and returns those
# declarations and the include directories they need.
_API_PROGRAM = r'''
import json
import os
import re
import subprocess
import sys

SPEC = json.loads(sys.argv[1])
TREE = SPEC["tree"]
SKIP = {".git", ".atesor", "node_modules", ".hg", ".svn"}
# Headers here belong to extras, tests or other projects, not the API.
SIDE = {"contrib", "test", "tests", "testing", "example", "examples",
        "doc", "docs", "third_party", "thirdparty", "vendor", "external",
        "deps", "bench", "benches", "benchmark", "benchmarks", "fuzz",
        "fuzzing", "demo", "demos", "samples", "win32", "msdos", "amiga",
        "os400", "qnx", "watcom", "old"}
SUFFIXES = (".h", ".hh", ".hpp", ".hxx", ".h++")
BUILD_DIRS = ("build", "_build", "builddir", "out")
IDENT = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
INCLUDE = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')
CXX_ONLY = re.compile(
    r"^\s*(class|namespace|template)\b|\bstd::|#\s*include\s*<(iostream|"
    r"string|vector|memory|fstream)>",
    re.MULTILINE,
)


def nm(args):
    try:
        proc = subprocess.run(
            ["nm"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.decode("utf-8", "replace")


symbols, cxx = set(), False
for rel in SPEC["libraries"]:
    path = os.path.join(TREE, rel)
    shared = ".so" in os.path.basename(rel)
    flags = ["-D", "--defined-only"] if shared else ["-g", "--defined-only"]
    for line in nm(flags + ["-C", path]).splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] not in ("T", "W", "i"):
            continue
        name = parts[2].split("@")[0].strip()
        if "(" in name or "::" in name:
            cxx = True
            name = name.split("(")[0].split("::")[-1].strip()
        if re.match(r"^[A-Za-z]\w*$", name):
            symbols.add(name)

headers = []
for dirpath, dirnames, filenames in os.walk(TREE):
    dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
    for name in sorted(filenames):
        if name.endswith(SUFFIXES):
            headers.append(os.path.relpath(os.path.join(dirpath, name), TREE))
    if len(headers) > 4000:
        break

texts, scores = {}, []
for rel in headers:
    try:
        with open(os.path.join(TREE, rel), "rb") as handle:
            text = handle.read(524288).decode("utf-8", "replace")
    except OSError:
        continue
    found = set(IDENT.findall(text)) & symbols
    if found:
        texts[rel] = (text, found)
        side = any(part.lower() in SIDE for part in rel.split("/")[:-1])
        # A C library's API is never in a C++-only header.
        foreign = not cxx and bool(CXX_ONLY.search(text))
        scores.append((side or foreign, -len(found), rel.count("/"), rel))
scores.sort()
main_scores = [item for item in scores if not item[0]] or scores
top = -main_scores[0][1] if main_scores else 0
chosen = [
    rel for _, score, _, rel in main_scores
    if -score >= max(1, top // 4)
][: SPEC["headers"]]


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


prototypes, seen = [], set()
for rel in chosen:
    text, found = texts[rel]
    clean = strip_comments(text)
    for match in IDENT.finditer(clean):
        name = match.group(1)
        if name not in found or name in seen:
            continue
        start = max(
            clean.rfind(";", 0, match.start()),
            clean.rfind("}", 0, match.start()),
            clean.rfind("{", 0, match.start()),
        )
        ends = [
            pos for pos in (
                clean.find(";", match.end()),
                clean.find("{", match.end()),
            ) if pos >= 0
        ]
        if not ends:
            continue
        end = min(ends)
        segment = clean[start + 1:end]
        lines = [
            line for line in segment.split("\n")
            if not line.strip().startswith("#")
        ]
        statement = " ".join(" ".join(lines).split())
        if not statement or len(statement) > 400:
            continue
        seen.add(name)
        prototypes.append(statement + ";")
        if len(prototypes) >= SPEC["prototypes"]:
            break
    if len(prototypes) >= SPEC["prototypes"]:
        break

by_name = {}
for rel in headers:
    by_name.setdefault(os.path.basename(rel), []).append(rel)


def build_first(rel):
    return (0 if rel.split("/")[0] in BUILD_DIRS else 1, rel)


include_dirs = []


def add(rel):
    directory = os.path.dirname(rel) or "."
    if directory not in include_dirs:
        include_dirs.append(directory)


for rel in chosen:
    add(rel)
# A header in <dir>/<name>/x.h is often included as <name/x.h>, also by
# the library's own headers (brotli: encode.h includes <brotli/port.h>),
# so the parent directory goes in too.
for rel in chosen:
    parent = os.path.dirname(os.path.dirname(rel))
    if parent and parent not in include_dirs:
        include_dirs.append(parent)
for rel in chosen:
    for name in INCLUDE.findall(texts[rel][0]):
        for candidate in sorted(by_name.get(os.path.basename(name), []),
                                key=build_first):
            add(candidate)
sys.stdout.write("@@MARKER@@" + json.dumps({
    "symbols": sorted(symbols)[:4000],
    "cxx": cxx,
    "headers": chosen,
    "include_dirs": include_dirs[:8],
    "prototypes": prototypes,
}))
'''

_MARKER = "ATESOR_WL_JSON "
_WRITE_PROGRAM = _WRITE_PROGRAM.replace("@@MARKER@@", _MARKER)
_API_PROGRAM = _API_PROGRAM.replace("@@MARKER@@", _MARKER)

# Runs in the sandbox. Argument: JSON {"tree", "libraries", "helpers"}.
# For each helper program it counts the library's exported functions
# that the helper imports (dynamic link) or contains (static link, not
# stripped). A helper with none does not exercise the library.
_USES_PROGRAM = r'''
import json
import os
import subprocess
import sys

SPEC = json.loads(sys.argv[1])
TREE = SPEC["tree"]


def nm(args):
    try:
        proc = subprocess.run(
            ["nm"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.decode("utf-8", "replace")


def names(output, kinds):
    found = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-2] in kinds:
            found.add(parts[-1].split("@")[0])
    return found


exported = set()
for rel in SPEC["libraries"]:
    path = os.path.join(TREE, rel)
    shared = ".so" in os.path.basename(rel)
    flags = ["-D", "--defined-only"] if shared else ["-g", "--defined-only"]
    exported |= names(nm(flags + [path]), {"T", "W", "i"})
uses = {}
for rel in SPEC["helpers"]:
    path = os.path.join(TREE, rel)
    used = names(nm(["-D", "--undefined-only", path]), {"U", "w"})
    used |= names(nm(["--defined-only", path]), {"T", "t", "W", "i"})
    uses[rel] = len(used & exported)
sys.stdout.write("@@MARKER@@" + json.dumps(
    {"exported": len(exported), "uses": uses}
) + "\n")
'''.replace("@@MARKER@@", _MARKER)


def _run_program(
    program: str,
    spec: Dict[str, Any],
    timeout: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Run one of the in-sandbox Python programs above.

    Returns:
        ``(payload, "")`` parsed from the program's marker line (the
        last one wins), or ``(None, error)`` with the exit code and
        the tail of its output.
    """
    # validate=False: the programs are fixed constants and their only
    # argument is a JSON spec in one argv element.
    result = execute_command(
        ["python3", "-c", program, json.dumps(spec)],
        timeout=timeout,
        validate=False,
    )
    for line in reversed((result.stdout or "").splitlines()):
        if line.startswith(_MARKER):
            try:
                payload = json.loads(line[len(_MARKER):])
            except ValueError:
                break
            if isinstance(payload, dict):
                return payload, ""
            break
    detail = (result.stderr or result.stdout or "").strip()[-300:]
    return None, f"exit {result.exit_code}: {detail}"


def _sh(
    command: str,
    cwd: str,
    timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
) -> Any:
    # validate=False: Atesor builds these commands; every path in them
    # is quoted with shlex.
    return execute_command(
        command, cwd=cwd, timeout=timeout, validate=False, extra_env=env
    )


def _tail(text: Optional[str], limit: int = 1500) -> str:
    return (text or "").strip()[-limit:]


def write_files(tree: str, files: Dict[str, str]) -> Optional[str]:
    """Write files below ``tree`` in the sandbox.

    Returns:
        None on success, or the error.
    """
    if not files:
        return None
    encoded = {
        rel: base64.b64encode(text.encode("utf-8")).decode("ascii")
        for rel, text in files.items()
    }
    payload, error = _run_program(
        _WRITE_PROGRAM, {"tree": tree, "files": encoded}, timeout=300
    )
    return None if payload is not None else f"could not write: {error}"


def install_workload(
    workload: bench.Workload,
    tree: str,
    env: Dict[str, str],
    timeout: int,
    paths: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Write the workload's files into a tree and build them there.

    A harness builds in every tree with that tree's compiler settings,
    so it is part of the build under test. ``paths`` gives the tree's
    path of each ``{path:REL}`` artifact in the build commands.

    Returns:
        None on success, or the error with the build output.
    """
    error = write_files(tree, workload.files)
    if error:
        return error
    for command in workload.build:
        command = bench.render_paths(command, paths)
        done = _sh(command, tree, timeout=timeout, env=env)
        if not done.success:
            output = _tail(
                "\n".join(part for part in (done.stdout, done.stderr)
                          if part), 1800
            )
            return f"`{command[:80]}` exited {done.exit_code}:\n{output}"
    return None


# Running a candidate on the baseline


def _input_path(root: str, kind: Optional[str], seed: str) -> str:
    return f"{root}/inputs/{seed}.{kind}"


def rewrite_for_tree(command: str, repo_path: str, dest: str) -> str:
    """Point absolute paths into ``repo_path`` at the tree ``dest``.

    ``repo_path`` is the verified tree of the port, not the baseline
    copy.
    """
    pattern = re.escape(repo_path.rstrip("/")) + r"(?=$|[/\s'\";&|)])"
    return re.sub(pattern, lambda _: dest.rstrip("/"), command)


def baseline_job(
    workload: bench.Workload,
    tree: str,
    input_path: Optional[str],
    output_path: Optional[str],
    repo_path: Optional[str] = None,
    paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return the runner job of a workload in one tree.

    A test command from the port can name the verified tree; with
    ``repo_path`` those paths point at ``tree`` instead. ``paths`` gives
    the tree's path of each artifact that its build put in another
    directory.
    """
    if workload.mode == "tests":
        command = workload.command
        if repo_path:
            command = rewrite_for_tree(command, repo_path, tree)
    else:
        command = bench.render(workload, tree, input_path, output_path,
                               paths)
    cwd = f"{tree}/{workload.workdir}" if workload.workdir else tree
    job = {
        "cmd": command,
        "cwd": cwd,
        "env": bench.render_env(workload, input_path),
        "output": output_path,
    }
    if input_path:
        # The runner stops a session whose run changes this file.
        job["input"] = input_path
    return job


def _write_input(ctx: WorkloadContext, workload: bench.Workload) -> str:
    """Write the measure input; return its path, or "" for no input.

    Raises:
        ValueError: When the input directory or file cannot be written.
    """
    if not workload.input_kind:
        return ""
    inputs = shlex.quote(ctx.root + "/inputs")
    if not _sh(f"mkdir -p {inputs}", ctx.tree).success:
        raise ValueError("the input directory could not be created")
    path = _input_path(ctx.root, workload.input_kind, "atesor-measure")
    error = bench.write_input(
        workload.input_kind, workload.input_mb, "atesor-measure", path
    )
    if error:
        raise ValueError(f"the input could not be written: {error}")
    return path


def _two_runs(
    ctx: WorkloadContext,
    workload: bench.Workload,
    input_path: str,
) -> Tuple[Optional[float], str]:
    """Run the workload twice on the baseline tree.

    A crash of the baseline build also goes into ``ctx.crashes``.

    Returns:
        ``(median seconds, "")``, or ``(None, problem)`` when a run
        fails, prints no check line, writes to stderr, or the two
        outputs differ.
    """
    job = baseline_job(workload, ctx.tree, input_path or None,
                       f"{ctx.root}/out-B", ctx.state.repo_path, ctx.paths)
    payload = bench.run_jobs(
        {
            "jobs": {"B": job},
            "rounds": 2,
            "seed": 3,
            "hash_output": workload.deterministic,
            "check_prefix": workload.check_prefix,
            "keep_stderr": True,
            "timeout": bench.RUN_TIMEOUT,
        },
        timeout=1800,
        extra_env=ctx.qemu_env,
    )
    runs = [run for run in payload.get("runs", []) if run.get("job") == "B"]
    if payload.get("error"):
        reason, _ = bench.failure_reason(payload)
        signal = next(
            (
                bench.crash_signal(run.get("rc"))
                for run in runs
                if bench.crash_signal(run.get("rc"))
            ),
            None,
        )
        if signal:
            # The baseline tree is the verified build with no tune
            # flags. A crash here is a fault of the port, not of the
            # workload: the machine cannot run the code it built.
            crash = (
                f"the verified build crashed with {signal} while running "
                f"the workload on this machine: {reason}"
            )
            if crash not in ctx.crashes:
                ctx.crashes.append(crash)
            return None, crash
        return None, reason
    complaint = next(
        (_tail(run.get("err_tail"), 300) for run in runs
         if run.get("err_tail")),
        "",
    )
    if complaint:
        # It exited 0, so it claims success, and still reported an
        # error. Its own error path does not return 1.
        return None, (
            "it exited 0 and still wrote an error to stderr, so its "
            "error path reports success: "
            f"{complaint}"
        )
    if workload.check_prefix and any(
        run.get("check_lines") == 0 for run in runs
    ):
        shown = _tail(runs[-1].get("out_tail"), 300) if runs else ""
        return None, (
            f"it printed no {workload.check_prefix} line; stdout ended "
            f"with: {shown or '(nothing)'}"
        )
    if workload.deterministic and len({run.get("sha") for run in runs}) != 1:
        return None, (
            "it printed different results in two equal runs (an "
            "uninitialized variable, or a value that does not come from "
            "the input)"
        )
    if not runs:
        return None, "the runner returned no run"
    return bench.median([run["wall_ns"] / 1e9 for run in runs]), ""


def size_by_iterations(
    ctx: WorkloadContext,
    workload: bench.Workload,
) -> Optional[str]:
    """Check a harness or benchmark on the baseline; set ``iterations``.

    Starts with one iteration, then scales the count so that one run
    takes about :data:`TARGET_SECONDS` on native hardware. Under QEMU
    one iteration is enough: QEMU measures no time.

    Returns:
        None when the workload is usable, else the problem.
    """
    try:
        input_path = _write_input(ctx, workload)
    except ValueError as exc:
        return str(exc)
    low, high = bench.RUN_SECONDS
    workload.iterations = 1
    for _ in range(4):
        seconds, problem = _two_runs(ctx, workload, input_path)
        if seconds is None:
            return problem
        workload.run_seconds = round(seconds, 3)
        if ctx.kind != "native" or low <= seconds <= high:
            return None
        if seconds > high and workload.iterations == 1:
            return (
                f"one iteration takes {seconds:.1f} s; it must take less "
                f"than {high:.0f} s (use less data or less work per "
                "iteration)"
            )
        scaled = int(workload.iterations * TARGET_SECONDS / max(seconds, 1e-3))
        workload.iterations = max(1, min(MAX_ITERATIONS, scaled))
    return (
        f"the run could not be sized to {low}-{high} s "
        f"(last run {workload.run_seconds} s with N={workload.iterations})"
    )


def size_by_input(
    ctx: WorkloadContext,
    workload: bench.Workload,
) -> Optional[str]:
    """Check a program run on the baseline; scale its input size.

    Returns:
        None when the workload is usable, else the problem.
    """
    if ctx.kind != "native" and workload.input_kind:
        workload.input_mb = min(workload.input_mb, 4)
    low, high = bench.RUN_SECONDS
    for attempt in range(3):
        try:
            input_path = _write_input(ctx, workload)
        except ValueError as exc:
            return str(exc)
        seconds, problem = _two_runs(ctx, workload, input_path)
        if seconds is None:
            return problem
        workload.run_seconds = round(seconds, 3)
        if ctx.kind != "native" or low <= seconds <= high:
            return None
        if not workload.input_kind or attempt == 2:
            break
        factor = TARGET_SECONDS / max(seconds, 1e-3)
        workload.input_mb = int(
            min(bench.MAX_INPUT_MB, max(1, round(workload.input_mb * factor)))
        )
    if workload.run_seconds > high:
        if workload.input_kind and workload.input_mb <= 1:
            return (
                f"one run takes {workload.run_seconds:.1f} s with the "
                f"smallest input (1 MiB); the limit is {high:.0f} s, so "
                "the command must do less work per byte (for example a "
                "faster mode or level of the program)"
            )
        return (
            f"one run takes {workload.run_seconds:.1f} s; the limit is "
            f"{high:.0f} s (use a smaller input or less work)"
        )
    # A fast run is noisy but usable; the statistics say how noisy.
    return None


def check_duration(
    ctx: WorkloadContext,
    workload: bench.Workload,
) -> Optional[str]:
    """Time one baseline run of a test workload on native hardware.

    Sets ``workload.run_seconds``.

    Returns:
        None under QEMU (no run) or when the run takes at most the upper
        bound of :data:`bench.RUN_SECONDS`, else the problem.
    """
    if ctx.kind != "native":
        return None
    job = baseline_job(workload, ctx.tree, None, None, ctx.state.repo_path,
                       ctx.paths)
    payload = bench.run_jobs(
        {"jobs": {"B": job}, "rounds": 1, "seed": 3,
         "timeout": bench.RUN_TIMEOUT},
        timeout=bench.RUN_TIMEOUT + 120,
    )
    if payload.get("error"):
        return f"it fails on the baseline: {payload['error']}"
    runs = [run for run in payload.get("runs", []) if run.get("job") == "B"]
    if not runs:
        return "the baseline run gave no timing"
    seconds = runs[0]["wall_ns"] / 1e9
    workload.run_seconds = round(seconds, 3)
    if seconds > bench.RUN_SECONDS[1]:
        return (
            f"one run takes {seconds:.1f} s (limit "
            f"{bench.RUN_SECONDS[1]:.0f} s)"
        )
    return None


# Agent replies


def _input_fields(data: Dict[str, Any]) -> Optional[str]:
    """Return why an agent answer's input fields are unusable, or None."""
    kind = data.get("input_kind")
    if kind is not None and kind not in bench.INPUT_KINDS:
        return f"input_kind must be one of {list(bench.INPUT_KINDS)} or null"
    if kind is not None:
        try:
            size = int(data.get("input_mb") or 0)
        except (TypeError, ValueError):
            return "input_mb must be a number"
        if not 1 <= size <= 256:
            return "input_mb must be between 1 and 256"
    if not str(data.get("description") or "").strip():
        return "description is missing"
    return None


def _source_problem(source: Any, language: str) -> Optional[str]:
    """Return why a harness source is unusable, or None.

    Enforces the harness contract: a complete file within the size
    cap, a check line, and none of the calls that would make runs
    differ from each other or reach outside the sandbox.
    """
    if not isinstance(source, str) or len(source.strip()) < 80:
        return "source must be the complete file as one JSON string"
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return f"source is longer than {MAX_SOURCE_BYTES} bytes"
    if CHECK_PREFIX not in source:
        return f"the source never prints a {CHECK_PREFIX} line"
    family = "c" if language in ("c", "c++") else language
    match = _FORBIDDEN[family].search(source)
    if match:
        return f"the source uses {match.group(0)!r}, which is not allowed"
    return None


def check_program_proposal(
    data: Dict[str, Any],
    programs: Sequence[str],
) -> Optional[str]:
    """Check the agent's program run; return the problem, or None.

    ``{"binary": null, "reason": "..."}`` says that no program can work
    on generated input.
    """
    if "binary" in data and data["binary"] is None:
        if str(data.get("reason") or "").strip():
            return None
        return "say why no program fits in \"reason\""
    binary = str(data.get("binary") or "").strip()
    command = str(data.get("command") or "").strip()
    if binary not in programs:
        return f"binary {binary!r} is not one of the built programs"
    if data.get("input_kind") is not None and "{input}" not in command:
        return "the command does not read {input}"
    problem = _input_fields(data)
    if problem:
        return problem
    return bench.check_workload_command(command)


def _description(data: Dict[str, Any]) -> str:
    """Return the agent's one-line description without a final period.

    The size the agent asked for (input_mb) can change in the sizing, so
    it goes out of the text; the report adds the real size.
    """
    text = " ".join(str(data.get("description") or "").split())
    size = data.get("input_mb")
    if isinstance(size, int) and not isinstance(size, bool):
        text = re.sub(
            rf"\b{size}\s*(?:[KMG]i?B|megabytes?)\b(?:\s+of\b)?\s*", "",
            text,
            flags=re.IGNORECASE,
        )
    return " ".join(text.split()).rstrip(". ")[:200]


# Generated input is never a file in the package's own format. A harness
# that expects one fails (libjpeg-turbo: "Not a JPEG file", then "Input
# file size 8390144 does not match expected 2764800 bytes").
_INPUT_RULE = (
    "- The input is generated text or binary bytes of any size, never a "
    "file in the library's own format (no image, audio or compressed "
    "file). Use the bytes as raw material: an encoder or compressor "
    "encodes them; to time a decoder, encode the bytes once in memory "
    "and decode that result in the loop. Derive every size from the "
    "input length (for example as many full pixel rows as the bytes "
    "fill); never require an exact size.\n"
)

# Atesor compares two runs of the harness on the baseline build. A
# harness that reports success after an error, or prints a value that
# does not come from the input, fails that comparison and gives no
# reason. These rules keep both out.
_ERROR_RULE = (
    "- Check the result of every library call. On any error, print the "
    "reason to stderr and leave at once with status 1. Never print the "
    f"{CHECK_PREFIX} line on an error path: it reports success, and a "
    "shared cleanup block that prints it makes an error look like a "
    "pass.\n"
    "- Give every variable its value where you declare it. A jump over "
    "a declaration leaves the variable unset, and an unset value "
    "differs between two runs.\n"
)

# A lossy codec never returns its input. A harness that asserts
# equality fails on correct output, and the agent cannot see why from
# the error alone, so it repeats the mistake.
_LOSSY_RULE = (
    "- Some libraries lose data by design (audio, image and video "
    "codecs, and any quality or quantization setting). Decoding what "
    "such a library encoded gives data close to the input, never equal "
    "to it. Never compare the result with the input for equality "
    "there. Check the status codes and the sizes the API returns, and "
    "build the printed value from the decoded data itself.\n"
)

# Both an invalid frame size and a leftover-sized last block make a
# library reject the call, so this rule goes in every harness prompt.
_VALUES_RULE = (
    "- Pass only values that the API accepts. When the API takes a "
    "fixed set of sizes (for example an audio frame of 2.5, 5, 10, 20, "
    "40 or 60 ms), choose one of them and use whole units of the "
    "input; never pass a size computed from a leftover.\n"
)


def _feedback(
    prompt: str,
    data: Dict[str, Any],
    problem: str,
    earlier: Sequence[str] = (),
) -> str:
    """Return the prompt again, with the errors of every round so far.

    ``earlier`` holds the problems of the rounds before this one. An
    answer that fixes only the last error meets an earlier one again,
    so every round sees them all.
    """
    previous = json.dumps(
        {key: data.get(key) for key in sorted(data)}, indent=1
    )[-6000:]
    history = ""
    if earlier:
        lines = "\n".join(f"- {text[:300]}" for text in earlier)
        history = (
            "\nEarlier answers failed with these errors. Your new answer "
            f"must not bring any of them back:\n{lines}\n"
        )
    return (
        f"{prompt}\n\nYour previous answer did not work:\n{problem}\n"
        f"{history}\n"
        f"Your previous answer was:\n{previous}\n\n"
        "Fix the problem and reply with the complete JSON again."
    )


def _headline(problem: str) -> str:
    """Return the most telling line of an error: a compiler error first."""
    lines = [line.strip() for line in problem.splitlines() if line.strip()]
    for line in lines:
        if re.search(r"\berror\b|undefined reference|not found", line,
                     re.IGNORECASE) and not line.startswith("`"):
            return line
    return lines[0] if lines else ""


class _NoFit(Exception):
    """The agent says that no answer of this kind fits the package."""


def _error_key(problem: str) -> str:
    """Return an error without its numbers, to compare two rounds."""
    return re.sub(r"\d+(\.\d+)?", "#", _headline(problem))


def _agent_rounds(
    ctx: WorkloadContext,
    phase: str,
    prompt: str,
    check: Callable[[Dict[str, Any]], Optional[str]],
    attempt: Callable[[Dict[str, Any]], Tuple[Optional[bench.Workload], str]],
) -> Tuple[Optional[bench.Workload], str]:
    """Ask the agent, try its answer on the baseline, feed back errors.

    After :data:`AGENT_ROUNDS` rounds, another round runs only while
    each error is new, up to :data:`AGENT_MAX_ROUNDS`. An error that
    an earlier round met stops the rounds.
    """
    if ctx.agent is None:
        return None, "no LLM is available for the optimizer agent"
    current, problem = prompt, ""
    # Every error seen so far, not only the last one: an answer that
    # brings back the error of two rounds ago shows no progress either.
    previous: set = set()
    earlier: List[str] = []
    round_number = 0
    while round_number < AGENT_MAX_ROUNDS:
        round_number += 1
        if ctx.time_left() < AGENT_MIN_SECONDS:
            return None, "the time budget is used up"
        data, error = ctx.agent(phase, current, check)
        if data is None:
            return None, f"the optimizer agent gave no usable answer ({error})"
        try:
            workload, problem = attempt(data)
        except _NoFit as exc:
            ctx.say(f"the optimizer agent finds no fitting run: {exc}")
            return None, f"no fitting run: {exc}"
        if workload is not None:
            return workload, ""
        first = problem.strip().splitlines()[0] if problem.strip() else ""
        detail = _headline(problem)
        if detail and detail != first:
            first = f"{first.split(':')[0]}: {detail}"
        ctx.say(f"round {round_number}: {first[:160]}")
        key = _error_key(problem)
        if round_number >= AGENT_ROUNDS and key in previous:
            break
        previous.add(key)
        current = _feedback(prompt, data, problem, earlier)
        if problem not in earlier:
            earlier.append(problem)
    return None, f"{round_number} rounds failed; the last error: {problem}"


# Program runs


def _help_texts(ctx: WorkloadContext, programs: Sequence[str]) -> str:
    blocks = []
    for rel in programs[:MAX_PROGRAMS]:
        # The verified path can name another build's hash directory.
        path = bench.shell_path(ctx.tree, rel, ctx.paths)
        done = _sh(
            f"timeout 10 {path} --help </dev/null 2>&1 | head -n {HELP_LINES}",
            ctx.tree,
            timeout=60,
            env=ctx.qemu_env,
        )
        text = "\n".join(
            "    " + line for line in (done.stdout or "").splitlines()
        )
        blocks.append(f"- {rel}\n{text}" if text.strip() else f"- {rel}")
    return "\n".join(blocks)


def program_prompt(
    package: PackageProfile,
    name: str,
    help_texts: str,
) -> str:
    """Return the prompt that asks the agent for a program run."""
    return (
        "You choose a benchmark run for a ported package. Atesor runs it "
        "on the RISC-V machine many times and compares optimized builds "
        "of the package by its run time.\n"
        f"Package: {name}\n"
        f"Purpose: {package.purpose or 'unknown'}\n"
        "Programs built in the tree (paths relative to the tree), with "
        "the first lines of their --help output:\n"
        f"{help_texts}\n\n"
        "Write ONE shell command that runs one program on a generated "
        "input file and does the work that real users do with it.\n"
        "Rules:\n"
        "- Write {bin} for the program, {input} for the input file and "
        "{output} for an output file.\n"
        "- Every run must do the same work and print the same output (no "
        "timestamps, no random seeds, no temporary names).\n"
        "- If the program writes a different file each time (for example "
        "an encryption), chain a decode step with |.\n"
        "- No network, no package managers, no git, no rm, no sudo.\n"
        f"- input_kind is one of {list(bench.INPUT_KINDS)}, or null when "
        "the program needs no input file. Atesor scales input_mb so one "
        "run takes about 2 seconds. The smallest input is 1 MiB: a "
        "program that needs more than 5 seconds for 1 MiB must use a "
        "faster mode or level.\n"
        "- The description says what the run does, not the input size "
        "(Atesor adds the size).\n\n"
        "Reply with JSON only:\n"
        '{"binary": "<one path from the list>", "command": "...", '
        '"input_kind": "text", "input_mb": 16, '
        '"description": "<one line: what the run does>"}\n'
        "If no program can do real work on such generated input (for "
        "example each one needs an image, audio or video file), reply "
        '{"binary": null, "reason": "<why>"} instead.'
    )


def _program_workload(
    ctx: WorkloadContext,
    programs: Sequence[str],
) -> Tuple[Optional[bench.Workload], str]:
    ctx.say(
        "the optimizer agent chooses a run of "
        + (
            posixpath.basename(programs[0])
            if len(programs) == 1
            else f"one of {len(programs)} programs"
        )
    )
    prompt = program_prompt(
        ctx.package, ctx.state.repo_name, _help_texts(ctx, programs)
    )

    def attempt(data: Dict[str, Any]) -> Tuple[Optional[bench.Workload], str]:
        if data.get("binary") is None:
            raise _NoFit(" ".join(str(data.get("reason")).split())[:200])
        kind = data.get("input_kind")
        workload = bench.Workload(
            mode="program",
            command=str(data["command"]).strip(),
            binary=str(data["binary"]).strip(),
            input_kind=kind,
            input_mb=int(data.get("input_mb") or 16) if kind else 0,
            description=_description(data),
            origin="agent",
        )
        problem = size_by_input(ctx, workload)
        return (None, problem) if problem else (workload, "")

    return _agent_rounds(
        ctx,
        "workload-program",
        prompt,
        lambda data: check_program_proposal(data, programs),
        attempt,
    )


# C and C++ harness


def library_api(ctx: WorkloadContext) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return the exported API of the package's libraries, or an error."""
    payload, error = _run_program(
        _API_PROGRAM,
        {
            "tree": ctx.tree,
            "libraries": [
                ctx.paths.get(rel, rel) for rel in ctx.package.libraries
            ],
            "headers": 3,
            "prototypes": 80,
        },
        timeout=900,
    )
    if payload is None:
        return None, f"the API scan failed: {error}"
    if not payload.get("symbols"):
        return None, "the libraries export no function"
    if not payload.get("prototypes"):
        return None, "no header declares an exported function"
    return payload, ""


def c_harness_build(
    api: Dict[str, Any],
    libraries: Sequence[str],
    language: str,
) -> Tuple[str, List[str]]:
    """Return the harness source path and its build commands.

    The harness links the shared libraries when the build made any,
    else the static ones. A shared link also checks with ldd that the
    harness loads the library of the tree it runs in, not a system copy.
    """
    suffix = "cpp" if language == "c++" else "c"
    source = f"{HARNESS_DIR}/harness.{suffix}"
    binary = f"{HARNESS_DIR}/harness"
    compiler = '"$CXX" -std=gnu++17' if language == "c++" else '"$CC"'
    includes = " ".join(
        f"-I{shlex.quote(directory)}" for directory in api["include_dirs"]
    )
    shared = [rel for rel in libraries if ".so" in posixpath.basename(rel)]
    static = [rel for rel in libraries if rel.endswith(".a")]
    # A build that names its output directory after a hash of its flags
    # puts each library somewhere else in every tree: the placeholder
    # becomes the path that the tree's inspection found
    # (install_workload).
    words = [bench.path_word(rel) for rel in shared or static]
    names = " ".join(words)
    if shared:
        links = names + " " + " ".join(
            f'-Wl,--disable-new-dtags,-rpath,"$PWD/$(dirname {word})"'
            for word in words
        )
    else:
        links = (
            f"-Wl,--start-group {names} -Wl,--end-group -lpthread -ldl"
        )
    commands = [
        f"mkdir -p {HARNESS_DIR} && {compiler} -O2 -w "
        f"{includes} {source} -o {binary} {links} -lm"
    ]
    if shared:
        commands.append(
            f'ldd {binary} | grep -F "$PWD/" >/dev/null || '
            '{ echo "the harness does not load the library of this tree:" '
            f">&2; ldd {binary} >&2; exit 1; }}"
        )
    return source, commands


def c_harness_prompt(
    package: PackageProfile,
    name: str,
    api: Dict[str, Any],
    language: str,
) -> str:
    """Return the prompt that asks the agent for a C or C++ harness."""
    headers = "\n".join(
        f'#include "{posixpath.basename(rel)}"' for rel in api["headers"]
    )
    prototypes = "\n".join(f"  {line}" for line in api["prototypes"])
    title = "C++" if language == "c++" else "C"
    return (
        f"You write a benchmark driver (a harness) in {title} for a "
        "ported library. Atesor compiles it against each optimized build "
        "of the library, runs it on the RISC-V machine many times and "
        "compares the builds by its run time.\n"
        f"Package: {name}\n"
        f"Purpose: {package.purpose or 'unknown'}\n"
        f"Library files: {', '.join(package.libraries)}\n"
        "Headers that declare the API (the build adds their include "
        f"directories):\n{headers}\n"
        "Copy those #include lines as they stand. The library's file "
        "name is not a header name: a library libfoo.so does not give "
        "a header \"libfoo.h\". Include no other header of this "
        "library, only the C standard ones you need.\n"
        "Functions that the library exports, as its headers declare "
        f"them:\n{prototypes}\n\n"
        f"Write ONE {title} source file with int main(int argc, char "
        "**argv):\n"
        "- argv[1] is the path of an input file with generated data "
        "(/dev/null when you ask for no input). argv[2] is a positive "
        "number N.\n"
        + _INPUT_RULE
        + "- Read the input file into memory once. Then do the library's "
        "main work N times on that data: the operations that real users "
        "of this library spend their time in.\n"
        "- Each of the N repetitions does the same work. One repetition "
        "should take 10-500 ms on a 1.5 GHz in-order core.\n"
        + _ERROR_RULE
        + _VALUES_RULE
        + _LOSSY_RULE
        + f"- At the end print exactly one line: \"{CHECK_PREFIX} <value>\", "
        "where the value depends on the results of all N repetitions "
        "(for example a sum of output bytes, or a checksum the library "
        "computes). The same input and N must print the same line.\n"
        "- Use only the C standard library and this library's API. No "
        "threads, no clocks, no random numbers, no other files, no "
        "network.\n\n"
        "Reply with JSON only. Put the whole file in \"source\" as one "
        "JSON string (escape newlines as \\n and quotes as \\\"):\n"
        '{"input_kind": "binary", "input_mb": 8, '
        '"description": "<one line: what the harness does>", '
        '"source": "<the complete file>"}\n'
        f"input_kind is one of {list(bench.INPUT_KINDS)} or null. The "
        "description says what the harness does, not the input size."
    )


def _c_harness(ctx: WorkloadContext) -> Tuple[Optional[bench.Workload], str]:
    api, error = library_api(ctx)
    if api is None:
        return None, error
    language = "c++" if api.get("cxx") else "c"
    title = "C++" if language == "c++" else "C"
    ctx.say(
        f"the optimizer agent writes a {title} harness for "
        f"{len(api['prototypes'])} exported functions"
    )
    source_path, build = c_harness_build(
        api, ctx.package.libraries, language
    )
    symbols = set(api["symbols"])
    prompt = c_harness_prompt(ctx.package, ctx.state.repo_name, api, language)

    def check(data: Dict[str, Any]) -> Optional[str]:
        source = data.get("source")
        problem = _source_problem(source, language) or _input_fields(data)
        if problem:
            return problem
        if "main" not in source:
            return "the source has no main function"
        used = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", source)) & symbols
        if not used:
            return "the source calls no function that the library exports"
        return None

    def attempt(data: Dict[str, Any]) -> Tuple[Optional[bench.Workload], str]:
        kind = data.get("input_kind")
        workload = bench.Workload(
            mode="harness",
            command="{bin} {input} {iters}",
            binary=f"{HARNESS_DIR}/harness",
            input_kind=kind,
            input_mb=int(data.get("input_mb") or 8) if kind else 0,
            description=_description(data),
            check_prefix=CHECK_PREFIX,
            files={source_path: str(data["source"])},
            build=list(build),
            source_language=language,
            origin="agent",
        )
        error = install_workload(workload, ctx.tree, ctx.env,
                                 ctx.build_timeout, ctx.paths)
        if error:
            return None, f"the harness did not build: {error}"
        problem = size_by_iterations(ctx, workload)
        return (None, problem) if problem else (workload, "")

    return _agent_rounds(ctx, "harness-c", prompt, check, attempt)


# Go


def go_benchmarks(
    ctx: WorkloadContext,
) -> Tuple[Optional[bench.Workload], str]:
    """Return the module's own Go benchmarks as a workload, or why not.

    The package with the most benchmarks wins. Its test binary is built
    in every tree; the run uses a fixed iteration count, so B and V do
    the same work.
    """
    listing = _sh("go test -list '^Benchmark' ./...", ctx.tree, timeout=900,
                  env=ctx.env)
    if not listing.success:
        return None, "go test -list failed"
    package, count, names = None, 0, 0
    for line in (listing.stdout or "").splitlines():
        if line.startswith("Benchmark"):
            names += 1
        elif line.startswith(("ok", "?")):
            parts = line.split()
            if line.startswith("ok") and len(parts) >= 2 and names > count:
                package, count = parts[1], names
            names = 0
    if not package:
        return None, "the Go module has no benchmarks"
    workdir, error = _go_package_dir(ctx, package)
    if error:
        return None, error
    workload = bench.Workload(
        mode="gobench",
        command=(
            "{bin} -test.run '^$' -test.bench . "
            "-test.benchtime {iters}x -test.count 1"
        ),
        binary=bench.GOBENCH_BINARY,
        description=f"{count} Go benchmarks of {package}",
        package=package,
        deterministic=False,
        workdir=workdir,
        build=[
            f"go test -c -o {bench.GOBENCH_BINARY} {shlex.quote(package)}"
        ],
        origin="upstream",
    )
    error = install_workload(workload, ctx.tree, ctx.env, ctx.build_timeout,
                             ctx.paths)
    if error:
        return None, f"the benchmark binary did not build: {error}"
    problem = size_by_iterations(ctx, workload)
    if problem:
        return None, problem
    workload.description += f", {workload.iterations} iterations each"
    return workload, ""


def _go_package_dir(
    ctx: WorkloadContext,
    package: str,
) -> Tuple[Optional[str], str]:
    located = _sh(f"go list -f '{{{{.Dir}}}}' {shlex.quote(package)}",
                  ctx.tree, env=ctx.env)
    lines = (located.stdout or "").strip().splitlines()
    directory = lines[-1].strip() if located.success and lines else ""
    tree = ctx.tree.rstrip("/")
    if directory == tree:
        return None, ""
    if directory.startswith(tree + "/"):
        return directory[len(tree) + 1:], ""
    return None, f"the package {package} is not in the tree"


def _go_packages(ctx: WorkloadContext) -> List[Dict[str, Any]]:
    """Return the module's three largest non-main packages.

    Sorted by Go file count so the package most worth benchmarking
    comes first; the cap keeps the agent prompt short. Testdata and
    vendored packages are skipped.
    """
    listing = _sh(
        "go list -f '{{.ImportPath}}|{{.Name}}|{{.Dir}}|{{len .GoFiles}}' "
        "./...",
        ctx.tree,
        timeout=900,
        env=ctx.env,
    )
    packages = []
    for line in (listing.stdout or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) != 4 or parts[1] == "main":
            continue
        if "/testdata" in parts[0] or "/vendor/" in parts[0]:
            continue
        try:
            files = int(parts[3])
        except ValueError:
            continue
        packages.append({"import": parts[0], "name": parts[1],
                         "dir": parts[2], "files": files})
    packages.sort(key=lambda item: (-item["files"], item["import"]))
    return packages[:3]


def _go_harness(ctx: WorkloadContext) -> Tuple[Optional[bench.Workload], str]:
    packages = _go_packages(ctx)
    if not packages:
        return None, "the module has no library package"
    docs = []
    for item in packages:
        done = _sh(f"go doc -short {shlex.quote(item['import'])} | head -n 40",
                   ctx.tree, env=ctx.env)
        docs.append(
            f"package {item['name']} (import \"{item['import']}\"):\n"
            + "\n".join("  " + line for line in
                        (done.stdout or "").splitlines())
        )
    ctx.say("the optimizer agent writes a Go benchmark for the package API")
    prompt = (
        "You write a Go benchmark for a ported Go module. Atesor builds "
        "it with `go test -c` in each optimized build, runs it on the "
        "RISC-V machine with a fixed iteration count and compares the "
        "builds by run time.\n"
        f"Module: {ctx.state.repo_name}\n"
        f"Purpose: {ctx.package.purpose or 'unknown'}\n"
        "Packages and their API (go doc -short):\n"
        + "\n".join(docs)
        + "\n\nWrite ONE Go test file for one of these packages:\n"
        "- Use the external test package (package <name>_test) and "
        "import the package.\n"
        f"- Define func {GO_BENCHMARK}(b *testing.B). Read the file named "
        "by the environment variable ATESOR_INPUT once (it is /dev/null "
        "when you ask for no input), call b.ResetTimer(), then do the "
        "package's main work b.N times.\n"
        + _INPUT_RULE
        + _VALUES_RULE
        + _LOSSY_RULE
        + "- Check every result and call b.Fatal on an error.\n"
        f"- After the loop call fmt.Println(\"{CHECK_PREFIX}\", value), "
        "where the value depends on the results of all b.N iterations "
        "and on nothing else.\n"
        "- No network, no other files, no time.Now, no math/rand, no "
        "goroutines that outlive the benchmark.\n\n"
        "Reply with JSON only. Put the whole file in \"source\" as one "
        "JSON string:\n"
        '{"package": "<import path>", "input_kind": "json", '
        '"input_mb": 8, "description": "<one line>", "source": "<file>"}\n'
        f"input_kind is one of {list(bench.INPUT_KINDS)} or null."
    )
    by_import = {item["import"]: item for item in packages}

    def check(data: Dict[str, Any]) -> Optional[str]:
        problem = _source_problem(data.get("source"), "go")
        problem = problem or _input_fields(data)
        if problem:
            return problem
        if data.get("package") not in by_import:
            return f"package must be one of {sorted(by_import)}"
        source = data["source"]
        if not re.search(
            rf"func\s+{GO_BENCHMARK}\s*\(\s*\w+\s+\*testing\.B\s*\)", source
        ):
            return f"the source has no func {GO_BENCHMARK}(b *testing.B)"
        name = by_import[data["package"]]["name"]
        if not re.search(rf"^package\s+{re.escape(name)}(_test)?\s*$",
                         source, re.MULTILINE):
            return f"the package clause must be {name}_test"
        return None

    def attempt(data: Dict[str, Any]) -> Tuple[Optional[bench.Workload], str]:
        item = by_import[data["package"]]
        workdir, error = _go_package_dir(ctx, item["import"])
        if error:
            return None, error
        kind = data.get("input_kind")
        rel = f"{workdir}/{GO_HARNESS_FILE}" if workdir else GO_HARNESS_FILE
        workload = bench.Workload(
            mode="gobench",
            command=(
                f"{{bin}} -test.run '^$' -test.bench '^{GO_BENCHMARK}$' "
                "-test.benchtime {iters}x -test.count 1"
            ),
            binary=bench.GOBENCH_BINARY,
            input_kind=kind,
            input_mb=int(data.get("input_mb") or 8) if kind else 0,
            description=_description(data),
            package=item["import"],
            workdir=workdir,
            check_prefix=CHECK_PREFIX,
            files={rel: str(data["source"])},
            build=[
                f"go test -c -o {bench.GOBENCH_BINARY} "
                f"{shlex.quote(item['import'])}"
            ],
            env={"ATESOR_INPUT": "{input}"},
            source_language="go",
            origin="agent",
        )
        error = install_workload(workload, ctx.tree, ctx.env,
                                 ctx.build_timeout, ctx.paths)
        if error:
            return None, f"the benchmark did not build: {error}"
        problem = size_by_iterations(ctx, workload)
        return (None, problem) if problem else (workload, "")

    return _agent_rounds(ctx, "harness-go", prompt, check, attempt)


# Rust


def _rust_crate(ctx: WorkloadContext) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return the package's library crate, or why there is none.

    A workspace can hold several crates; a library target at the tree
    root wins over one in a subdirectory.
    """
    done = _sh("cargo metadata --no-deps --format-version 1", ctx.tree,
               timeout=600, env=ctx.env)
    try:
        metadata = json.loads(done.stdout or "")
    except ValueError:
        return None, "cargo metadata failed"
    tree = ctx.tree.rstrip("/")
    best = None
    for package in metadata.get("packages") or []:
        manifest = str(package.get("manifest_path") or "")
        directory = posixpath.dirname(manifest)
        for target in package.get("targets") or []:
            kinds = set(target.get("kind") or [])
            if not kinds & {"lib", "rlib", "dylib", "cdylib", "staticlib"}:
                continue
            rank = 0 if directory == tree else 1
            if best is None or rank < best["rank"]:
                best = {
                    "rank": rank,
                    "package": package.get("name"),
                    "crate": str(target.get("name")).replace("-", "_"),
                    "dir": directory,
                    "src": target.get("src_path"),
                }
    if best is None:
        return None, "the Cargo package has no library target"
    if not str(best["dir"]).startswith(tree):
        return None, "the library crate is not in the tree"
    return best, ""


def _rust_harness(
    ctx: WorkloadContext,
) -> Tuple[Optional[bench.Workload], str]:
    crate, error = _rust_crate(ctx)
    if crate is None:
        return None, error
    items = _sh(
        "grep -hE '^\\s*pub (fn|struct|enum|trait|mod|use|type|const) ' "
        f"{shlex.quote(str(crate['src']))} | head -n 80",
        ctx.tree,
    )
    readme = _sh(
        "for f in README.md README README.rst; do "
        '[ -f "$f" ] && head -n 80 "$f" && break; done; true',
        crate["dir"],
    )
    ctx.say(
        f"the optimizer agent writes a Cargo example for crate "
        f"{crate['crate']}"
    )
    prompt = (
        "You write a benchmark program for a ported Rust crate, as a "
        "Cargo example. Atesor builds it with `cargo build --release "
        f"--example {RUST_EXAMPLE}` in each optimized build, runs it on "
        "the RISC-V machine many times and compares the builds by run "
        "time.\n"
        f"Crate: {crate['crate']} (package {crate['package']})\n"
        f"Purpose: {ctx.package.purpose or 'unknown'}\n"
        "Public items of the library:\n"
        f"{_tail(items.stdout, 4000)}\n"
        "README excerpt:\n"
        f"{_tail(readme.stdout, 3000)}\n\n"
        f"Write ONE file examples/{RUST_EXAMPLE}.rs with fn main():\n"
        "- args[1] is the path of an input file with generated data "
        "(/dev/null when you ask for no input); args[2] is a positive "
        "number N.\n"
        + _INPUT_RULE
        + _VALUES_RULE
        + _LOSSY_RULE
        + "- Read the input once, then do the crate's main work N times.\n"
        "- Check every result; on an error print it to stderr and call "
        "std::process::exit(1).\n"
        f"- At the end print one line: println!(\"{CHECK_PREFIX} {{}}\", "
        "value), where the value depends on the results of all N "
        "repetitions.\n"
        "- Use only std and this crate. No threads, no time, no random "
        "numbers, no network.\n\n"
        "Reply with JSON only. Put the whole file in \"source\" as one "
        "JSON string:\n"
        '{"input_kind": "text", "input_mb": 8, "description": "<one '
        'line>", "source": "<file>"}\n'
        f"input_kind is one of {list(bench.INPUT_KINDS)} or null."
    )
    tree = ctx.tree.rstrip("/")
    rel_dir = crate["dir"][len(tree):].lstrip("/")
    source_rel = posixpath.join(rel_dir, "examples", f"{RUST_EXAMPLE}.rs")

    def check(data: Dict[str, Any]) -> Optional[str]:
        problem = _source_problem(data.get("source"), "rust")
        problem = problem or _input_fields(data)
        if problem:
            return problem
        if "fn main" not in data["source"]:
            return "the source has no fn main"
        if crate["crate"] not in data["source"]:
            return f"the source does not use the crate {crate['crate']}"
        return None

    def attempt(data: Dict[str, Any]) -> Tuple[Optional[bench.Workload], str]:
        kind = data.get("input_kind")
        workload = bench.Workload(
            mode="harness",
            command="{bin} {input} {iters}",
            binary=f"target/release/examples/{RUST_EXAMPLE}",
            input_kind=kind,
            input_mb=int(data.get("input_mb") or 8) if kind else 0,
            description=_description(data),
            package=crate["package"],
            check_prefix=CHECK_PREFIX,
            files={source_rel: str(data["source"])},
            build=[
                f"cargo build --release --example {RUST_EXAMPLE} -p "
                f"{shlex.quote(str(crate['package']))}"
            ],
            source_language="rust",
            origin="agent",
        )
        error = install_workload(workload, ctx.tree, ctx.env,
                                 ctx.build_timeout, ctx.paths)
        if error:
            return None, f"the example did not build: {error}"
        problem = size_by_iterations(ctx, workload)
        return (None, problem) if problem else (workload, "")

    return _agent_rounds(ctx, "harness-rust", prompt, check, attempt)


# Choice


def _test_workload(
    ctx: WorkloadContext,
) -> Tuple[Optional[bench.Workload], str]:
    tests = getattr(ctx.state, "package_tests", None) or {}
    if ctx.package.language == "go":
        # go test caches results, so a second run does no work.
        return None, "Go tests are not timed (go test caches results)"
    if not tests.get("command") or tests.get("status") != "passed":
        return None, (
            f"the package tests did not pass in the port "
            f"({tests.get('status') or 'not run'})"
        )
    workload = bench.Workload(
        mode="tests",
        command=tests["command"],
        description=f"the package tests ({tests.get('framework')})",
        weak=True,
        deterministic=False,
        origin="tests",
    )
    problem = check_duration(ctx, workload)
    return (None, problem) if problem else (workload, "")


def helpers_using_library(
    ctx: WorkloadContext,
) -> Tuple[List[str], str]:
    """Return the test and example programs that use the library.

    A test helper that does not call the package's library (a checksum
    tool of the test suite) times code that has nothing to do with the
    package.

    Returns:
        ``(helpers, "")``, or ``([], why none qualifies)``.
    """
    package = ctx.package
    if not package.libraries:
        return [], "the package builds no library for its test programs"
    payload, error = _run_program(
        _USES_PROGRAM,
        {
            "tree": ctx.tree,
            "libraries": [
                ctx.paths.get(rel, rel) for rel in package.libraries
            ],
            "helpers": [ctx.paths.get(rel, rel) for rel in package.helpers],
        },
        timeout=600,
    )
    if payload is None:
        return [], f"the library check of the test programs failed: {error}"
    uses = payload.get("uses") or {}
    chosen = [
        rel for rel in package.helpers if uses.get(ctx.paths.get(rel, rel))
    ]
    if not chosen:
        return [], "no test or example program uses the package's library"
    return chosen, ""


def _helper_workload(
    ctx: WorkloadContext,
) -> Tuple[Optional[bench.Workload], str]:
    helpers, problem = helpers_using_library(ctx)
    if not helpers:
        return None, problem
    left_out = len(ctx.package.helpers) - len(helpers)
    if left_out:
        ctx.say(
            f"{left_out} of {len(ctx.package.helpers)} test programs do not "
            "use the package's library; they are left out"
        )
    return _program_workload(ctx, helpers)


def strategies(package: PackageProfile) -> List[str]:
    """Return the workload sources to try, in order, for a package type.

    Args:
        package: The package profile.

    Returns:
        Names from ``go-benchmarks``, ``program``, ``harness``,
        ``helper-program`` and ``tests``.
    """
    order: List[str] = []
    if package.language == "go":
        order.append("go-benchmarks")
    if package.programs:
        order.append("program")
    if package.libraries or package.language == "go":
        order.append("harness")
    if package.helpers:
        order.append("helper-program")
    order.append("tests")
    return order


def choose_workload(
    ctx: WorkloadContext,
) -> Tuple[Optional[bench.Workload], List[str]]:
    """Choose and check the workload for the package type.

    Returns:
        ``(workload or None, notes)``. Each note says why one source
        gave no workload.
    """
    notes: List[str] = []
    runners: Dict[str, Callable[[], Tuple[Optional[bench.Workload], str]]]
    runners = {
        "go-benchmarks": lambda: go_benchmarks(ctx),
        "program": lambda: _program_workload(ctx, ctx.package.programs),
        "harness": lambda: {
            "go": _go_harness,
            "rust": _rust_harness,
        }.get(ctx.package.language, _c_harness)(ctx),
        "helper-program": lambda: _helper_workload(ctx),
        "tests": lambda: _test_workload(ctx),
    }
    for name in strategies(ctx.package):
        try:
            workload, reason = runners[name]()
        except Exception as exc:  # one source failing must not stop it
            logger.exception(f"Workload source {name} failed")
            workload, reason = None, f"{type(exc).__name__}: {exc}"
        if workload is not None:
            return workload, notes
        notes.append(f"{name}: {reason}")
    return None, notes


def describe(workload: bench.Workload) -> str:
    """Return one line about a workload for the console and the report."""
    who = {
        "agent": "by the optimizer agent",
        "upstream": "from the package",
        "tests": "from the package",
    }.get(workload.origin, "")
    language = {"c": "C", "c++": "C++", "rust": "Rust", "go": "Go"}.get(
        workload.source_language or "", ""
    )
    kind = {
        "program": "program run",
        "harness": f"{language} harness".strip(),
        "gobench": "Go benchmark",
        "tests": "package tests (weak signal)",
    }.get(workload.mode, workload.mode)
    size = f", {workload.input_mb} MiB {workload.input_kind} input" if (
        workload.input_kind
    ) else ""
    count = f", N={workload.iterations}" if workload.iterations else ""
    seconds = (
        f", {workload.run_seconds:.2f} s per run"
        if workload.run_seconds
        else ""
    )
    return (
        f"{kind} {who}".strip()
        + f": {workload.description}{size}{count}{seconds}"
    )
