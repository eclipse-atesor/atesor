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

"""Build, check and measure optimized builds ("tunes") of a verified port.

Performance mode (``--optimize``) runs this after a verified build:

1. Probe the machine (:mod:`src.machine`) and profile the package
   (:mod:`src.workload`): its language, and whether it builds a
   library, programs or both.
2. Rebuild the verified build in a copy, the baseline tree
   ``<repo>.opt/base``. Each tune is compared with this copy, built the
   same way, so only the tune's change counts. Its code must equal the
   verified build's code; Atesor says so when it does not.
3. Choose the workload for the package type (:mod:`src.workload`): the
   package's own benchmarks, a program run, or a harness that the
   optimizer agent writes against the library API. It must do fixed
   work per run.
4. Turn the probe into tunes. C/C++: core tuning (``-mtune``), machine
   ISA (``-march``), vector ISA, optimization level, link-time
   optimization, a second compiler, then profile-guided optimization
   (PGO) on top of the best. Go: the GORISCV64 level, then PGO. Rust:
   the target CPU or features, then LTO. A tune that the build already
   has (for example ``-O3``) is skipped.
5. Build each tune in a fresh copy of the patched tree at
   ``<repo>.opt/<id>``: compiled outputs are removed, then the commands
   of the verified build run again. Package installs, install steps,
   source control steps and writes outside the tree are not repeated.
   The flags reach the compiler through wrapper scripts (``CC``/``CXX``
   and ``MAKEFLAGS``), so the project's own flags stay and each compiler
   call is logged. The workload's harness builds in each tree too.
6. Gates: (1) the artifacts exist, are riscv64, and the tune's settings
   reached them; code identical to the baseline means no effect, and
   code identical to an earlier tune reuses its result. Every later
   step uses the artifact paths that this check found. (2) The package
   tests pass when the baseline tests pass; an artifact that the tests
   build again gets the tune's build back. (3) The workload output
   equals the baseline output, and a run does not change its input.
7. Measure on native hardware only (:mod:`src.bench`); under QEMU a
   tune gets the gates and no speed claim. A faster tune must win again
   in a second check (another code layout on both builds, or a second
   session for Go) before Atesor accepts it. Once a tune is faster, a
   later tune runs only when the budget still covers it and that second
   check; a gain that got no second timing is "not confirmed", not
   "no gain".

The verified build never changes. The accepted tune's riscv64 artifacts
are copied to ``<repo>.opt/accepted``; every other tree is removed.
"""

import json
import logging
import os
import posixpath
import re
import shlex
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import bench
from . import workload as workloads
from .machine import (
    MachineProfile,
    align_flags,
    check_integrity,
    forbidden_classes,
    goriscv64_for,
    integrity_failures,
    march_for,
    probe_machine,
    rust_flags_for,
    tune_for,
)
from .target import is_native
from .tools import execute_command
from .workload import rewrite_for_tree

logger = logging.getLogger(__name__)

BUDGET_SECONDS = 3600
# Six ladder tunes at most (c_ladder yields up to six), plus one PGO.
MAX_VARIANTS = 7
BUILD_TIMEOUT = 1800
MIN_STEP_SECONDS = 300
# PGO builds and runs twice: a training build and run, then the tune.
PGO_COST_FACTOR = 2.0
OPT_SUFFIX = ".opt"
MEASURE_SEED = "atesor-measure"
TRAIN_SEED = "atesor-train"
# Another function alignment changes the code layout without changing
# the code; the second check applies it to both builds, so a gain from
# a lucky layout cannot pass.
SECOND_LAYOUT_FLAG = "-falign-functions=64"
# LLVM takes the alignment as log2 bytes: 6 is 64, the same change.
RUST_SECOND_LAYOUT = "-C llvm-args=-align-all-functions=6"
BASE_ID = "base"
# Stands for the variant tree in stored flags; variant_env() fills it.
TREE_TOKEN = "@TREE@"
# The ISA scan reads at most this many artifacts, and this many bytes of
# code in each. Python in the sandbox is much slower under QEMU.
SCAN_LIMITS = {"native": (8, 1 << 25), "qemu": (4, 1 << 20)}
# Top-level directories that hold build output only. They are removed
# from a variant tree when git tracks no file in them.
BUILD_DIRS = (
    "build",
    "_build",
    "builddir",
    "target",
    "cmake-build-release",
    "cmake-build-debug",
)
_LANGUAGES = {
    "go": "go",
    "cargo": "rust",
    "rust": "rust",
}
# A variant build never repeats these steps. The packages are installed
# already; an install step would overwrite the baseline's files; the
# copied tree holds the result of every source control step and
# download.
_PACKAGE_STEP = re.compile(
    r"^(apt-get|apt|aptitude|apk|dnf|yum|zypper|pacman|dpkg|rustup)\b"
    r"|^(pip3?|python3?\s+-m\s+pip)\s+install\b"
)
_INSTALL_STEP = re.compile(
    r"^(make|gmake|ninja|samu|meson)\b[^|;&]*\binstall\b"
    r"|^cmake\s+--install\b"
    r"|^(go|cargo)\s+install\b"
    r"|^ldconfig\b"
)
_SOURCE_STEP = re.compile(
    r"^git\s+(clone|fetch|pull|checkout|switch|submodule|reset|restore|"
    r"apply|am|stash|cherry-pick|merge|rebase)\b"
    r"|^(patch|wget|curl)\b"
)
_OUTPUT_OPTIONS = (
    "-o",
    "-O",
    "-B",
    "-C",
    "--output",
    "--out-dir",
    "--target-dir",
    "--build",
    "--directory",
)
_COPY_COMMANDS = ("cp", "mv", "install", "ln", "rsync")
_HARMLESS = ("cd", "export", "true", "echo", "ls", "pwd", ":")
_SHELL_OPERATORS = ("|", "||", "&", ";", "<", ">", ">>")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Compiler calls of configure checks, not of the package's sources.
_PROBE_CALL = re.compile(
    r"CMakeScratch|CMakeTmp|TryCompile|cmTC_|conftest|sanitycheck|"
    r"CompilerId|CMakeC(XX)?CompilerABI|meson-private"
)
_SOURCE_SUFFIX = re.compile(r"\.(c|cc|cpp|cxx|c\+\+|C|m|mm|S|s)$")

# Verdicts of a tune that a measurement timed.
_TIMED = ("faster", "slower", "no change", "not confirmed", "not checked")
UNRELIABLE_BASELINE = (
    "the unchanged baseline failed 2 timing runs, so no timing can be "
    "trusted (the machine or the workload gives wrong results)"
)

# The techniques, in the words the console and the report use.
CORE_TUNING = "core tuning"
MACHINE_ISA = "machine ISA"
VECTOR_ISA = "vector ISA"
OPT_LEVEL = "optimization level"
LTO = "link-time optimization"
COMPILER = "compiler"
PGO = "profile-guided optimization"
TARGET_CPU = "target CPU"

ExecuteTests = Callable[
    [str, Optional[Dict[str, str]]], Optional[Dict[str, Any]]
]
Progress = Callable[[str], None]


@dataclass
class Variant:
    """One optimized build to try: a tune.

    Attributes:
        vid: A short id, also the tree name, for example ``tune2``.
        title: The whole setting in one line, for example
            ``GCC -mtune=X -march=Y``.
        language: ``c``, ``go`` or ``rust``.
        compiler: The C compiler command for ``c`` variants, or None.
        flags: Flags the wrapper appends to every compiler call.
            :data:`TREE_TOKEN` stands for the variant tree.
        env: Extra environment (GORISCV64, RUSTFLAGS, CARGO_PROFILE_*).
        isa: The ``-march``, GORISCV64 or RUSTFLAGS ISA setting, or None.
        note: Why the variant may be risky, or None.
        pgo: ``generate`` or ``use`` for PGO builds, else None.
        label: The console name, for example ``Tune-2``.
        technique: What the tune changes, for example ``machine ISA``.
        value: The new setting of this tune, for example
            ``-march=rv64gc_zba_zbb``.
        keeps: The label of the tune whose settings this one keeps, or
            None.
    """

    vid: str
    title: str
    language: str
    compiler: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    isa: Optional[str] = None
    note: Optional[str] = None
    pgo: Optional[str] = None
    label: str = ""
    technique: str = ""
    value: str = ""
    keeps: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def language_of(build_system: Optional[str]) -> str:
    """Return ``go``, ``rust`` or ``c`` for a build system name."""
    return _LANGUAGES.get((build_system or "").lower(), "c")


def _cxx_for(compiler: str) -> str:
    """Return the C++ driver of a C compiler command.

    A versioned clang maps to its versioned C++ name (``clang-19`` to
    ``clang++-19``); an unknown compiler name comes back unchanged.
    """
    if compiler == "gcc":
        return "g++"
    if compiler.startswith("clang"):
        return compiler.replace("clang", "clang++", 1)
    return compiler


def _best_clang(profile: MachineProfile) -> Optional[str]:
    """Return the newest probed clang that emits bitmanip code, or None."""
    compilers = profile.toolchains.get("compilers", {})
    usable = [
        name
        for name, facts in compilers.items()
        if facts.get("family") == "clang" and facts.get("emits_bitmanip")
    ]
    if not usable:
        return None
    # The highest version suffix wins; plain "clang" sorts first.
    return sorted(usable, key=lambda name: (len(name), name))[-1]


def c_ladder(profile: MachineProfile) -> List[Variant]:
    """Return the C/C++ tunes: core, ISA, vector, -O3, LTO, then clang.

    Each GCC tune keeps the flags of the tunes before it. A tune whose
    precondition fails (no known core, no ISA extras, no vector unit,
    no LTO support, no usable clang) is left out. Without GCC in the
    probe, the list is empty.
    """
    compilers = profile.toolchains.get("compilers", {})
    if "gcc" not in compilers:
        return []
    variants: List[Variant] = []
    align = list(align_flags(profile, "gcc"))
    flags = list(align)

    def add(technique: str, value: List[str], title: str, **extra) -> None:
        if not variants:
            # The first tune also shows the alignment flag it brings.
            value = value + [flag for flag in align if flag not in value]
        variants.append(
            Variant(
                f"tune{len(variants) + 1}",
                title,
                "c",
                extra.pop("compiler", "gcc"),
                extra.pop("flags", list(flags)),
                technique=technique,
                value=" ".join(value),
                **extra,
            )
        )

    tune = tune_for(profile, "gcc")
    if tune:
        flags.insert(0, tune)
        add(CORE_TUNING, [tune], f"GCC {' '.join(flags)}")
    march = march_for(profile, "gcc")
    if march:
        flags.append(f"-march={march}")
        add(
            MACHINE_ISA,
            [f"-march={march}"],
            f"GCC {' '.join(flags)}",
            isa=march,
        )
    vector = march_for(profile, "gcc", include_vector=True)
    if vector and vector != march:
        flags = [flag for flag in flags if not flag.startswith("-march=")]
        flags.append(f"-march={vector}")
        march = vector
        add(
            VECTOR_ISA,
            [f"-march={vector}"],
            f"GCC {' '.join(flags)}",
            isa=vector,
            note=f"VLEN {profile.vlen or 'unknown'}",
        )
    flags.append("-O3")
    add(OPT_LEVEL, ["-O3"], f"GCC {' '.join(flags)}", isa=march)
    gcc_flags = compilers["gcc"].get("flags") or []
    if "-flto=auto" in gcc_flags and "-ffat-lto-objects" in gcc_flags:
        lto = ["-flto=auto", "-ffat-lto-objects"]
        add(
            LTO,
            lto,
            f"GCC {' '.join(flags + lto)}",
            flags=flags + lto,
            isa=march,
            note="fat LTO objects, so static archives still link",
        )
    clang = _best_clang(profile)
    if clang:
        clang_flags = []
        clang_tune = tune_for(profile, clang)
        if clang_tune:
            clang_flags.append(clang_tune)
        clang_flags += align_flags(profile, clang)
        clang_march = march_for(profile, clang, include_vector=True)
        if clang_march:
            clang_flags.append(f"-march={clang_march}")
        clang_flags.append("-O3")
        add(
            COMPILER,
            [clang],
            f"{clang} {' '.join(clang_flags)}",
            compiler=clang,
            flags=clang_flags,
            isa=clang_march,
            note="the same kind of flags with another compiler",
        )
    labels: Dict[str, str] = {}
    for index, variant in enumerate(variants, 1):
        variant.label = f"Tune-{index}"
        labels[variant.vid] = variant.label
    for index, variant in enumerate(variants):
        if variant.technique == COMPILER:
            gcc_tunes = [v for v in variants if v.compiler == "gcc"]
            best = [v for v in gcc_tunes if v.technique == OPT_LEVEL]
            variant.keeps = best[-1].label if best else None
        elif index and variants[index - 1].compiler == "gcc":
            variant.keeps = variants[index - 1].label
    return variants


def go_ladder(profile: MachineProfile) -> List[Variant]:
    """Return the Go tunes: the GORISCV64 level from the probe.

    The list is empty when the probe gives ``rva20u64``, Go's default
    level.
    """
    level, note = goriscv64_for(profile)
    if level == "rva20u64":
        return []
    return [
        Variant(
            "tune1",
            f"GORISCV64={level}",
            "go",
            env={"GORISCV64": level},
            isa=level,
            note=note,
            label="Tune-1",
            technique=MACHINE_ISA,
            value=f"GORISCV64={level}",
        )
    ]


def rust_ladder(profile: MachineProfile) -> List[Variant]:
    """Return the Rust tunes: target CPU or features, then fat LTO."""
    variants: List[Variant] = []
    env: Dict[str, str] = {}
    rustflags = rust_flags_for(profile)
    if rustflags:
        env = {"RUSTFLAGS": rustflags}
        variants.append(
            Variant(
                "tune1",
                f"RUSTFLAGS={rustflags}",
                "rust",
                env=dict(env),
                isa=rustflags,
                technique=(
                    TARGET_CPU if "target-cpu" in rustflags else MACHINE_ISA
                ),
                value=rustflags,
            )
        )
    lto_env = dict(
        env,
        CARGO_PROFILE_RELEASE_LTO="fat",
        CARGO_PROFILE_RELEASE_CODEGEN_UNITS="1",
    )
    variants.append(
        Variant(
            f"tune{len(variants) + 1}",
            "fat LTO, one codegen unit"
            + (f", RUSTFLAGS={rustflags}" if rustflags else ""),
            "rust",
            env=lto_env,
            isa=rustflags,
            technique=LTO,
            value="lto=fat, codegen-units=1",
        )
    )
    for index, variant in enumerate(variants, 1):
        variant.label = f"Tune-{index}"
        if index > 1:
            variant.keeps = variants[index - 2].label
    return variants


def build_ladder(profile: MachineProfile, language: str) -> List[Variant]:
    """Return the tunes to try, leaving room for one PGO tune."""
    if language == "go":
        ladder = go_ladder(profile)
    elif language == "rust":
        ladder = rust_ladder(profile)
    else:
        ladder = c_ladder(profile)
    return ladder[: MAX_VARIANTS - 1]


def build_flags(
    log_lines: Sequence[str],
    tree: Optional[str] = None,
) -> Dict[str, Any]:
    """Return what the verified build's own compiler calls already use.

    Args:
        log_lines: The wrapper log of the baseline tree: the arguments
            of each compiler call, before any tune flag.
        tree: The baseline tree. A compile of an absolute source path
            outside it (a build system's compiler check) does not count.

    Returns:
        ``{"compiles", "levels", "lto"}``: the number of source
        compiles, the optimization levels they use (``-O0`` when none
        is given) and whether any uses LTO. Configure checks do not
        count.
    """
    levels = set()
    compiles = 0
    lto = False
    root = (tree or "").rstrip("/") + "/"
    for line in log_lines:
        if _PROBE_CALL.search(line):
            continue
        try:
            words = shlex.split(line)
        except ValueError:
            words = line.split()
        sources = [word for word in words if _SOURCE_SUFFIX.search(word)]
        if "-c" not in words or not sources:
            continue
        if tree and all(
            word.startswith("/") and not word.startswith(root)
            for word in sources
        ):
            continue
        compiles += 1
        level = "-O0"
        for word in words:
            if word == "-O":
                level = "-O1"
            elif re.match(r"^-O(\d|s|z|g|fast)$", word):
                level = word
            elif word.startswith("-flto") and word != "-fno-lto":
                lto = True
        levels.add(level)
    return {"compiles": compiles, "levels": sorted(levels), "lto": lto}


def _skip_redundant_bases(
    ladder: List[Variant],
    redundant: Dict[str, str],
) -> None:
    """Point "on top of" past tunes that the build has already.

    Tune-4 keeps the flags of Tune-3; when Tune-3 is skipped (the build
    has -O3 already), Tune-4 really builds on Tune-2.
    """
    by_label = {variant.label: variant for variant in ladder}
    for variant in ladder:
        if variant.technique == COMPILER:
            continue
        kept = by_label.get(variant.keeps or "")
        while kept is not None and kept.vid in redundant:
            variant.keeps = kept.keeps
            kept = by_label.get(kept.keeps or "")


def redundant_tunes(
    ladder: List[Variant],
    flags: Dict[str, Any],
) -> Dict[str, str]:
    """Return ``vid -> reason`` for the tunes that the build has already.

    ``-O3`` is redundant when every source compile uses ``-O3``; C/C++
    LTO is redundant when a source compile already uses ``-flto``.
    """
    reasons: Dict[str, str] = {}
    for variant in ladder:
        if (
            variant.technique == OPT_LEVEL
            and flags.get("compiles")
            and (flags.get("levels") == ["-O3"])
        ):
            reasons[variant.vid] = (
                f"the build compiles all {flags['compiles']} sources with "
                "-O3 already"
            )
        if (
            variant.technique == LTO
            and variant.language == "c"
            and (flags.get("lto"))
        ):
            reasons[variant.vid] = "the build uses -flto already"
    return reasons


def _words(segment: str) -> List[str]:
    """Split a shell segment into words, tolerating unbalanced quotes."""
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _command_words(words: List[str]) -> List[str]:
    """Return the command words past any leading sudo, env or VAR=value."""
    index = 0
    while index < len(words) and (
        words[index] in ("sudo", "env") or _ASSIGNMENT.match(words[index])
    ):
        index += 1
    return words[index:]


def _until_operator(words: List[str]) -> List[str]:
    """Return the words before the first shell operator or redirection."""
    for index, word in enumerate(words):
        if word in _SHELL_OPERATORS or re.match(r"^\d*>", word):
            return words[:index]
    return words


def _outside(path: str, repo_path: str) -> bool:
    """Return True for an absolute path outside the tree.

    ``/dev`` counts as inside: a command may redirect to /dev/null.
    """
    if not path.startswith("/") or path.startswith("/dev/"):
        return False
    repo = repo_path.rstrip("/")
    normal = posixpath.normpath(path)
    return normal != repo and not normal.startswith(repo + "/")


def classify_segment(segment: str, repo_path: str) -> str:
    """Return how a variant build treats one ``&&`` part of a command.

    Returns:
        ``keep`` to run it; ``drop`` to skip it (package and install
        steps, source control steps, writes outside the tree);
        ``leave`` to skip it and the rest of its chain (a ``cd`` out of
        the tree, which builds or installs something else).
    """
    words = _command_words(_words(segment))
    if not words:
        return "keep"
    text = " ".join(words)
    if (
        _PACKAGE_STEP.search(text)
        or _INSTALL_STEP.search(text)
        or _SOURCE_STEP.search(text)
    ):
        return "drop"
    if words[0] == "cd":
        target = words[1] if len(words) > 1 else "~"
        if target.startswith("~") or _outside(target, repo_path):
            return "leave"
        return "keep"
    if words[0] in _COPY_COMMANDS:
        operands = [
            word
            for word in _until_operator(words)[1:]
            if not word.startswith("-")
        ]
        if operands and _outside(operands[-1], repo_path):
            return "drop"
    for index, word in enumerate(words):
        value = None
        if word in _OUTPUT_OPTIONS or word in (">", ">>"):
            value = words[index + 1] if index + 1 < len(words) else None
        elif word.startswith("--") and "=" in word:
            name, _, value = word.partition("=")
            if name not in _OUTPUT_OPTIONS:
                value = None
        elif word.startswith("-o") and len(word) > 2:
            value = word[2:]
        if value and _outside(value, repo_path):
            return "drop"
    return "keep"


def replayable(
    command: str,
    repo_path: str,
) -> Tuple[Optional[str], List[str]]:
    """Return the part of a build command that a variant runs again.

    Args:
        command: A command that ran in the verified build.
        repo_path: The baseline tree in the sandbox.

    Returns:
        ``(command or None, dropped parts)``. None when nothing that
        builds is left.
    """
    kept: List[str] = []
    dropped: List[str] = []
    segments = [part.strip() for part in command.split("&&")]
    for index, segment in enumerate(segments):
        if not segment:
            continue
        kind = classify_segment(segment, repo_path)
        if kind == "keep":
            kept.append(segment)
        elif kind == "leave":
            dropped.extend(part for part in segments[index:] if part)
            break
        else:
            dropped.append(segment)
    useful = [
        part
        for part in kept
        if (_command_words(_words(part)) or [":"])[0] not in _HARMLESS
    ]
    if not useful:
        return None, dropped
    return " && ".join(kept), dropped


def replay_commands(state: Any) -> Tuple[List[str], List[str]]:
    """Return the commands that built the verified baseline, in order.

    Each plan command maps to the command that really ran: the builder
    rewrites some (an autotools bootstrap, a Go flag), and a retry may
    store its result under the command it ran.

    Returns:
        ``(commands, dropped parts)``.
    """
    commands: List[str] = []
    dropped: List[str] = []
    plan = getattr(state, "build_plan", None)
    if not plan:
        return commands, dropped
    for phase in plan.phases:
        for command in phase.commands:
            result = state.get_cached_command_result(command)
            if result is not None and not result.success:
                retried = state.get_cached_command_result(result.command)
                if retried is not None and retried.success:
                    result = retried
            executed = getattr(result, "command", None) or command
            kept, skipped = replayable(executed, state.repo_path)
            dropped += skipped
            if kept:
                commands.append(kept)
    return commands, dropped


def variant_env(variant: Variant, dest: str) -> Dict[str, str]:
    """Return the environment for one variant build in ``dest``."""
    env = {
        key: value.replace(TREE_TOKEN, dest)
        for key, value in variant.env.items()
    }
    if variant.language == "c" and variant.compiler:
        wrapper_dir = f"{dest}/.atesor"
        flags = [flag.replace(TREE_TOKEN, dest) for flag in variant.flags]
        env.update(
            CC=f"{wrapper_dir}/cc",
            CXX=f"{wrapper_dir}/cxx",
            ATESOR_REAL_CC=variant.compiler,
            ATESOR_REAL_CXX=_cxx_for(variant.compiler),
            # Only the tune's own flags. A -Wno-error here would also
            # reach configure checks that compile with -Werror (zlib's
            # hidden-visibility check passed with it and changed the
            # code), so the copy would not be the verified build.
            ATESOR_CC_FLAGS=" ".join(flags),
            ATESOR_CC_LOG=f"{wrapper_dir}/cc.log",
            # GNU make takes MAKEFLAGS variables as command-line values,
            # so a Makefile's own "CC = gcc" cannot win over them.
            MAKEFLAGS=f"CC={wrapper_dir}/cc CXX={wrapper_dir}/cxx",
        )
    return env


# Runs in the sandbox. Argument: JSON {"src", "dest", "root",
# "build_dirs"}. Copies the patched tree, removes compiled outputs that
# git does not track, and writes the compiler wrappers.
_PREPARE_PROGRAM = r"""
import json
import os
import shutil
import subprocess
import sys

SPEC = json.loads(sys.argv[1])
src, dest, root = SPEC["src"], SPEC["dest"], SPEC["root"].rstrip("/")
if not root.endswith(".opt") or os.path.dirname(dest.rstrip("/")) != root:
    sys.exit("unsafe variant path: " + dest)
if os.path.lexists(dest):
    shutil.rmtree(dest)
os.makedirs(root, exist_ok=True)
subprocess.run(["cp", "-a", src, dest], check=True)
tracked = None
if os.path.isdir(os.path.join(dest, ".git")):
    listing = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", dest, "ls-files", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if listing.returncode == 0:
        tracked = set(
            item.decode("utf-8", "surrogateescape")
            for item in listing.stdout.split(b"\0")
            if item
        )
tracked_dirs = set()
for item in tracked or ():
    parts = item.split("/")[:-1]
    for depth in range(1, len(parts) + 1):
        tracked_dirs.add("/".join(parts[:depth]))
removed = {"dirs": [], "files": 0}


def is_build_dir(path, rel, name, top):
    # True for a directory of build output that git does not track.
    if tracked is None or rel in tracked_dirs:
        return False
    return (
        (top and name in SPEC["build_dirs"])
        or name == "CMakeFiles"
        or os.path.isfile(os.path.join(path, "CMakeCache.txt"))
        or os.path.isdir(os.path.join(path, "meson-private"))
    )


for dirpath, dirnames, filenames in os.walk(dest):
    rel_dir = os.path.relpath(dirpath, dest)
    rel_dir = "" if rel_dir == "." else rel_dir
    keep = []
    for name in dirnames:
        path = os.path.join(dirpath, name)
        rel = os.path.join(rel_dir, name) if rel_dir else name
        if name == ".git" or os.path.islink(path):
            continue
        if is_build_dir(path, rel, name, not rel_dir):
            shutil.rmtree(path)
            removed["dirs"].append(rel)
            continue
        keep.append(name)
    dirnames[:] = keep
    for name in filenames:
        path = os.path.join(dirpath, name)
        rel = os.path.join(rel_dir, name) if rel_dir else name
        if os.path.islink(path) or (tracked is not None and rel in tracked):
            continue
        drop = name in ("CMakeCache.txt", "config.cache") or name.endswith(
            (".lo", ".la", ".gch", ".pch")
        )
        if not drop:
            try:
                with open(path, "rb") as handle:
                    head = handle.read(8)
            except OSError:
                continue
            drop = head[:4] == b"\x7fELF" or head == b"!<arch>\n"
        if drop:
            os.remove(path)
            removed["files"] += 1
wrapper_dir = os.path.join(dest, ".atesor")
os.makedirs(wrapper_dir, exist_ok=True)
for name, real in (("cc", "ATESOR_REAL_CC"), ("cxx", "ATESOR_REAL_CXX")):
    lines = [
        "#!/bin/sh",
        "# Atesor wrapper: the call's own arguments first, the tune",
        "# flags last, so they win over the project's own (of repeated",
        "# options like -O and -march, the compiler uses the last one).",
        'printf "%s\\n" "$*" >> "${ATESOR_CC_LOG:-/dev/null}"',
        "exec $" + real + ' "$@" $ATESOR_CC_FLAGS',
    ]
    path = os.path.join(wrapper_dir, name)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
sys.stdout.write("@@MARKER@@" + json.dumps(removed) + "\n")
"""

# Runs in the sandbox. Argument: JSON {"tree", "artifacts" (relative
# paths), "log", "read_log", "go", "max_scans", "max_bytes"}. Reports
# the compiler calls and, per artifact: existence, size, SHA-256,
# e_machine, the RISC-V attribute tags and a count of instructions per
# extension class.
_INSPECT_PROGRAM = r"""
import glob
import hashlib
import json
import os
import re
import struct
import subprocess
import sys

SPEC = json.loads(sys.argv[1])


def uleb(data, pos):
    # Decode the ULEB128 value at pos; return (value, next position).
    value, shift = 0, 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            break
    return value, pos


def attributes(data):
    # Parse a .riscv.attributes section body into {tag: value}.
    tags = {}
    if not data or data[:1] != b"A":
        return tags
    pos = 1
    while pos + 4 <= len(data):
        length = struct.unpack_from("<I", data, pos)[0]
        end = pos + length
        name_end = data.index(b"\0", pos + 4)
        vendor = data[pos + 4:name_end]
        sub = name_end + 1
        while vendor == b"riscv" and sub + 5 <= end:
            start = sub
            tag, sub = uleb(data, sub)
            size = struct.unpack_from("<I", data, sub)[0]
            body_end = start + size
            sub += 4
            while tag == 1 and sub < body_end:
                attr, sub = uleb(data, sub)
                if attr % 2:
                    stop = data.index(b"\0", sub)
                    tags[attr] = data[sub:stop].decode("ascii", "replace")
                    sub = stop + 1
                else:
                    tags[attr], sub = uleb(data, sub)
            sub = body_end
        pos = end if length else len(data)
    return tags


def classify32(w):
    # The extension class of a 32-bit instruction word, or None.
    op, f3 = w & 0x7F, (w >> 12) & 7
    f7, rs2, top6 = w >> 25, (w >> 20) & 0x1F, w >> 26
    imm = w >> 20
    if op == 0x33:
        if f7 == 0x10 and f3 in (2, 4, 6):
            return "zba"
        if f7 == 0x20 and f3 in (4, 6, 7):
            return "zbb"
        if f7 == 0x05 and f3 in (4, 5, 6, 7):
            return "zbb"
        if f7 == 0x30 and f3 in (1, 5):
            return "zbb"
        if f7 in (0x14, 0x24, 0x34) and f3 == 1:
            return "zbs"
        if f7 == 0x24 and f3 == 5:
            return "zbs"
        if f7 == 0x07 and f3 in (5, 7):
            return "zicond"
        if f7 == 0x05 and f3 in (1, 2, 3):
            return "zbc"
        return None
    if op == 0x3B:
        if f7 == 0x04 and f3 == 0:
            return "zba"
        if f7 == 0x10 and f3 in (2, 4, 6):
            return "zba"
        if f7 == 0x04 and f3 == 4 and rs2 == 0:
            return "zbb"
        if f7 == 0x30 and f3 in (1, 5):
            return "zbb"
        return None
    if op == 0x13:
        if f3 == 1:
            if top6 in (0x0A, 0x12, 0x1A):
                return "zbs"
            if imm in (0x600, 0x601, 0x602, 0x604, 0x605):
                return "zbb"
        if f3 == 5:
            if top6 == 0x12:
                return "zbs"
            if top6 == 0x18 or imm in (0x287, 0x6B8):
                return "zbb"
        return None
    if op == 0x1B:
        if f3 == 1 and top6 == 0x02:
            return "zba"
        if f3 == 1 and imm in (0x600, 0x601, 0x602):
            return "zbb"
        if f3 == 5 and f7 == 0x30:
            return "zbb"
        return None
    if op == 0x53:
        if f7 in (0x78, 0x79, 0x7A, 0x7B) and rs2 == 1 and f3 == 0:
            return "zfa"
        if (f7 & 0x7C) == 0x14 and f3 in (2, 3):
            return "zfa"
        if (f7 & 0x7C) == 0x20 and rs2 in (4, 5):
            return "zfa"
        if f7 == 0x61 and rs2 == 8 and f3 == 1:
            return "zfa"
        if (f7 & 0x7C) == 0x50 and f3 in (4, 5):
            return "zfa"
        if f7 & 3 == 2:
            return "zfh"
        if f7 in (0x20, 0x21) and rs2 == 2:
            return "zfh"
        return None
    if op in (0x07, 0x27):
        if f3 == 1:
            return "zfh"
        if f3 in (0, 5, 6, 7):
            return "v"
        return None
    if op == 0x57:
        return "v"
    return None


def classify16(h):
    # The extension class of a 16-bit instruction word, or None.
    quadrant, f3 = h & 3, (h >> 13) & 7
    if quadrant == 0 and f3 == 4:
        return "zcb"
    if quadrant == 1 and f3 == 4 and (h >> 10) & 0x3F == 0x27:
        if (h >> 5) & 3 in (2, 3):
            return "zcb"
    return None


def is_elf(data):
    # A little-endian 64-bit ELF image.
    return (
        len(data) > 5
        and data[:4] == b"\x7fELF"
        and data[4:6] == b"\x02\x01"
    )


def elf_facts(data, budget, code):
    # One ELF image: its machine, attributes and instruction counts.
    # Its code (the executable sections) goes into ``code``: two trees
    # that differ only in an embedded path (a build-tree RPATH) have
    # the same code. Returns (facts, counts, code bytes, bytes scanned).
    facts = {"machine": struct.unpack_from("<H", data, 18)[0]}
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    counts, scanned, size = {}, 0, 0
    for index in range(shnum):
        base = shoff + index * shentsize
        sh_type, sh_flags = struct.unpack_from("<IQ", data, base + 4)
        offset, length = struct.unpack_from("<QQ", data, base + 24)
        body = data[offset:offset + length]
        if sh_type == 0x70000003:
            tags = attributes(body)
            facts["arch"] = tags.get(5)
            facts["unaligned"] = tags.get(6)
        if sh_type == 1 and sh_flags & 0x4:
            code.update(body)
            size += len(body)
        if sh_type != 1 or not sh_flags & 0x4 or budget <= 0:
            continue
        pos = 0
        limit = min(len(body), max(0, budget - scanned))
        while pos + 2 <= limit:
            half = body[pos] | body[pos + 1] << 8
            if half & 3 != 3:
                name = classify16(half)
                pos += 2
            elif pos + 4 <= limit:
                name = classify32(struct.unpack_from("<I", body, pos)[0])
                pos += 4
            else:
                break
            if name:
                counts[name] = counts.get(name, 0) + 1
        scanned += pos
        if pos < len(body):
            facts["partial"] = True
    return facts, counts, size, scanned


def members(data):
    # The ELF objects of an ar archive (GNU or BSD names), in order.
    pos = 8
    while pos + 60 <= len(data):
        header = data[pos:pos + 60]
        name = header[:16].decode("latin-1").strip()
        try:
            size = int(header[48:58].decode("latin-1").strip() or "0")
        except ValueError:
            return
        body = data[pos + 60:pos + 60 + size]
        pos += 60 + size + (size & 1)
        if name.startswith("#1/"):
            try:
                body = body[int(name[3:]):]
            except ValueError:
                continue
        if is_elf(body):
            yield body


def scan(path, budget):
    # One artifact: identity and hashes, ELF/archive facts, instruction
    # counts (only while the byte budget lasts).
    info = {"exists": os.path.isfile(path)}
    if not info["exists"]:
        return info
    with open(path, "rb") as handle:
        data = handle.read()
    info["size"] = len(data)
    info["sha256"] = hashlib.sha256(data).hexdigest()
    code = hashlib.sha256()
    if is_elf(data):
        info["elf"] = True
        facts, counts, _, _ = elf_facts(data, budget, code)
        info.update(facts)
        # None: the artifact is past the scan limit and was not scanned.
        info["counts"] = counts if budget > 0 else None
        info["code_sha256"] = code.hexdigest()
        return info
    info["elf"] = False
    if data[:8] != b"!<arch>\n":
        return info
    # A static archive: the facts and the code of its objects.
    info["archive"] = True
    counts, scanned, size, machines = {}, 0, 0, []
    for member in members(data):
        facts, found, length, used = elf_facts(member, budget - scanned,
                                               code)
        machines.append(facts["machine"])
        size += length
        scanned += used
        if facts.get("arch") and "arch" not in info:
            info["arch"] = facts["arch"]
        if facts.get("unaligned") == 1:
            info["unaligned"] = 1
        if facts.get("partial"):
            info["partial"] = True
        for name, number in found.items():
            counts[name] = counts.get(name, 0) + number
    info["members"] = len(machines)
    if machines:
        other = [machine for machine in machines if machine != 243]
        info["machine"] = other[0] if other else 243
    info["counts"] = counts if budget > 0 else None
    # Objects with no code (LTO intermediate code only) leave nothing
    # to compare.
    if size:
        info["code_sha256"] = code.hexdigest()
    return info


def go_settings(path):
    # The GORISCV64 and -pgo build settings embedded in a Go binary.
    settings = {}
    proc = subprocess.run(
        ["go", "version", "-m", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        text = line.strip()
        if not text.startswith("build"):
            continue
        setting = text[len("build"):].strip()
        if setting.startswith("GORISCV64="):
            settings["goriscv64"] = setting.split("=", 1)[1]
        elif setting.startswith("-pgo="):
            settings["pgo"] = setting.split("=", 1)[1]
    return settings


HASH_PART = re.compile(r"^[0-9a-fA-F]{16,}$")


def relocate(tree, rel):
    # A build that names an output directory after a hash of its flags
    # (lz4: lib/cachedObjs/<md5>/liblz4.a) writes each tune's artifact
    # to another directory: find the newest file at the same path with
    # any directory in the hash places.
    parts = rel.split("/")
    if not any(HASH_PART.match(part) for part in parts):
        return None
    pattern = "/".join(
        "*" if HASH_PART.match(part) else glob.escape(part)
        for part in parts
    )
    found = [
        path for path in glob.glob(os.path.join(glob.escape(tree), pattern))
        if os.path.isfile(path) and not os.path.islink(path)
    ]
    if not found:
        return None
    return os.path.relpath(max(found, key=os.path.getmtime), tree)


def main():
    tree = SPEC["tree"]
    result = {"cc_calls": None, "artifacts": {}}
    log = SPEC.get("log")
    if log:
        result["cc_calls"] = 0
        if os.path.isfile(log):
            with open(log, "rb") as handle:
                data = handle.read()
            result["cc_calls"] = data.count(b"\n")
            if SPEC.get("read_log"):
                text = data[-(1 << 19):].decode("utf-8", "replace")
                result["log_lines"] = text.splitlines()[-5000:]
    for index, rel in enumerate(SPEC["artifacts"]):
        actual = rel
        if not os.path.isfile(os.path.join(tree, rel)):
            actual = relocate(tree, rel) or rel
        path = os.path.join(tree, actual)
        # Only the first max_scans artifacts get their instructions
        # read; the rest report counts=None (see SCAN_LIMITS).
        budget = SPEC["max_bytes"] if index < SPEC["max_scans"] else 0
        try:
            info = scan(path, budget)
        except (OSError, struct.error, ValueError, IndexError) as exc:
            info = {"exists": os.path.isfile(path), "error": str(exc)[:200]}
        if actual != rel:
            info["path"] = actual
        if SPEC.get("go") and info.get("elf"):
            try:
                info.update(go_settings(path))
            except OSError:
                pass
        result["artifacts"][rel] = info
    sys.stdout.write("@@MARKER@@" + json.dumps(result) + "\n")


main()
"""

# Runs in the sandbox. Argument: JSON {"root", "reset", "keep_tree",
# "artifacts", "remove"}. "reset" removes the whole root. Otherwise it
# copies the accepted artifacts to <root>/accepted and removes the
# named entries of the root.
_FINISH_PROGRAM = r"""
import json
import os
import shutil
import sys

SPEC = json.loads(sys.argv[1])
root = SPEC["root"].rstrip("/")
if not root.endswith(".opt") or not os.path.isabs(root):
    sys.exit("unsafe root: " + root)
copied = []
if SPEC.get("reset"):
    if os.path.isdir(root) and not os.path.islink(root):
        shutil.rmtree(root)
    sys.stdout.write("@@MARKER@@" + json.dumps({"copied": []}) + "\n")
    sys.exit(0)
keep = SPEC.get("keep_tree")
if keep:
    target = os.path.join(root, "accepted")
    if os.path.lexists(target):
        shutil.rmtree(target)
    for rel in SPEC.get("artifacts", []):
        source = os.path.join(keep, rel)
        if os.path.isfile(source) and not os.path.islink(source):
            path = os.path.join(target, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copy2(source, path)
            copied.append(rel)
for name in SPEC.get("remove", []):
    path = os.path.normpath(os.path.join(root, name))
    if os.path.dirname(path) != root or name == "accepted":
        continue
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)
if os.path.isdir(root) and not os.listdir(root):
    os.rmdir(root)
sys.stdout.write("@@MARKER@@" + json.dumps({"copied": copied}) + "\n")
"""

_MARKER = "ATESOR_OPT_JSON "
_PREPARE_PROGRAM = _PREPARE_PROGRAM.replace("@@MARKER@@", _MARKER)
_INSPECT_PROGRAM = _INSPECT_PROGRAM.replace("@@MARKER@@", _MARKER)
_FINISH_PROGRAM = _FINISH_PROGRAM.replace("@@MARKER@@", _MARKER)


def _run_program(
    program: str,
    spec: Dict[str, Any],
    timeout: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
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
                payload = json.loads(line[len(_MARKER) :])
            except ValueError:
                break
            if isinstance(payload, dict):
                return payload, ""
            break
    detail = (result.stderr or result.stdout or "").strip()[-300:]
    return None, f"exit {result.exit_code}: {detail}"


def prepare_tree(src: str, dest: str, root: str) -> Optional[str]:
    """Copy the patched tree to ``dest`` and remove compiled outputs.

    It also writes the compiler wrappers to ``<dest>/.atesor``.

    Returns:
        None on success, or the error.
    """
    payload, error = _run_program(
        _PREPARE_PROGRAM,
        {
            "src": src,
            "dest": dest,
            "root": root,
            "build_dirs": list(BUILD_DIRS),
        },
        timeout=1800,
    )
    if payload is None:
        return f"could not prepare the tree: {error}"
    return None


def build_variant(
    commands: Sequence[str],
    repo_path: str,
    dest: str,
    env: Dict[str, str],
    timeout: int,
) -> Optional[str]:
    """Run the baseline build commands again in the variant tree.

    Returns:
        None on success, or the failing command and its error output.
    """
    for command in commands:
        rewritten = rewrite_for_tree(command, repo_path, dest)
        result = execute_command(
            rewritten, cwd=dest, timeout=timeout, extra_env=env
        )
        if not result.success:
            tail = (result.stderr or result.stdout or "").strip()[-400:]
            return f"`{rewritten[:160]}` exited {result.exit_code}: {tail}"
    return None


def inspect_tree(
    tree: str,
    artifacts: Sequence[str],
    log: Optional[str],
    go: bool,
    kind: str,
    read_log: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Inspect the artifacts of one tree in the sandbox.

    Args:
        tree: The tree in the sandbox.
        artifacts: The tree-relative artifact paths.
        log: The compiler wrapper log, or None.
        go: True to read the Go build settings of each artifact.
        kind: ``native`` or ``qemu`` (the scan limits).
        read_log: True to return the wrapper log lines too.

    Returns:
        ``(facts, "")`` or ``(None, error)``.
    """
    max_scans, max_bytes = SCAN_LIMITS.get(kind, SCAN_LIMITS["qemu"])
    return _run_program(
        _INSPECT_PROGRAM,
        {
            "tree": tree,
            "artifacts": list(artifacts),
            "log": log,
            "read_log": read_log,
            "go": go,
            "max_scans": max_scans,
            "max_bytes": max_bytes,
        },
        timeout=1800,
    )


def _code_hash(info: Dict[str, Any]) -> Optional[str]:
    """Return the hash of an artifact's code, or of the whole file."""
    return info.get("code_sha256") or info.get("sha256")


def _elf_artifacts(facts: Dict[str, Any]) -> List[str]:
    return [
        rel
        for rel, info in (facts.get("artifacts") or {}).items()
        if info.get("elf")
    ]


def _code_artifacts(facts: Dict[str, Any]) -> List[str]:
    """Return the artifacts whose code Atesor compares.

    ELF files, and static archives whose objects have code. An archive
    of LTO intermediate code only has no code to compare.
    """
    return [
        rel
        for rel, info in (facts.get("artifacts") or {}).items()
        if info.get("elf") or (info.get("archive") and info.get("code_sha256"))
    ]


def gate_build_facts(
    facts: Dict[str, Any],
    variant: Variant,
    profile: MachineProfile,
    baseline: Optional[Dict[str, Any]] = None,
    required: Optional[Sequence[str]] = None,
) -> Tuple[str, str, List[str]]:
    """Run gate 1: check the artifacts and prove the variant reached them.

    Args:
        facts: The inspection of the variant tree.
        variant: The variant.
        profile: The machine profile.
        baseline: The inspection of the baseline tree, or None.
        required: The artifacts that the measurement needs: the
            package's libraries and the timed program. A missing
            artifact outside this list is a warning, because it cannot
            change a timing. None requires every artifact, which is
            what the rebuild of the verified build gets.

    Returns:
        ``(outcome, reason, warnings)``; the outcome is ``passed``,
        ``failed`` or ``no effect``. Instructions from extensions the
        machine lacks are warnings only: runtime dispatch (ifunc,
        hwprobe) can contain them legitimately; gates 2 and 3 decide.
    """
    warnings: List[str] = []
    artifacts = facts.get("artifacts") or {}
    if not artifacts:
        return "failed", "the build has no artifact to check", warnings
    for rel, info in artifacts.items():
        if not info.get("exists"):
            if required is not None and rel not in required:
                # A tune's configure step can drop a build-time helper
                # (a coverage or instruction-set test program) while
                # the library and the timed program build. The timing
                # never runs that helper, so the tune stays valid.
                warnings.append(
                    f"{rel} was not built; the measurement does not use "
                    "it, so it does not fail the tune"
                )
                continue
            return "failed", f"{rel} was not built", warnings
        machine = info.get("machine")
        if (info.get("elf") or info.get("archive")) and machine not in (
            None,
            243,
        ):
            return "failed", f"{rel} is not riscv64", warnings
    elf = _elf_artifacts(facts)
    code = _code_artifacts(facts)
    if variant.language == "c" and not facts.get("cc_calls"):
        return (
            "failed",
            "the flags did not reach the compiler (no wrapper call)",
            warnings,
        )
    if variant.language == "go":
        settings = [artifacts[rel] for rel in elf]
        if variant.isa and not any(
            item.get("goriscv64") == variant.isa for item in settings
        ):
            return (
                "failed",
                f"no artifact was built with GORISCV64={variant.isa}",
                warnings,
            )
        if variant.pgo == "use" and not any(
            item.get("pgo") for item in settings
        ):
            return "failed", "no artifact was built with -pgo", warnings
    base_artifacts = (baseline or {}).get("artifacts") or {}
    if code and all(
        _code_hash(artifacts[rel])
        and _code_hash(artifacts[rel])
        == _code_hash(base_artifacts.get(rel) or {})
        for rel in code
    ):
        return (
            "no effect",
            "every riscv64 artifact has the same code as the baseline",
            warnings,
        )
    missing = set(forbidden_classes(profile))
    for rel in code:
        info = artifacts[rel]
        before = (base_artifacts.get(rel) or {}).get("counts") or {}
        found = sorted(
            name
            for name in (info.get("counts") or {})
            if name in missing and name not in before
        )
        if found:
            warnings.append(
                f"{rel} contains {', '.join(found)} instructions that the "
                "machine lacks (runtime dispatch, or a SIGILL risk)"
            )
        if info.get("unaligned") == 1 and profile.misaligned != "fast":
            warnings.append(
                f"{rel} may access memory misaligned, which is "
                f"{profile.misaligned} on this machine"
            )
        if info.get("partial"):
            warnings.append(f"{rel} was scanned in part only")
    return "passed", "artifacts built with the variant settings", warnings


def gate_tests(
    baseline: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Run gate 2: the variant must pass the tests the baseline passes.

    Returns:
        ``(passed, reason)``. The gate passes as not decisive when the
        baseline tests did not pass.
    """
    if not baseline or baseline.get("status") != "passed":
        status = (baseline or {}).get("status", "not run")
        return True, f"not decisive: the baseline tests were {status}"
    if not result:
        return False, "the tests did not run"
    if result.get("status") == "passed":
        return True, "the package tests pass"
    return False, f"the package tests {result.get('status')}"


def _baseline_artifacts(state: Any) -> List[str]:
    """Return the tree-relative paths of the verified artifacts.

    Primary artifacts come first. A path outside the repo tree does
    not count: a variant build only rebuilds what the tree holds.
    """
    repo = state.repo_path.rstrip("/") + "/"
    ordered = sorted(
        state.curated_artifacts or state.build_artifacts or [],
        key=lambda item: 0 if item.get("role") == "primary" else 1,
    )
    paths: List[str] = []
    for artifact in ordered:
        path = artifact.get("filepath") or artifact.get("path") or ""
        if path.startswith(repo):
            rel = posixpath.normpath(path[len(repo) :])
            if rel and not rel.startswith("..") and rel not in paths:
                paths.append(rel)
    return paths


def _mkdir(path: str) -> bool:
    return execute_command(
        ["mkdir", "-p", path], timeout=120, validate=False
    ).success


def _input_path(root: str, workload: bench.Workload, seed: str) -> str:
    return f"{root}/inputs/{seed}.{workload.input_kind}"


def _job(
    workload: bench.Workload,
    tree: str,
    input_path: Optional[str],
    output_path: Optional[str],
    env: Optional[Dict[str, str]] = None,
    repo_path: Optional[str] = None,
    paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Return the runner job of the workload for one tree.

    ``env`` (the variant's GORISCV64, RUSTFLAGS, ...) merges into the
    job's own environment.
    """
    job = workloads.baseline_job(
        workload, tree, input_path, output_path, repo_path, paths
    )
    job["env"] = dict(job["env"], **(env or {}))
    return job


def _jobs_for(
    workload: bench.Workload,
    context: Dict[str, Any],
    tree: str,
    env: Dict[str, str],
    seed: str = MEASURE_SEED,
    base: Optional[Tuple[str, Dict[str, str]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return the baseline and variant jobs for one variant tree.

    ``base`` is ``(tree, env)`` of another baseline build, for example
    the baseline in the second code layout; the baseline tree otherwise.
    """
    root = context["root"]
    repo_path = context["state"].repo_path
    base_tree, base_env = base or context["base"]
    paths = context.get("tree_paths") or {}
    input_path = (
        _input_path(root, workload, seed) if workload.input_kind else None
    )
    base_job = _job(
        workload,
        base_tree,
        input_path,
        f"{root}/out-B",
        base_env,
        repo_path,
        paths.get(base_tree),
    )
    variant = _job(
        workload,
        tree,
        input_path,
        f"{root}/out-V",
        env,
        repo_path,
        paths.get(tree),
    )
    return base_job, variant


def _other_containers() -> List[str]:
    """Return the other running Atesor containers on the native machine.

    Their builds would disturb the timings.
    """
    from . import sandbox
    from .platforms import get_container_name

    try:
        done = sandbox.run_raw(
            ["podman", "ps", "--format", "{{.Names}}"], timeout=60
        )
    except Exception as exc:  # the check is advice, never a stop
        logger.warning(f"Could not list the containers: {exc}")
        return []
    if done.returncode != 0:
        return []
    own = get_container_name()
    return [
        name
        for name in (done.stdout or "").split()
        if name.startswith("atesor") and name != own
    ]


class _Budget:
    """Track the session's remaining seconds against a deadline.

    ``left()`` may go negative; callers compare it with a floor.
    """

    def __init__(self, seconds: float, clock: Callable[[], float]):
        self.clock = clock
        self.deadline = clock() + seconds

    def left(self) -> float:
        return self.deadline - self.clock()


class _Reporter:
    """Writes the console lines of performance mode.

    Each line is ``<key> : <text>``; a continuation line is indented
    under the text.
    """

    WIDTH = 8

    def __init__(self, progress: Progress):
        self.progress = progress

    def line(self, key: str, text: str) -> None:
        self.progress(f"{key:<{self.WIDTH}} : {text}")

    def more(self, text: str) -> None:
        self.progress(f"{'':<{self.WIDTH}}   {text}")


def _step_timeout(context: Dict[str, Any], limit: int) -> int:
    """Cap a step's timeout at the remaining budget, with a 60 s floor."""
    return int(min(limit, max(60, context["budget"].left())))


def _remove(context: Dict[str, Any], names: Sequence[str]) -> None:
    _run_program(
        _FINISH_PROGRAM,
        {"root": context["root"], "remove": list(names)},
        timeout=1800,
    )


def _percent(ratio: float) -> float:
    return round(abs(1.0 - ratio) * 100, 1)


def _change(ratio: float) -> str:
    """Return ``10.8% faster`` or ``1.7% slower`` for a time ratio."""
    if _percent(ratio) == 0:
        return "no difference"
    return f"{_percent(ratio)}% {'faster' if ratio < 1 else 'slower'}"


def tune_result(record: Dict[str, Any]) -> str:
    """Return the short result of one tune for the summary.

    A timed tune gives its change in percent; any other tune says why
    it has no time.
    """
    verdict = str(record.get("verdict") or "")
    ratio = record.get("ratio")
    timed = {
        "faster": "",
        "slower": "",
        "no change": " (inside the noise)",
        "not checked": " (a faster tune won)",
    }
    if verdict == "not confirmed" and ratio:
        return f"{_change(ratio)} ({_unconfirmed_text(record)})"
    if verdict in timed and ratio:
        return _change(ratio) + timed[verdict]
    if verdict.startswith("same as"):
        text = f"same code as {record.get('same_as_label')}"
        return text + (f", {_change(ratio)}" if ratio else "")
    words = {
        "no effect": "no effect (same code as the baseline)",
        "build failed": "build failed",
        "gate 1 failed": "check failed",
        "gate 2 failed": "package tests failed",
        "gate 3 failed": "wrong output",
        "measurement failed": "not timed",
        "unstable": "not timed (the board changed during the run)",
        "not measured": "not timed",
        "skipped": "skipped",
        "checked": "built and checked (QEMU: no time)",
        "built": "built (QEMU: no time)",
    }
    return words.get(verdict, verdict or "unknown")


def _unconfirmed_text(record: Dict[str, Any]) -> str:
    """Return why a tune that was faster once is not confirmed.

    Only a second check that timed the tune again (slower or no change)
    contradicts the gain; any other reason is named as it is.
    """
    check = record.get("second_check") or {}
    reason = " ".join(str(record.get("reason") or "").split())
    if not reason or check.get("verdict") in ("slower", "no change"):
        return "did not hold in the second check"
    return f"not confirmed: {reason[:100]}"


def headline(result: Dict[str, Any]) -> str:
    """Return the answer to "did Atesor optimize it?" in one line."""
    status = result.get("status")
    if status == "accepted":
        ratio = result.get("accepted_ratio") or 1.0
        return (
            f"YES: {result.get('accepted_label')} "
            f"({result.get('accepted_title')}) is {_change(ratio)} than "
            "the verified build"
        )
    if status == "checked":
        return (
            "NOT TIMED: QEMU checks the builds and their output; only "
            "native hardware measures speed"
        )
    if status in ("no gain", "not confirmed"):
        timed = [
            record
            for record in result.get("variants") or []
            if record.get("ratio")
            and record.get("verdict")
            in (
                "faster",
                "slower",
                "no change",
                "not confirmed",
            )
        ]
        best = min(timed, key=lambda r: r["ratio"]) if timed else None
        text = (
            "NO: no faster tune was confirmed"
            if status == "not confirmed"
            else "NO: no tune was clearly faster than the verified build"
        )
        if best:
            text += (
                f" (best: {best['variant'].get('label')}, "
                f"{tune_result(best)})"
            )
        return text
    if status == "not measured":
        return "NO: no tune could be timed"
    if status == "skipped":
        return "NO: performance mode was skipped"
    return "NO: performance mode stopped with an error"


def result_text(record: Dict[str, Any]) -> str:
    """Return one line about a tune's result for the console."""
    verdict = str(record.get("verdict") or "")
    reason = " ".join(str(record.get("reason") or "").split())
    ratio = record.get("ratio")
    comparison = (record.get("measurement") or {}).get("comparison") or {}
    interval = (
        f"95% CI {comparison['ci_low']}-{comparison['ci_high']}"
        if comparison
        else ""
    )
    if verdict in ("faster", "slower", "no change") and ratio:
        noise = ", inside the noise" if verdict == "no change" else ""
        return f"{_change(ratio)}{noise}: time ratio {ratio} ({interval})"
    if verdict.startswith("same as"):
        return f"same code as {record.get('same_as_label')}: {reason}"
    words = {
        "no effect": "no effect",
        "build failed": "build failed",
        "gate 1 failed": "check failed",
        "gate 2 failed": "package tests failed",
        "gate 3 failed": "wrong output",
        "measurement failed": "measurement failed",
        "unstable": "unstable measurement",
        "not measured": "not measured",
        "not confirmed": "not confirmed",
        "not checked": "not checked again",
        "skipped": "skipped",
        "checked": "checked",
        "built": "built",
    }
    head = words.get(verdict, verdict or "unknown")
    first = reason.split("\n")[0][:220]
    return f"{head}: {first}" if first else head


# The package tests can rebuild an artifact in place with flags of their
# own (a test target that builds the program again with debug flags).
# The tree must keep the build that gate 1 checked, so its artifacts
# wait here while the tests run.
KEEP_DIR = ".atesor/keep"
_KEEP_SCRIPT = (
    'for f in "$@"; do mkdir -p "{keep}/$(dirname "$f")" '
    '&& cp -pL "$f" "{keep}/$f" || exit 1; done'
).format(keep=KEEP_DIR)
# Prints each artifact that it puts back; _restore_artifacts collects
# those lines to report them.
_RESTORE_SCRIPT = (
    'for f in "$@"; do cmp -s "{keep}/$f" "$f" && continue; '
    'rm -f "$f" && mkdir -p "$(dirname "$f")" && cp -p "{keep}/$f" "$f" '
    '&& echo "$f" || exit 1; done'
).format(keep=KEEP_DIR)


def _keep_artifacts(tree: str, rels: Sequence[str]) -> str:
    """Copy the artifacts of a tree aside; return an error or ""."""
    if not rels:
        return ""
    done = execute_command(
        ["sh", "-c", _KEEP_SCRIPT, "sh"] + list(rels),
        cwd=tree,
        timeout=600,
        validate=False,
    )
    if done.success:
        return ""
    return " ".join((done.stderr or done.stdout or "").split())[-300:]


def _restore_artifacts(
    tree: str,
    rels: Sequence[str],
) -> Tuple[List[str], str]:
    """Put back each artifact that changed since :func:`_keep_artifacts`.

    Returns:
        ``(the artifacts put back, error or "")``.
    """
    if not rels:
        return [], ""
    done = execute_command(
        ["sh", "-c", _RESTORE_SCRIPT, "sh"] + list(rels),
        cwd=tree,
        timeout=600,
        validate=False,
    )
    restored = [line for line in (done.stdout or "").splitlines() if line]
    if done.success:
        return restored, ""
    return restored, " ".join((done.stderr or "").split())[-300:] or (
        f"exit {done.exit_code}"
    )


def _build_and_check(
    variant: Variant,
    context: Dict[str, Any],
    run_gate2: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build one variant and run gates 1 and 2.

    Args:
        variant: The variant.
        context: The run context.
        run_gate2: False for helper builds that never ship (the PGO
            training build, the baselines).

    Returns:
        ``(record, env)``. The record has a ``verdict`` when the
        variant stops here.
    """
    state, root = context["state"], context["root"]
    dest = f"{root}/{variant.vid}"
    context["trees"].append(variant.vid)
    record: Dict[str, Any] = {"variant": variant.to_dict(), "tree": dest}
    env = variant_env(variant, dest)
    build_env = dict(env, **(context["qemu_env"] or {}))
    timeout = _step_timeout(context, BUILD_TIMEOUT)
    started = time.time()
    error = prepare_tree(state.repo_path, dest, root)
    if not error:
        error = build_variant(
            context["commands"], state.repo_path, dest, build_env, timeout
        )
    record["build_seconds"] = round(time.time() - started, 1)
    if error:
        record.update(verdict="build failed", reason=error)
        return record, env
    # Gate 1 reads the compiler log before the harness adds its own
    # compiler calls to it: only the package's calls prove the flags.
    facts, error = inspect_tree(
        dest,
        context["artifacts"],
        f"{dest}/.atesor/cc.log" if variant.language == "c" else None,
        variant.language == "go",
        context["kind"],
        read_log=variant.vid == BASE_ID,
    )
    if facts is None:
        record.update(verdict="gate 1 failed", reason=error)
        return record, env
    record["facts"] = facts
    outcome, reason, warnings = gate_build_facts(
        facts,
        variant,
        context["profile"],
        context.get("baseline_facts"),
        context.get("required_artifacts"),
    )
    record["gate1"] = {
        "outcome": outcome,
        "reason": reason,
        "warnings": warnings,
        "cc_calls": facts.get("cc_calls"),
    }
    # A gate that passes a tune in spite of something it noticed must
    # say so: a reader who sees only the percent cannot know that an
    # artifact went missing or that the code holds an instruction the
    # machine lacks.
    for warning in warnings:
        context["report"].more(f"note: {warning}")
        if warning not in context["result"]["warnings"]:
            context["result"]["warnings"].append(warning)
    record["elf_artifacts"] = _elf_artifacts(facts)
    # Every artifact the tune built, static archives too: the accepted
    # tune ships all of them, not only its ELF files.
    record["built_artifacts"] = [
        info.get("path") or rel
        for rel, info in (facts.get("artifacts") or {}).items()
        if info.get("exists")
    ]
    record["artifact_hashes"] = sorted(
        (rel, _code_hash(facts["artifacts"][rel] or {}))
        for rel in _code_artifacts(facts)
    )
    # Every later step uses the paths that the inspection found right
    # after this build: a build that names its output directory after a
    # hash of its flags puts the artifacts somewhere else in every tree,
    # and the package tests can add more such directories.
    paths = {
        rel: info["path"]
        for rel, info in (facts.get("artifacts") or {}).items()
        if info.get("path")
    }
    context.setdefault("tree_paths", {})[dest] = paths
    if outcome != "passed":
        verdict = "no effect" if outcome == "no effect" else "gate 1 failed"
        record.update(verdict=verdict, reason=reason)
        return record, env
    workload = context.get("workload")
    if workload is not None and (workload.build or workload.files):
        problem = workloads.install_workload(
            workload, dest, build_env, timeout, paths
        )
        if problem:
            record.update(
                verdict="build failed",
                reason=f"the workload did not build: {problem}",
            )
            return record, env
    baseline_tests = getattr(state, "package_tests", None) or {}
    run_tests = context.get("run_tests")
    if not run_gate2 or variant.pgo == "generate":
        # A helper build (a PGO training build, a baseline) never
        # ships, so its tests do not decide anything.
        record["gate2"] = {"passed": True, "reason": "helper build"}
    elif baseline_tests.get("status") != "passed":
        passed, reason = gate_tests(baseline_tests, None)
        record["gate2"] = {"passed": passed, "reason": reason}
    elif run_tests is None:
        record["gate2"] = {"passed": True, "reason": "not run"}
    else:
        built = record["built_artifacts"]
        problem = _keep_artifacts(dest, built)
        if problem:
            record.update(
                verdict="gate 2 failed",
                reason=f"the artifacts could not be copied aside before "
                f"the package tests: {problem}",
            )
            return record, env
        tested = run_tests(dest, build_env)
        restored, problem = _restore_artifacts(dest, built)
        passed, reason = gate_tests(baseline_tests, tested)
        record["gate2"] = {
            "passed": passed,
            "reason": reason,
            "status": (tested or {}).get("status"),
        }
        if restored:
            record["gate2"]["restored"] = restored
            context["report"].more(
                f"the package tests rebuilt {', '.join(restored)[:200]}; "
                "Atesor put back the tune's build"
            )
        if problem:
            record.update(
                verdict="gate 2 failed",
                reason=f"the package tests changed the artifacts and "
                f"Atesor could not put them back: {problem}",
            )
        elif not passed:
            record.update(verdict="gate 2 failed", reason=reason)
    return record, env


def _p_text(p_value: Optional[float]) -> str:
    """Return a p-value as short console text, for example ``p=0.03``."""
    if p_value is None:
        return "p=?"
    if p_value < 1e-6:
        return "p<1e-06"
    return f"p={p_value:.2g}"


def _measure(
    record: Dict[str, Any],
    env: Dict[str, str],
    context: Dict[str, Any],
    base: Optional[Tuple[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run gate 3, and on native hardware the timing protocol.

    ``base`` is ``(tree, env)`` of the baseline to compare with; the
    baseline tree by default.
    """
    workload = context.get("workload")
    if workload is None:
        record.update(
            verdict="built",
            reason="QEMU: it builds and passes gates 1-2; there is no "
            "workload to run",
        )
        return record
    if context["kind"] != "native" and not workload.deterministic:
        record.update(
            verdict="built",
            reason="QEMU: it builds and passes gates 1-2; this workload "
            "has no output to compare",
        )
        return record
    needed = bench.protocol_seconds(workload.run_seconds)
    if context["kind"] == "native" and needed > context["budget"].left():
        record.update(
            verdict="not measured",
            reason=f"the time budget is too short for a measurement "
            f"(about {needed / 60:.0f} min needed)",
        )
        return record
    base_job, variant_job = _jobs_for(
        workload, context, record["tree"], env, base=base
    )
    if context["kind"] != "native":
        checked = bench.check_once(
            base_job, variant_job, workload, extra_env=context["qemu_env"]
        )
        record["gate3"] = checked
        _write_input_again(workload, context, checked)
        if checked["status"] != "checked":
            record.update(verdict="gate 3 failed", reason=checked["reason"])
        else:
            record.update(
                verdict="checked",
                reason="QEMU: it builds, passes the gates and gives the "
                "baseline's output; QEMU measures no time",
            )
        return record
    outcome = bench.measure(
        base_job,
        variant_job,
        workload,
        pin=context["pin"],
        timeout=int(max(120, context["budget"].left())),
    )
    record["measurement"] = outcome
    _write_input_again(workload, context, outcome)
    status = outcome.get("status")
    if status == "failed":
        verdict = (
            "gate 3 failed"
            if "differs" in (outcome.get("reason") or "")
            else "measurement failed"
        )
        record.update(verdict=verdict, reason=outcome.get("reason"))
        if outcome.get("failed_job") == "B":
            # The unchanged baseline passed its checks before; a failure
            # now means a flaky workload or a machine that computes
            # wrong results. No tune can be compared with it.
            context.setdefault("baseline_failures", []).append(
                outcome.get("reason")
            )
    elif status == "unstable":
        record.update(verdict="unstable", reason=outcome.get("reason"))
    else:
        comparison = outcome.get("comparison") or {}
        record.update(
            verdict=comparison.get("verdict", "no change"),
            ratio=comparison.get("ratio"),
            reason=(
                f"median time ratio {comparison.get('ratio')} "
                f"(95% interval {comparison.get('ci_low')}-"
                f"{comparison.get('ci_high')}, "
                f"{_p_text(comparison.get('p_value'))})"
            ),
        )
    return record


def _write_input_again(
    workload: bench.Workload,
    context: Dict[str, Any],
    outcome: Dict[str, Any],
) -> None:
    """Write the measure input again after a run changed it.

    When that fails, no later tune can be timed on the chosen work.
    """
    if not (outcome.get("input_changed") and workload.input_kind):
        return
    error = bench.write_input(
        workload.input_kind,
        workload.input_mb,
        MEASURE_SEED,
        _input_path(context["root"], workload, MEASURE_SEED),
    )
    if error:
        context["input_lost"] = (
            "a run changed the input file and Atesor could not write it "
            f"again: {error}"
        )


def _second_check_seconds(
    context: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    best: Dict[str, Any],
) -> float:
    """Return the time that the second check of ``best`` needs.

    Go measures a second session. C/C++ and Rust build the tune again
    in another code layout, gate and time it, and build the baseline in
    that layout once: about the longest tune so far plus a base build.
    """
    language = best["variant"].get("language")
    workload = context.get("workload")
    if language == "go" and workload is not None:
        return bench.protocol_seconds(workload.run_seconds)
    tried = [r["seconds"] for r in records if r.get("seconds")]
    layout = (
        0.0
        if language in context.get("layout_baselines", {})
        else float(context.get("base_seconds") or 0.0)
    )
    return (max(tried) if tried else 0.0) + layout


def _kept_for_second_check(
    context: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    factor: float = 1.0,
) -> str:
    """Return why the next step leaves the budget to the second check.

    Once a tune is faster, the next step runs only when the budget
    still covers it (``factor`` times the longest tune so far; PGO
    builds and runs twice) and the second check of the best tune.
    Else that gain would stay unconfirmed.

    Returns:
        The reason to skip the step, or "".
    """
    faster = [
        r for r in records if r.get("verdict") == "faster" and r.get("ratio")
    ]
    tried = [r["seconds"] for r in records if r.get("seconds")]
    if not faster or not tried:
        return ""
    best = min(faster, key=lambda r: r["ratio"])
    need = factor * max(tried) + _second_check_seconds(context, records, best)
    if context["budget"].left() >= need:
        return ""
    return (
        "the rest of the time budget is kept for the second check of "
        f"{best['variant'].get('label') or best['variant'].get('vid')}"
    )


def _timing_stopped(context: Dict[str, Any]) -> str:
    """Return why no later tune can be timed, or ""."""
    if len(context.get("baseline_failures") or []) >= 2:
        return UNRELIABLE_BASELINE
    return context.get("input_lost") or ""


def _announce(variant: Variant, context: Dict[str, Any]) -> None:
    """Print the first line of a tune: what it tries."""
    if variant.technique == COMPILER:
        extra = f" (the flags of {variant.keeps})" if variant.keeps else ""
    elif variant.technique == PGO:
        extra = f" on top of {variant.keeps}" if variant.keeps else ""
    else:
        extra = f", on top of {variant.keeps}" if variant.keeps else ""
    context["report"].line(
        variant.label, f"{variant.technique} = {variant.value}{extra}"
    )


def _try_variant(
    variant: Variant,
    context: Dict[str, Any],
    base: Optional[Tuple[str, Dict[str, str]]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Build, gate and measure one variant; remove its tree unless faster.

    A variant whose riscv64 artifacts have the same code as an earlier
    variant's gets that result and is not measured again.
    """
    if not quiet:
        _announce(variant, context)
    clock = context["budget"].clock
    started = clock()
    record, env = _build_and_check(variant, context)
    built = context.setdefault("built", {})
    key = (repr(record.get("artifact_hashes")), repr(base))
    twin = built.get(key) if not record.get("verdict") else None
    if twin is not None:
        twin_variant = twin["variant"]
        record.update(
            verdict=f"same as {twin_variant['vid']}",
            same_as=twin_variant["vid"],
            same_as_label=twin_variant.get("label") or twin_variant["vid"],
            ratio=twin.get("ratio"),
            reason=(f"result reused ({result_text(twin)})"),
        )
    elif not record.get("verdict"):
        record = _measure(record, env, context, base=base)
        if record.get("artifact_hashes"):
            built[key] = record
    record["env"] = env
    record.pop("facts", None)
    # Build, gates and timing: the budget plan uses it.
    record["seconds"] = round(clock() - started, 1)
    if not quiet:
        context["report"].more(result_text(record))
    logger.info(
        f"Variant {variant.vid}: {record['verdict']} ({record.get('reason')})"
    )
    if record["verdict"] != "faster":
        _remove(context, [variant.vid])
    return record


def _second_check(
    record: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Confirm a faster variant in a second layout or a second session."""
    variant = Variant(**record["variant"])
    report = context["report"]
    if variant.language == "go":
        label = "a second measurement session"
        report.line("Check", f"{variant.label} again, in {label}")
        again = _measure(
            {"variant": record["variant"], "tree": record["tree"]},
            record["env"],
            context,
        )
    else:
        if variant.language == "rust":
            env = dict(variant.env)
            env["RUSTFLAGS"] = (
                env.get("RUSTFLAGS", "") + " " + RUST_SECOND_LAYOUT
            ).strip()
            second = Variant(**dict(variant.to_dict(), env=env))
            knob = RUST_SECOND_LAYOUT
        else:
            second = Variant(
                **dict(
                    variant.to_dict(),
                    flags=list(variant.flags) + [SECOND_LAYOUT_FLAG],
                )
            )
            knob = SECOND_LAYOUT_FLAG
        label = f"a second code layout ({knob} on both builds)"
        report.line("Check", f"{variant.label} again, in {label}")
        base, error = _layout_baseline(variant.language, context)
        if base is None:
            record["verdict"] = "not confirmed"
            record["reason"] = f"no second-layout baseline: {error}"
            record["second_check"] = {
                "check": label,
                "verdict": None,
                "reason": error,
            }
            report.more(f"not confirmed: no second-layout baseline: {error}")
            return record
        second.vid = f"{variant.vid}-layout2"
        second.title = f"{variant.title}, second layout"
        again = _try_variant(second, context, base=base, quiet=True)
        _remove(context, [second.vid])
    record["second_check"] = {
        "check": label,
        "verdict": again.get("verdict"),
        "ratio": again.get("ratio"),
        "reason": again.get("reason"),
    }
    if again.get("verdict") != "faster":
        record["verdict"] = "not confirmed"
        if again.get("verdict") in ("slower", "no change"):
            record["reason"] = (
                f"the gain did not hold in {label}: {result_text(again)}"
            )
        else:
            # The check gave no time: nothing contradicts the gain.
            record["reason"] = (
                f"the second check in {label} gave no time: "
                f"{result_text(again)}"
            )
            if again.get("verdict") == "not measured":
                record["unchecked"] = "time"
        report.more(f"not confirmed: {result_text(again)}")
    else:
        report.more(f"confirmed: {result_text(again)}")
    return record


def _layout_baseline(
    language: str,
    context: Dict[str, Any],
) -> Tuple[Optional[Tuple[str, Dict[str, str]]], str]:
    """Return ``(tree, env)`` of the baseline in the second layout.

    The layout flag has its own cost; a variant in the second layout is
    compared with the baseline in the same layout, so only the
    variant's own change counts. Atesor builds it once per language.

    Returns:
        ``((tree, env), "")``, or ``(None, error)``.
    """
    cache = context.setdefault("layout_baselines", {})
    if language in cache:
        return cache[language]
    if language == "rust":
        variant = Variant(
            "base-layout2",
            "baseline, second layout",
            "rust",
            env={"RUSTFLAGS": RUST_SECOND_LAYOUT},
        )
    else:
        # The verified build used the image's default compiler (CC=gcc).
        variant = Variant(
            "base-layout2",
            "baseline, second layout",
            "c",
            "gcc",
            flags=[SECOND_LAYOUT_FLAG],
        )
    record, env = _build_and_check(variant, context, run_gate2=False)
    if record.get("verdict") == "no effect":
        # The flag changed nothing: the baseline tree is that baseline.
        result: Tuple[Optional[Tuple[str, Dict[str, str]]], str] = (
            context["base"],
            "",
        )
    elif record.get("verdict"):
        result = (None, f"{record['verdict']}: {record.get('reason')}")
    else:
        result = ((record["tree"], env), "")
    cache[language] = result
    return result


def _pgo_flags(family: str, profile_dir: str, mode: str) -> List[str]:
    """Return a compiler family's PGO flags for ``generate`` or ``use``.

    ``profile_dir`` must live outside the variant trees: the training
    tree is removed before the final build reads the profile.
    """
    if family == "clang":
        if mode == "generate":
            return [f"-fprofile-generate={profile_dir}"]
        return [
            f"-fprofile-use={profile_dir}/merged.profdata",
            "-Wno-profile-instr-unprofiled",
            "-Wno-profile-instr-out-of-date",
        ]
    # -fprofile-prefix-path makes the profile names relative to the
    # tree, so the training tree and the final tree may differ.
    prefix = f"-fprofile-prefix-path={TREE_TOKEN}"
    if mode == "generate":
        return [
            f"-fprofile-generate={profile_dir}",
            "-fprofile-update=atomic",
            prefix,
        ]
    # Configure tests compile different conftest programs to the same
    # object name; their stale profiles must not stop the build.
    return [
        f"-fprofile-use={profile_dir}",
        prefix,
        "-Wno-missing-profile",
        "-Wno-coverage-mismatch",
    ]


def _pgo_base(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the variant that PGO builds on, or None."""
    faster = [r for r in records if r.get("verdict") == "faster"]
    if faster:
        return min(faster, key=lambda r: r["ratio"])
    measured = [
        r
        for r in records
        if r.get("verdict") == "no change" and r.get("ratio") is not None
    ]
    if measured:
        return min(measured, key=lambda r: r["ratio"])
    passed = [r for r in records if r.get("verdict") in ("checked", "built")]
    return passed[-1] if passed else None


def _pgo_variant(
    base: Dict[str, Any],
    context: Dict[str, Any],
    label: str,
    skip: str = "",
) -> Optional[Dict[str, Any]]:
    """Train a profile on top of ``base``, then build and measure.

    With ``skip``, the tune is announced and skipped for that reason.
    It returns None when PGO does not apply: no workload, a Rust tune,
    or a Go workload that is not a Go benchmark.
    """
    root = context["root"]
    workload = context.get("workload")
    variant = Variant(**base["variant"])
    if workload is None or variant.language not in ("c", "go"):
        return None
    if variant.language == "go" and workload.mode != "gobench":
        return None
    report = context["report"]
    vid = f"tune{int(label.split('-')[-1])}"
    profile_dir = f"{root}/pgo-{vid}"
    pgo_info = dict(
        variant.to_dict(),
        vid=vid,
        label=label,
        technique=PGO,
        # C/C++ trains on a second input when the workload reads one; Go
        # trains on its benchmarks with the timing input.
        value=(
            "PGO, trained on a second input"
            if workload.input_kind and variant.language != "go"
            else "PGO, trained on the workload"
        ),
        keeps=variant.label,
        title=f"{variant.title} + PGO",
    )
    report.line(
        label, f"{PGO} = {pgo_info['value']}, on top of " f"{variant.label}"
    )
    if skip:
        record = {"variant": pgo_info, "verdict": "skipped", "reason": skip}
        report.more(result_text(record))
        return record
    context["trees"].append(f"pgo-{vid}")
    failed = {"variant": pgo_info, "verdict": "build failed"}

    def fail(reason: str) -> Dict[str, Any]:
        record = dict(failed, reason=reason)
        report.more(result_text(record))
        return record

    if not _mkdir(profile_dir):
        return fail("the profile directory is missing")
    if variant.language == "go":
        profile_file = f"{profile_dir}/default.pgo"
        base_tree = context["base"][0]
        workdir = (
            f"{base_tree}/{workload.workdir}"
            if workload.workdir
            else base_tree
        )
        env = bench.render_env(
            workload,
            (
                _input_path(root, workload, MEASURE_SEED)
                if workload.input_kind
                else None
            ),
        )
        bench_filter = (
            "^BenchmarkAtesor$" if workload.origin == "agent" else "."
        )
        trained = execute_command(
            [
                f"{base_tree}/{bench.GOBENCH_BINARY}",
                "-test.run",
                "^$",
                "-test.bench",
                bench_filter,
                "-test.benchtime",
                "1s",
                "-test.count",
                "1",
                "-test.cpuprofile",
                profile_file,
            ],
            cwd=workdir,
            timeout=_step_timeout(context, 1800),
            validate=False,
            extra_env=env or None,
        )
        if not trained.success:
            return fail("the Go CPU profile was not recorded")
        flags = "-buildvcs=false -pgo=" + profile_file
        pgo = Variant(
            **dict(
                pgo_info,
                env=dict(variant.env, GOFLAGS=flags),
                pgo="use",
                note="trained and measured on the same benchmarks",
            )
        )
        record = _try_variant(pgo, context, quiet=True)
        report.more(result_text(record))
        return record
    compilers = context["profile"].toolchains.get("compilers", {})
    family = compilers.get(variant.compiler or "", {}).get("family", "gcc")
    no_cache = dict(variant.env, CCACHE_DISABLE="1")
    train = Variant(
        **dict(
            pgo_info,
            vid=f"{vid}-train",
            title=f"{variant.title} (PGO training build)",
            flags=list(variant.flags)
            + _pgo_flags(family, profile_dir, "generate"),
            env=no_cache,
            pgo="generate",
        )
    )
    record, env = _build_and_check(train, context)
    if record.get("verdict"):
        _remove(context, [train.vid])
        return fail(f"training build: {record['reason']}")
    if workload.input_kind:
        error = bench.write_input(
            workload.input_kind,
            workload.input_mb,
            TRAIN_SEED,
            _input_path(root, workload, TRAIN_SEED),
        )
        if error:
            _remove(context, [train.vid])
            return fail(f"training input: {error}")
    _, train_job = _jobs_for(
        workload, context, record["tree"], env, seed=TRAIN_SEED
    )
    trained = bench.run_jobs(
        {
            "jobs": {"T": train_job},
            "rounds": 3,
            "seed": 5,
            "timeout": bench.RUN_TIMEOUT,
        },
        timeout=_step_timeout(context, 3600),
        extra_env=context["qemu_env"],
    )
    _remove(context, [train.vid])
    if trained.get("error"):
        return fail(f"training run: {trained['error']}")
    if family == "clang":
        tool = variant.compiler.replace("clang", "llvm-profdata", 1)
        merged = execute_command(
            f"{tool} merge -o {profile_dir}/merged.profdata "
            f"{profile_dir}/*.profraw",
            timeout=600,
            validate=False,
        )
        if not merged.success:
            return fail(f"{tool} could not merge the profile")
    pgo = Variant(
        **dict(
            pgo_info,
            flags=list(variant.flags) + _pgo_flags(family, profile_dir, "use"),
            env=no_cache,
            pgo="use",
        )
    )
    record = _try_variant(pgo, context, quiet=True)
    record["pgo"] = {"family": family, "training_seed": TRAIN_SEED}
    report.more(result_text(record))
    return record


def _finish(
    context: Dict[str, Any],
    accepted: Optional[Dict[str, Any]],
) -> List[str]:
    """Copy the accepted artifacts, then remove every working entry."""
    remove = sorted(set(context["trees"]) | {"inputs"})
    remove += ["out-B", "out-V"]
    payload, _ = _run_program(
        _FINISH_PROGRAM,
        {
            "root": context["root"],
            "keep_tree": accepted["tree"] if accepted else None,
            "artifacts": (
                accepted.get("built_artifacts")
                or accepted.get("elf_artifacts", [])
                if accepted
                else []
            ),
            "remove": remove,
        },
        timeout=1800,
    )
    return (payload or {}).get("copied", [])


def _compact(record: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the bulky fields of a variant record for the result."""
    keep = dict(record)
    keep.pop("env", None)
    keep.pop("facts", None)
    measurement = keep.get("measurement")
    if isinstance(measurement, dict):
        keep["measurement"] = {
            key: measurement[key]
            for key in (
                "status",
                "reason",
                "cv",
                "mde",
                "rounds",
                "warnings",
                "comparison",
                "outputs_match",
            )
            if key in measurement
        }
    return keep


def _build_base(
    context: Dict[str, Any],
    language: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Rebuild the verified build in ``<root>/base``, the baseline tree.

    Returns:
        ``(record, "")``, or ``(None, error)`` when the build fails.
    """
    if language == "c":
        # The verified build used the image's default compiler (CC=gcc).
        variant = Variant(BASE_ID, "the verified build, rebuilt", "c", "gcc")
    else:
        variant = Variant(BASE_ID, "the verified build, rebuilt", language)
    clock = context["budget"].clock
    started = clock()
    record, env = _build_and_check(variant, context, run_gate2=False)
    # The second check builds a baseline in another layout once; the
    # budget plan counts this build time for it.
    context["base_seconds"] = clock() - started
    verdict = record.get("verdict")
    if verdict and verdict != "no effect":
        return None, f"{verdict}: {record.get('reason')}"
    context["base"] = (record["tree"], env)
    return record, ""


def _same_code(
    first: Optional[Dict[str, Any]],
    second: Optional[Dict[str, Any]],
) -> Tuple[int, int]:
    """Return ``(artifacts with equal code, artifacts with code)``."""
    left = (first or {}).get("artifacts") or {}
    right = (second or {}).get("artifacts") or {}
    code = _code_artifacts(first or {})
    same = sum(
        1
        for rel in code
        if _code_hash(left[rel])
        and _code_hash(left[rel]) == _code_hash(right.get(rel) or {})
    )
    return same, len(code)


def run_optimization(
    state: Any,
    agent: Optional[workloads.AskAgent] = None,
    run_tests: Optional[ExecuteTests] = None,
    budget_seconds: Optional[float] = None,
    clock: Callable[[], float] = time.time,
    progress: Optional[Progress] = None,
) -> Dict[str, Any]:
    """Probe, build, check and measure the tunes of a verified port.

    Args:
        state: The AgentState of the port.
        agent: Asks the optimizer agent (an LLM) one question with a
            reply check, or None: then no workload comes from the agent.
        run_tests: Runs the package tests in a tree with an extra
            environment; None skips gate 2.
        budget_seconds: The time limit. ATESOR_OPTIMIZE_BUDGET, or one
            hour, by default. Atesor checks it before each step and
            caps each build with it.
        clock: The clock, for tests.
        progress: Receives one console line per step, or None.

    Returns:
        The optimization record for ``state.optimization``. Its status
        is ``accepted``, ``no gain``, ``not confirmed``,
        ``not measured``, ``checked`` (QEMU), ``skipped`` or ``error``.
        A failed tune is recorded and dropped; nothing here raises for
        it.
    """
    started = clock()
    report = _Reporter(progress or (lambda message: None))
    if budget_seconds is None:
        try:
            budget_seconds = float(
                os.environ.get("ATESOR_OPTIMIZE_BUDGET", BUDGET_SECONDS)
            )
        except ValueError:
            budget_seconds = BUDGET_SECONDS
    kind = "native" if is_native() else "qemu"
    result: Dict[str, Any] = {
        "status": "error",
        "kind": kind,
        "budget_seconds": budget_seconds,
        "variants": [],
        "warnings": [],
        "notes": [],
        "accepted": None,
    }

    def stop(status: str, reason: str) -> Dict[str, Any]:
        result.update(status=status, reason=reason)
        result["duration_seconds"] = round(clock() - started, 1)
        result["headline"] = headline(result)
        report.line("Result", result["headline"])
        report.more(f"why: {reason}")
        return result

    if getattr(state, "verification_status", None) != "verified":
        return stop(
            "skipped", "the port is not verified, so there is no baseline"
        )
    emulated = getattr(state, "emulate_cpu", None)
    if kind == "native" and emulated:
        result["warnings"].append(
            "--emulate-cpu applies to QEMU runs only; it was ignored."
        )
    profile, error = probe_machine(kind, emulated if kind == "qemu" else None)
    if profile is None:
        return stop("error", f"machine probe: {error}")
    state.machine_profile = profile.to_dict()
    result["machine_label"] = profile.label
    result["warnings"] += profile.warnings
    where = (
        "native"
        if kind == "native"
        else f"QEMU, CPU {profile.emulated_cpu or 'max'}"
    )
    report.line(
        "Machine",
        f"{where}, {profile.label}, core {profile.core_name or 'unknown'}, "
        f"misaligned access {profile.misaligned}"
        + (
            f" ({profile.misaligned_source})"
            if profile.misaligned_source != "none"
            else ""
        ),
    )
    if kind == "native":
        # Before anything is built: a machine that gives wrong data for
        # unchanged code makes every timing meaningless.
        check, error = check_integrity(profile.allowed_cpus)
        if check is None:
            result["warnings"].append(
                f"The machine check could not run ({error})."
            )
        else:
            result["machine_check"] = check
            failed = integrity_failures(check)
            runs = check["rounds"] * len(check["bad"])
            if failed:
                wrong = sum(max(count, 1) for count in failed.values())
                where = ", ".join(
                    f"CPU {cpu}: {'crashed' if count < 0 else count}"
                    for cpu, count in sorted(failed.items())
                )
                report.line(
                    "Health",
                    f"{wrong} of {runs} zlib round trips gave wrong data "
                    f"({where})",
                )
                result["advice"] = (
                    "make the machine stable first (a lower CPU "
                    "frequency, a memory test), then run again"
                )
                stop_reason = (
                    f"the machine gives wrong results: {wrong} of {runs} "
                    f"zlib round trips of {check['size_mb']} MiB, with "
                    f"unchanged code, did not give the data back ({where})"
                )
                result = stop("not measured", stop_reason)
                report.more(f"fix: {result['advice']}")
                return result
            report.line(
                "Health",
                f"{runs} zlib round trips on {len(check['bad'])} CPUs gave "
                f"the same data ({check['seconds']} s)",
            )
    plan = getattr(state, "build_plan", None)
    language = language_of(plan.build_system if plan else None)
    result["language"] = language
    package = workloads.profile_package(state, language)
    result["package"] = package.to_dict()
    report.line("Package", package.describe())
    commands, dropped = replay_commands(state)
    result["notes"] += [f"not repeated: {part}" for part in dropped[:20]]
    ladder = build_ladder(profile, language)
    if language == "rust" and any("RUSTFLAGS" in cmd for cmd in commands):
        # The build's own RUSTFLAGS would replace the variant's value.
        ladder = [v for v in ladder if v.technique == LTO]
        for index, variant in enumerate(ladder, 1):
            variant.env = {
                k: x for k, x in variant.env.items() if k != "RUSTFLAGS"
            }
            variant.isa, variant.keeps = None, None
            variant.vid, variant.label = f"tune{index}", f"Tune-{index}"
        result["notes"].append(
            "the build sets RUSTFLAGS itself, so no target-CPU tune"
        )
    if not ladder:
        return stop(
            "skipped", "no tune applies to this machine and build system"
        )
    artifacts = _baseline_artifacts(state)
    if not commands or not artifacts:
        return stop(
            "skipped",
            "the verified build has no repeatable command or no artifact "
            "in the tree",
        )
    root = state.repo_path.rstrip("/") + OPT_SUFFIX
    qemu_env = (
        {"QEMU_CPU": profile.emulated_cpu}
        if kind == "qemu" and profile.emulated_cpu
        else None
    )
    _run_program(_FINISH_PROGRAM, {"root": root, "reset": True}, timeout=1800)
    verified_facts, error = inspect_tree(
        state.repo_path, artifacts, None, language == "go", kind
    )
    if verified_facts is None:
        return stop("error", f"baseline inspection: {error}")
    # A tune must rebuild what the verified tree holds now; a listed
    # artifact that is gone would fail gate 1 for every tune.
    artifacts = [
        rel
        for rel in artifacts
        if (verified_facts.get("artifacts") or {}).get(rel, {}).get("exists")
    ]
    if not artifacts:
        return stop(
            "skipped", "no listed artifact exists in the verified tree"
        )
    context: Dict[str, Any] = {
        "state": state,
        "profile": profile,
        "root": root,
        "kind": kind,
        "budget": _Budget(budget_seconds, clock),
        "commands": commands,
        "artifacts": artifacts,
        "baseline_facts": verified_facts,
        "workload": None,
        "run_tests": run_tests,
        "qemu_env": qemu_env,
        "pin": max(profile.allowed_cpus) if profile.allowed_cpus else None,
        "trees": [],
        "report": report,
        "result": result,
    }
    if context["budget"].left() < MIN_STEP_SECONDS:
        return stop(
            "not measured",
            f"the time budget ({budget_seconds:.0f} s) is shorter than one "
            f"step ({MIN_STEP_SECONDS} s)",
        )
    report.line("Baseline", "rebuilding the verified build in a copy")
    base_record, error = _build_base(context, language)
    if base_record is None:
        _finish(context, None)
        return stop("error", f"the verified build did not rebuild: {error}")
    base_facts = base_record.get("facts") or {}
    same, total = _same_code(verified_facts, base_facts)
    result["baseline"] = {
        "build_seconds": base_record.get("build_seconds"),
        "same_code": same,
        "artifacts": total,
    }
    if not total:
        report.more(
            f"built in {base_record.get('build_seconds')} s; it has no "
            "ELF file or static archive with code to compare"
        )
    elif same == total:
        report.more(
            f"built in {base_record.get('build_seconds')} s; "
            + (
                "its riscv64 artifact has"
                if total == 1
                else f"all {total} riscv64 artifacts have"
            )
            + " the code of the verified build"
        )
    else:
        warning = (
            f"the rebuilt copy differs from the verified build in "
            f"{total - same} of {total} riscv64 artifacts (the build is "
            "not reproducible); every tune is compared with the copy"
        )
        result["warnings"].append(warning)
        report.more(warning)
    # From here on a tune is compared with the copy, built the same way.
    context["baseline_facts"] = base_facts
    redundant: Dict[str, str] = {}
    if language == "c":
        flags = build_flags(
            base_facts.get("log_lines") or [], context["base"][0]
        )
        result["build_flags"] = flags
        redundant = redundant_tunes(ladder, flags)
        _skip_redundant_bases(ladder, redundant)
    ctx = workloads.WorkloadContext(
        state=state,
        package=package,
        tree=context["base"][0],
        env=dict(context["base"][1], **(qemu_env or {})),
        root=root,
        kind=kind,
        qemu_env=qemu_env,
        agent=agent,
        say=report.more,
        time_left=context["budget"].left,
        build_timeout=_step_timeout(context, BUILD_TIMEOUT),
        paths=(context.get("tree_paths") or {}).get(context["base"][0], {}),
    )
    report.line(
        "Workload", "choosing one for a " + package.describe().split(" (")[0]
    )
    workload, notes = workloads.choose_workload(ctx)
    result["notes"] += notes
    result["workload"] = workload.to_dict() if workload else None
    context["workload"] = workload
    # Gate 1 fails a tune that misses one of these: the package's
    # libraries and the program that the workload times. With neither,
    # every artifact stays required. Every other artifact of the
    # verified build becomes a warning when a tune does not build it
    # (see :func:`gate_build_facts`). The rebuild of the verified build
    # ran before this point and needed every artifact; the
    # second-layout baseline, built later, gets this list.
    needed = set(package.libraries)
    if workload is not None and workload.binary:
        needed.add(workload.binary)
    context["required_artifacts"] = sorted(needed) or None
    # The baseline build carries no tune flags, so a crash there is a
    # fault of the port. It needs its own line: a reader must not read
    # it as a workload that Atesor could not write.
    for crash in ctx.crashes:
        result["warnings"].append(crash)
        report.more(f"port problem: {crash}")
    if workload is None:
        reasons = "; ".join(notes) or "no source applies"
        report.more(f"none: {reasons[:400]}")
        if kind == "native":
            _finish(context, None)
            if ctx.crashes:
                return stop(
                    "not measured",
                    f"{ctx.crashes[0]}; no tune was built or measured",
                )
            return stop(
                "not measured",
                "no workload, so no tune was built or measured "
                f"({reasons[:300]})",
            )
    else:
        report.more(workloads.describe(workload))
    if kind == "native" and workload is not None:
        busy = _other_containers()
        if busy:
            result["warnings"].append(
                "other Atesor containers run on the machine ("
                + ", ".join(busy)
                + "): their work may disturb the timings"
            )
    records: List[Dict[str, Any]] = []
    for variant in ladder:
        if variant.vid in redundant:
            _announce(variant, context)
            record = {
                "variant": variant.to_dict(),
                "verdict": "skipped",
                "reason": redundant[variant.vid],
            }
            report.more(result_text(record))
            records.append(record)
            continue
        if context["budget"].left() < MIN_STEP_SECONDS:
            _announce(variant, context)
            record = {
                "variant": variant.to_dict(),
                "verdict": "skipped",
                "reason": "the time budget is used up",
            }
            report.more(result_text(record))
            records.append(record)
            continue
        if _timing_stopped(context):
            _announce(variant, context)
            record = {
                "variant": variant.to_dict(),
                "verdict": "skipped",
                "reason": _timing_stopped(context),
            }
            report.more(result_text(record))
            records.append(record)
            continue
        kept = _kept_for_second_check(context, records)
        if kept:
            _announce(variant, context)
            record = {
                "variant": variant.to_dict(),
                "verdict": "skipped",
                "reason": kept,
            }
            report.more(result_text(record))
            records.append(record)
            continue
        records.append(_try_variant(variant, context))
    pgo_base = _pgo_base(records)
    if _timing_stopped(context):
        pgo_base = None
    if pgo_base is not None and context["budget"].left() >= MIN_STEP_SECONDS:
        label = f"Tune-{len(ladder) + 1}"
        pgo = _pgo_variant(
            pgo_base,
            context,
            label,
            skip=_kept_for_second_check(context, records, PGO_COST_FACTOR),
        )
        if pgo is not None:
            records.append(pgo)
    accepted = None
    candidates = sorted(
        (r for r in records if r.get("verdict") == "faster"),
        key=lambda r: r["ratio"],
    )
    for record in candidates:
        if accepted is not None:
            record["verdict"] = "not checked"
            record["reason"] = (
                f"{accepted['variant'].get('label')} was faster and passed "
                "the second check"
            )
            continue
        if context["budget"].left() < MIN_STEP_SECONDS:
            record["verdict"] = "not confirmed"
            record["reason"] = "the time budget ended before the second check"
            record["unchecked"] = "time"
            continue
        if _timing_stopped(context):
            record["verdict"] = "not confirmed"
            record["reason"] = _timing_stopped(context)
            record["unchecked"] = "stopped"
            continue
        _second_check(record, context)
        if record.get("verdict") == "faster":
            accepted = record
    copied = _finish(context, accepted)
    result["variants"] = [_compact(record) for record in records]
    result["duration_seconds"] = round(clock() - started, 1)
    if accepted:
        variant = accepted["variant"]
        ratio = accepted.get("ratio")
        result.update(
            status="accepted",
            accepted=variant["vid"],
            accepted_label=variant.get("label"),
            accepted_title=variant["title"],
            accepted_ratio=ratio,
            accepted_dir=f"{root}/accepted",
            accepted_artifacts=copied,
            reason=(
                f"{variant.get('label')} ({variant['title']}) is "
                f"{_change(ratio)} than the verified build (time ratio "
                f"{ratio}), and the gain held in the second check"
            ),
        )
    elif kind != "native":
        result.update(
            status="checked",
            reason="QEMU: the tunes were built and checked; only native "
            "hardware measures speed",
        )
    elif not any(r.get("verdict") in _TIMED for r in records):
        failures = context.get("baseline_failures") or []
        result.update(
            status="not measured",
            reason=(
                f"the unchanged baseline failed {len(failures)} timing "
                f"runs ({failures[0]}); the machine or the workload gives "
                "wrong results"
                if failures
                else "every tune stopped before its timing: "
                + "; ".join(
                    f"{r['variant'].get('label')}: {result_text(r)}"
                    for r in records
                )[:600]
            ),
        )
        if failures:
            result["warnings"].append(
                "The unchanged baseline passed its checks, then failed "
                f"{len(failures)} timing runs: {failures[0]}. Wrong "
                "results from an unchanged program point at the machine "
                "(memory or CPU errors) or at a flaky workload."
            )
            result["advice"] = (
                "check the machine (a memory test, a lower CPU frequency), "
                "then run again"
            )
    else:
        unconfirmed = [
            r for r in records if r.get("verdict") == "not confirmed"
        ]
        causes: Dict[str, List[str]] = {}
        for record in unconfirmed:
            cause = (
                "not in the second check"
                if _unconfirmed_text(record).startswith("did not hold")
                else " ".join(str(record.get("reason") or "").split())
            )
            causes.setdefault(cause, []).append(
                record["variant"].get("label") or record["variant"]["vid"]
            )
        # A gain whose second check got no time is not a measured "no
        # gain".
        untested = any(r.get("unchecked") for r in unconfirmed)
        result.update(
            status="not confirmed" if untested else "no gain",
            reason=(
                "faster in the first measurement, but "
                + "; ".join(
                    f"{cause}: {', '.join(labels)}"
                    for cause, labels in causes.items()
                )
                if unconfirmed
                else "no tune was faster than the verified build"
            ),
        )
        if any(r.get("unchecked") == "time" for r in unconfirmed):
            result["advice"] = (
                "run again with a larger ATESOR_OPTIMIZE_BUDGET (this run "
                f"had {budget_seconds:.0f} s)"
            )
    result["headline"] = headline(result)
    report.line("Result", result["headline"])
    if not accepted:
        report.more(f"why: {result['reason']}")
    if result.get("advice"):
        report.more(f"fix: {result['advice']}")
    return result
