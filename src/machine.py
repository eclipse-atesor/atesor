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

"""Probe a riscv64 machine and derive optimization rules from it.

Performance mode (``--optimize``) probes the machine first, on a
native board or in a QEMU-backed container, and every later step reads
that result. One Python program runs inside the sandbox and prints raw
facts as one JSON document. The host turns them into a
:class:`MachineProfile` with :func:`evaluate_probe`. The rule
functions at the end of this module then choose the compiler, Go and
Rust flags that fit that machine.

Design rules:

* Real extensions, never profile names. ``-march`` lists only the
  extensions that the probe found and that the compiler accepts and
  emits. A SiFive P550 lacks Zbs, so an RVA22 build can stop with
  SIGILL there.
* Test instructions win. Under QEMU older than 8.1, ``/proc/cpuinfo``
  shows the host CPU and ``riscv_hwprobe`` is missing. A test
  instruction in a child process (SIGILL means absent) is the ground
  truth on every machine.
* Under QEMU without a named CPU (QEMU's ``max`` default), no ISA
  variant is built: no real board has that extension set.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .tools import DockerConfig, execute_command

logger = logging.getLogger(__name__)

PROBE_TIMEOUT = 600
_OUTPUT_MARKER = "ATESOR_PROBE_JSON "

# riscv_hwprobe (Linux uapi asm/hwprobe.h). Key numbers and the bits
# of RISCV_HWPROBE_KEY_IMA_EXT_0 that Atesor reads.
_KEY_MVENDORID = 0
_KEY_MARCHID = 1
_KEY_MIMPID = 2
_KEY_BASE_BEHAVIOR = 3
_KEY_IMA_EXT_0 = 4
_KEY_CPUPERF_0 = 5
_KEY_MISALIGNED_SCALAR_PERF = 9
_IMA_EXT_0_BITS = {
    0: "fd",
    1: "c",
    2: "v",
    3: "zba",
    4: "zbb",
    5: "zbs",
    6: "zicboz",
    7: "zbc",
    16: "zkt",
    17: "zvbb",
    27: "zfh",
    28: "zfhmin",
    29: "zihintntl",
    30: "zvfh",
    31: "zvfhmin",
    32: "zfa",
    33: "ztso",
    34: "zacas",
    35: "zicond",
    36: "zihintpause",
    42: "zimop",
    43: "zca",
    44: "zcb",
    45: "zcd",
    46: "zcf",
    47: "zcmop",
    48: "zawrs",
    50: "zicntr",
    51: "zihpm",
    55: "zicbom",
    60: "zicbop",
}
_MISALIGNED = {
    0: "unknown",
    1: "emulated",
    2: "slow",
    3: "fast",
    4: "unsupported",
}

# Extensions that a variant may add to ``-march``, in canonical order:
# single letters first, then Z extensions by category (I, F, B).
VECTOR_EXTENSION = "v"
ISA_CANDIDATES = ("zicond", "zfa", "zfhmin", "zba", "zbb", "zbs")
# The timing test: a misaligned 8-byte load that costs at most this
# many aligned loads is fast; at this many or more, the kernel or the
# firmware emulates it after a trap.
MISALIGNED_FAST_RATIO = 4.0
MISALIGNED_EMULATED_RATIO = 40.0

# Profile checks for the label only. Kernels do not report every
# mandatory item, so a real RVA22 machine can show a lower label.
_RVA22_CHECK = (
    "zba",
    "zbb",
    "zbs",
    "zfhmin",
    "zicbom",
    "zicboz",
    "zihintpause",
)
_RVA23_CHECK = _RVA22_CHECK + (
    "v",
    "zicond",
    "zfa",
    "zcb",
    "zvbb",
    "zvfhmin",
    "zimop",
    "zcmop",
    "zawrs",
)
_LABEL_EXTRAS = (
    "v",
    "zba",
    "zbb",
    "zbs",
    "zbc",
    "zicond",
    "zfa",
    "zfh",
    "zfhmin",
    "zcb",
    "zvbb",
)
_BASE_LETTERS = ("i", "m", "a", "f", "d", "c")

# The core table: how to recognize a core and which tune names suit it.
# A core matches by a device-tree ``compatible`` string (also the
# ``uarch`` line of /proc/cpuinfo), or by mvendorid AND marchid. One
# vendor makes many cores, so a vendor id alone never matches; T-Head
# cores report marchid 0, so they match by ``compatible`` only. Each
# tune list is in order of preference; the probe tests which names each
# compiler accepts, and the first accepted name wins. A tune never
# changes the ISA, so a wrong guess costs speed only. ``rust_cpu`` does
# change the ISA: rust_flags_for() uses it only when rustc's feature
# list for that CPU fits the probed machine.
CORE_TABLE: Tuple[Dict[str, Any], ...] = (
    {
        "name": "sifive-u74",
        "compatible": ("sifive,u74-mc", "sifive,u74"),
        "mvendorid": 0x489,
        "marchid": 0x8000000000000007,
        "gcc_tunes": ("sifive-u74", "sifive-7-series"),
        "clang_tunes": ("sifive-u74", "sifive-7-series"),
        "rust_cpu": "sifive-u74",
    },
    {
        "name": "sifive-p550",
        "compatible": ("sifive,p550", "sifive,p500"),
        "mvendorid": 0x489,
        "marchid": 0x8000000000000008,
        "gcc_tunes": ("sifive-p550", "sifive-p400-series"),
        "clang_tunes": ("sifive-p550", "sifive-p450"),
        "rust_cpu": "sifive-p550",
    },
    {
        "name": "spacemit-x60",
        "compatible": ("spacemit,x60",),
        "mvendorid": 0x710,
        "marchid": 0x8000000058000001,
        "gcc_tunes": ("spacemit-x60",),
        "clang_tunes": ("spacemit-x60",),
        "rust_cpu": "spacemit-x60",
    },
    {
        "name": "thead-c906",
        "compatible": ("thead,c906",),
        "mvendorid": None,
        "marchid": None,
        "gcc_tunes": ("thead-c906",),
        "clang_tunes": (),
        "rust_cpu": None,
    },
    {
        "name": "thead-c910",
        "compatible": ("thead,c910", "thead,c920"),
        "mvendorid": None,
        "marchid": None,
        "gcc_tunes": ("xt-c910", "xt-c920"),
        "clang_tunes": (),
        "rust_cpu": None,
    },
)


def _unique(names: Sequence[str]) -> Tuple[str, ...]:
    seen: List[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


# Every tune name of the table; the probe tests each one per compiler.
TUNE_CANDIDATES = _unique(
    [name for row in CORE_TABLE for name in row["gcc_tunes"]]
    + [name for row in CORE_TABLE for name in row["clang_tunes"]]
)
# The Rust CPUs whose rustc feature lists the probe reads.
RUST_CPU_CANDIDATES = _unique(
    [row["rust_cpu"] for row in CORE_TABLE if row["rust_cpu"]]
)

# Go: the extensions each GORISCV64 level needs, with the first Go
# release that knows the level.
_GO_LEVELS = (
    ("rva23u64", (1, 25), ("zba", "zbb", "zbs", "v", "zicond", "zfa")),
    ("rva22u64", (1, 23), ("zba", "zbb", "zbs")),
)
# Go releases up to this one emit no Zbs instruction under rva22u64,
# so rva22u64 may run on a machine that lacks only Zbs (gate 2 checks).
_GO_NO_ZBS_UNTIL = (1, 26)
# Rust target features that are stable in Rust 1.90 and that a variant
# may enable from the probe.
_RUST_STABLE_FEATURES = ("zba", "zbb", "zbs")

# Runs inside the sandbox. It must stay self-contained and compatible
# with Python 3.8+. It collects raw facts only; the host parses them.
# Argument: a JSON object with "compilers" and "isa_candidates".
_PROBE_PROGRAM = r'''
import ctypes
import glob
import json
import os
import re
import shutil
import subprocess
import sys

ARGS = json.loads(sys.argv[1])
EXTRA_PATH = ("/usr/local/go/bin", "/root/.cargo/bin", "/root/go/bin")
os.environ["PATH"] = ":".join(
    [os.environ.get("PATH", "/usr/bin:/bin")] + list(EXTRA_PATH)
)

# One test instruction per extension: the 32-bit words, the call
# signature (result:arguments) and the arguments. A ``ret`` follows.
TESTS = {
    "zba": ([0x20B52533], "i:ii", [3, 5]),
    "zbb": ([0x40B57533], "i:ii", [15, 5]),
    "zbs": ([0x28151513], "i:ii", [4, 0]),
    "zicond": ([0x0EB55533], "i:ii", [7, 0]),
    "zfa": ([0xA2B54553], "i:dd", [1.0, 2.0]),
    "zfhmin": ([0x40250553], "f:f", [1.0]),
    "v": ([0x0C0072D7, 0xC2202573], "i:", []),
}

CHILD = """
import ctypes, json, mmap, struct, sys
words, sig, args = json.loads(sys.argv[1])
code = b"".join(struct.pack("<I", w) for w in words + [0x00008067])
prot = mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
buf = mmap.mmap(-1, mmap.PAGESIZE, prot=prot)
buf.write(code)
addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
libc = ctypes.CDLL(None, use_errno=True)
try:
    libc.syscall(
        ctypes.c_long(259),
        ctypes.c_void_p(addr),
        ctypes.c_void_p(addr + len(code)),
        ctypes.c_ulong(0),
    )
except Exception:
    pass
kinds = {"i": ctypes.c_uint64, "d": ctypes.c_double, "f": ctypes.c_float}
ret, params = sig.split(":")
func = ctypes.CFUNCTYPE(kinds[ret], *[kinds[p] for p in params])(addr)
value = func(*args)
if ret == "f":
    value = repr(value)
sys.stdout.write(json.dumps({"value": value}))
"""

# Times 8-byte loads: f(p, n) runs "ld t0, 0(a0)" n times. The same
# loop runs on an aligned address, then on that address plus one. A
# core that handles a misaligned load in hardware needs a few cycles
# more; a trap to the kernel or the firmware costs hundreds.
MISALIGNED_CHILD = """
import ctypes, json, mmap, struct, sys, time
words = [0x00053283, 0xFFF58593, 0xFE059CE3, 0x00008067]
code = b"".join(struct.pack("<I", w) for w in words)
prot = mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
buf = mmap.mmap(-1, mmap.PAGESIZE, prot=prot)
buf.write(code)
addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
libc = ctypes.CDLL(None, use_errno=True)
try:
    libc.syscall(
        ctypes.c_long(259),
        ctypes.c_void_p(addr),
        ctypes.c_void_p(addr + len(code)),
        ctypes.c_ulong(0),
    )
except Exception:
    pass
kind = ctypes.c_uint64
func = ctypes.CFUNCTYPE(kind, kind, kind)(addr)
data = ctypes.create_string_buffer(512)
base = (ctypes.addressof(data) + 63) & ~63


def per_load(pointer, count):
    best = None
    for _ in range(3):
        start = time.perf_counter()
        func(pointer, count)
        spent = time.perf_counter() - start
        best = spent if best is None else min(best, spent)
    return best / count


per_load(base, 1000)
aligned = per_load(base, 2000000)
misaligned = per_load(base + 1, 20000)
sys.stdout.write(json.dumps({
    "aligned_ns": aligned * 1e9,
    "misaligned_ns": misaligned * 1e9,
}))
"""


def read(path, limit=262144):
    try:
        with open(path, "rb") as handle:
            return handle.read(limit).decode("utf-8", "replace")
    except OSError:
        return None


def run(argv, stdin=None, timeout=120):
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"rc": None, "out": "", "err": str(exc)[:200]}
    return {
        "rc": proc.returncode,
        "out": proc.stdout.decode("utf-8", "replace")[-20000:],
        "err": proc.stderr.decode("utf-8", "replace")[-2000:],
    }


def hwprobe():
    class Pair(ctypes.Structure):
        _fields_ = [("key", ctypes.c_int64), ("value", ctypes.c_uint64)]

    keys = list(range(17))
    pairs = (Pair * len(keys))()
    for index, key in enumerate(keys):
        pairs[index].key = key
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        rc = libc.syscall(
            ctypes.c_long(258),
            pairs,
            ctypes.c_size_t(len(keys)),
            ctypes.c_size_t(0),
            ctypes.c_void_p(0),
            ctypes.c_uint(0),
        )
    except Exception as exc:
        return {"error": str(exc)[:200]}
    if rc != 0:
        return {"error": "errno " + str(ctypes.get_errno())}
    return {
        "pairs": [
            [pair.key, pair.value] for pair in pairs if pair.key != -1
        ]
    }


def test_instructions():
    results = {}
    for name, (words, sig, args) in TESTS.items():
        try:
            proc = subprocess.run(
                [sys.executable, "-c", CHILD, json.dumps([words, sig, args])],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            results[name] = {"rc": None, "err": str(exc)[:200]}
            continue
        entry = {"rc": proc.returncode}
        if proc.returncode == 0:
            try:
                entry["value"] = json.loads(proc.stdout.decode())["value"]
            except (ValueError, KeyError):
                entry["rc"] = 99
        else:
            entry["err"] = proc.stderr.decode("utf-8", "replace")[-300:]
        results[name] = entry
    return results


def misaligned_test():
    try:
        proc = subprocess.run(
            [sys.executable, "-c", MISALIGNED_CHILD],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"rc": None, "err": str(exc)[:200]}
    entry = {"rc": proc.returncode}
    if proc.returncode == 0:
        try:
            entry.update(json.loads(proc.stdout.decode()))
        except ValueError:
            entry["rc"] = 99
    else:
        entry["err"] = proc.stderr.decode("utf-8", "replace")[-300:]
    return entry


def rust_cpu_features(cpus_text):
    # rustc's cfg output for the default CPU and for each candidate
    # CPU that this rustc knows.
    known = set(
        line.split()[0]
        for line in cpus_text.splitlines()
        if line.startswith(" ") and line.split()
    )
    features = {}
    pattern = re.compile(r'target_feature="([^"]+)"')
    default = run(["rustc", "--print", "cfg"])
    if default["rc"] == 0:
        features["(default)"] = pattern.findall(default["out"])
    for cpu in ARGS.get("rust_cpus", []):
        if cpu not in known:
            continue
        done = run(["rustc", "--print", "cfg", "-C", "target-cpu=" + cpu])
        if done["rc"] == 0:
            features[cpu] = pattern.findall(done["out"])
    return features


EMIT_SOURCE = (
    "unsigned long f(unsigned long a, unsigned long b)"
    "{return (a << 3) + b;}\n"
    "unsigned long g(unsigned long a, unsigned long b){return a & ~b;}\n"
)


def compiler_facts(compiler):
    facts = {"version": run([compiler, "--version"])["out"].split("\n")[0]}
    base = [compiler, "-x", "c", "-c", "-o", "/dev/null", "-"]
    accepts = []
    for ext in ARGS["isa_candidates"]:
        march = "rv64gcv" if ext == "v" else "rv64gc_" + ext
        if run(base + ["-march=" + march], b"int x;\n")["rc"] == 0:
            accepts.append(ext)
    facts["accepts"] = accepts
    tunes = []
    for name in ARGS["tune_candidates"]:
        if run(base + ["-mtune=" + name], b"int x;\n")["rc"] == 0:
            tunes.append(name)
    facts["tunes"] = tunes
    flags = []
    for flag in ("-mstrict-align", "-flto=auto", "-ffat-lto-objects"):
        if run(base + [flag], b"int x;\n")["rc"] == 0:
            flags.append(flag)
    facts["flags"] = flags
    if "zba" in accepts and "zbb" in accepts:
        asm = run(
            [
                compiler,
                "-x",
                "c",
                "-O2",
                "-march=rv64gc_zba_zbb",
                "-S",
                "-o",
                "-",
                "-",
            ],
            EMIT_SOURCE.encode(),
        )
        facts["emit_asm"] = asm["out"][-4000:]
    return facts


def main():
    result = {
        "cpuinfo": read("/proc/cpuinfo"),
        "dt_model": read("/proc/device-tree/model"),
        "dt_compatible": read("/proc/device-tree/cpus/cpu@0/compatible"),
        "hwprobe": hwprobe(),
        "tests": test_instructions(),
        "misaligned_test": misaligned_test(),
        "nproc": len(os.sched_getaffinity(0)),
        "cpus": sorted(os.sched_getaffinity(0)),
        "meminfo": read("/proc/meminfo", 4096),
        "loadavg": read("/proc/loadavg", 256),
        "uname": list(os.uname()),
        "perf_event_paranoid": read("/proc/sys/kernel/perf_event_paranoid"),
        "perf": shutil.which("perf"),
        "musl": bool(glob.glob("/lib/ld-musl-*")),
        "caches": {},
        "cpufreq": {},
        "thermal": [],
        "compilers": {},
        "go": None,
        "rustc": None,
        "rust_cpus": None,
        "rust_cpu_features": {},
    }
    try:
        result["glibc"] = os.confstr("CS_GNU_LIBC_VERSION")
    except (ValueError, OSError):
        result["glibc"] = None
    cache_dir = "/sys/devices/system/cpu/cpu0/cache"
    for index in sorted(glob.glob(cache_dir + "/index*")):
        entry = {}
        for name in ("level", "type", "size"):
            value = read(os.path.join(index, name), 64)
            entry[name] = value.strip() if value else None
        result["caches"][os.path.basename(index)] = entry
    policy = "/sys/devices/system/cpu/cpufreq/policy0"
    for name in (
        "scaling_governor",
        "scaling_cur_freq",
        "scaling_max_freq",
        "cpuinfo_max_freq",
    ):
        value = read(os.path.join(policy, name), 64)
        result["cpufreq"][name] = value.strip() if value else None
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        value = read(zone, 64)
        if value and value.strip().lstrip("-").isdigit():
            result["thermal"].append(int(value.strip()))
    for compiler in ARGS["compilers"]:
        if shutil.which(compiler):
            result["compilers"][compiler] = compiler_facts(compiler)
    if shutil.which("go"):
        result["go"] = run(["go", "version"])["out"].strip()
    if shutil.which("rustc"):
        result["rustc"] = run(["rustc", "--version"])["out"].strip()
        cpus = run(["rustc", "--print", "target-cpus"])["out"]
        result["rust_cpus"] = cpus[-20000:]
        result["rust_cpu_features"] = rust_cpu_features(cpus)
    sys.stdout.write("@@MARKER@@" + json.dumps(result) + "\n")


main()
'''.replace("@@MARKER@@", _OUTPUT_MARKER)

PROBE_COMPILERS = (
    "gcc",
    "clang",
    "clang-19",
    "clang-20",
    "clang-21",
    "clang-22",
)


@dataclass
class MachineProfile:
    """What the probe found about one machine.

    Attributes:
        kind: ``native`` or ``qemu``.
        emulated_cpu: The ``QEMU_CPU`` value in use, or None.
        extensions: Lowercase extension names (single letters and Z/S
            extensions) that the machine runs.
        vlen: The vector register length in bits, or None without V.
        core: Raw identity facts: ``mvendorid``, ``marchid``,
            ``mimpid``, ``uarch``, ``model`` and ``compatible``.
        core_name: The core table entry that matched, or None.
        misaligned: Scalar misaligned access speed: ``fast``, ``slow``,
            ``emulated``, ``unsupported`` or ``unknown``.
        misaligned_source: ``hwprobe``, ``timing test`` or ``none``.
        misaligned_test: The timing test result: the time of one
            aligned and one misaligned load in ns, or the exit code.
        cores: The number of usable CPUs.
        allowed_cpus: The CPU numbers the sandbox may run on; a
            measurement pins its runs to one of them.
        mem_kib: Total memory in KiB, or None.
        kernel: The kernel release.
        libc: ``glibc <version>``, ``musl`` or None.
        board: Governor, frequencies (kHz), temperatures (millidegrees
            C), ``perf_event_paranoid``, the perf path and the load.
        toolchains: Facts per compiler, plus ``go`` and ``rust``.
        label: A short description, for example ``rv64gc+zba+zbb``.
        sources: Which sources answered: ``cpuinfo``, ``hwprobe``,
            ``tests``.
        tests: The result per test instruction: ``present``,
            ``absent`` or ``error``.
        warnings: Plain-language warnings for the report.
    """

    kind: str
    emulated_cpu: Optional[str]
    extensions: List[str]
    vlen: Optional[int] = None
    core: Dict[str, Any] = field(default_factory=dict)
    core_name: Optional[str] = None
    misaligned: str = "unknown"
    misaligned_source: str = "none"
    misaligned_test: Dict[str, Any] = field(default_factory=dict)
    cores: Optional[int] = None
    allowed_cpus: List[int] = field(default_factory=list)
    mem_kib: Optional[int] = None
    kernel: Optional[str] = None
    libc: Optional[str] = None
    board: Dict[str, Any] = field(default_factory=dict)
    toolchains: Dict[str, Any] = field(default_factory=dict)
    label: str = "unknown"
    sources: List[str] = field(default_factory=list)
    tests: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        """Return True when the machine has every named extension."""
        present = set(self.extensions)
        return all(name in present for name in names)

    @property
    def isa_variants_allowed(self) -> bool:
        """Return True when ISA variants may target this machine.

        Under QEMU without a named CPU, or with ``max``, the extension
        set is QEMU's ``max``, which matches no real board.
        """
        return self.kind == "native" or is_named_model(self.emulated_cpu)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MachineProfile":
        """Build a profile from :meth:`to_dict` output.

        Unknown keys are dropped and missing required ones get empty
        values, so a partial or older record still loads.
        """
        known = {name for name in cls.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in known}
        values.setdefault("kind", "unknown")
        values.setdefault("emulated_cpu", None)
        values.setdefault("extensions", [])
        return cls(**values)


def is_named_model(emulated_cpu: Optional[str]) -> bool:
    """Return True for a QEMU_CPU value that models a real CPU.

    ``max`` (QEMU's default since 8.2, with or without extra
    properties) enables every extension QEMU knows, so it is no target.
    """
    base = (emulated_cpu or "").split(",")[0].strip().lower()
    return bool(base) and base != "max"


def parse_isa_string(isa: str) -> Set[str]:
    """Return the extensions in an ISA string such as ``rv64imafdc_zba``.

    ``g`` expands to ``imafd`` plus Zicsr and Zifencei. Version
    suffixes such as ``zba1p0`` are dropped.

    Args:
        isa: The ISA string, as in ``/proc/cpuinfo``.

    Returns:
        The set of lowercase extension names; empty when the string is
        not a riscv ISA string.
    """
    text = (isa or "").strip().lower()
    match = re.match(r"^rv(?:32|64|128)([a-z0-9]*)((?:_[a-z0-9]+)*)$", text)
    if not match:
        return set()
    extensions: Set[str] = set()
    letters = re.sub(r"\d+p\d+", "", match.group(1))
    for letter in letters:
        if letter == "g":
            extensions.update(("i", "m", "a", "f", "d"))
            extensions.update(("zicsr", "zifencei"))
        elif letter.isalpha():
            extensions.add(letter)
    for part in match.group(2).split("_"):
        name = re.sub(r"\d+p\d+$", "", part)
        if name:
            extensions.add(name)
    return extensions


def parse_cpuinfo(text: Optional[str]) -> Dict[str, str]:
    """Return the riscv fields of the first CPU in ``/proc/cpuinfo``.

    Args:
        text: The file content, or None.

    Returns:
        The fields ``isa``, ``hart isa``, ``mmu``, ``uarch``,
        ``mvendorid``, ``marchid`` and ``mimpid`` that are present.
        An x86 cpuinfo (old QEMU shows the host's) gives an empty dict.
    """
    wanted = (
        "isa",
        "hart isa",
        "mmu",
        "uarch",
        "mvendorid",
        "marchid",
        "mimpid",
    )
    fields: Dict[str, str] = {}
    for line in (text or "").splitlines():
        if not line.strip():
            if fields:
                break
            continue
        key, sep, value = line.partition(":")
        key = key.strip()
        if sep and key in wanted and key not in fields:
            fields[key] = value.strip()
    if not parse_isa_string(fields.get("isa", "")):
        return {}
    return fields


def hwprobe_extensions(pairs: Dict[int, int]) -> Set[str]:
    """Return the extensions that ``riscv_hwprobe`` key/value pairs show.

    Args:
        pairs: The key to value map of one ``riscv_hwprobe`` call.

    Returns:
        Lowercase extension names. BASE_BEHAVIOR_IMA adds I, M and A;
        bit 0 of IMA_EXT_0 adds F and D.
    """
    extensions: Set[str] = set()
    if pairs.get(_KEY_BASE_BEHAVIOR, 0) & 1:
        extensions.update(("i", "m", "a"))
    bits = pairs.get(_KEY_IMA_EXT_0, 0)
    for bit, name in _IMA_EXT_0_BITS.items():
        if bits & (1 << bit):
            if name == "fd":
                extensions.update(("f", "d"))
            else:
                extensions.add(name)
    return extensions


def misaligned_speed(pairs: Dict[int, int]) -> str:
    """Return the scalar misaligned access speed from hwprobe pairs.

    The MISALIGNED_SCALAR_PERF key (Linux 6.11+) wins over the older
    CPUPERF_0 key (Linux 6.4+).
    """
    if _KEY_MISALIGNED_SCALAR_PERF in pairs:
        value = pairs[_KEY_MISALIGNED_SCALAR_PERF]
    elif _KEY_CPUPERF_0 in pairs:
        value = pairs[_KEY_CPUPERF_0] & 7
    else:
        return "unknown"
    return _MISALIGNED.get(value, "unknown")


def misaligned_from_timing(test: Optional[Dict[str, Any]]) -> str:
    """Return the misaligned access speed from the probe's timing test.

    SIGBUS (the child died, or a shell reported 128 + 7) means the
    machine does not support misaligned access at all.
    """
    test = test or {}
    code = test.get("rc")
    if code in (-7, 135):
        return "unsupported"
    aligned = test.get("aligned_ns")
    misaligned = test.get("misaligned_ns")
    if code != 0 or not aligned or not misaligned or aligned <= 0:
        return "unknown"
    ratio = misaligned / aligned
    if ratio <= MISALIGNED_FAST_RATIO:
        return "fast"
    if ratio >= MISALIGNED_EMULATED_RATIO:
        return "emulated"
    return "slow"


def match_core(core: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the core table entry that matches the identity facts.

    A device-tree ``compatible`` string or the cpuinfo ``uarch`` value
    matches first. Otherwise both ``mvendorid`` and ``marchid`` must
    equal the entry's values; an entry without them matches by name
    only.
    """
    names = set(core.get("compatible") or [])
    if core.get("uarch"):
        names.add(str(core["uarch"]).strip().lower())
    for entry in CORE_TABLE:
        if names & set(entry["compatible"]):
            return entry
    vendor = core.get("mvendorid")
    arch = core.get("marchid")
    if vendor is None or arch is None:
        return None
    for entry in CORE_TABLE:
        if entry["mvendorid"] is None or entry["marchid"] is None:
            continue
        if vendor == entry["mvendorid"] and arch == entry["marchid"]:
            return entry
    return None


def profile_label(extensions: Set[str]) -> str:
    """Return a short label for an extension set.

    ``RVA23U64`` or ``RVA22U64`` when every checked item is present;
    otherwise ``rv64gc`` plus the notable extras, for example
    ``rv64gc+zba+zbb``.
    """
    if all(name in extensions for name in _RVA23_CHECK):
        return "RVA23U64"
    if all(name in extensions for name in _RVA22_CHECK):
        return "RVA22U64"
    if not all(letter in extensions for letter in _BASE_LETTERS):
        return "below rv64gc"
    extras = [name for name in _LABEL_EXTRAS if name in extensions]
    return "+".join(["rv64gc"] + extras)


def _parse_int(text: Any) -> Optional[int]:
    try:
        return int(str(text).strip(), 0)
    except (TypeError, ValueError):
        return None


def _version_tuple(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _emits_bitmanip(asm: Optional[str]) -> bool:
    text = asm or ""
    return "sh3add" in text and "andn" in text


def _compiler_family(name: str) -> str:
    return "clang" if name.startswith("clang") else "gcc"


def _toolchains(payload: Dict[str, Any]) -> Dict[str, Any]:
    compilers: Dict[str, Any] = {}
    for name, facts in (payload.get("compilers") or {}).items():
        compilers[name] = {
            "family": _compiler_family(name),
            "version": facts.get("version") or "",
            "accepts": list(facts.get("accepts") or []),
            "tunes": list(facts.get("tunes") or []),
            "flags": list(facts.get("flags") or []),
            "emits_bitmanip": _emits_bitmanip(facts.get("emit_asm")),
        }
    go_text = payload.get("go") or ""
    go_version = re.search(r"go(\d+\.\d+(?:\.\d+)?)", go_text)
    rust_text = payload.get("rustc") or ""
    rust_version = re.search(r"rustc (\d+\.\d+\.\d+)", rust_text)
    rust_cpus = [
        line.split()[0]
        for line in (payload.get("rust_cpus") or "").splitlines()
        if line.startswith(" ") and line.split()
    ]
    features = payload.get("rust_cpu_features") or {}
    return {
        "compilers": compilers,
        "go": go_version.group(1) if go_version else None,
        "rust": rust_version.group(1) if rust_version else None,
        "rust_cpus": rust_cpus,
        "rust_cpu_features": {
            str(cpu): [str(name) for name in names]
            for cpu, names in features.items()
            if isinstance(names, list)
        },
    }


def evaluate_probe(
    payload: Dict[str, Any],
    kind: str,
    emulated_cpu: Optional[str] = None,
) -> MachineProfile:
    """Turn the raw probe output into a :class:`MachineProfile`.

    Args:
        payload: The JSON document printed by the in-sandbox program.
        kind: ``native`` or ``qemu``, from the run's target.
        emulated_cpu: The ``QEMU_CPU`` value in use, or None.

    Returns:
        The machine profile. Test instruction results win over
        ``/proc/cpuinfo`` and ``riscv_hwprobe``.
    """
    warnings: List[str] = []
    sources: List[str] = []
    cpuinfo = parse_cpuinfo(payload.get("cpuinfo"))
    extensions: Set[str] = set()
    if cpuinfo:
        sources.append("cpuinfo")
        extensions |= parse_isa_string(cpuinfo.get("isa", ""))
        extensions |= parse_isa_string(cpuinfo.get("hart isa", ""))
    elif kind == "qemu":
        warnings.append(
            "/proc/cpuinfo shows no riscv64 CPU. QEMU older than 8.1 "
            "shows the host's, so extensions come from test instructions."
        )

    pairs: Dict[int, int] = {}
    hw = payload.get("hwprobe") or {}
    uname = payload.get("uname") or []
    release = _version_tuple(uname[2]) if len(uname) > 2 else None
    if "pairs" in hw:
        sources.append("hwprobe")
        pairs = {int(key): int(value) for key, value in hw["pairs"]}
        extensions |= hwprobe_extensions(pairs)
    elif (
        kind == "native"
        and hw.get("error") == "errno 38"
        and release
        and release >= (6, 4)
    ):
        # Seen on the Megrez: the host kernel answers, but podman's
        # default seccomp profile does not list the syscall.
        warnings.append(
            f"riscv_hwprobe is blocked in the container (errno 38), "
            f"although Linux {uname[2]} has it: the container's seccomp "
            "profile does not list it. Extensions come from cpuinfo and "
            "test instructions; the misaligned access speed comes from "
            "a timing test."
        )
    else:
        warnings.append(
            "riscv_hwprobe is not available "
            f"({hw.get('error', 'no data')}): it needs Linux 6.4 or "
            "QEMU 8.1."
        )

    tests: Dict[str, str] = {}
    vlen: Optional[int] = None
    for name, result in (payload.get("tests") or {}).items():
        code = result.get("rc")
        if code == 0:
            tests[name] = "present"
            extensions.add(name)
            if name == VECTOR_EXTENSION:
                vlenb = _parse_int(result.get("value"))
                vlen = vlenb * 8 if vlenb else None
        elif code in (-4, 132):
            # SIGILL: the child died, or a shell reported 128 + 4.
            tests[name] = "absent"
            extensions.discard(name)
        else:
            tests[name] = "error"
    if any(value != "error" for value in tests.values()):
        sources.append("tests")
    if tests and all(value == "error" for value in tests.values()):
        warnings.append(
            "Test instructions could not run (an executable memory map "
            "is needed); extensions come from cpuinfo and hwprobe only."
        )
    if not sources:
        warnings.append(
            "No source answered; Atesor assumes plain rv64gc."
        )
    if not all(letter in extensions for letter in _BASE_LETTERS):
        # The probe itself runs rv64gc code (python3), so the base is
        # present even when no source says so.
        extensions.update(_BASE_LETTERS)

    core = {
        "mvendorid": pairs.get(_KEY_MVENDORID)
        if _KEY_MVENDORID in pairs
        else _parse_int(cpuinfo.get("mvendorid")),
        "marchid": pairs.get(_KEY_MARCHID)
        if _KEY_MARCHID in pairs
        else _parse_int(cpuinfo.get("marchid")),
        "mimpid": pairs.get(_KEY_MIMPID)
        if _KEY_MIMPID in pairs
        else _parse_int(cpuinfo.get("mimpid")),
        "uarch": cpuinfo.get("uarch"),
        "model": (payload.get("dt_model") or "").strip("\x00\n ") or None,
        "compatible": [
            part
            for part in (payload.get("dt_compatible") or "").split("\x00")
            if part
        ],
    }
    if cpuinfo.get("uarch") == "qemu" and kind == "native":
        warnings.append("The native machine reports a QEMU CPU.")
    entry = match_core(core) if kind == "native" else None

    meminfo = payload.get("meminfo") or ""
    mem_match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
    glibc = payload.get("glibc")
    libc = glibc if glibc else ("musl" if payload.get("musl") else None)
    loadavg = (payload.get("loadavg") or "").split()
    cpufreq = payload.get("cpufreq") or {}
    board = {
        "governor": cpufreq.get("scaling_governor"),
        "cur_freq_khz": _parse_int(cpufreq.get("scaling_cur_freq")),
        "max_freq_khz": _parse_int(cpufreq.get("scaling_max_freq")),
        "hw_max_freq_khz": _parse_int(cpufreq.get("cpuinfo_max_freq")),
        "temperatures_mc": list(payload.get("thermal") or []),
        "perf_event_paranoid": _parse_int(
            payload.get("perf_event_paranoid")
        ),
        "perf": payload.get("perf"),
        "loadavg_1m": float(loadavg[0]) if loadavg else None,
        "caches": payload.get("caches") or {},
    }

    misaligned = misaligned_speed(pairs)
    misaligned_source = "hwprobe" if misaligned != "unknown" else "none"
    timing = payload.get("misaligned_test") or {}
    if misaligned == "unknown" and kind == "native":
        # Under QEMU the timing shows the emulator, not a machine.
        misaligned = misaligned_from_timing(timing)
        if misaligned != "unknown":
            misaligned_source = "timing test"
    if misaligned == "unknown":
        warnings.append(
            "The misaligned access speed is unknown, so the variants "
            "keep -mstrict-align."
        )

    toolchains = _toolchains(payload)
    for name, facts in toolchains["compilers"].items():
        if (
            "zba" in facts["accepts"]
            and "zbb" in facts["accepts"]
            and not facts["emits_bitmanip"]
        ):
            warnings.append(
                f"{name} ({facts['version']}) accepts Zba/Zbb but emits "
                "no such instruction; Atesor builds no ISA variant with it."
            )

    profile = MachineProfile(
        kind=kind,
        emulated_cpu=emulated_cpu,
        extensions=sorted(extensions),
        vlen=vlen,
        core=core,
        core_name=entry["name"] if entry else None,
        misaligned=misaligned,
        misaligned_source=misaligned_source,
        misaligned_test={
            key: (round(value, 3) if isinstance(value, float) else value)
            for key, value in timing.items()
            if key in ("rc", "aligned_ns", "misaligned_ns", "err")
        },
        cores=_parse_int(payload.get("nproc")),
        allowed_cpus=[
            int(cpu)
            for cpu in payload.get("cpus") or []
            if isinstance(cpu, int)
        ],
        mem_kib=int(mem_match.group(1)) if mem_match else None,
        kernel=uname[2] if len(uname) > 2 else None,
        libc=libc,
        board=board,
        toolchains=toolchains,
        label=profile_label(extensions),
        sources=sources,
        tests=tests,
        warnings=warnings,
    )
    if kind == "qemu" and not is_named_model(emulated_cpu):
        profile.warnings.append(
            "QEMU runs its max CPU (the default since QEMU 8.2), which no "
            "real board matches; ISA variants are off. Use --emulate-cpu "
            "NAME to target a real CPU model."
        )
    return profile


# Runs in the sandbox. Argument: JSON {"cpus", "rounds", "size_mb"}.
# One child per CPU, all at once and pinned: each compresses and
# decompresses the same buffer "rounds" times with Python's zlib and
# counts the round trips that do not give the buffer back. A child
# that dies counts as -1. Unchanged code must always give the same
# data; a machine that does not (an unstable clock, bad memory)
# makes every speed number meaningless.
_INTEGRITY_PROGRAM = r'''
import json
import os
import sys
import time
import zlib

SPEC = json.loads(sys.argv[1])
size = int(SPEC["size_mb"]) << 20
allowed = os.sched_getaffinity(0)
cpus = [int(cpu) for cpu in SPEC["cpus"] if int(cpu) in allowed]
cpus = cpus or sorted(allowed)


def work(cpu):
    os.sched_setaffinity(0, {cpu})
    text = b"riscv vector atesor port build "
    data = os.urandom(size // 2)
    data += (text * (size // len(text) + 1))[: size - len(data)]
    bad = 0
    for _ in range(int(SPEC["rounds"])):
        if zlib.decompress(zlib.compress(data, 6)) != data:
            bad += 1
    return bad


start = time.time()
children = {}
for cpu in cpus:
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_end)
        try:
            answer = str(work(cpu))
        except Exception:
            answer = "-1"
        os.write(write_end, answer.encode())
        os._exit(0)
    os.close(write_end)
    children[cpu] = (pid, read_end)
bad = {}
for cpu, (pid, read_end) in children.items():
    chunks = []
    while True:
        chunk = os.read(read_end, 64)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_end)
    os.waitpid(pid, 0)
    try:
        bad[str(cpu)] = int(b"".join(chunks).decode() or "-1")
    except ValueError:
        bad[str(cpu)] = -1
sys.stdout.write("@@MARKER@@" + json.dumps({
    "bad": bad,
    "rounds": int(SPEC["rounds"]),
    "size_mb": int(SPEC["size_mb"]),
    "seconds": round(time.time() - start, 1),
}) + "\n")
'''.replace("@@MARKER@@", _OUTPUT_MARKER)

INTEGRITY_ROUNDS = 10
INTEGRITY_MB = 8


def check_integrity(
    cpus: Sequence[int],
    rounds: int = INTEGRITY_ROUNDS,
    size_mb: int = INTEGRITY_MB,
    timeout: int = 600,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Check that the machine computes the same data twice, on each CPU.

    Every CPU runs ``rounds`` zlib round trips of ``size_mb`` MiB at
    the same time. On the Megrez at 1.8 GHz, 3 of 40 such round trips
    gave wrong data; at 1.4 GHz, 0 of 160 did.

    Args:
        cpus: The CPUs to test; all allowed CPUs when empty.
        rounds: Round trips per CPU.
        size_mb: The buffer size in MiB.
        timeout: The timeout of the whole check in seconds.

    Returns:
        ``(result, "")`` with ``bad`` (CPU number to wrong round trips,
        -1 for a crash), ``rounds``, ``size_mb`` and ``seconds``; or
        ``(None, error)`` when the check could not run.
    """
    # validate=False: the program is a fixed constant and its only
    # argument is JSON passed as one argv element.
    result = execute_command(
        [
            "python3",
            "-c",
            _INTEGRITY_PROGRAM,
            json.dumps(
                {"cpus": list(cpus), "rounds": rounds, "size_mb": size_mb}
            ),
        ],
        timeout=timeout,
        validate=False,
    )
    payload = _parse_probe_output(result.stdout)
    if payload is None or not isinstance(payload.get("bad"), dict):
        detail = (result.stderr or result.stdout or "").strip()[-300:]
        return None, f"exit {result.exit_code}: {detail}"
    return payload, ""


def integrity_failures(result: Dict[str, Any]) -> Dict[str, int]:
    """Return ``CPU -> wrong round trips`` for the CPUs that failed.

    A crashed child (-1) counts as a failure too.
    """
    return {
        cpu: count
        for cpu, count in (result.get("bad") or {}).items()
        if count != 0
    }


def _parse_probe_output(stdout: str) -> Optional[Dict[str, Any]]:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_OUTPUT_MARKER):
            try:
                payload = json.loads(line[len(_OUTPUT_MARKER):])
            except ValueError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def probe_machine(
    kind: str,
    emulated_cpu: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout: int = PROBE_TIMEOUT,
) -> Tuple[Optional[MachineProfile], Optional[str]]:
    """Run the probe in the sandbox and evaluate it.

    Args:
        kind: ``native`` or ``qemu``.
        emulated_cpu: A ``QEMU_CPU`` value for QEMU runs, or None for
            QEMU's default CPU. Ignored on native.
        cwd: The sandbox directory to run in; the workspace by default.
        timeout: The probe timeout in seconds.

    Returns:
        ``(profile, None)`` on success, ``(None, error)`` when the
        probe could not run.
    """
    extra_env = None
    if kind == "qemu" and emulated_cpu:
        extra_env = {"QEMU_CPU": emulated_cpu}
    arguments = {
        "compilers": list(PROBE_COMPILERS),
        "isa_candidates": list(ISA_CANDIDATES) + [VECTOR_EXTENSION],
        "tune_candidates": list(TUNE_CANDIDATES),
        "rust_cpus": list(RUST_CPU_CANDIDATES),
    }
    # validate=False: the program is a fixed constant and its only
    # argument is JSON passed as one argv element.
    result = execute_command(
        ["python3", "-c", _PROBE_PROGRAM, json.dumps(arguments)],
        cwd=cwd or DockerConfig.WORKSPACE_PATH,
        timeout=timeout,
        validate=False,
        extra_env=extra_env,
    )
    payload = _parse_probe_output(result.stdout)
    if payload is None:
        detail = (result.stderr or result.stdout or "").strip()[-300:]
        error = f"exit {result.exit_code}: {detail}"
        logger.warning(f"Machine probe could not run ({error})")
        return None, error
    profile = evaluate_probe(payload, kind, emulated_cpu)
    logger.info(
        f"Machine probe: {profile.label} ({profile.kind}), "
        f"core {profile.core_name or 'unknown'}, "
        f"misaligned {profile.misaligned}"
    )
    return profile, None


# ---------------------------------------------------------------------------
# Rules: from a profile to flags
# ---------------------------------------------------------------------------


def march_for(
    profile: MachineProfile,
    compiler: str,
    include_vector: bool = False,
) -> Optional[str]:
    """Return the ``-march`` value for a variant, or None.

    The value is rv64gc plus every candidate extension that the machine
    has and the compiler accepts, in canonical order. None when ISA
    variants are off for this machine, when the compiler fails the
    emit test, or when nothing would be added.

    Args:
        profile: The machine profile.
        compiler: The compiler command, for example ``gcc``.
        include_vector: Add V when the machine has it (vector phase).

    Returns:
        For example ``rv64gc_zba_zbb``, or None.
    """
    if not profile.isa_variants_allowed:
        return None
    facts = profile.toolchains.get("compilers", {}).get(compiler)
    if not facts or not facts.get("emits_bitmanip"):
        return None
    if not profile.has(*_BASE_LETTERS):
        return None
    accepted = set(facts.get("accepts") or [])
    extras = [
        name
        for name in ISA_CANDIDATES
        if profile.has(name) and name in accepted
    ]
    vector = (
        include_vector
        and profile.has(VECTOR_EXTENSION)
        and VECTOR_EXTENSION in accepted
    )
    if not extras and not vector:
        return None
    base = "rv64gcv" if vector else "rv64gc"
    return base + "".join(f"_{name}" for name in extras)


def tune_for(profile: MachineProfile, compiler: str) -> Optional[str]:
    """Return the ``-mtune=...`` flag for the probed core, or None.

    The first name in the core table's list that the compiler accepted
    during the probe wins. An unknown core gets no tune flag.
    """
    if not profile.core_name:
        return None
    entry = next(
        (row for row in CORE_TABLE if row["name"] == profile.core_name),
        None,
    )
    facts = profile.toolchains.get("compilers", {}).get(compiler)
    if not entry or not facts:
        return None
    key = "clang_tunes" if facts["family"] == "clang" else "gcc_tunes"
    accepted = set(facts.get("tunes") or [])
    for name in entry[key]:
        if name in accepted:
            return f"-mtune={name}"
    return None


def align_flags(profile: MachineProfile, compiler: str) -> List[str]:
    """Return ``["-mstrict-align"]`` unless misaligned access is fast.

    On a core that traps on a misaligned access, the kernel or the
    firmware emulates each one (hundreds of cycles), so a variant lets
    the compiler emit them only when the probe found them fast.
    """
    if profile.misaligned == "fast":
        return []
    facts = profile.toolchains.get("compilers", {}).get(compiler) or {}
    if "-mstrict-align" in (facts.get("flags") or []):
        return ["-mstrict-align"]
    return []


def goriscv64_for(profile: MachineProfile) -> Tuple[str, Optional[str]]:
    """Return the GORISCV64 level for the machine and an exception note.

    The highest level whose extensions the machine has in full wins.
    One exception: ``rva22u64`` on a machine that lacks only Zbs, with
    a Go release that emits no Zbs (gate 2 then checks the binary).

    Returns:
        ``(level, note)``; the note is None unless the exception
        applies. ``rva20u64`` is Go's default and changes nothing.
    """
    version = _version_tuple(profile.toolchains.get("go"))
    if not version or not profile.isa_variants_allowed:
        return "rva20u64", None
    for level, first, needs in _GO_LEVELS:
        if version >= first and profile.has(*needs):
            return level, None
    if (
        version >= (1, 23)
        and version[:2] <= _GO_NO_ZBS_UNTIL
        and profile.has("zba", "zbb")
        and not profile.has("zbs")
    ):
        return (
            "rva22u64",
            "the machine lacks Zbs; this Go release emits no Zbs, and the "
            "package tests on the machine decide",
        )
    return "rva20u64", None


def rust_cpu_fits(profile: MachineProfile, cpu: str) -> bool:
    """Return True when ``-C target-cpu=cpu`` is safe on this machine.

    rustc enables every feature of a CPU model. Each feature that the
    model adds to rustc's default must be an extension the probe found;
    an ``unaligned-*`` feature needs fast misaligned access, and the
    ``zve*``/``zvl*`` vector features need V. Without rustc's feature
    lists the answer is False.
    """
    features = profile.toolchains.get("rust_cpu_features") or {}
    if cpu not in features or "(default)" not in features:
        return False
    extra = set(features[cpu]) - set(features["(default)"])
    for name in extra:
        if name.startswith("unaligned-"):
            if profile.misaligned != "fast":
                return False
        elif name.startswith(("zve", "zvl")):
            if not profile.has(VECTOR_EXTENSION):
                return False
        elif not profile.has(name):
            return False
    return True


def rust_flags_for(profile: MachineProfile) -> Optional[str]:
    """Return the RUSTFLAGS value for a Rust variant, or None.

    ``-C target-cpu`` from the core table when rustc knows that CPU and
    its features fit the machine (:func:`rust_cpu_fits`); otherwise
    ``-C target-feature`` with the probed stable features.
    """
    tools = profile.toolchains
    if not tools.get("rust") or not profile.isa_variants_allowed:
        return None
    entry = next(
        (row for row in CORE_TABLE if row["name"] == profile.core_name),
        None,
    )
    cpu = entry["rust_cpu"] if entry else None
    if (
        cpu
        and cpu in (tools.get("rust_cpus") or [])
        and rust_cpu_fits(profile, cpu)
    ):
        return f"-C target-cpu={cpu}"
    features = [
        f"+{name}" for name in _RUST_STABLE_FEATURES if profile.has(name)
    ]
    if not features:
        return None
    return "-C target-feature=" + ",".join(features)


def forbidden_classes(profile: MachineProfile) -> List[str]:
    """Return the instruction classes the machine cannot run.

    The variant ISA scan (gate 1) warns when an artifact contains any
    of them.
    """
    classes = ("zba", "zbb", "zbs", "zicond", "zfa", "zfh", "v", "zcb")
    missing = []
    for name in classes:
        if name == "zfh":
            if not profile.has("zfh") and not profile.has("zfhmin"):
                missing.append(name)
        elif not profile.has(name):
            missing.append(name)
    return missing


def qemu_cpu_for(profile: MachineProfile) -> str:
    """Return a ``QEMU_CPU`` string that models the probed machine.

    The model is QEMU's bare ``rv64i`` CPU plus the probed extensions
    that QEMU knows, so a variant that uses anything else stops with
    SIGILL under emulation. QEMU 9.0 or later is needed.
    """
    flags = [f"{letter}=true" for letter in ("m", "a", "f", "d", "c")]
    flags += ["zicsr=true", "zifencei=true"]
    known = (
        "zicntr",
        "zihpm",
        "zba",
        "zbb",
        "zbs",
        "zicond",
        "zfa",
        "zfh",
        "zfhmin",
        "zcb",
    )
    flags += [f"{name}=true" for name in known if profile.has(name)]
    if profile.has(VECTOR_EXTENSION):
        flags.append("v=true")
        if profile.vlen:
            flags.append(f"vlen={profile.vlen}")
    return ",".join(["rv64i"] + flags)


def _timing_text(profile: MachineProfile) -> str:
    test = profile.misaligned_test or {}
    aligned, misaligned = test.get("aligned_ns"), test.get("misaligned_ns")
    if profile.misaligned_source != "timing test" or not aligned:
        return ""
    return (
        f"; one load takes {aligned:.2f} ns aligned, "
        f"{misaligned:.2f} ns misaligned"
    )


def describe(profile: MachineProfile) -> List[str]:
    """Return human-readable lines that summarize a profile."""
    core = profile.core_name or "unknown core"
    lines = [
        f"Machine:     {profile.kind}"
        + (f" (QEMU_CPU={profile.emulated_cpu})" if profile.emulated_cpu
           else ""),
        f"ISA label:   {profile.label}",
        "Extensions:  " + " ".join(profile.extensions),
        f"Vector:      {f'VLEN {profile.vlen}' if profile.vlen else 'no'}",
        f"Core:        {core} ({profile.core.get('model') or 'no model'})",
        f"Misaligned:  {profile.misaligned} "
        f"(source: {profile.misaligned_source}{_timing_text(profile)})",
        f"CPUs:        {profile.cores}",
        f"Kernel/libc: {profile.kernel} / {profile.libc}",
        f"Sources:     {', '.join(profile.sources) or 'none'}",
    ]
    governor = profile.board.get("governor")
    if governor:
        lines.append(
            f"Governor:    {governor} at "
            f"{profile.board.get('cur_freq_khz')} kHz"
        )
    for name, facts in profile.toolchains.get("compilers", {}).items():
        lines.append(f"Compiler:    {name}: {facts['version']}")
    for warning in profile.warnings:
        lines.append(f"Warning:     {warning}")
    return lines


def describe_rules(profile: MachineProfile) -> List[str]:
    """Return the settings that performance mode derives from a profile.

    On a native machine the last line gives the ``--emulate-cpu`` value
    that models it under QEMU.
    """
    lines = []
    for name in sorted(profile.toolchains.get("compilers", {})):
        flags = []
        tune = tune_for(profile, name)
        if tune:
            flags.append(tune)
        flags += align_flags(profile, name)
        march = march_for(profile, name)
        if march:
            flags.append(f"-march={march}")
        lines.append(
            f"Flags:       {name}: "
            + (" ".join(flags) or "no change from the defaults")
        )
    if profile.toolchains.get("go"):
        level, note = goriscv64_for(profile)
        lines.append(
            f"Go:          GORISCV64={level}" + (f" ({note})" if note else "")
        )
    if profile.toolchains.get("rust"):
        lines.append(
            f"Rust:        RUSTFLAGS={rust_flags_for(profile) or '(default)'}"
        )
    if profile.kind == "native":
        lines.append(f"QEMU model:  --emulate-cpu '{qemu_cpu_for(profile)}'")
    return lines


def tune_names(compilers: Sequence[str]) -> List[str]:
    """Return the tune candidates, for tests and documentation."""
    del compilers
    return list(TUNE_CANDIDATES)
