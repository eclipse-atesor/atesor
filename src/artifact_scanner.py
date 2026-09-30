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

"""Verify that a build produced riscv64 machine code.

The scanner runs one Python program inside the sandbox (every sandbox
image ships ``python3``). The program walks the repository, reads the
ELF header of each executable and shared library and of each member
of each static archive, asks git which files are committed, and
prints one JSON document. The verdict is computed on the host by
:func:`evaluate_scan`.

Design rules:

* The ELF header is the evidence. ``e_machine`` must be ``EM_RISCV``
  and ``EI_CLASS`` must be ``ELFCLASS64``: riscv32 is not a riscv64
  port. The text of ``file(1)`` is not parsed.
* Every build output counts. One x86-64 build output fails the
  verification even when other outputs are riscv64, and one foreign
  member fails a static archive.
* Files committed to git and not modified by the build are
  pre-existing (vendored blobs, test fixtures). They are reported but
  they never decide the verdict. ``node_modules`` is not scanned: it
  holds downloaded dependencies, not build outputs.
* When the scan cannot run, the result is UNVERIFIED, never VERIFIED.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .tools import execute_command

logger = logging.getLogger(__name__)

VERIFIED = "verified"
WRONG_ARCH = "wrong_arch"
UNVERIFIED = "unverified"

EM_RISCV = 243

# Deliverable directories outside the repository where ``make install``,
# ``go install`` and ``cargo install`` put their outputs.
OUTSIDE_DIRS = (
    "/usr/local/bin",
    "/usr/local/sbin",
    "/usr/local/lib",
    "/usr/local/lib64",
    "/root/go/bin",
    "/root/.cargo/bin",
)

_MACHINE_NAMES = {
    3: "x86",
    8: "mips",
    20: "ppc",
    21: "ppc64",
    22: "s390x",
    40: "arm",
    62: "x86-64",
    183: "aarch64",
    247: "bpf",
    258: "loongarch",
}

# e_type values (ELF spec).
_ET_REL = 1
_ET_DYN = 3
_ET_CORE = 4

# RISC-V e_flags (RISC-V ELF psABI).
_EF_RISCV_FLOAT_ABI = 0x6
_EF_RISCV_RVE = 0x8
_FLOAT_ABI_SUFFIX = {0x0: "", 0x2: "f", 0x4: "d", 0x6: "q"}

# The ABI of the Alpine, Debian and Ubuntu riscv64 ports.
_SYSTEM_ABI = "lp64d"

_MAX_EXPECTED_NAMES = 10
_SAFE_BASENAME = re.compile(r"^[A-Za-z0-9._+-]+$")
_OUTPUT_MARKER = "ATESOR_SCAN_JSON "

# Runs inside the sandbox. It must stay self-contained and compatible
# with Python 3.8+ (Ubuntu jammy, Debian trixie and Alpine images).
# Arguments: root, max_files, JSON list of expected basenames,
# minimum mtime (epoch seconds) for files outside the repository.
_SCAN_PROGRAM = r'''
import json
import os
import stat
import struct
import subprocess
import sys

ELF_MAGIC = b"\x7fELF"
AR_MAGIC = b"!<arch>\n"
THIN_AR_MAGIC = b"!<thin>\n"
AR_INDEX_NAMES = (b"/", b"/SYM64/", b"//", b"__.SYMDEF", b"__.SYMDEF SORTED")
OUTSIDE_DIRS = %(outside_dirs)s
# node_modules holds downloaded dependencies, not build outputs: npm
# packages may ship x86 prebuilds that would fail a good riscv64 build.
SKIP_DIRS = (".git", "node_modules")
MAX_ENTRIES = 3000
OUTSIDE_DEPTH = 3


def elf_ident(head):
    if len(head) < 52 or head[:4] != ELF_MAGIC:
        return None
    ei_class, ei_data = head[4], head[5]
    if ei_class not in (1, 2) or ei_data not in (1, 2):
        return None
    order = "<" if ei_data == 1 else ">"
    e_type, e_machine = struct.unpack(order + "HH", head[16:20])
    if ei_class == 2:
        if len(head) < 64:
            return None
        flags = struct.unpack(order + "I", head[48:52])[0]
    else:
        flags = struct.unpack(order + "I", head[36:40])[0]
    return [32 * ei_class, ei_data, e_type, e_machine, flags]


def archive_members(handle, size):
    counts = {}
    other = 0
    pos = 8
    while pos + 60 <= size:
        handle.seek(pos)
        header = handle.read(60)
        if len(header) < 60 or header[58:60] != b"`\n":
            break
        name = header[:16].rstrip(b" ")
        try:
            member_size = int(header[48:58].strip() or b"0")
        except ValueError:
            break
        data_pos = pos + 60
        pos = data_pos + member_size + (member_size & 1)
        if name in AR_INDEX_NAMES:
            continue
        if name.startswith(b"#1/"):
            try:
                name_len = int(name[3:])
            except ValueError:
                continue
            data_pos += name_len
            member_size -= name_len
        if member_size <= 0:
            continue
        handle.seek(data_pos)
        ident = elf_ident(handle.read(min(64, member_size)))
        if ident is None:
            other += 1
            continue
        key = "%%d/%%d/%%d/%%d" %% (ident[0], ident[1], ident[3], ident[4])
        counts[key] = counts.get(key, 0) + 1
    return counts, other


def classify(path, size):
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
            if head[:8] == AR_MAGIC:
                counts, other = archive_members(handle, size)
                return {"ar": counts, "other": other}
            if head[:8] == THIN_AR_MAGIC:
                return {"ar": {}, "other": 0, "thin": True}
            ident = elf_ident(head)
    except OSError:
        return None
    if ident is None:
        return None
    return {"elf": ident}


def is_library(name):
    return name.endswith(".a") or name.endswith(".so") or ".so." in name


def is_riscv64(info):
    elf = info.get("elf")
    if elf:
        # elf[2] == 4 is a core dump, which the verdict ignores.
        return elf[0] == 64 and elf[1] == 1 and elf[2] != 4 and elf[3] == 243
    members = info.get("ar") or {}
    return bool(members) and all(
        key.startswith("64/1/243/") for key in members
    )


def git(root, args):
    try:
        proc = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", root] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def git_paths(root, args):
    out = git(root, args)
    if out is None:
        return None
    return set(
        p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p
    )


def scan_tree(root, max_files, tracked, result):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_size < 52:
                continue
            executable = bool(st.st_mode & 0o111)
            if not (executable or is_library(name)):
                continue
            if result["examined"] >= max_files:
                result["truncated"] = True
                return
            result["examined"] += 1
            info = classify(path, st.st_size)
            if info is None:
                continue
            rel = os.path.relpath(path, root)
            is_tracked = rel in tracked
            if len(result["files"]) >= MAX_ENTRIES and is_riscv64(info):
                # The list is full. Keep reading, and list only what can
                # fail the verdict, so no foreign output is missed.
                if not is_tracked:
                    result["unlisted_riscv64"] += 1
                continue
            info.update(
                path=rel,
                size=st.st_size,
                exec=executable,
                tracked=is_tracked,
            )
            result["files"].append(info)


def scan_outside(names, min_mtime, result):
    for base in OUTSIDE_DIRS:
        if not os.path.isdir(base):
            continue
        base_depth = base.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(base):
            if dirpath.rstrip("/").count("/") - base_depth >= OUTSIDE_DEPTH:
                dirnames[:] = []
            for name in filenames:
                if not any(
                    name == n or name.startswith(n + ".") for n in names
                ):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode) or st.st_mtime < min_mtime:
                    continue
                info = classify(path, st.st_size)
                if info is None:
                    continue
                info.update(
                    path=path,
                    size=st.st_size,
                    exec=bool(st.st_mode & 0o111),
                    tracked=False,
                )
                result["outside"].append(info)


def main():
    root = sys.argv[1]
    max_files = int(sys.argv[2])
    names = json.loads(sys.argv[3])
    min_mtime = float(sys.argv[4])
    result = {
        "root": root,
        "files": [],
        "outside": [],
        "examined": 0,
        "truncated": False,
        "unlisted_riscv64": 0,
        "commit": None,
        "tracked_known": False,
    }
    if not os.path.isdir(root):
        result["error"] = "repository directory does not exist: " + root
    else:
        head = git(root, ["rev-parse", "HEAD"])
        if head:
            result["commit"] = head.decode("ascii", "replace").strip()
        tracked = git_paths(root, ["ls-files", "-z", "--recurse-submodules"])
        if tracked is None:
            tracked = git_paths(root, ["ls-files", "-z"])
        modified = git_paths(root, ["ls-files", "-z", "-m"]) or set()
        result["tracked_known"] = tracked is not None
        scan_tree(root, max_files, (tracked or set()) - modified, result)
    if names:
        scan_outside(names, min_mtime, result)
    sys.stdout.write("%(marker)s" + json.dumps(result) + "\n")


main()
''' % {
    "outside_dirs": repr(OUTSIDE_DIRS),
    "marker": _OUTPUT_MARKER,
}


@dataclass
class Artifact:
    """One ELF file or static archive found by the scan.

    Attributes:
        path: Absolute path inside the sandbox.
        type: ``binary``, ``library_shared``, ``library_static`` or
            ``object``.
        arch: Architecture label, e.g. ``riscv64``, ``riscv32``,
            ``x86-64``, ``aarch64``. An archive with members of more
            than one architecture is labelled ``mixed(<a>+<b>)``.
        abi: The RISC-V psABI (``lp64d``, ...) or None.
        size: File size in bytes.
        prebuilt: True when the file is committed to git and the build
            did not modify it. Prebuilt files never decide the verdict.
    """

    path: str
    type: str
    arch: str
    abi: Optional[str]
    size: int
    prebuilt: bool = False

    @property
    def is_riscv64(self) -> bool:
        """Return True when the artifact is riscv64 machine code."""
        return self.arch == "riscv64"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "path": self.path,
            "type": self.type,
            "arch": self.arch,
            "abi": self.abi,
            "size": self.size,
            "prebuilt": self.prebuilt,
        }


@dataclass
class VerificationResult:
    """The verdict of one artifact scan.

    Attributes:
        status: ``verified``, ``wrong_arch`` or ``unverified``.
        reason: One-line explanation of the status.
        riscv64: riscv64 build outputs (the evidence for ``verified``).
        foreign: Build outputs for another architecture.
        prebuilt_foreign: Committed non-riscv64 binaries. They do not
            fail the build, but the build must not link or run them.
        abi_warnings: riscv64 outputs whose ABI is not the system ABI.
        source_commit: The ``git rev-parse HEAD`` of the scanned tree.
        examined: Number of candidate files that were read.
        truncated: True when the scan stopped at the file limit before
            it read every candidate. It then cannot prove the port.
        scan_error: Why the scan could not run, or None.
        expected_missing: Expected artifacts that were not found.
        unlisted_riscv64: riscv64 build outputs read after the scan
            list was full. They are counted but not listed. Foreign
            outputs are always listed.
    """

    status: str
    reason: str
    riscv64: List[Artifact] = field(default_factory=list)
    foreign: List[Artifact] = field(default_factory=list)
    prebuilt_foreign: List[Artifact] = field(default_factory=list)
    abi_warnings: List[str] = field(default_factory=list)
    source_commit: Optional[str] = None
    examined: int = 0
    truncated: bool = False
    scan_error: Optional[str] = None
    expected_missing: List[str] = field(default_factory=list)
    unlisted_riscv64: int = 0

    @property
    def verified(self) -> bool:
        """Return True when riscv64 build outputs were proven."""
        return self.status == VERIFIED

    def to_dict(self, max_items: int = 200) -> Dict[str, Any]:
        """Return a JSON-serialisable summary for state and reports.

        Args:
            max_items: Maximum entries kept per artifact list. The
                counts always cover the full lists.

        Returns:
            A dictionary with the status, counts and artifact lists.
        """
        return {
            "status": self.status,
            "verified": self.verified,
            "reason": self.reason,
            "source_commit": self.source_commit,
            "counts": {
                "riscv64": len(self.riscv64) + self.unlisted_riscv64,
                "foreign": len(self.foreign),
                "prebuilt_foreign": len(self.prebuilt_foreign),
            },
            "riscv64": [a.to_dict() for a in self.riscv64[:max_items]],
            "foreign": [a.to_dict() for a in self.foreign[:max_items]],
            "prebuilt_foreign": [
                a.to_dict() for a in self.prebuilt_foreign[:max_items]
            ],
            "abi_warnings": self.abi_warnings[:max_items],
            "examined": self.examined,
            "truncated": self.truncated,
            "scan_error": self.scan_error,
            "expected_missing": list(self.expected_missing),
        }


def describe_elf(
    bits: int, data: int, machine: int, flags: int
) -> "tuple[str, Optional[str]]":
    """Return the architecture label and RISC-V ABI of an ELF header.

    Args:
        bits: 32 or 64 (``EI_CLASS``).
        data: 1 for little-endian, 2 for big-endian (``EI_DATA``).
        machine: The ``e_machine`` value.
        flags: The ``e_flags`` value.

    Returns:
        ``(arch, abi)``, e.g. ``("riscv64", "lp64d")`` or
        ``("x86-64", None)``.
    """
    if machine == EM_RISCV:
        arch = "riscv64" if bits == 64 else "riscv32"
        base = "lp64" if bits == 64 else "ilp32"
        if flags & _EF_RISCV_RVE:
            base += "e"
        abi = base + _FLOAT_ABI_SUFFIX[flags & _EF_RISCV_FLOAT_ABI]
    else:
        arch = _MACHINE_NAMES.get(machine, f"em-{machine}")
        if machine in (8, 20, 258) and bits == 64:
            arch += "64"
        abi = None
    if data == 2:
        arch += "-be"
    return arch, abi


def _parse_member_key(key: str) -> "tuple[str, Optional[str]]":
    bits, data, machine, flags = (int(part) for part in key.split("/"))
    return describe_elf(bits, data, machine, flags)


def _artifact_from_entry(
    entry: Dict[str, Any], root: str
) -> Optional[Artifact]:
    """Build an :class:`Artifact` from one scan entry.

    Returns None for entries that carry no verifiable machine code:
    core dumps, thin archives and archives without ELF members.
    """
    path = entry.get("path", "")
    if not path.startswith("/"):
        path = f"{root.rstrip('/')}/{path}"
    name = path.rsplit("/", 1)[-1]
    size = int(entry.get("size", 0))
    prebuilt = bool(entry.get("tracked"))

    if "elf" in entry:
        bits, data, e_type, machine, flags = entry["elf"]
        if e_type == _ET_CORE:
            return None
        arch, abi = describe_elf(bits, data, machine, flags)
        if e_type == _ET_REL:
            kind = "object"
        elif e_type == _ET_DYN and (name.endswith(".so") or ".so." in name):
            kind = "library_shared"
        else:
            kind = "binary"
        return Artifact(path, kind, arch, abi, size, prebuilt)

    members = entry.get("ar") or {}
    if not members:
        return None
    described = [_parse_member_key(key) for key in members]
    arches = sorted({arch for arch, _abi in described})
    abis = sorted({abi for _arch, abi in described if abi})
    arch = arches[0] if len(arches) == 1 else f"mixed({'+'.join(arches)})"
    abi = None
    if abis:
        abi = abis[0] if len(abis) == 1 else "mixed"
    return Artifact(path, "library_static", arch, abi, size, prebuilt)


def _safe_expected_names(names: Sequence[str]) -> List[str]:
    """Return the expected artifact names that are plain basenames."""
    safe: List[str] = []
    for raw in names or ():
        name = str(raw).strip()
        if name in ("", ".", "..") or not _SAFE_BASENAME.match(name):
            continue
        if name not in safe:
            safe.append(name)
    return safe[:_MAX_EXPECTED_NAMES]


def evaluate_scan(
    payload: Dict[str, Any],
    repo_path: str,
    expected_names: Sequence[str] = (),
) -> VerificationResult:
    """Compute the verification verdict from a scan payload.

    Args:
        payload: The JSON document printed by the in-sandbox program.
        repo_path: The repository path that was scanned.
        expected_names: The artifact basenames the analysis expected.

    Returns:
        The verdict. Foreign build outputs give ``wrong_arch``;
        otherwise riscv64 build outputs give ``verified``; otherwise
        the result is ``unverified``.
    """
    riscv64: List[Artifact] = []
    foreign: List[Artifact] = []
    prebuilt_foreign: List[Artifact] = []
    prebuilt_riscv64 = 0
    for entry in payload.get("files", []):
        artifact = _artifact_from_entry(entry, repo_path)
        if artifact is None:
            continue
        if artifact.prebuilt:
            if artifact.is_riscv64:
                prebuilt_riscv64 += 1
            else:
                prebuilt_foreign.append(artifact)
        elif artifact.is_riscv64:
            riscv64.append(artifact)
        else:
            foreign.append(artifact)

    outside = [
        artifact
        for artifact in (
            _artifact_from_entry(entry, repo_path)
            for entry in payload.get("outside", [])
        )
        if artifact is not None
    ]

    result = VerificationResult(
        status=UNVERIFIED,
        reason="",
        riscv64=riscv64,
        foreign=foreign,
        prebuilt_foreign=prebuilt_foreign,
        source_commit=payload.get("commit"),
        examined=int(payload.get("examined", 0)),
        truncated=bool(payload.get("truncated")),
        scan_error=payload.get("error"),
        unlisted_riscv64=int(payload.get("unlisted_riscv64", 0)),
    )

    if not foreign and not riscv64 and not result.unlisted_riscv64:
        result.riscv64 = [a for a in outside if a.is_riscv64]
        result.foreign = [a for a in outside if not a.is_riscv64]

    result.abi_warnings = [
        f"{a.path}: {a.abi} (system ABI is {_SYSTEM_ABI})"
        for a in result.riscv64
        if a.abi != _SYSTEM_ABI
    ]

    if result.foreign:
        sample = ", ".join(
            f"{a.path} ({a.arch})" for a in result.foreign[:5]
        )
        more = len(result.foreign) - 5
        result.status = WRONG_ARCH
        result.reason = (
            f"{len(result.foreign)} build output(s) are not riscv64: "
            f"{sample}" + (f" and {more} more" if more > 0 else "")
        )
    elif result.truncated:
        # Outputs after the file limit were never read; one of them
        # could be foreign, so the riscv64 ones found prove nothing.
        result.status = UNVERIFIED
        result.reason = (
            f"the scan stopped after {result.examined} candidate files, "
            "before it read every build output"
        )
    elif result.riscv64 or result.unlisted_riscv64:
        result.status = VERIFIED
        where = (
            ""
            if riscv64 or result.unlisted_riscv64
            else " (installed outside the repository)"
        )
        result.reason = (
            f"{len(result.riscv64) + result.unlisted_riscv64} riscv64 ELF "
            f"build output(s) verified by header{where}"
        )
    else:
        found = {a.path.rsplit("/", 1)[-1] for a in outside}
        result.expected_missing = [
            name
            for name in _safe_expected_names(expected_names)
            if not any(f == name or f.startswith(name + ".") for f in found)
        ]
        if result.scan_error:
            result.reason = f"artifact scan failed: {result.scan_error}"
        else:
            result.reason = (
                "the build produced no ELF executable or library "
                f"({result.examined} candidate file(s) examined"
                + (
                    f", {prebuilt_riscv64 + len(prebuilt_foreign)} "
                    "committed binaries ignored"
                    if prebuilt_riscv64 or prebuilt_foreign
                    else ""
                )
                + ")"
            )
    return result


def _parse_scan_output(stdout: str) -> Optional[Dict[str, Any]]:
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_OUTPUT_MARKER):
            try:
                payload = json.loads(line[len(_OUTPUT_MARKER):])
            except ValueError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


class ArtifactScanner:
    """Scan a repository in the sandbox and verify its build outputs.

    Attributes:
        repo_path: The repository path inside the sandbox.
        timeout: Scan timeout in seconds.
        max_files: Maximum number of candidate files to read.
    """

    def __init__(
        self,
        repo_path: str,
        timeout: int = 900,
        max_files: int = 20000,
    ) -> None:
        self.repo_path = repo_path
        self.timeout = timeout
        self.max_files = max_files

    def scan(
        self,
        expected_names: Sequence[str] = (),
        since: Optional[float] = None,
    ) -> VerificationResult:
        """Scan the repository and return the verification verdict.

        Args:
            expected_names: Artifact basenames the analysis expects.
                They are also searched in :data:`OUTSIDE_DIRS` when the
                repository holds no build output.
            since: Epoch seconds. Files outside the repository older
                than this are ignored, so an earlier package's
                ``make install`` in a shared sandbox is not evidence.

        Returns:
            The :class:`VerificationResult`. A scan that cannot run
            gives ``unverified`` with ``scan_error`` set.
        """
        names = _safe_expected_names(expected_names)
        min_mtime = (since - 600) if since else 0
        # validate=False: every argument is fixed or shlex-quoted, and
        # the CommandValidator whitelist cannot match a multi-line
        # ``python3 -c`` program.
        command = [
            "python3",
            "-c",
            _SCAN_PROGRAM,
            self.repo_path,
            str(self.max_files),
            json.dumps(names),
            str(int(min_mtime)),
        ]
        result = execute_command(
            command, timeout=self.timeout, validate=False
        )
        payload = _parse_scan_output(result.stdout)
        if payload is None:
            detail = (result.stderr or result.stdout or "").strip()
            if result.exit_code == 127:
                detail = "python3 is not installed in the sandbox"
            error = f"exit {result.exit_code}: {detail[-300:]}"
            logger.warning(f"Artifact scan could not run ({error})")
            return VerificationResult(
                status=UNVERIFIED,
                reason=f"artifact scan failed: {error}",
                scan_error=error,
                expected_missing=names,
            )
        verdict = evaluate_scan(payload, self.repo_path, names)
        logger.info(f"Artifact verification: {verdict.reason}")
        return verdict
