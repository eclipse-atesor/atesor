"""Tests for src/artifact_scanner.py.

The in-sandbox scan program runs for real here, with the host's
python3, over files that carry real ELF and ar headers.
"""

import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src import artifact_scanner
from src.artifact_scanner import (
    UNVERIFIED,
    VERIFIED,
    WRONG_ARCH,
    ArtifactScanner,
    _parse_scan_output,
    describe_elf,
    evaluate_scan,
)
from src.state import CommandResult

EM_X86_64 = 62
EM_AARCH64 = 183
EM_RISCV = 243
RVC_DOUBLE = 0x5  # EF_RISCV_RVC | EF_RISCV_FLOAT_ABI_DOUBLE


def _elf(machine: int, bits: int = 64, e_type: int = 2, flags: int = 0):
    """Return a little-endian ELF header padded to 128 bytes."""
    ident = b"\x7fELF" + bytes([bits // 32, 1, 1]) + b"\0" * 9
    if bits == 64:
        body = struct.pack(
            "<HHIQQQIHHHHHH", e_type, machine, 1, 0, 64, 0, flags,
            64, 56, 0, 64, 0, 0,
        )
    else:
        body = struct.pack(
            "<HHIIIIIHHHHHH", e_type, machine, 1, 0, 52, 0, flags,
            52, 32, 0, 40, 0, 0,
        )
    return (ident + body).ljust(128, b"\0")


def _ar(members):
    """Return a GNU ar archive with a symbol table and ``members``."""
    out = b"!<arch>\n"
    entries = [(b"/", b"\0" * 8)] + [
        (name.encode() + b"/", data) for name, data in members
    ]
    for name, data in entries:
        out += (
            name.ljust(16)
            + b"0".ljust(12)
            + b"0".ljust(6)
            + b"0".ljust(6)
            + b"644".ljust(8)
            + str(len(data)).encode().ljust(10)
            + b"`\n"
        )
        out += data + (b"\n" if len(data) % 2 else b"")
    return out


class _RepoCase(unittest.TestCase):
    """A temporary repository that the real scan program reads."""

    def setUp(self) -> None:
        """Create an empty repository directory."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = self._tmp.name

    def put(self, rel: str, data: bytes, executable: bool = True) -> None:
        """Write ``data`` to ``rel`` in the repository."""
        path = os.path.join(self.repo, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        os.chmod(path, 0o755 if executable else 0o644)

    def run_scan(self, expected=(), max_files=20000, max_entries=None):
        """Run the in-sandbox program on the host and evaluate it.

        ``max_entries`` lowers the program's list limit, so a test can
        fill the list without writing thousands of files.
        """
        import json

        program = artifact_scanner._SCAN_PROGRAM
        if max_entries is not None:
            self.assertIn("MAX_ENTRIES = 3000\n", program)
            program = program.replace(
                "MAX_ENTRIES = 3000\n", f"MAX_ENTRIES = {max_entries}\n"
            )
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                self.repo,
                str(max_files),
                json.dumps(list(expected)),
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = _parse_scan_output(done.stdout)
        self.assertIsNotNone(payload, done.stderr)
        return evaluate_scan(payload, self.repo, expected)


class TestDescribeElf(unittest.TestCase):
    """Architecture and ABI come from the ELF header fields."""

    def test_detect_each_architecture(self) -> None:
        """Machine and class map to a stable label."""
        cases = [
            ((64, 1, EM_RISCV, RVC_DOUBLE), ("riscv64", "lp64d")),
            ((32, 1, EM_RISCV, RVC_DOUBLE), ("riscv32", "ilp32d")),
            ((64, 1, EM_X86_64, 0), ("x86-64", None)),
            ((64, 1, EM_AARCH64, 0), ("aarch64", None)),
            ((32, 1, 3, 0), ("x86", None)),
            ((64, 1, 9999, 0), ("em-9999", None)),
            ((64, 2, EM_RISCV, 0), ("riscv64-be", "lp64")),
        ]
        for fields, expected in cases:
            with self.subTest(fields=fields):
                self.assertEqual(describe_elf(*fields), expected)

    def test_riscv_float_abi_and_rve(self) -> None:
        """The e_flags float ABI bits and RVE bit set the psABI name."""
        self.assertEqual(describe_elf(64, 1, EM_RISCV, 0x0)[1], "lp64")
        self.assertEqual(describe_elf(64, 1, EM_RISCV, 0x2)[1], "lp64f")
        self.assertEqual(describe_elf(64, 1, EM_RISCV, 0x6)[1], "lp64q")
        self.assertEqual(describe_elf(32, 1, EM_RISCV, 0x8)[1], "ilp32e")


class TestVerifyBuildSuccess(_RepoCase):
    """The verdict over real files: one foreign output fails the port."""

    def test_no_artifacts_is_unverified(self) -> None:
        """A tree without ELF output is not a verified port."""
        self.put("configure", b"#!/bin/sh\necho configure\n" * 4)

        verdict = self.run_scan()

        self.assertEqual(verdict.status, UNVERIFIED)
        self.assertIn("no ELF executable or library", verdict.reason)

    def test_riscv_artifacts_succeed(self) -> None:
        """A riscv64 lp64d executable proves the port."""
        self.put("build/tool", _elf(EM_RISCV, flags=RVC_DOUBLE))

        verdict = self.run_scan()

        self.assertEqual(verdict.status, VERIFIED)
        self.assertEqual(verdict.riscv64[0].abi, "lp64d")
        self.assertEqual(verdict.riscv64[0].type, "binary")
        self.assertEqual(verdict.abi_warnings, [])

    def test_x64_only_artifacts_fail(self) -> None:
        """An x86-64 build output is a wrong-architecture build."""
        self.put("tool", _elf(EM_X86_64))

        verdict = self.run_scan()

        self.assertEqual(verdict.status, WRONG_ARCH)
        self.assertIn("x86-64", verdict.reason)

    def test_mixed_arches_with_riscv_fail(self) -> None:
        """One x86-64 output fails the port even beside riscv64 ones.

        The old scanner passed this case when any file was RISC-V.
        """
        self.put("a/tool", _elf(EM_RISCV, flags=RVC_DOUBLE))
        self.put("b/helper", _elf(EM_X86_64))

        verdict = self.run_scan()

        self.assertEqual(verdict.status, WRONG_ARCH)
        self.assertEqual([a.arch for a in verdict.foreign], ["x86-64"])

    def test_riscv32_is_not_a_riscv64_port(self) -> None:
        """ELFCLASS32 RISC-V output fails a riscv64 port."""
        self.put("tool", _elf(EM_RISCV, bits=32, flags=RVC_DOUBLE))

        verdict = self.run_scan()

        self.assertEqual(verdict.status, WRONG_ARCH)
        self.assertEqual(verdict.foreign[0].arch, "riscv32")

    def test_unknown_arch_artifacts_fail(self) -> None:
        """An unknown machine value is foreign, not riscv64."""
        self.put("tool", _elf(9999))

        verdict = self.run_scan()

        self.assertEqual(verdict.status, WRONG_ARCH)
        self.assertEqual(verdict.foreign[0].arch, "em-9999")

    def test_non_system_abi_is_a_warning_not_a_failure(self) -> None:
        """riscv64 output with another float ABI still verifies."""
        self.put("tool", _elf(EM_RISCV, flags=0x1))

        verdict = self.run_scan()

        self.assertEqual(verdict.status, VERIFIED)
        self.assertEqual(len(verdict.abi_warnings), 1)
        self.assertIn("lp64", verdict.abi_warnings[0])


class TestArchives(_RepoCase):
    """Every member of a static archive is checked, not the first."""

    def test_riscv_archive_is_verified_as_static_library(self) -> None:
        """An archive of riscv64 objects verifies the port."""
        obj = _elf(EM_RISCV, e_type=1, flags=RVC_DOUBLE)
        self.put("libz.a", _ar([("a.o", obj), ("b.o", obj)]), False)

        verdict = self.run_scan()

        self.assertEqual(verdict.status, VERIFIED)
        self.assertEqual(verdict.riscv64[0].type, "library_static")

    def test_one_foreign_member_fails_the_archive(self) -> None:
        """A single x86-64 member makes the archive mixed and foreign."""
        riscv = _elf(EM_RISCV, e_type=1, flags=RVC_DOUBLE)
        x86 = _elf(EM_X86_64, e_type=1)
        self.put(
            "libmix.a", _ar([("a.o", riscv), ("z.o", x86)]), False
        )

        verdict = self.run_scan()

        self.assertEqual(verdict.status, WRONG_ARCH)
        self.assertTrue(verdict.foreign[0].arch.startswith("mixed("))

    def test_thin_archive_and_elfless_archive_are_skipped(self) -> None:
        """Archives without readable ELF members give no evidence."""
        self.put("libthin.a", b"!<thin>\n".ljust(80, b"\0"), False)
        self.put("libdata.a", _ar([("x.txt", b"hello" * 20)]), False)

        verdict = self.run_scan()

        self.assertEqual(verdict.status, UNVERIFIED)


class TestScanIntegration(_RepoCase):
    """The real program walks the tree and asks git for committed files."""

    def _git(self, *args: str) -> None:
        subprocess.run(
            ["git", "-C", self.repo, *args],
            check=True,
            capture_output=True,
        )

    def test_committed_foreign_blob_does_not_decide(self) -> None:
        """A vendored x86-64 blob is reported, the new build decides."""
        self.put("vendor/prebuilt.so", _elf(EM_X86_64, e_type=3))
        self._git("init", "-q")
        self._git(
            "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"
        )
        self._git(
            "-c", "user.name=t", "-c", "user.email=t@t",
            "commit", "-q", "-m", "vendored blob",
        )
        self.put("build/tool", _elf(EM_RISCV, flags=RVC_DOUBLE))

        verdict = self.run_scan()

        self.assertEqual(verdict.status, VERIFIED)
        self.assertEqual(len(verdict.prebuilt_foreign), 1)
        self.assertEqual(len(verdict.source_commit or ""), 40)

    def test_node_modules_is_not_scanned(self) -> None:
        """Downloaded npm prebuilds are dependencies, not outputs."""
        self.put("node_modules/pkg/prebuilds/x.node", _elf(EM_X86_64))
        self.put("out/tool", _elf(EM_RISCV, flags=RVC_DOUBLE))

        verdict = self.run_scan()

        self.assertEqual(verdict.status, VERIFIED)
        self.assertEqual(verdict.foreign, [])

    def test_full_list_still_reads_foreign_outputs(self) -> None:
        """Outputs after the list limit are read, and foreign ones fail.

        The old program stopped at the list limit and verified the
        riscv64 outputs it had listed.
        """
        for name in ("a1", "a2", "a3"):
            self.put(f"a/{name}", _elf(EM_RISCV, flags=RVC_DOUBLE))
        self.put("z/helper", _elf(EM_X86_64))

        verdict = self.run_scan(max_entries=2)

        self.assertEqual(verdict.status, WRONG_ARCH)
        self.assertEqual([a.arch for a in verdict.foreign], ["x86-64"])
        self.assertFalse(verdict.truncated)

    def test_full_list_counts_riscv64_outputs(self) -> None:
        """riscv64 outputs after the list limit are counted, not listed."""
        for name in ("t1", "t2", "t3", "t4"):
            self.put(f"bin/{name}", _elf(EM_RISCV, flags=RVC_DOUBLE))

        verdict = self.run_scan(max_entries=2)

        self.assertEqual(verdict.status, VERIFIED)
        self.assertEqual(len(verdict.riscv64), 2)
        self.assertEqual(verdict.unlisted_riscv64, 2)
        self.assertEqual(verdict.to_dict()["counts"]["riscv64"], 4)
        self.assertIn("4 riscv64 ELF build output(s)", verdict.reason)

    def test_file_limit_makes_the_scan_unverified(self) -> None:
        """A scan that stops before it reads every candidate proves nothing."""
        for name in ("t1", "t2", "t3"):
            self.put(f"bin/{name}", _elf(EM_RISCV, flags=RVC_DOUBLE))

        verdict = self.run_scan(max_files=2)

        self.assertTrue(verdict.truncated)
        self.assertEqual(verdict.status, UNVERIFIED)
        self.assertIn("stopped after 2 candidate files", verdict.reason)

    def test_committed_riscv64_after_the_limit_is_not_counted(self) -> None:
        """Committed binaries past the list limit are not build outputs."""
        for name in ("p1", "p2", "p3"):
            self.put(f"a/{name}", _elf(EM_RISCV, flags=RVC_DOUBLE))
        self._git("init", "-q")
        self._git(
            "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"
        )
        self._git(
            "-c", "user.name=t", "-c", "user.email=t@t",
            "commit", "-q", "-m", "vendored riscv64 blobs",
        )
        self.put("b/tool", _elf(EM_RISCV, flags=RVC_DOUBLE))

        verdict = self.run_scan(max_entries=2)

        self.assertEqual(verdict.status, VERIFIED)
        self.assertEqual(verdict.riscv64, [])
        self.assertEqual(verdict.unlisted_riscv64, 1)
        self.assertEqual(
            verdict.reason, "1 riscv64 ELF build output(s) verified by header"
        )

    def test_symlinks_and_small_files_are_skipped(self) -> None:
        """Links and files shorter than an ELF header are not read."""
        self.put("tool", _elf(EM_RISCV, flags=RVC_DOUBLE))
        os.symlink(
            os.path.join(self.repo, "tool"),
            os.path.join(self.repo, "alias"),
        )
        self.put("tiny", b"\x7fELF")

        verdict = self.run_scan()

        self.assertEqual(verdict.status, VERIFIED)
        self.assertEqual(len(verdict.riscv64), 1)


class TestEvaluateScanOutside(unittest.TestCase):
    """Installed outputs outside the tree count only when it has none."""

    def _entry(self, path, machine, bits=64):
        return {
            "path": path,
            "size": 128,
            "exec": True,
            "tracked": False,
            "elf": [bits, 1, 2, machine, RVC_DOUBLE],
        }

    def test_installed_riscv_output_verifies_an_empty_tree(self) -> None:
        """Output that make install put in /usr/local/bin is evidence."""
        payload = {
            "files": [],
            "outside": [self._entry("/usr/local/bin/tool", EM_RISCV)],
            "examined": 3,
        }

        verdict = evaluate_scan(payload, "/workspace/repos/x", ["tool"])

        self.assertEqual(verdict.status, VERIFIED)
        self.assertIn("outside the repository", verdict.reason)

    def test_missing_expected_names_are_listed(self) -> None:
        """Expected artifacts that never appeared are reported."""
        payload = {"files": [], "outside": [], "examined": 0}

        verdict = evaluate_scan(
            payload, "/workspace/repos/x", ["tool", "../evil", "tool"]
        )

        self.assertEqual(verdict.status, UNVERIFIED)
        self.assertEqual(verdict.expected_missing, ["tool"])

    def test_summary_counts_by_list(self) -> None:
        """to_dict keeps full counts even when lists are cut."""
        payload = {
            "files": [
                {
                    "path": f"bin/t{i}",
                    "size": 1,
                    "exec": True,
                    "tracked": False,
                    "elf": [64, 1, 2, EM_RISCV, RVC_DOUBLE],
                }
                for i in range(5)
            ],
            "outside": [],
            "examined": 5,
        }

        summary = evaluate_scan(payload, "/r").to_dict(max_items=2)

        self.assertEqual(summary["counts"]["riscv64"], 5)
        self.assertEqual(len(summary["riscv64"]), 2)
        self.assertTrue(summary["verified"])
        self.assertFalse(summary["truncated"])

    def test_truncated_scan_is_unverified(self) -> None:
        """riscv64 outputs do not verify a scan that stopped early."""
        payload = {
            "files": [self._entry("bin/tool", EM_RISCV)],
            "outside": [self._entry("/usr/local/bin/tool", EM_RISCV)],
            "examined": 20000,
            "truncated": True,
        }

        verdict = evaluate_scan(payload, "/r", ["tool"])

        self.assertEqual(verdict.status, UNVERIFIED)
        self.assertFalse(verdict.to_dict()["verified"])
        self.assertIn("stopped after 20000 candidate files", verdict.reason)

    def test_truncated_scan_with_foreign_output_is_wrong_arch(self) -> None:
        """A foreign output that was read fails the port anyway."""
        payload = {
            "files": [
                self._entry("bin/tool", EM_RISCV),
                self._entry("bin/helper", EM_X86_64),
            ],
            "outside": [],
            "examined": 20000,
            "truncated": True,
        }

        verdict = evaluate_scan(payload, "/r")

        self.assertEqual(verdict.status, WRONG_ARCH)


class TestArtifactScannerRun(unittest.TestCase):
    """ArtifactScanner.scan never reports success when it cannot run."""

    def test_missing_python_is_unverified(self) -> None:
        """Exit 127 without output becomes a clear scan error."""
        failed = CommandResult("python3", 127, "", "not found", 0.1)
        with mock.patch.object(
            artifact_scanner, "execute_command", return_value=failed
        ):
            verdict = ArtifactScanner("/workspace/repos/x").scan(["tool"])

        self.assertEqual(verdict.status, UNVERIFIED)
        self.assertIn("python3 is not installed", verdict.scan_error)
        self.assertEqual(verdict.expected_missing, ["tool"])

    def test_scan_runs_unvalidated_python_in_the_sandbox(self) -> None:
        """The program is one python3 -c call with quoted arguments."""
        ok = CommandResult(
            "python3",
            0,
            'ATESOR_SCAN_JSON {"files": [], "outside": [], "examined": 0}',
            "",
            0.1,
        )
        with mock.patch.object(
            artifact_scanner, "execute_command", return_value=ok
        ) as execute:
            ArtifactScanner("/workspace/repos/x").scan()

        command = execute.call_args.args[0]
        self.assertEqual(command[:2], ["python3", "-c"])
        self.assertEqual(command[3], "/workspace/repos/x")
        self.assertFalse(execute.call_args.kwargs["validate"])


if __name__ == "__main__":
    unittest.main()
