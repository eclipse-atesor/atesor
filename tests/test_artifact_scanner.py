"""Tests for src/artifact_scanner.py.

Covers architecture detection and build verification.
"""

import unittest
from unittest import mock

from src.artifact_scanner import ArtifactScanner
from src.state import CommandResult


def _result(stdout: str = "", exit_code: int = 0) -> CommandResult:
    """Create a CommandResult for mocked shell execution."""
    return CommandResult(
        stdout=stdout,
        stderr="",
        command="",
        exit_code=exit_code,
        duration_seconds=0.0,
    )


class TestDetectArchitecture(unittest.TestCase):
    """_detect_architecture must classify `file` output correctly."""

    CASES = [
        (
            "ELF 64-bit LSB pie executable, UCB RISC-V, RVC, double-float ABI",
            "RISC-V",
        ),
        ("ELF 64-bit LSB executable, riscv64", "RISC-V"),
        ("ELF 64-bit LSB executable, x86-64, version 1 (SYSV)", "x64"),
        ("ELF 64-bit LSB executable, x86_64", "x64"),
        ("ELF 64-bit LSB executable, ARM aarch64", "ARM64"),
        ("ELF 64-bit LSB executable, arm64", "ARM64"),
        ("ELF 32-bit LSB executable, ARM, EABI5", "ARM32"),
        ("ELF 32-bit LSB executable, Intel 80386", "x86"),
        ("ASCII text", None),
        ("", None),
    ]

    def test_detect_each_architecture(self) -> None:
        """Test detect each architecture."""
        sc = ArtifactScanner("workspace/build")
        for info, expected in self.CASES:
            with self.subTest(info=info):
                self.assertEqual(sc._detect_architecture(info), expected)

    def test_case_insensitive(self) -> None:
        """Test case insensitive."""
        sc = ArtifactScanner("workspace/build")
        self.assertEqual(sc._detect_architecture("RISC-V 64-bit"), "RISC-V")
        self.assertEqual(sc._detect_architecture("X86_64 binary"), "x64")


class TestVerifyBuildSuccess(unittest.TestCase):
    """Tests for VerifyBuildSuccess."""

    def _scanner_with(self, artifacts):
        """Scanner with."""
        sc = ArtifactScanner("workspace/build")
        sc.artifacts = artifacts
        return sc

    def test_no_artifacts_fails(self) -> None:
        """Test no artifacts fails."""
        sc = self._scanner_with([])
        ok, msg = sc.verify_build_success()
        self.assertFalse(ok)
        self.assertIn("No build artifacts", msg)

    def test_riscv_artifacts_succeed(self) -> None:
        """Test riscv artifacts succeed."""
        sc = self._scanner_with(
            [
                {
                    "filepath": "/build/a",
                    "type": "binary",
                    "architecture": "RISC-V",
                }
            ]
        )
        ok, msg = sc.verify_build_success()
        self.assertTrue(ok)
        self.assertIn("RISC-V", msg)

    def test_x64_only_artifacts_fail(self) -> None:
        """Test x64 only artifacts fail."""
        sc = self._scanner_with(
            [{"filepath": "/build/a", "type": "binary", "architecture": "x64"}]
        )
        ok, msg = sc.verify_build_success()
        self.assertFalse(ok)
        self.assertIn("not for RISC-V", msg)
        self.assertIn("x64", msg)

    def test_mixed_arches_with_riscv_succeed(self) -> None:
        # If even one RISC-V artifact is found, build counts as success
        """Test mixed arches with riscv succeed."""
        sc = self._scanner_with(
            [
                {
                    "filepath": "/build/a",
                    "type": "binary",
                    "architecture": "RISC-V",
                },
                {
                    "filepath": "/build/b",
                    "type": "binary",
                    "architecture": "x64",
                },
            ]
        )
        ok, _ = sc.verify_build_success()
        self.assertTrue(ok)

    def test_unknown_arch_artifacts_fail(self) -> None:
        """Test unknown arch artifacts fail."""
        sc = self._scanner_with(
            [{"filepath": "/build/a", "type": "binary", "architecture": ""}]
        )
        ok, msg = sc.verify_build_success()
        self.assertFalse(ok)
        self.assertIn("could not detect architecture", msg)


class TestGetSummary(unittest.TestCase):
    """Tests for GetSummary."""

    def test_summary_counts_by_type_and_arch(self) -> None:
        """Test summary counts by type and arch."""
        sc = ArtifactScanner("workspace/build")
        sc.artifacts = [
            {"type": "binary", "architecture": "RISC-V"},
            {"type": "binary", "architecture": "RISC-V"},
            {"type": "library_static", "architecture": "RISC-V"},
            {"type": "library_shared", "architecture": "x64"},
        ]
        s = sc.get_summary()
        self.assertEqual(s["total_artifacts"], 4)
        self.assertEqual(
            s["by_type"],
            {"binary": 2, "library_static": 1, "library_shared": 1},
        )
        self.assertEqual(
            s["by_arch" if "by_arch" in s else "by_architecture"],
            {"RISC-V": 3, "x64": 1},
        )
        self.assertTrue(s["has_riscv"])

    def test_empty_summary(self) -> None:
        """Test empty summary."""
        sc = ArtifactScanner("workspace/build")
        s = sc.get_summary()
        self.assertEqual(s["total_artifacts"], 0)
        self.assertFalse(s["has_riscv"])


class TestScanIntegration(unittest.TestCase):
    """Verify the scan() pipeline calls the right helpers."""

    def test_scan_calls_find_and_file(self) -> None:
        """Test scan calls find and file."""
        sc = ArtifactScanner("workspace/build", cwd="workspace/build")

        # Build a sequence of fake responses for the find/file/stat chain
        def fake_exec(cmd, cwd=None, **kw):
            """Fake exec."""
            if cmd.startswith("find") and "executable" in cmd:
                return _result("workspace/build/foo\n")
            if cmd.startswith("find") and "*.a" in cmd:
                return _result("")
            if cmd.startswith("find") and "*.so*" in cmd:
                return _result("")
            if cmd.startswith("file "):
                return _result(
                    "workspace/build/foo: "
                    "ELF 64-bit LSB executable, UCB RISC-V"
                )
            if "stat" in cmd:
                return _result("12345")
            return _result("")

        with mock.patch(
            "src.artifact_scanner.execute_command", side_effect=fake_exec
        ):
            artifacts = sc.scan()

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["type"], "binary")
        self.assertEqual(artifacts[0]["architecture"], "RISC-V")
        self.assertEqual(artifacts[0]["size_bytes"], 12345)

    def test_scan_checks_static_and_shared_libraries(self) -> None:
        """Scan checks static archives and shared libraries."""
        sc = ArtifactScanner("workspace/build", cwd="workspace/build")

        def fake_exec(cmd, cwd=None, **kw):
            """Return deterministic find, file, archive, and stat output."""
            if cmd.startswith("find") and "executable" in cmd:
                return _result("")
            if cmd.startswith("find") and "*.a" in cmd:
                return _result("workspace/build/libfoo.a\n")
            if cmd.startswith("find") and "*.so*" in cmd:
                return _result("workspace/build/libfoo.so\n")
            if cmd.startswith("file workspace/build/libfoo.a"):
                return _result("workspace/build/libfoo.a: current ar archive")
            if cmd.startswith("file workspace/build/libfoo.so"):
                return _result(
                    "workspace/build/libfoo.so: "
                    "ELF 64-bit LSB shared object, x86-64"
                )
            if cmd.startswith("cd "):
                return _result(
                    "foo.o: ELF 64-bit LSB relocatable, UCB RISC-V"
                )
            if "stat" in cmd:
                return _result("77")
            return _result("")

        with mock.patch(
            "src.artifact_scanner.execute_command", side_effect=fake_exec
        ):
            artifacts = sc.scan()

        self.assertEqual([a["type"] for a in artifacts],
                         ["library_static", "library_shared"])
        self.assertEqual(artifacts[0]["architecture"], "RISC-V")
        self.assertEqual(artifacts[1]["architecture"], "x64")

    def test_scan_handles_no_artifacts(self) -> None:
        """Test scan handles no artifacts."""
        sc = ArtifactScanner("workspace/build")
        with mock.patch(
            "src.artifact_scanner.execute_command", return_value=_result("")
        ):
            artifacts = sc.scan()
        self.assertEqual(artifacts, [])

    def test_check_artifact_skips_failed_file_command(self) -> None:
        """Failed file command leaves artifacts unchanged."""
        sc = ArtifactScanner("workspace/build")

        with mock.patch(
            "src.artifact_scanner.execute_command",
            return_value=_result(exit_code=1),
        ):
            sc._check_artifact("workspace/build/app", "binary")

        self.assertEqual(sc.artifacts, [])

    def test_check_artifact_records_unknown_arch_and_bad_size(self) -> None:
        """Unknown architecture and invalid stat output are tolerated."""
        sc = ArtifactScanner("workspace/build")

        def fake_exec(cmd, cwd=None, **kw):
            """Return text file output then an invalid file size."""
            if cmd.startswith("file "):
                return _result("workspace/build/app: ASCII text")
            return _result("not-a-number")

        with mock.patch(
            "src.artifact_scanner.execute_command", side_effect=fake_exec
        ):
            sc._check_artifact("workspace/build/app", "binary")

        self.assertIsNone(sc.artifacts[0]["architecture"])
        self.assertEqual(sc.artifacts[0]["size_bytes"], 0)

    def test_archive_architecture_returns_none_without_output(self) -> None:
        """Archive detection returns None when extraction gives no output."""
        sc = ArtifactScanner("workspace/build")

        with mock.patch(
            "src.artifact_scanner.execute_command",
            return_value=_result(""),
        ):
            self.assertIsNone(
                sc._get_archive_architecture("workspace/build/libfoo.a")
            )

    def test_archive_architecture_returns_none_on_failure(self) -> None:
        """Archive detection returns None when extraction fails."""
        sc = ArtifactScanner("workspace/build")

        with mock.patch(
            "src.artifact_scanner.execute_command",
            return_value=_result(exit_code=1),
        ):
            self.assertIsNone(
                sc._get_archive_architecture("workspace/build/libfoo.a")
            )

    def test_archive_architecture_returns_none_on_exception(self) -> None:
        """Archive detection catches execute_command exceptions."""
        sc = ArtifactScanner("workspace/build")

        with mock.patch(
            "src.artifact_scanner.execute_command",
            side_effect=RuntimeError("boom"),
        ):
            self.assertIsNone(
                sc._get_archive_architecture("workspace/build/libfoo.a")
            )

    def test_file_size_returns_zero_on_failed_stat(self) -> None:
        """File-size helper returns zero when stat fails."""
        sc = ArtifactScanner("workspace/build")

        with mock.patch(
            "src.artifact_scanner.execute_command",
            return_value=_result(exit_code=1),
        ):
            self.assertEqual(sc._get_file_size("workspace/build/app"), 0)


if __name__ == "__main__":
    unittest.main()
