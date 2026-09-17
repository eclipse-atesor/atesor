"""Tests for src/evidence.py — repo evidence bundling for LLM prompts."""

import os
import tempfile
import unittest
from unittest import mock

from src.evidence import collect_build_evidence, error_context_excerpts


class TestCollectBuildEvidence(unittest.TestCase):
    """Tests for collect_build_evidence."""

    def setUp(self) -> None:
        """Create a fake repo with build files and docs."""
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        with open(os.path.join(self.repo, "CMakeLists.txt"), "w") as f:
            f.write(
                "cmake_minimum_required(VERSION 3.10)\n"
                "project(demo C)\n"
                "find_package(ZLIB REQUIRED)\n"
            )
        with open(os.path.join(self.repo, "README.md"), "w") as f:
            f.write("# Demo\n\nBuild with cmake and make.\n")
        os.makedirs(os.path.join(self.repo, "src"))

    def tearDown(self) -> None:
        """Remove the fake repo."""
        self.tmp.cleanup()

    def test_bundle_contains_real_file_contents(self) -> None:
        """The bundle must contain the ACTUAL build file contents."""
        bundle = collect_build_evidence(self.repo)
        self.assertIn("find_package(ZLIB REQUIRED)", bundle)
        self.assertIn("### CMakeLists.txt", bundle)
        self.assertIn("Build with cmake and make.", bundle)

    def test_bundle_contains_top_level_listing(self) -> None:
        """The bundle lists top-level entries, marking directories."""
        bundle = collect_build_evidence(self.repo)
        self.assertIn("### Top-level files", bundle)
        self.assertIn("src/", bundle)

    def test_missing_files_are_skipped(self) -> None:
        """Absent build files produce no empty sections."""
        bundle = collect_build_evidence(self.repo)
        self.assertNotIn("### go.mod", bundle)
        self.assertNotIn("### Cargo.toml", bundle)

    def test_large_file_is_truncated(self) -> None:
        """Oversized files are cut to the per-file cap."""
        with open(os.path.join(self.repo, "Makefile"), "w") as f:
            f.write("x" * 10000)
        bundle = collect_build_evidence(self.repo)
        self.assertIn("[... truncated ...]", bundle)
        self.assertLess(len(bundle), 20000)

    def test_unreadable_repo_degrades_gracefully(self) -> None:
        """A nonexistent repo yields a bundle, not an exception."""
        bundle = collect_build_evidence("/nonexistent/path/xyz")
        self.assertIn("repository not readable", bundle)


class TestErrorContextExcerpts(unittest.TestCase):
    """Tests for error_context_excerpts."""

    def setUp(self) -> None:
        """Create a fake repo with a source file."""
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        os.makedirs(os.path.join(self.repo, "src"))
        with open(os.path.join(self.repo, "src", "main.c"), "w") as f:
            f.write("\n".join(f"line{i}" for i in range(1, 51)))

    def tearDown(self) -> None:
        """Remove the fake repo."""
        self.tmp.cleanup()

    def test_extracts_context_around_error_line(self) -> None:
        """A file:line reference yields numbered source context."""
        error = (
            "src/main.c:20:5: error: unknown type name 'simd_t'\n"
            "make: *** [main.o] Error 1"
        )
        out = error_context_excerpts(error, self.repo)
        self.assertIn("src/main.c (around line 20)", out)
        self.assertIn("line20", out)
        self.assertIn("line10", out)  # ±12 lines of context
        self.assertNotIn("line50", out)

    def test_ignores_files_outside_repo(self) -> None:
        """Path traversal references are never followed."""
        error = "../../etc/passwd:1: error"
        self.assertEqual(error_context_excerpts(error, self.repo), "")

    def test_ignores_unreadable_references(self) -> None:
        """References to missing files are skipped silently."""
        error = "src/ghost.c:5: error: boom"
        self.assertEqual(error_context_excerpts(error, self.repo), "")

    def test_empty_error_returns_empty(self) -> None:
        """No error text, no excerpts."""
        self.assertEqual(error_context_excerpts("", self.repo), "")

    def test_deduplicates_repeated_references(self) -> None:
        """Multiple refs to one file yield a single excerpt."""
        error = "src/main.c:5: error a\nsrc/main.c:6: error b"
        out = error_context_excerpts(error, self.repo)
        self.assertEqual(out.count("### src/main.c"), 1)


class TestEvidenceEdgeCases(unittest.TestCase):
    """Cover caps, budgets, and containment branches."""

    def setUp(self) -> None:
        """Create an empty temp repo."""
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name

    def tearDown(self) -> None:
        """Remove the temp repo."""
        self.tmp.cleanup()

    def test_unreadable_build_file_is_skipped(self) -> None:
        """Test unreadable build file is skipped."""
        path = os.path.join(self.repo, "Makefile")
        with open(path, "w") as f:
            f.write("all:\n\ttrue\n")
        real_open = open

        def failing_open(fname, *args, **kwargs):
            """Failing open."""
            if fname == path:
                raise OSError("permission denied")
            return real_open(fname, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=failing_open):
            out = collect_build_evidence(self.repo)
        self.assertNotIn("### Makefile", out)

    def test_whitespace_only_build_file_is_skipped(self) -> None:
        """Test whitespace only build file is skipped."""
        with open(os.path.join(self.repo, "Makefile"), "w") as f:
            f.write("   \n\t\n")
        out = collect_build_evidence(self.repo)
        self.assertNotIn("### Makefile", out)

    def test_top_level_listing_truncates_long_dirs(self) -> None:
        """Test top level listing truncates long dirs."""
        for i in range(70):
            with open(os.path.join(self.repo, f"f{i:03d}.txt"), "w") as f:
                f.write("x")
        out = collect_build_evidence(self.repo)
        self.assertIn("more entries ...]", out)

    def test_budget_exhaustion_stops_collection(self) -> None:
        """Test budget exhaustion stops collection."""
        big = "x" * 3000 + "\n"
        for name in (
            "go.mod",
            "Cargo.toml",
            "CMakeLists.txt",
            "configure.ac",
            "configure.in",
            "meson.build",
            "Makefile.am",
            "Makefile",
        ):
            with open(os.path.join(self.repo, name), "w") as f:
                f.write(big)
        with open(os.path.join(self.repo, "README.md"), "w") as f:
            f.write("# readme\n")
        out = collect_build_evidence(self.repo)
        # Budget (14000 chars) is spent before the doc loop runs.
        self.assertNotIn("README.md (excerpt)", out)
        self.assertNotIn("### Makefile\n", out)

    def test_error_refs_outside_repo_are_ignored(self) -> None:
        """Test error refs outside repo are ignored."""
        error = "../secrets/evil.c:3: error: nope"
        self.assertEqual(error_context_excerpts(error, self.repo), "")

    def test_error_excerpts_capped_at_three_files(self) -> None:
        """Test error excerpts capped at three files."""
        refs = []
        for name in ("a.c", "b.c", "c.c", "d.c"):
            with open(os.path.join(self.repo, name), "w") as f:
                f.write("int main(void) { return 0; }\n")
            refs.append(f"{name}:1: error: boom")
        out = error_context_excerpts("\n".join(refs), self.repo)
        self.assertEqual(out.count("### "), 3)
        self.assertNotIn("d.c", out)


if __name__ == "__main__":
    unittest.main()
