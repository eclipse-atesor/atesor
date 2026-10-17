"""Tests for src/evidence.py — repo evidence bundling for LLM prompts."""

import os
import tempfile
import unittest
from unittest import mock

from src.evidence import (
    collect_build_evidence,
    cut_text,
    error_context_excerpts,
    failure_digest,
    first_error_line,
    has_error_line,
    top_level_listing,
)


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
        bundle = collect_build_evidence(self.repo)
        self.assertNotIn("### go.mod", bundle)
        self.assertNotIn("### Cargo.toml", bundle)

    def test_large_file_is_truncated(self) -> None:
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
        error = "src/ghost.c:5: error: boom"
        self.assertEqual(error_context_excerpts(error, self.repo), "")

    def test_empty_error_returns_empty(self) -> None:
        self.assertEqual(error_context_excerpts("", self.repo), "")

    def test_deduplicates_repeated_references(self) -> None:
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
        path = os.path.join(self.repo, "Makefile")
        with open(path, "w") as f:
            f.write("all:\n\ttrue\n")
        real_open = open

        def failing_open(fname, *args, **kwargs):
            if fname == path:
                raise OSError("permission denied")
            return real_open(fname, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=failing_open):
            out = collect_build_evidence(self.repo)
        self.assertNotIn("### Makefile", out)

    def test_whitespace_only_build_file_is_skipped(self) -> None:
        with open(os.path.join(self.repo, "Makefile"), "w") as f:
            f.write("   \n\t\n")
        out = collect_build_evidence(self.repo)
        self.assertNotIn("### Makefile", out)

    def test_top_level_listing_truncates_long_dirs(self) -> None:
        for i in range(70):
            with open(os.path.join(self.repo, f"f{i:03d}.txt"), "w") as f:
                f.write("x")
        out = collect_build_evidence(self.repo)
        self.assertIn("more entries ...]", out)

    def test_budget_exhaustion_stops_collection(self) -> None:
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
        error = "../secrets/evil.c:3: error: nope"
        self.assertEqual(error_context_excerpts(error, self.repo), "")

    def test_error_excerpts_capped_at_three_files(self) -> None:
        refs = []
        for name in ("a.c", "b.c", "c.c", "d.c"):
            with open(os.path.join(self.repo, name), "w") as f:
                f.write("int main(void) { return 0; }\n")
            refs.append(f"{name}:1: error: boom")
        out = error_context_excerpts("\n".join(refs), self.repo)
        self.assertEqual(out.count("### "), 3)
        self.assertNotIn("d.c", out)


class TestFailureEvidence(unittest.TestCase):
    """The fixer sees the decisive error lines, not the log head."""

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

    def test_digest_keeps_earlier_error_lines_and_the_tail(self) -> None:
        noise = "\n".join(
            f"compiling unit {i} of the tree" for i in range(400)
        )
        output = (
            "Build command failed: make (exit 2) - warning: x\n"
            "src/main.c:20:5: error: unknown type name 'simd_t'\n"
            + noise
            + "\nmake: *** [main.o] Error 1"
        )
        key, tail = failure_digest(output, tail_chars=300)
        self.assertEqual(
            key, "src/main.c:20:5: error: unknown type name 'simd_t'"
        )
        self.assertTrue(tail.endswith("make: *** [main.o] Error 1"))
        self.assertTrue(tail.startswith("[... "))
        self.assertIn("earlier characters cut ...]\ncompiling unit", tail)

    def test_short_output_is_only_the_tail_without_the_prefix(self) -> None:
        output = (
            "Build command failed: ./configure (exit 1) - "
            "configure: error: cannot find sources\nmore"
        )
        key, tail = failure_digest(output)
        self.assertEqual(key, "")
        self.assertEqual(tail, "configure: error: cannot find sources\nmore")
        self.assertEqual(failure_digest("no prefix here")[1], "no prefix here")

    def test_first_error_line_skips_warnings_and_make_recipe_lines(
        self,
    ) -> None:
        message = (
            "Build command failed: make gawk (exit 2) - "
            "awkgram.y: warning: 36 shift/reduce conflicts\n"
            "make: *** No rule to make target 'support/libsupport.a'.  Stop.\n"
            "make: *** [Makefile:900: all] Error 2"
        )
        self.assertEqual(
            first_error_line(message),
            "make: *** No rule to make target 'support/libsupport.a'.  Stop.",
        )
        self.assertEqual(
            first_error_line("make[1]: *** [Makefile:9: x.o] Error 1"), ""
        )
        autoconf = (
            "configure.ac:244: warning: macro 'AM_GNU_GETTEXT' not found\n"
            "configure.ac:15: error: autoconf-archive is required"
        )
        self.assertEqual(
            first_error_line(autoconf),
            "configure.ac:15: error: autoconf-archive is required",
        )

    def test_hyphenated_names_are_not_error_lines(self) -> None:
        self.assertFalse(has_error_line("Unpacking node-error-ex (1.3.2-4)"))
        self.assertFalse(has_error_line("gcc -Werror=format -c a.c"))
        self.assertTrue(has_error_line("configure: error: no C compiler"))
        self.assertTrue(has_error_line("Configuration failed."))

    def test_digest_bounds_many_key_lines(self) -> None:
        errors = "\n".join(f"a{i}.c:1: error: e{i}" for i in range(40))
        output = errors + "\n" + "\n".join("plain" for _ in range(600))
        key, _ = failure_digest(output)
        lines = key.splitlines()
        self.assertEqual(len(lines), 21)
        self.assertIn("20 more matching lines", lines[12])
        self.assertEqual(lines[-1], "a39.c:1: error: e39")

    def test_absolute_repo_paths_get_excerpts(self) -> None:
        error = (
            f"In file included from {self.repo}/src/main.c:3:\n"
            f"{self.repo}/src/main.c:20:5: error: boom"
        )
        out = error_context_excerpts(error, self.repo)
        self.assertIn("src/main.c (around line 20)", out)

    def test_error_line_reference_comes_first(self) -> None:
        with open(os.path.join(self.repo, "src", "other.c"), "w") as f:
            f.write("x\n" * 30)
        error = "src/other.c:2: warning: unused\n" "src/main.c:20: error: boom"
        out = error_context_excerpts(error, self.repo)
        self.assertLess(out.index("src/main.c"), out.index("src/other.c"))

    def test_out_of_tree_build_paths_resolve(self) -> None:
        out = error_context_excerpts(
            "../src/main.c:20: error: boom", self.repo
        )
        self.assertIn("src/main.c (around line 20)", out)

    def test_symlink_out_of_the_repo_is_not_read(self) -> None:
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        with open(os.path.join(outside.name, "secret.c"), "w") as f:
            f.write("secret\n" * 30)
        os.symlink(outside.name, os.path.join(self.repo, "link"))
        out = error_context_excerpts("link/secret.c:3: error: x", self.repo)
        self.assertEqual(out, "")

    def test_top_level_listing(self) -> None:
        self.assertIn("src/", top_level_listing(self.repo))


class TestCutText(unittest.TestCase):
    """Cuts fall on line or word boundaries and say what was left out."""

    def test_cut_on_a_line_boundary(self) -> None:
        text = "\n".join(f"line {i:03d} of the file" for i in range(100))
        out = cut_text(text, 300)
        kept, marker = out.rsplit("\n", 1)
        self.assertTrue(kept.endswith("of the file"))
        self.assertIn("[... truncated ...]", marker)
        self.assertIn("more lines left out", marker)
        self.assertLessEqual(len(kept), 300)

    def test_cut_on_a_word_boundary_without_newlines(self) -> None:
        out = cut_text("word " * 100, 52)
        self.assertTrue(out.split("\n")[0].endswith("word"))
        self.assertIn("more characters left out", out)

    def test_short_text_is_unchanged(self) -> None:
        self.assertEqual(cut_text("short", 100), "short")


if __name__ == "__main__":
    unittest.main()
