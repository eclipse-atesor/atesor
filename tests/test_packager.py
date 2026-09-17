"""Tests for src.packager."""

import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from src.packager import _add_repo_tree, _safe_zip_path, package_build


class TestPackageBuild(unittest.TestCase):
    """Cover the happy path and key edge cases of package_build()."""

    def setUp(self) -> None:
        """Create temporary repo, recipe, and package directories."""
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.repo_name = "amass"
        self.repo_path = os.path.join(self.root, "repos", self.repo_name)
        self.recipe_path = os.path.join(
            self.root, "output", f"{self.repo_name}_recipe.md"
        )
        self.packages_dir = os.path.join(self.root, "packages")
        os.makedirs(self.repo_path)
        os.makedirs(os.path.dirname(self.recipe_path))
        os.makedirs(self.packages_dir)
        with open(self.recipe_path, "w") as f:
            f.write("# Build recipe\n")
        with open(os.path.join(self.repo_path, "main.go"), "w") as f:
            f.write("package main\n")
        os.makedirs(os.path.join(self.repo_path, "internal"))
        with open(os.path.join(self.repo_path, "internal", "x.go"), "w") as f:
            f.write("package internal\n")

    def tearDown(self) -> None:
        """Clean up the temporary package workspace."""
        self.tmp.cleanup()

    def _build(self, **overrides) -> str:
        kwargs = dict(
            repo_name=self.repo_name,
            repo_path=self.repo_path,
            recipe_path=self.recipe_path,
            platform_name="debian",
            packages_dir=self.packages_dir,
            repo_url="https://github.com/owasp-amass/amass",
        )
        kwargs.update(overrides)
        return package_build(**kwargs)

    def test_filename_format(self) -> None:
        """Filename uses repo, timestamp, platform, and zip suffix."""
        zp = self._build()
        base = os.path.basename(zp)
        self.assertTrue(base.startswith(f"{self.repo_name}-"))
        self.assertTrue(base.endswith("-debian.zip"))
        # repo-YYYYMMDD-HHMMSS-platform.zip = 4 segments split by '-'.
        self.assertEqual(len(base[: -len(".zip")].split("-")), 4)

    def test_layout_recipe_at_root_and_repo_subtree(self) -> None:
        """Zip contains root recipe, manifest, and repo files."""
        zp = self._build()
        with zipfile.ZipFile(zp) as zf:
            names = set(zf.namelist())
        self.assertIn("build_recipe.md", names)
        self.assertIn("manifest.json", names)
        self.assertIn(f"{self.repo_name}/main.go", names)
        self.assertIn(f"{self.repo_name}/internal/x.go", names)

    def test_manifest_contents(self) -> None:
        """Manifest records repo, platform, recipe, source, and URL."""
        zp = self._build()
        with zipfile.ZipFile(zp) as zf:
            with zf.open("manifest.json") as f:
                m = json.load(f)
        self.assertEqual(m["repo_name"], self.repo_name)
        self.assertEqual(m["platform"], "debian")
        self.assertEqual(m["recipe_filename_in_zip"], "build_recipe.md")
        self.assertEqual(m["source_root_in_zip"], self.repo_name)
        self.assertEqual(m["repo_url"], "https://github.com/owasp-amass/amass")

    def test_dot_git_excluded(self) -> None:
        """Package excludes .git entries from the zip archive."""
        git_dir = os.path.join(self.repo_path, ".git")
        os.makedirs(git_dir)
        with open(os.path.join(git_dir, "HEAD"), "w") as f:
            f.write("ref: refs/heads/main\n")
        zp = self._build()
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
        self.assertFalse(
            any(".git/" in n or n.endswith("/.git") for n in names),
            msg=f"unexpected .git entries: {names}",
        )

    def test_symlinks_skipped_for_security(self) -> None:
        """Package skips symlinks while keeping real repo files."""
        # A symlink pointing OUTSIDE the repo must not be packaged.
        outside = os.path.join(self.root, "secret.txt")
        with open(outside, "w") as f:
            f.write("SECRET\n")
        os.symlink(outside, os.path.join(self.repo_path, "leak.txt"))
        os.symlink("/nonexistent", os.path.join(self.repo_path, "dangling"))

        zp = self._build()
        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
        self.assertNotIn(f"{self.repo_name}/leak.txt", names)
        self.assertNotIn(f"{self.repo_name}/dangling", names)
        # Real files still made it in.
        self.assertIn(f"{self.repo_name}/main.go", names)

    def test_missing_repo_dir_raises(self) -> None:
        """Missing repository directory raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self._build(
                repo_path=os.path.join(self.root, "does", "not", "exist")
            )

    def test_missing_recipe_raises(self) -> None:
        """Missing recipe file raises FileNotFoundError."""
        os.remove(self.recipe_path)
        with self.assertRaises(FileNotFoundError):
            self._build()

    def test_collision_appends_suffix(self) -> None:
        """Second package path gets a suffix instead of clobbering."""
        zp1 = self._build()
        # Same name, same second — packager must avoid clobbering.
        # Create a sentinel at the would-be path.
        sentinel = zp1
        with open(sentinel, "ab"):
            pass
        zp2 = self._build()
        self.assertNotEqual(zp1, zp2)
        self.assertTrue(os.path.isfile(zp1))
        self.assertTrue(os.path.isfile(zp2))

    def test_logs_included_at_zip_root(self) -> None:
        """Existing agent and batch logs are added at the zip root."""
        agent_log = os.path.join(self.root, f"agent_{self.repo_name}.log")
        batch_log = os.path.join(self.root, f"{self.repo_name}.log")
        with open(agent_log, "w") as f:
            f.write("AGENT LOG CONTENT\n")
        with open(batch_log, "w") as f:
            f.write("BATCH LOG CONTENT\n")

        zp = self._build(agent_log_path=agent_log, batch_log_path=batch_log)
        with zipfile.ZipFile(zp) as zf:
            names = set(zf.namelist())
            self.assertIn(f"agent_{self.repo_name}.log", names)
            self.assertIn(f"{self.repo_name}.log", names)
            self.assertEqual(
                zf.read(f"agent_{self.repo_name}.log").decode(),
                "AGENT LOG CONTENT\n",
            )
            self.assertEqual(
                zf.read(f"{self.repo_name}.log").decode(),
                "BATCH LOG CONTENT\n",
            )
            manifest = json.loads(zf.read("manifest.json").decode())
        self.assertEqual(
            sorted(manifest["logs_in_zip"]),
            sorted([f"{self.repo_name}.log", f"agent_{self.repo_name}.log"]),
        )

    def test_missing_logs_are_skipped_not_fatal(self) -> None:
        """Missing log paths add no zip entries and do not fail."""
        # Caller passes paths but neither file exists → no zip entries,
        # no exception, manifest reflects absence.
        zp = self._build(
            agent_log_path=os.path.join(self.root, "nope-agent.log"),
            batch_log_path=os.path.join(self.root, "nope-batch.log"),
        )
        with zipfile.ZipFile(zp) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("manifest.json").decode())
        self.assertNotIn(f"agent_{self.repo_name}.log", names)
        self.assertNotIn(f"{self.repo_name}.log", names)
        self.assertEqual(manifest["logs_in_zip"], [])

    def test_no_log_args_means_no_log_entries(self) -> None:
        """Omitted log paths produce no log entries in the archive."""
        zp = self._build()
        with zipfile.ZipFile(zp) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("manifest.json").decode())
        # No *.log entries at all when caller didn't pass paths.
        self.assertFalse(
            any(n.endswith(".log") for n in names),
            msg=f"unexpected .log entries: {names}",
        )
        self.assertEqual(manifest["logs_in_zip"], [])


class TestPackagerEdgeCases(unittest.TestCase):
    """Cover collision numbering, symlink skips, unreadable files."""

    def setUp(self) -> None:
        """Create temp dirs."""
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self) -> None:
        """Remove temp dirs."""
        self.tmp.cleanup()

    def test_safe_zip_path_increments_past_collisions(self) -> None:
        """Test safe zip path increments past collisions."""
        for name in ("pkg.zip", "pkg.1.zip"):
            with open(os.path.join(self.root, name), "w") as f:
                f.write("x")
        self.assertEqual(
            _safe_zip_path(self.root, "pkg.zip"),
            os.path.join(self.root, "pkg.2.zip"),
        )

    def test_symlinked_directory_is_skipped(self) -> None:
        """Test symlinked directory is skipped."""
        repo = os.path.join(self.root, "repo")
        outside = os.path.join(self.root, "outside")
        os.makedirs(repo)
        os.makedirs(outside)
        with open(os.path.join(outside, "secret.txt"), "w") as f:
            f.write("secret")
        with open(os.path.join(repo, "main.c"), "w") as f:
            f.write("int main;")
        os.symlink(outside, os.path.join(repo, "link"))
        zip_path = os.path.join(self.root, "out.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            added, skipped = _add_repo_tree(zf, repo, "r")
        self.assertEqual(added, 1)
        self.assertEqual(skipped, 1)
        with zipfile.ZipFile(zip_path) as zf:
            self.assertNotIn("r/link/secret.txt", zf.namelist())

    def test_unreadable_file_is_skipped(self) -> None:
        """Test unreadable file is skipped."""
        repo = os.path.join(self.root, "repo")
        os.makedirs(repo)
        with open(os.path.join(repo, "main.c"), "w") as f:
            f.write("int main;")
        zf = mock.MagicMock()
        zf.write.side_effect = OSError("unreadable")
        added, skipped = _add_repo_tree(zf, repo, "r")
        self.assertEqual(added, 0)
        self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()
