"""Checks for the JSON data files that Atesor writes under data/.

A line-by-line git merge of data/recipe_cache.json (commit 6b50080) left
35 packages twice, mixed fields of neighboring packages and lost 87.
json.load() hides a duplicate key, so the first test reads every key of
every object, and the second test looks for mixed fields.
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / ".github" / "scripts" / "merge_data_artifacts.py"


def _load_script():
    """Import merge_data_artifacts.py, which is not in a package."""
    spec = importlib.util.spec_from_file_location(
        "merge_data_artifacts", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _duplicate_keys(path: Path) -> list:
    """Return each key that one JSON object in the file holds twice."""
    found = []

    def hook(pairs):
        """Record the duplicate keys of one object."""
        counts = Counter(key for key, _ in pairs)
        found.extend(key for key, count in counts.items() if count > 1)
        return dict(pairs)

    with open(path, encoding="utf-8") as handle:
        json.load(handle, object_pairs_hook=hook)
    return found


def _recipe(built: str) -> dict:
    """Return a minimal recipe that was built at a given time."""
    return {"repo_url": "https://example.com/r", "last_built": built}


class TestDataFilesHaveNoDuplicateKeys(unittest.TestCase):
    """Every JSON file under data/ is free of duplicate keys."""

    def test_no_duplicate_keys(self) -> None:
        """No JSON object under data/ holds the same key twice."""
        paths = sorted((_ROOT / "data").rglob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=str(path.relative_to(_ROOT))):
                self.assertEqual([], _duplicate_keys(path))

    def test_recipes_name_their_own_package(self) -> None:
        """No recipe points at the files of another package."""
        cache = json.loads(
            (_ROOT / "data" / "recipe_cache.json").read_text("utf-8")
        )
        mixed = []
        for name, sandboxes in cache["packages"].items():
            for sandbox, recipe in sandboxes.items():
                match = re.match(
                    r"output/(.+)_recipe\.md$", recipe.get("recipe_file") or ""
                )
                if match and match.group(1) != name:
                    mixed.append((name, sandbox, recipe["recipe_file"]))
                for artifact in recipe.get("artifacts") or []:
                    found = re.match(
                        r"/workspace/repos/([^/]+)/", artifact.get("path", "")
                    )
                    if found and found.group(1) != name:
                        mixed.append((name, sandbox, artifact["path"]))
        self.assertEqual([], mixed)


class TestRecipeCacheMerge(unittest.TestCase):
    """Tests for the duplicate-safe merge in merge_data_artifacts.py."""

    def setUp(self) -> None:
        """Import the script and make a scratch folder."""
        self.script = _load_script()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name: str, text: str) -> Path:
        """Write a file in the scratch folder."""
        path = Path(self._tmp.name) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_duplicates_keep_the_newest_recipe(self) -> None:
        """A duplicate keeps its newest recipe, not its last copy."""
        text = (
            '{"version": "2.0", "packages": {'
            '"ecoji": {"debian-riscv64": {"last_built": "2026-07-06T07:03"}},'
            '"zlib": {"alpine-riscv64": {"last_built": "2026-05-01T00:00"}},'
            '"ecoji": {"debian-riscv64": {"last_built": "2026-06-03T01:29"},'
            ' "alpine-riscv64": {"last_built": "2026-05-21T00:00"}}}}'
        )
        cache = self.script._load_recipe_cache(self._write("dup.json", text))
        ecoji = cache["packages"]["ecoji"]
        self.assertEqual(
            "2026-07-06T07:03", ecoji["debian-riscv64"]["last_built"]
        )
        self.assertIn("alpine-riscv64", ecoji)
        self.assertEqual(["ecoji", "zlib"], sorted(cache["packages"]))

    def test_merge_caches_mode_writes_one_clean_cache(self) -> None:
        """The conflict mode keeps both sides; the newest recipe wins."""
        ours = {
            "version": "2.0",
            "packages": {
                "a": {"debian-riscv64": _recipe("2026-07-01T00:00:00")},
                "b": {"debian-riscv64": _recipe("2026-01-01T00:00:00")},
            },
        }
        theirs = {
            "version": "2.0",
            "packages": {
                "b": {"debian-riscv64": _recipe("2026-02-01T00:00:00")},
                "c": {"alpine-riscv64": _recipe("2026-03-01T00:00:00")},
            },
        }
        paths = [
            str(self._write("ours.json", json.dumps(ours))),
            str(self._write("theirs.json", json.dumps(theirs))),
        ]
        output = Path(self._tmp.name) / "merged.json"
        argv = ["merge_data_artifacts.py", "--merge-caches", *paths]
        argv.extend(["--output", str(output)])
        with (
            mock.patch("sys.argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(0, self.script.main())
        merged = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(["a", "b", "c"], sorted(merged["packages"]))
        self.assertEqual(
            "2026-02-01T00:00:00",
            merged["packages"]["b"]["debian-riscv64"]["last_built"],
        )
        self.assertEqual([], _duplicate_keys(output))

    def test_output_matches_the_app_writer(self) -> None:
        """The script writes the same text as src/memory.py."""
        data = {"version": "2.0", "packages": {"a": {"s": {"n": "it’s …"}}}}
        path = Path(self._tmp.name) / "out.json"
        self.script._save_json(path, data)
        self.assertEqual(
            json.dumps(data, indent=2, ensure_ascii=False),
            path.read_text(encoding="utf-8"),
        )


class TestMergeDriver(unittest.TestCase):
    """Tests for the git merge driver mode of merge_data_artifacts.py."""

    def setUp(self) -> None:
        """Import the script and make a scratch folder."""
        self.script = _load_script()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    @staticmethod
    def _cache(**packages) -> dict:
        """Return a cache whose packages each have one debian recipe."""
        return {
            "version": "2.0",
            "packages": {
                name: {"debian-riscv64": _recipe(built)}
                for name, built in packages.items()
            },
        }

    def test_three_way_rules(self) -> None:
        """Each side's change survives; a removal holds; newest wins."""
        base = self._cache(a="2026-01-01", d="2026-01-01", e="2026-01-01")
        ours = self._cache(
            a="2026-02-01", d="2026-05-01", e="2026-01-01", c="2026-04-01"
        )
        theirs = self._cache(a="2026-03-01", d="2026-01-01", b="2026-03-02")
        merged = self.script.merge_three_way(base, ours, theirs)["packages"]
        built = {
            n: s["debian-riscv64"]["last_built"] for n, s in merged.items()
        }
        self.assertEqual(
            {
                "a": "2026-03-01",
                "d": "2026-05-01",
                "c": "2026-04-01",
                "b": "2026-03-02",
            },
            built,
        )

    def test_a_changed_recipe_beats_a_removal(self) -> None:
        """One side removes a recipe that the other side changed."""
        base = self._cache(a="2026-01-01")
        ours = self._cache()
        theirs = self._cache(a="2026-02-01")
        merged = self.script.merge_three_way(base, ours, theirs)
        self.assertEqual(
            "2026-02-01",
            merged["packages"]["a"]["debian-riscv64"]["last_built"],
        )

    def _run_driver(self, base: str, ours: str, theirs: str):
        """Run the driver CLI on three files; return the exit and ours."""
        paths = []
        for name, text in (("base", base), ("ours", ours), ("theirs", theirs)):
            path = Path(self._tmp.name) / name
            path.write_text(text, encoding="utf-8")
            paths.append(str(path))
        argv = ["merge_data_artifacts.py", "--merge-driver", *paths]
        with (
            mock.patch("sys.argv", argv),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = self.script.main()
        return code, Path(paths[1]).read_text(encoding="utf-8")

    def test_driver_writes_ours_and_exits_zero(self) -> None:
        """Git reads the merged cache from the OURS file (%A)."""
        code, text = self._run_driver(
            json.dumps(self._cache(a="2026-01-01")),
            json.dumps(self._cache(a="2026-01-01", c="2026-04-01")),
            json.dumps(self._cache(a="2026-01-01", b="2026-03-01")),
        )
        self.assertEqual(0, code)
        self.assertEqual(["a", "c", "b"], list(json.loads(text)["packages"]))

    def test_driver_without_ancestor_merges_both_sides(self) -> None:
        """An empty ancestor file (no common history) is an empty cache."""
        code, text = self._run_driver(
            "",
            json.dumps(self._cache(a="2026-01-01")),
            json.dumps(self._cache(b="2026-01-01")),
        )
        self.assertEqual(0, code)
        self.assertEqual(["a", "b"], list(json.loads(text)["packages"]))

    def test_driver_reports_a_conflict_on_bad_json(self) -> None:
        """Conflict markers or broken JSON give a normal git conflict."""
        code, _ = self._run_driver(
            json.dumps(self._cache()),
            "<<<<<<< ours",
            json.dumps(self._cache()),
        )
        self.assertEqual(1, code)


@unittest.skipUnless(shutil.which("git"), "needs git")
class TestMergeDriverWithGit(unittest.TestCase):
    """The setup script, then a real git merge with no conflict."""

    def _git(self, *args: str, cwd: str) -> subprocess.CompletedProcess:
        """Run git in the scratch repository, isolated from user config."""
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
            + list(args),
            cwd=cwd,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def setUp(self) -> None:
        """Make a scratch repository with the repo's merge files."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = os.path.join(self._tmp.name, "repo")
        self.env = dict(os.environ)
        self.env.update(
            {
                "HOME": self._tmp.name,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            }
        )
        scripts = os.path.join(self.repo, ".github", "scripts")
        os.makedirs(scripts)
        os.makedirs(os.path.join(self.repo, "data"))
        shutil.copy(_ROOT / ".gitattributes", self.repo)
        shutil.copy(_SCRIPT, scripts)
        shutil.copy(
            _ROOT / ".github/scripts/setup_git_merge_driver.sh", scripts
        )
        subprocess.run(
            ["git", "-c", "init.defaultBranch=main", "init", "-q", self.repo],
            env=self.env,
            check=True,
            timeout=60,
        )

    def _commit(self, packages: dict, message: str) -> None:
        """Write the cache with these packages and commit it."""
        cache = {
            "version": "2.0",
            "packages": {
                name: {"debian-riscv64": _recipe(built)}
                for name, built in packages.items()
            },
        }
        path = os.path.join(self.repo, "data", "recipe_cache.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
        self._git("add", "-A", cwd=self.repo)
        self._git("commit", "-qm", message, cwd=self.repo)

    def test_merge_needs_no_hand_work(self) -> None:
        """After the setup, both branches' recipes merge by meaning."""
        self._commit({"a": "2026-01-01"}, "base")
        self._git("checkout", "-qb", "side", cwd=self.repo)
        self._commit({"a": "2026-03-01", "b": "2026-03-02"}, "side")
        self._git("checkout", "-q", "main", cwd=self.repo)
        self._commit({"a": "2026-02-01", "c": "2026-04-01"}, "main")
        setup = subprocess.run(
            ["sh", ".github/scripts/setup_git_merge_driver.sh"],
            cwd=self.repo,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(0, setup.returncode, setup.stderr)
        merge = self._git("merge", "--no-edit", "side", cwd=self.repo)
        self.assertEqual(0, merge.returncode, merge.stdout + merge.stderr)
        path = os.path.join(self.repo, "data", "recipe_cache.json")
        with open(path, encoding="utf-8") as handle:
            packages = json.load(handle)["packages"]
        self.assertEqual(
            "2026-03-01", packages["a"]["debian-riscv64"]["last_built"]
        )
        self.assertEqual(["a", "c", "b"], list(packages))

    def test_setup_works_from_another_folder(self) -> None:
        """The setup finds its own clone, even from inside another repo."""
        other = os.path.join(self._tmp.name, "other")
        subprocess.run(
            ["git", "init", "-q", other], env=self.env, check=True, timeout=60
        )
        script = os.path.join(
            self.repo, ".github", "scripts", "setup_git_merge_driver.sh"
        )
        for cwd in (self._tmp.name, other):
            with self.subTest(cwd=cwd):
                setup = subprocess.run(
                    ["sh", script],
                    cwd=cwd,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(0, setup.returncode, setup.stderr)
        key = "merge.recipe-cache.driver"
        own = self._git("config", "--get", key, cwd=self.repo)
        self.assertIn("--merge-driver", own.stdout)
        self.assertEqual(
            "", self._git("config", "--get", key, cwd=other).stdout
        )
        attributes = os.path.join(self.repo, ".git", "info", "attributes")
        with open(attributes, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(
            1, lines.count("data/recipe_cache.json merge=recipe-cache")
        )

    def test_merge_without_setup_stops_at_a_conflict(self) -> None:
        """Without the driver, git never merges the cache as text."""
        self._commit({"a": "2026-01-01"}, "base")
        self._git("checkout", "-qb", "side", cwd=self.repo)
        self._commit({"a": "2026-01-01", "b": "2026-03-02"}, "side")
        self._git("checkout", "-q", "main", cwd=self.repo)
        self._commit({"a": "2026-01-01", "c": "2026-04-01"}, "main")
        merge = self._git("merge", "--no-edit", "side", cwd=self.repo)
        self.assertNotEqual(0, merge.returncode)
        self.assertIn("CONFLICT", merge.stdout + merge.stderr)


class TestGitMergeGuard(unittest.TestCase):
    """The recipe cache is never merged line by line."""

    def test_recipe_cache_is_never_merged_as_text(self) -> None:
        """.gitattributes marks the cache so git stops at a conflict."""
        text = (_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("data/recipe_cache.json merge=binary", text.splitlines())


if __name__ == "__main__":
    unittest.main()
