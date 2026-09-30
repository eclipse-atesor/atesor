"""Tests for the skip-already-released planner used by batch-port.

Covers ``.github/scripts/plan_remaining.py``:
* Regex parsing of release asset filenames (per-platform isolation).
* Declared-order preservation.
* Shard count math from the remaining set.
* Edge cases: empty release, everything released, malformed names,
  mixed platforms.

Also covers the polymorphic loader in
``.github/scripts/missing_pkgs.py``: it must accept BOTH the
``full.json`` schema (list of ``{name: ...}`` dicts) AND the
``remaining-<platform>.json`` schema (list of plain strings).
"""

import importlib.util
import json
import os
import pathlib
import tempfile
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(_REPO, rel_path),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plan_remaining = _load(
    ".github/scripts/plan_remaining.py",
    "plan_remaining_mod",
)
missing_pkgs = _load(
    ".github/scripts/missing_pkgs.py",
    "missing_pkgs_pr_mod",
)
slice_pkgs = _load(
    ".github/scripts/slice_pkgs.py",
    "slice_pkgs_mod",
)


def _entry_tuple(entry):
    """Return comparable fields from a missing_pkgs PackageEntry."""
    return (
        entry.name,
        entry.package_stem,
        entry.legacy_package_stem,
        entry.legacy_unique,
    )


class TestReleasedNames(unittest.TestCase):
    """``_released_names`` extracts package names from asset filenames."""

    def test_basic_match(self) -> None:
        """Matching alpine asset filenames yield their package names."""
        names = plan_remaining._released_names(
            [
                "afrog-20260604-143632-alpine.zip",
                "age-20260605-064655-alpine.zip",
            ],
            "alpine",
        )
        self.assertEqual(names, {"afrog", "age"})

    def test_per_platform_isolation(self) -> None:
        """Alpine assets must not mark a package released for debian."""
        names = plan_remaining._released_names(
            [
                "afrog-20260604-143632-alpine.zip",
                "age-20260604-092304-debian.zip",
            ],
            "debian",
        )
        self.assertEqual(names, {"age"})

    def test_hyphenated_package_name(self) -> None:
        """Hyphenated package names are extracted before the timestamp."""
        names = plan_remaining._released_names(
            ["lzip-1.0-20260604-150544-alpine.zip"],
            "alpine",
        )
        self.assertEqual(names, {"lzip-1.0"})

    def test_camelcase_package_name(self) -> None:
        """Mixed-case package names are preserved when extracted."""
        names = plan_remaining._released_names(
            ["AnalyticsRelationships-20260604-150544-debian.zip"],
            "debian",
        )
        self.assertEqual(names, {"AnalyticsRelationships"})

    def test_malformed_filename_ignored(self) -> None:
        """Malformed filenames are ignored while valid assets are kept."""
        names = plan_remaining._released_names(
            [
                "foo.zip",  # no pattern
                "bar-20260601-debian.zip",  # missing time
                "baz-2026-06-01-150000-alpine.zip",  # wrong date fmt
                "quux-20260601-150000-alpine.zip.bak",  # extra suffix
                "good-20260601-150000-alpine.zip",  # OK
            ],
            "alpine",
        )
        self.assertEqual(names, {"good"})

    def test_empty_input(self) -> None:
        """Empty asset inputs return an empty released-name set."""
        self.assertEqual(
            plan_remaining._released_names([], "alpine"),
            set(),
        )
        self.assertEqual(
            plan_remaining._released_names(["", "   ", "\n"], "alpine"),
            set(),
        )

    def test_unknown_platform_not_matched(self) -> None:
        """Unknown-platform assets do not match another platform."""
        names = plan_remaining._released_names(
            ["foo-20260601-150000-windows.zip"],
            "alpine",
        )
        self.assertEqual(names, set())


class TestComputePlan(unittest.TestCase):
    """``compute_plan`` skips released names + computes shard count."""

    def test_preserves_declared_order(self) -> None:
        """Remaining packages keep declared order after released names skip."""
        declared = ["c", "a", "b", "d"]
        released = {"a"}
        remaining, total, groups = plan_remaining.compute_plan(
            declared,
            released,
            group_size=50,
        )
        self.assertEqual(remaining, ["c", "b", "d"])
        self.assertEqual(total, 1)
        self.assertEqual(groups, [0])

    def test_new_owner_repo_stem_skips_released_package(self) -> None:
        """Owner-prefixed zip names are the primary release keys."""
        remaining, total, _ = plan_remaining.compute_plan(
            ["hc-cli", "urfave-cli"],
            {"hetznercloud-cli"},
            group_size=50,
            stem_of={
                "hc-cli": "hetznercloud-cli",
                "urfave-cli": "urfave-cli",
            },
            legacy_stem_of={"hc-cli": "cli", "urfave-cli": "cli"},
            legacy_unique={"hc-cli": False, "urfave-cli": False},
        )
        self.assertEqual(remaining, ["urfave-cli"])
        self.assertEqual(total, 1)

    def test_unique_legacy_stem_skips_released_package(self) -> None:
        """Old basename zips count only when the basename is unique."""
        remaining, total, _ = plan_remaining.compute_plan(
            ["zlib", "hc-cli", "urfave-cli"],
            {"zlib", "cli"},
            group_size=50,
            stem_of={
                "zlib": "madler-zlib",
                "hc-cli": "hetznercloud-cli",
                "urfave-cli": "urfave-cli",
            },
            legacy_stem_of={
                "zlib": "zlib",
                "hc-cli": "cli",
                "urfave-cli": "cli",
            },
            legacy_unique={
                "zlib": True,
                "hc-cli": False,
                "urfave-cli": False,
            },
        )
        self.assertEqual(remaining, ["hc-cli", "urfave-cli"])
        self.assertEqual(total, 1)

    def test_no_skips_when_release_empty(self) -> None:
        """Empty released set leaves all declared packages in one shard."""
        declared = ["a", "b", "c"]
        remaining, total, groups = plan_remaining.compute_plan(
            declared,
            set(),
            group_size=50,
        )
        self.assertEqual(remaining, declared)
        self.assertEqual(total, 1)
        self.assertEqual(groups, [0])

    def test_everything_released_yields_zero_shards(self) -> None:
        """All released packages yield no remaining packages or shards."""
        declared = ["a", "b"]
        released = {"a", "b"}
        remaining, total, groups = plan_remaining.compute_plan(
            declared,
            released,
            group_size=50,
        )
        self.assertEqual(remaining, [])
        self.assertEqual(total, 0)
        self.assertEqual(groups, [])

    def test_shard_math_matches_group_size(self) -> None:
        """Shard count uses ceil division over the remaining packages."""
        # 172 remaining at 50 per shard -> 4 shards (ceil)
        declared = [f"pkg{i}" for i in range(200)]
        released = {f"pkg{i}" for i in range(28)}  # leaves 172
        remaining, total, groups = plan_remaining.compute_plan(
            declared,
            released,
            group_size=50,
        )
        self.assertEqual(len(remaining), 172)
        self.assertEqual(total, 4)
        self.assertEqual(groups, [0, 1, 2, 3])

    def test_invalid_group_size(self) -> None:
        """Zero or negative group sizes raise ValueError."""
        with self.assertRaises(ValueError):
            plan_remaining.compute_plan(["a"], set(), group_size=0)
        with self.assertRaises(ValueError):
            plan_remaining.compute_plan(["a"], set(), group_size=-1)

    def test_released_not_in_declared_is_noop(self) -> None:
        """Released names outside declared packages do not affect planning."""
        # Spurious assets (e.g. a package no longer on the list)
        # must not affect the remaining computation.
        declared = ["a", "b"]
        released = {"a", "ghost"}
        remaining, total, _ = plan_remaining.compute_plan(
            declared,
            released,
            group_size=50,
        )
        self.assertEqual(remaining, ["b"])
        self.assertEqual(total, 1)


class TestLoadReleasedAssets(unittest.TestCase):
    """``_load_released_assets`` is robust to missing/empty files."""

    def test_missing_path(self) -> None:
        """Missing or unset asset paths load as an empty list."""
        self.assertEqual(plan_remaining._load_released_assets(None), [])
        self.assertEqual(
            plan_remaining._load_released_assets("/no/such/file"),
            [],
        )

    def test_strips_blanks(self) -> None:
        """Released asset loading strips blank lines from files."""
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".txt",
            delete=False,
        ) as fh:
            fh.write("a-20260601-150000-alpine.zip\n")
            fh.write("\n")
            fh.write("b-20260601-150000-alpine.zip\n")
            path = fh.name
        try:
            self.assertEqual(
                plan_remaining._load_released_assets(path),
                [
                    "a-20260601-150000-alpine.zip",
                    "b-20260601-150000-alpine.zip",
                ],
            )
        finally:
            os.unlink(path)


class TestMissingPkgsPolymorphicLoader(unittest.TestCase):
    """missing_pkgs must read both full.json and remaining-*.json."""

    def _write_json(self, payload: dict) -> str:
        fh = tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            delete=False,
        )
        json.dump(payload, fh)
        fh.close()
        return fh.name

    def test_loads_dict_schema(self) -> None:
        """Dict package entries load as (name, url-derived-stem) pairs."""
        # .github/packages/*.json shape. The zip stem comes from the
        # URL basename, not the name field (they differ for 91/705
        # full.json entries).
        path = self._write_json(
            {
                "packages": [
                    {"name": "a", "url": "https://github.com/x/a"},
                    {"name": "b", "url": "https://github.com/x/b-repo.git"},
                ],
            }
        )
        try:
            self.assertEqual(
                [_entry_tuple(e) for e in missing_pkgs._load_packages(path)],
                [
                    ("a", "x-a", "a", True),
                    ("b", "x-b-repo", "b-repo", True),
                ],
            )
        finally:
            os.unlink(path)

    def test_loads_string_schema(self) -> None:
        """String package entries load unchanged from remaining files."""
        # remaining-<platform>.json shape, written by plan_remaining.py.
        # The optional "stems" map overrides zip stems per name.
        path = self._write_json(
            {
                "packages": ["x", "y", "z"],
                "stems": {
                    "y": {
                        "package_stem": "org-y",
                        "legacy_package_stem": "y-binaries",
                        "legacy_unique": True,
                    },
                },
            }
        )
        try:
            self.assertEqual(
                [_entry_tuple(e) for e in missing_pkgs._load_packages(path)],
                [
                    ("x", "x", None, False),
                    ("y", "org-y", "y-binaries", True),
                    ("z", "z", None, False),
                ],
            )
        finally:
            os.unlink(path)

    def test_skips_entries_without_name(self) -> None:
        """Package entries without names are skipped during loading."""
        path = self._write_json(
            {
                "packages": [
                    {"name": "ok"},
                    {"url": "no-name"},  # dropped
                    "",  # dropped
                    "also-ok",
                ],
            }
        )
        try:
            self.assertEqual(
                [_entry_tuple(e) for e in missing_pkgs._load_packages(path)],
                [
                    ("ok", "ok", "ok", True),
                    ("also-ok", "also-ok", None, False),
                ],
            )
        finally:
            os.unlink(path)


class TestPlanRemainingWritesArtifact(unittest.TestCase):
    """End-to-end: ``_write_remaining`` + reload via missing_pkgs."""

    def test_roundtrip(self) -> None:
        """Written remaining packages round-trip through the loader."""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "remaining-alpine.json")
            plan_remaining._write_remaining(
                out,
                ["a", "b", "c"],
                stem_of={"a": "org-a", "b": "org-b", "c": "org-c"},
                legacy_stem_of={"a": "a", "b": "b-binaries", "c": "c"},
                legacy_unique={"a": True, "b": True, "c": True},
            )
            with open(out) as fh:
                data = json.load(fh)
            self.assertEqual(
                data,
                {
                    "packages": ["a", "b", "c"],
                    "stems": {
                        "a": {
                            "package_stem": "org-a",
                            "legacy_package_stem": "a",
                            "legacy_unique": True,
                        },
                        "b": {
                            "package_stem": "org-b",
                            "legacy_package_stem": "b-binaries",
                            "legacy_unique": True,
                        },
                        "c": {
                            "package_stem": "org-c",
                            "legacy_package_stem": "c",
                            "legacy_unique": True,
                        },
                    },
                },
            )
            # And the loader missing-pkgs uses must see the same names
            # plus the divergent stem for zip matching.
            self.assertEqual(
                [_entry_tuple(e) for e in missing_pkgs._load_packages(out)],
                [
                    ("a", "org-a", "a", True),
                    ("b", "org-b", "b-binaries", True),
                    ("c", "org-c", "c", True),
                ],
            )


class TestMissingPkgsBuiltMatching(unittest.TestCase):
    """missing_pkgs matches new stems and unique legacy stems."""

    def test_new_or_unique_legacy_stem_counts_as_built(self) -> None:
        """Ambiguous legacy stems do not hide missing packages."""
        built = {"hetznercloud-cli", "zlib", "cli"}
        cases = [
            (missing_pkgs.PackageEntry("hc", "hetznercloud-cli"), True),
            (missing_pkgs.PackageEntry("z", "madler-zlib", "zlib", True),
             True),
            (missing_pkgs.PackageEntry("cli-a", "owner-a-cli", "cli",
                                       False), False),
        ]
        for entry, expected in cases:
            with self.subTest(entry=entry):
                self.assertEqual(missing_pkgs._is_built(entry, built),
                                 expected)

    def test_built_names_extracts_new_owner_repo_stem(self) -> None:
        """Built-name parsing keeps the owner-repo stem intact."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "hetznercloud-cli-20260930-120000-debian.zip").touch()
            self.assertEqual(
                missing_pkgs._built_names(directory, "debian"),
                {"hetznercloud-cli"},
            )


class TestSlicePkgs(unittest.TestCase):
    """slice_pkgs must match the canonical ceil-division chunking."""

    def test_total_one_returns_full_list(self) -> None:
        """A single shard returns the full package list."""
        self.assertEqual(
            slice_pkgs.apply_shard(["a", "b", "c"], 0, 1),
            ["a", "b", "c"],
        )

    def test_partitions_cover_every_name_exactly_once(self) -> None:
        """Shard partitions rebuild every package name exactly once."""
        names = [f"p{i}" for i in range(173)]
        for total in [1, 2, 4, 7]:
            with self.subTest(total=total):
                rebuilt: list[str] = []
                for i in range(total):
                    rebuilt.extend(
                        slice_pkgs.apply_shard(names, i, total),
                    )
                self.assertEqual(rebuilt, names)

    def test_matches_missing_pkgs_apply_shard(self) -> None:
        """slice_pkgs sharding matches missing_pkgs sharding."""
        names = [f"p{i}" for i in range(50)]
        for total in [1, 2, 3, 7]:
            for idx in range(total):
                with self.subTest(total=total, idx=idx):
                    self.assertEqual(
                        slice_pkgs.apply_shard(names, idx, total),
                        missing_pkgs._apply_shard(names, idx, total),
                    )


if __name__ == "__main__":
    unittest.main()
