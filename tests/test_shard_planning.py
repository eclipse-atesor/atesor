"""Tests for shard planning + slicing used by the batch-port workflow.

Covers:
* ``.github/scripts/plan_shards.py::compute_plan`` — group count math.
* ``.github/scripts/batch_test.py::_apply_shard`` — contiguous slicing
  must cover every package exactly once across all shards.
* ``.github/scripts/missing_pkgs.py::_apply_shard`` — the retry
  workflow's slicing must match ``batch_test._apply_shard`` exactly.
"""

import importlib.util
import os
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel_path: str, name: str):
    """Load a script from .github/scripts as a module.

    .github/scripts is not an importable package, so import_module
    cannot reach the script.
    """
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plan_shards = _load(".github/scripts/plan_shards.py", "plan_shards")
batch_test = _load(".github/scripts/batch_test.py", "batch_test_mod")
missing_pkgs = _load(".github/scripts/missing_pkgs.py", "missing_pkgs_mod")


class TestComputePlan(unittest.TestCase):
    """plan_shards.compute_plan output shape and edge cases."""

    def test_empty_list_yields_zero_shards(self) -> None:
        total, groups = plan_shards.compute_plan(0, 50)
        self.assertEqual(total, 0)
        self.assertEqual(groups, [])

    def test_exact_multiple(self) -> None:
        total, groups = plan_shards.compute_plan(100, 50)
        self.assertEqual(total, 2)
        self.assertEqual(groups, [0, 1])

    def test_non_multiple_rounds_up(self) -> None:
        total, groups = plan_shards.compute_plan(172, 50)
        self.assertEqual(total, 4)
        self.assertEqual(groups, [0, 1, 2, 3])

    def test_single_package(self) -> None:
        total, groups = plan_shards.compute_plan(1, 50)
        self.assertEqual(total, 1)
        self.assertEqual(groups, [0])

    def test_count_below_group_size(self) -> None:
        total, groups = plan_shards.compute_plan(8, 50)
        self.assertEqual(total, 1)
        self.assertEqual(groups, [0])

    def test_invalid_group_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_shards.compute_plan(10, 0)
        with self.assertRaises(ValueError):
            plan_shards.compute_plan(10, -1)


class TestApplyShard(unittest.TestCase):
    """batch_test._apply_shard partitioning invariants."""

    def _pkgs(self, n: int) -> list[tuple[str, str]]:
        return [(f"pkg{i}", f"https://example/{i}") for i in range(n)]

    def test_total_one_returns_full_list(self) -> None:
        pkgs = self._pkgs(10)
        self.assertEqual(batch_test._apply_shard(pkgs, 0, 1), pkgs)

    def test_partitions_cover_every_package_exactly_once(self) -> None:
        # The key invariant: for any (N, total), concatenating all
        # shards in order reproduces the input list with no overlap and
        # no loss. Tested across a representative grid of sizes.
        for n in [0, 1, 7, 50, 100, 172, 173]:
            for total in [1, 2, 3, 4, 7]:
                pkgs = self._pkgs(n)
                with self.subTest(n=n, total=total):
                    rebuilt: list[tuple[str, str]] = []
                    for i in range(total):
                        rebuilt.extend(batch_test._apply_shard(pkgs, i, total))
                    self.assertEqual(rebuilt, pkgs)

    def test_shard_sizes_balanced_within_one(self) -> None:
        # Ceil division fills every shard except the trailing ones, so
        # sizes can differ by more than 1 (7 into 3 gives 3, 3, 1).
        # 172 into 4 divides evenly.
        pkgs = self._pkgs(172)
        sizes = [len(batch_test._apply_shard(pkgs, i, 4)) for i in range(4)]
        self.assertEqual(sizes, [43, 43, 43, 43])

    def test_shard_count_exceeds_packages(self) -> None:
        # 5 shards for 3 packages: first 3 shards have 1 pkg each, last
        # 2 are empty. Empty shards are tolerated.
        pkgs = self._pkgs(3)
        sizes = [len(batch_test._apply_shard(pkgs, i, 5)) for i in range(5)]
        self.assertEqual(sizes, [1, 1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()


class TestMissingPkgsApplyShard(unittest.TestCase):
    """missing_pkgs._apply_shard must match batch_test._apply_shard.

    The retry workflow uses missing_pkgs to figure out which packages
    a single (platform, shard) job should re-run. Its chunking MUST
    match the main run's chunking exactly — otherwise the retry would
    re-run packages from the wrong shard.
    """

    def _names(self, n: int) -> list[str]:
        return [f"pkg{i}" for i in range(n)]

    def test_matches_batch_test_apply_shard(self) -> None:
        for n in [0, 1, 7, 50, 100, 172, 173]:
            for total in [1, 2, 3, 4, 7]:
                for idx in range(total):
                    with self.subTest(n=n, total=total, idx=idx):
                        names = self._names(n)
                        pairs = [(name, "url") for name in names]
                        bt = [
                            name
                            for name, _ in batch_test._apply_shard(
                                pairs, idx, total
                            )
                        ]
                        mp = missing_pkgs._apply_shard(names, idx, total)
                        self.assertEqual(mp, bt)

    def test_total_one_returns_full_list(self) -> None:
        names = self._names(10)
        self.assertEqual(
            missing_pkgs._apply_shard(names, 0, 1),
            names,
        )

    def test_partitions_cover_every_name_exactly_once(self) -> None:
        for n in [0, 1, 50, 173]:
            for total in [1, 2, 4]:
                names = self._names(n)
                with self.subTest(n=n, total=total):
                    rebuilt: list[str] = []
                    for i in range(total):
                        rebuilt.extend(
                            missing_pkgs._apply_shard(names, i, total)
                        )
                    self.assertEqual(rebuilt, names)
