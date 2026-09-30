#!/usr/bin/env python3
"""Compute the list of packages from a list-JSON that have no zip yet.

Used by the retry workflow to figure out which packages failed in the
main run and need to be re-tried. A package is considered "missing" when
no zip in ``--packages-dir`` matches the canonical filename pattern
produced by ``src.packager.package_build``:

    <owner>-<repo>-<YYYYMMDD>-<HHMMSS>-<platform>.zip

A zip named with the legacy ``<repo>`` stem also counts, but only when
that stem is unique in the list: five catalog repos are named ``cli``.

The script emits the missing names, one per line, to stdout. With
``--format space`` they're space-joined on a single line, ready to pass
straight to ``batch_test.py`` as positional name filters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Iterable, NamedTuple

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.packager import legacy_package_stem, package_stem  # noqa: E402


class PackageEntry(NamedTuple):
    """Package name plus zip stems accepted for a completed build.

    A NamedTuple, not a dataclass: tests load this script with
    ``spec_from_file_location`` without registering it in
    ``sys.modules``, and a dataclass needs that entry to resolve its
    postponed annotations.
    """

    name: str
    package_stem: str
    legacy_package_stem: str | None = None
    legacy_unique: bool = False


def _stem_metadata(
    name: str,
    stems_map: dict,
) -> tuple[str, str | None, bool]:
    """Read new/legacy stem metadata from a remaining-list stems map."""
    raw = stems_map.get(name)
    if isinstance(raw, dict):
        new_stem = raw.get("package_stem") or raw.get("stem") or name
        legacy_stem = raw.get("legacy_package_stem")
        legacy_unique = bool(raw.get("legacy_unique", False))
        return str(new_stem), (
            str(legacy_stem) if legacy_stem else None
        ), legacy_unique
    if isinstance(raw, str):
        return raw, None, False
    return name, None, False


def _load_packages(list_path: str) -> list[PackageEntry]:
    """Return ordered package entries from a list JSON file.

    Accepts both schemas in use across the workflow:

    * ``.github/packages/*.json`` — ``{"packages": [{"name": ...,
      "url": ...}, ...]}``; stems are derived from the URL.
    * ``remaining-<platform>.json`` (from ``plan_remaining.py``) —
      ``{"packages": ["name1", ...], "stems": {...}}``; old string
      stem maps still work, and new maps include legacy-stem metadata.

    Mixed lists are tolerated; entries with neither a ``name`` key nor
    a string value are silently skipped.
    """
    with open(list_path) as fh:
        data = json.load(fh)
    pkgs = data.get("packages", [])
    stems_map = data.get("stems") or {}
    # One pass keeps the declared order, which _apply_shard relies on.
    # A None uniqueness marks a stem derived here; it is counted below
    # over the whole list. String entries carry the uniqueness that
    # plan_remaining.py computed over the full catalog.
    rows: list[tuple[str, str, str | None, bool | None]] = []
    for p in pkgs:
        if isinstance(p, str):
            if p:
                rows.append((p, *_stem_metadata(p, stems_map)))
            continue
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not name:
            continue
        url = p.get("url") or p.get("repo") or ""
        if url:
            rows.append(
                (name, package_stem(url), legacy_package_stem(url), None)
            )
        else:
            new_stem, legacy_stem, _unique = _stem_metadata(name, stems_map)
            rows.append((name, new_stem, legacy_stem or new_stem, None))
    counts: dict[str, int] = {}
    for _name, _new_stem, legacy_stem, unique in rows:
        if unique is None:
            counts[legacy_stem] = counts.get(legacy_stem, 0) + 1
    return [
        PackageEntry(
            name,
            new_stem,
            legacy_stem,
            counts[legacy_stem] == 1 if unique is None else unique,
        )
        for name, new_stem, legacy_stem, unique in rows
    ]


def _apply_shard(
    names: list,
    shard_index: int,
    shard_total: int,
) -> list:
    """Return the contiguous slice of ``names`` for one shard.

    Must mirror ``batch_test._apply_shard``: ceil-division chunks so
    every shard except possibly the last is full-sized, and every name
    appears in exactly one shard.
    """
    if shard_total <= 1:
        return list(names)
    n = len(names)
    chunk = -(-n // shard_total)  # ceil(n / shard_total)
    start = shard_index * chunk
    end = start + chunk
    return list(names[start:end])


def _built_names(packages_dir: str, platform: str) -> set[str]:
    """Return the set of repo names for which a zip exists on disk."""
    if not os.path.isdir(packages_dir):
        return set()
    pat = re.compile(
        rf"^(?P<name>.+?)-\d{{8}}-\d{{6}}-{re.escape(platform)}\.zip$"
    )
    out: set[str] = set()
    for fname in os.listdir(packages_dir):
        m = pat.match(fname)
        if m:
            out.add(m.group("name"))
    return out


def _is_built(entry: PackageEntry, built: set[str]) -> bool:
    """Return True when a new or unambiguous legacy zip exists."""
    if entry.package_stem in built:
        return True
    return bool(
        entry.legacy_unique
        and entry.legacy_package_stem
        and entry.legacy_package_stem in built
    )


def _emit(names: Iterable[str], fmt: str) -> None:
    if fmt == "space":
        print(" ".join(names))
    else:  # default: lines
        for n in names:
            print(n)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--list",
        required=True,
        help="Path to a package-list JSON (e.g. .github/packages/smoke.json).",
    )
    p.add_argument(
        "--packages-dir",
        required=True,
        help="Directory containing previously-built .zip artifacts.",
    )
    p.add_argument(
        "--platform",
        required=True,
        choices=["alpine", "debian"],
        help="Platform slug used in the zip filename.",
    )
    p.add_argument(
        "--format",
        choices=["lines", "space"],
        default="lines",
        help="Output format. 'space' is convenient for shell substitution.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help=(
            "Zero-based index of the shard to filter the declared list "
            "to before computing missing packages (requires "
            "--shard-total). Mirrors batch_test.py's contiguous "
            "ceil(N/total) chunking so the retry workflow can mirror "
            "the main run's shard layout."
        ),
    )
    p.add_argument(
        "--shard-total",
        type=int,
        default=None,
        help=(
            "Total shard count the declared list is split into "
            "(requires --shard-index). Pass 1 to disable sharding."
        ),
    )
    args = p.parse_args(argv)

    shard_index = args.shard_index
    shard_total = args.shard_total
    if (shard_index is None) != (shard_total is None):
        print(
            "[ERROR] --shard-index and --shard-total must be provided "
            "together.",
            file=sys.stderr,
        )
        return 2
    if shard_total is not None:
        if shard_total < 1:
            print(
                f"[ERROR] --shard-total must be >= 1 (got {shard_total}).",
                file=sys.stderr,
            )
            return 2
        if shard_index < 0 or shard_index >= shard_total:
            print(
                f"[ERROR] --shard-index={shard_index} out of range "
                f"[0, {shard_total}).",
                file=sys.stderr,
            )
            return 2

    declared_pairs = _load_packages(args.list)
    if shard_total is not None:
        declared_pairs = _apply_shard(declared_pairs, shard_index, shard_total)
    declared = [entry.name for entry in declared_pairs]
    built = _built_names(args.packages_dir, args.platform)
    missing = [entry.name for entry in declared_pairs if not _is_built(
        entry,
        built,
    )]

    _emit(missing, args.format)

    # Also surface a summary on stderr so CI logs are self-explanatory.
    shard_tag = (
        f" shard={shard_index + 1}/{shard_total}"
        if shard_total is not None
        else ""
    )
    print(
        f"[missing_pkgs] list={os.path.basename(args.list)}"
        f"{shard_tag} platform={args.platform} "
        f"declared={len(declared)} built={len(built)} "
        f"missing={len(missing)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
