#!/usr/bin/env python3
"""Compute the per-platform "remaining" package plan for batch-port.

Given:

* a package list JSON (e.g. ``.github/packages/full.json``),
* the set of asset filenames already published under the current
  monthly release tag and any overflow tags
  (``builds-YYYY-MM``, ``builds-YYYY-MM-01``, ``builds-YYYY-MM-02``,
  …; one name per line in ``--released-assets``),
* a target ``--platform`` slug (``alpine`` or ``debian``), and
* a shard ``--group-size``,

this script writes the ordered list of packages that still need
building for that platform to ``--remaining-out`` (as
``{"packages": [...]}``), and emits two ``KEY=VALUE`` lines suitable
for ``$GITHUB_OUTPUT``::

    <prefix>_total=<N>
    <prefix>_groups=[0, 1, ..., N-1]

where ``<prefix>`` defaults to the platform name. ``N`` is the number
of shards needed to cover the remaining packages at ``--group-size``
each (``ceil(remaining / group_size)``).

"Already released" means an asset under the current monthly tag (or
any of its ``-NN`` overflow tags created when a release fills its
1000-asset cap) whose filename matches the canonical pattern produced
by ``src.packager.package_build``::

    <package>-<YYYYMMDD>-<HHMMSS>-<platform>.zip

The platform is matched exactly, so ``foo-...-alpine.zip`` does not
mark ``foo`` as released for the ``debian`` platform.

When the remaining list is empty the script emits ``<prefix>_total=0``
and ``<prefix>_groups=[]`` — downstream matrix jobs are then skipped
naturally because their matrix has no entries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Iterable

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.packager import legacy_package_stem, package_stem  # noqa: E402

_PACKAGES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "packages")
)


def _resolve_list_path(value: str) -> str:
    """Bare name resolves under ``.github/packages/<name>.json``.

    Mirrors ``plan_shards.py::_resolve_list_path`` so the workflow can
    pass either ``--list full`` (production) or an explicit file path
    (tests, ad-hoc invocations).
    """
    if os.sep in value or value.endswith(".json"):
        return value
    return os.path.join(_PACKAGES_DIR, f"{value}.json")


def _load_packages(list_path: str) -> list[tuple[str, str, str, bool]]:
    """Return ordered ``(name, new_stem, legacy_stem, legacy_unique)``.

    ``new_stem`` uses the owner-prefixed package stem. ``legacy_stem``
    is accepted only when it is unique in the whole list, because a
    legacy ``cli-*.zip`` asset cannot identify which ``cli`` repo it
    belongs to.
    """
    with open(list_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    pkgs = data.get("packages", [])
    raw: list[tuple[str, str, str]] = []
    for p in pkgs:
        name = p.get("name")
        if not name:
            continue
        url = p.get("url") or p.get("repo") or ""
        new_stem = package_stem(url) if url else name
        legacy_stem = legacy_package_stem(url) if url else name
        raw.append((name, new_stem, legacy_stem))
    counts: dict[str, int] = {}
    for _name, _new_stem, legacy_stem in raw:
        counts[legacy_stem] = counts.get(legacy_stem, 0) + 1
    out: list[tuple[str, str, str, bool]] = []
    for name, new_stem, legacy_stem in raw:
        out.append((name, new_stem, legacy_stem, counts[legacy_stem] == 1))
    return out


def _released_names(
    asset_names: Iterable[str],
    platform: str,
) -> set[str]:
    """Return the set of package names already released for ``platform``.

    Filenames that don't match the canonical pattern are silently
    ignored — they can't be confidently attributed to a package, so we
    prefer to err on the side of rebuilding.
    """
    pat = re.compile(
        rf"^(?P<name>.+?)-\d{{8}}-\d{{6}}-{re.escape(platform)}\.zip$"
    )
    out: set[str] = set()
    for raw in asset_names:
        name = raw.strip()
        if not name:
            continue
        m = pat.match(name)
        if m:
            out.add(m.group("name"))
    return out


def _load_released_assets(path: str | None) -> list[str]:
    """Read released asset filenames from ``path`` (one per line).

    Missing or empty file is treated as "no assets" — that's the
    expected state on the first run of a new month.
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def compute_plan(
    declared: list[str],
    released: set[str],
    group_size: int,
    stem_of: dict[str, str] | None = None,
    legacy_stem_of: dict[str, str] | None = None,
    legacy_unique: dict[str, bool] | None = None,
) -> tuple[list[str], int, list[int]]:
    """Return ``(remaining, total_shards, group_indices)``.

    ``remaining`` preserves the declared order; ``total_shards`` is
    ``ceil(len(remaining) / group_size)`` (0 when empty). ``stem_of``
    maps a declared name to the zip filename stem to match against
    ``released`` (defaults to the name itself).
    """
    if group_size < 1:
        raise ValueError(f"group-size must be >= 1 (got {group_size})")
    stems = stem_of or {}
    legacy_stems = legacy_stem_of or {}
    legacy_ok = legacy_unique or {}
    remaining = []
    for name in declared:
        new_stem = stems.get(name, name)
        old_stem = legacy_stems.get(name)
        released_by_new = new_stem in released
        released_by_legacy = bool(
            old_stem and legacy_ok.get(name, False) and old_stem in released
        )
        if not released_by_new and not released_by_legacy:
            remaining.append(name)
    if not remaining:
        return remaining, 0, []
    total = math.ceil(len(remaining) / group_size)
    return remaining, total, list(range(total))


def _write_remaining(
    path: str,
    remaining: list[str],
    stem_of: dict[str, str] | None = None,
    legacy_stem_of: dict[str, str] | None = None,
    legacy_unique: dict[str, bool] | None = None,
) -> None:
    """Persist the remaining list as ``{"packages": [...]}`` JSON.

    The additive ``stems`` map lets the retry workflow's
    ``missing_pkgs.py`` match both owner-prefixed zips and unambiguous
    legacy zips. Readers that only consume ``packages`` are unaffected.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload: dict = {"packages": remaining}
    if stem_of:
        stem_payload: dict[str, dict[str, object] | str] = {}
        for name in remaining:
            if name not in stem_of:
                continue
            legacy_stem = (
                legacy_stem_of.get(name) if legacy_stem_of else None
            )
            unique = bool(legacy_unique and legacy_unique.get(name, False))
            stem_payload[name] = {
                "package_stem": stem_of[name],
                "legacy_package_stem": legacy_stem,
                "legacy_unique": unique,
            }
        if stem_payload:
            payload["stems"] = stem_payload
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--list",
        required=True,
        help="Path to a package-list JSON (e.g. .github/packages/full.json).",
    )
    p.add_argument(
        "--released-assets",
        required=False,
        default=None,
        help=(
            "Path to a file containing released asset filenames, one "
            "per line. Missing or empty file = no skips."
        ),
    )
    p.add_argument(
        "--platform",
        required=True,
        choices=["alpine", "debian"],
        help="Platform slug used in zip filenames.",
    )
    p.add_argument(
        "--group-size",
        required=True,
        type=int,
        help="Max packages per shard (matches main workflow's input).",
    )
    p.add_argument(
        "--remaining-out",
        required=True,
        help="Where to write the ordered remaining list as JSON.",
    )
    p.add_argument(
        "--output-key-prefix",
        default=None,
        help=(
            "Prefix for the emitted KEY=VALUE lines. Defaults to "
            "the platform slug, yielding e.g. ``debian_total`` and "
            "``debian_groups``."
        ),
    )
    args = p.parse_args(argv)

    prefix = args.output_key_prefix or args.platform

    declared_pairs = _load_packages(_resolve_list_path(args.list))
    declared = [name for name, _stem, _legacy, _unique in declared_pairs]
    stem_of = {name: stem for name, stem, _legacy, _unique in declared_pairs}
    legacy_stem_of = {
        name: legacy for name, _stem, legacy, _unique in declared_pairs
    }
    legacy_unique = {
        name: unique for name, _stem, _legacy, unique in declared_pairs
    }
    released_all = _load_released_assets(args.released_assets)
    released = _released_names(released_all, args.platform)

    remaining, total, groups = compute_plan(
        declared,
        released,
        args.group_size,
        stem_of=stem_of,
        legacy_stem_of=legacy_stem_of,
        legacy_unique=legacy_unique,
    )

    _write_remaining(
        args.remaining_out,
        remaining,
        stem_of=stem_of,
        legacy_stem_of=legacy_stem_of,
        legacy_unique=legacy_unique,
    )

    print(f"{prefix}_total={total}")
    print(f"{prefix}_groups={json.dumps(groups)}")

    print(
        f"[plan_remaining] list={os.path.basename(args.list)} "
        f"platform={args.platform} declared={len(declared)} "
        f"released_assets={len(released)} "
        f"skipped={len(declared) - len(remaining)} "
        f"remaining={len(remaining)} shards={total} "
        f"(group_size={args.group_size}) -> {args.remaining_out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
