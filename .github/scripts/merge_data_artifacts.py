#!/usr/bin/env python3
"""Merge data/ updates from shard artifacts into one canonical data/ tree.

With --merge-caches, the script merges recipe cache files by meaning:
the newest recipe of each package and sandbox wins. Use that mode to
resolve a git conflict in data/recipe_cache.json, which .gitattributes
keeps git from merging line by line.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class _Pairs(list):
    """A JSON object read as (key, value) pairs, so duplicates stay."""


def _plain(value: Any) -> Any:
    """Turn _Pairs back into dicts; a later duplicate key wins."""
    if isinstance(value, _Pairs):
        return {key: _plain(item) for key, item in value}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_recipe_cache(path: Path) -> dict[str, Any]:
    """Load a recipe cache, and merge any duplicate package or sandbox.

    A line-by-line git merge can leave a package, or a sandbox of it,
    twice in the file. json.load() keeps only the last copy, which can
    be the older recipe. Here the duplicates meet in _prefer_recipe(),
    so the newest recipe wins.
    """
    with path.open("r", encoding="utf-8") as fh:
        top = json.load(fh, object_pairs_hook=_Pairs)
    out: dict[str, Any] = {}
    for key, value in top:
        if key != "packages" or not isinstance(value, _Pairs):
            out[key] = _plain(value)
            continue
        packages = out.setdefault("packages", {})
        for pkg_name, sandboxes in value:
            slot = packages.setdefault(pkg_name, {})
            for sandbox_name, recipe in sandboxes:
                recipe = _plain(recipe)
                existing = slot.get(sandbox_name)
                slot[sandbox_name] = (
                    recipe
                    if existing is None
                    else _prefer_recipe(existing, recipe)
                )
    return out


def _save_json(path: Path, data: dict[str, Any]) -> None:
    # The same format as the writers in src/memory.py, so a CI write and
    # a local write of the same data give the same file.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _parse_time(value: str | None) -> datetime | None:
    # Accept "2026-06-18T12:45:04.618860" and "...Z" variants. Naive
    # timestamps come back as UTC-aware so a later _prefer_recipe()
    # comparison never mixes offset-naive and offset-aware datetimes.
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _prefer_recipe(
    current: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    c_dt = _parse_time(current.get("last_built"))
    i_dt = _parse_time(incoming.get("last_built"))
    if c_dt and i_dt:
        return incoming if i_dt >= c_dt else current
    if i_dt and not c_dt:
        return incoming
    if not i_dt and c_dt:
        return current
    # Fall back to incoming on tie/unknown; this lets later shards win.
    return incoming


def _merge_recipe_cache(
    base: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    out = dict(base)
    out.setdefault(
        "version", incoming.get("version", base.get("version", "2.0"))
    )
    out.setdefault("packages", {})

    base_pkgs = out["packages"]
    inc_pkgs = incoming.get("packages", {})

    for pkg_name, sandboxes in inc_pkgs.items():
        base_pkgs.setdefault(pkg_name, {})
        for sandbox_name, recipe in sandboxes.items():
            existing = base_pkgs[pkg_name].get(sandbox_name)
            if existing is None:
                base_pkgs[pkg_name][sandbox_name] = recipe
            else:
                base_pkgs[pkg_name][sandbox_name] = _prefer_recipe(
                    existing, recipe
                )
    return out


def _prefer_example(
    current: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    c_source = (current.get("source") or "").lower()
    i_source = (incoming.get("source") or "").lower()
    # Preserve curated/manual entries over auto-learned entries.
    if c_source == "manual" and i_source != "manual":
        return current
    if i_source == "manual" and c_source != "manual":
        return incoming

    c_dt = _parse_time(current.get("timestamp"))
    i_dt = _parse_time(incoming.get("timestamp"))
    if c_dt and i_dt:
        return incoming if i_dt >= c_dt else current
    if i_dt and not c_dt:
        return incoming
    if not i_dt and c_dt:
        return current
    return incoming


def _example_id(ex: dict[str, Any]) -> str:
    ex_id = ex.get("id")
    if ex_id:
        return str(ex_id)
    # Defensive fallback if malformed entry lacks id.
    return json.dumps(ex, sort_keys=True)


def _merge_examples_file(
    base: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    out = dict(base)
    merged: dict[str, dict[str, Any]] = {}

    for ex in base.get("examples", []):
        merged[_example_id(ex)] = ex
    for ex in incoming.get("examples", []):
        ex_id = _example_id(ex)
        existing = merged.get(ex_id)
        if existing is None:
            merged[ex_id] = ex
        else:
            merged[ex_id] = _prefer_example(existing, ex)

    out["examples"] = [merged[k] for k in sorted(merged.keys())]
    out.setdefault(
        "version", incoming.get("version", base.get("version", "2.0"))
    )
    return out


def _merge_recipe_cache_files(
    repo_data_dir: Path, artifact_root: Path
) -> bool:
    target = repo_data_dir / "recipe_cache.json"
    if not target.exists():
        return False

    merged = _load_recipe_cache(target)
    changed = False

    for path in artifact_root.rglob("data/recipe_cache.json"):
        incoming = _load_recipe_cache(path)
        before = json.dumps(merged, sort_keys=True)
        merged = _merge_recipe_cache(merged, incoming)
        after = json.dumps(merged, sort_keys=True)
        if after != before:
            changed = True

    if changed:
        _save_json(target, merged)
    return changed


def _merge_examples_files(repo_data_dir: Path, artifact_root: Path) -> bool:
    examples_dir = repo_data_dir / "examples"
    if not examples_dir.exists():
        return False

    changed = False
    for target in examples_dir.glob("*_examples.json"):
        merged = _load_json(target)
        target_changed = False
        suffix = f"data/examples/{target.name}"
        for path in artifact_root.rglob(target.name):
            if not str(path).endswith(suffix):
                continue
            incoming = _load_json(path)
            before = json.dumps(merged, sort_keys=True)
            merged = _merge_examples_file(merged, incoming)
            after = json.dumps(merged, sort_keys=True)
            if after != before:
                target_changed = True
        if target_changed:
            _save_json(target, merged)
            changed = True
    return changed


def merge_three_way(
    base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any]
) -> dict[str, Any]:
    """Merge two edits of one recipe cache, as a git merge driver does.

    A recipe that one side removed stays removed, unless the other side
    changed it. A recipe that only one side changed takes that change.
    When both sides changed it, the newest recipe wins.

    Args:
        base: The common ancestor.
        ours: The current branch.
        theirs: The branch that is merged in.

    Returns:
        The merged recipe cache.
    """
    base_pkgs = base.get("packages", {})
    ours_pkgs = ours.get("packages", {})
    theirs_pkgs = theirs.get("packages", {})
    out: dict[str, Any] = {
        "version": ours.get("version", theirs.get("version", "2.0")),
        "packages": {},
    }
    names = list(ours_pkgs)
    names.extend(name for name in theirs_pkgs if name not in ours_pkgs)
    for name in names:
        o_sbs = ours_pkgs.get(name, {})
        t_sbs = theirs_pkgs.get(name, {})
        b_sbs = base_pkgs.get(name, {})
        merged: dict[str, Any] = {}
        sandboxes = list(o_sbs)
        sandboxes.extend(sb for sb in t_sbs if sb not in o_sbs)
        for sb in sandboxes:
            o, t, b = o_sbs.get(sb), t_sbs.get(sb), b_sbs.get(sb)
            if o is not None and t is not None:
                if o == t or t == b:
                    merged[sb] = o
                elif o == b:
                    merged[sb] = t
                else:
                    merged[sb] = _prefer_recipe(o, t)
            elif o is not None:
                if o != b:
                    merged[sb] = o
            elif t != b:
                merged[sb] = t
        if merged:
            out["packages"][name] = merged
    return out


def _load_cache_or_empty(path: Path) -> dict[str, Any]:
    """Load a recipe cache; an empty file (no ancestor) is an empty cache."""
    if path.stat().st_size == 0:
        return {"version": "2.0", "packages": {}}
    return _load_recipe_cache(path)


def merge_cache_files(paths: list[Path]) -> dict[str, Any]:
    """Merge recipe cache files in order; the newest recipe wins.

    Args:
        paths: The recipe cache files, for example ours and theirs.

    Returns:
        One recipe cache with no duplicate package or sandbox.
    """
    merged: dict[str, Any] = {"version": "2.0", "packages": {}}
    for path in paths:
        merged = _merge_recipe_cache(merged, _load_recipe_cache(path))
    return merged


def main() -> int:
    """Run the CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        help="Root directory where shard artifacts were downloaded.",
    )
    parser.add_argument(
        "--repo-data-dir",
        help="Repository data directory (typically ./data).",
    )
    parser.add_argument(
        "--merge-caches",
        nargs="+",
        metavar="FILE",
        help="Merge these recipe cache files (the newest recipe wins) "
        "into --output.",
    )
    parser.add_argument(
        "--output",
        help="The file that --merge-caches writes.",
    )
    parser.add_argument(
        "--merge-driver",
        nargs=3,
        metavar=("BASE", "OURS", "THEIRS"),
        help="Git merge driver mode (%%O %%A %%B): merge the recipe cache "
        "by meaning and write the result to OURS. See "
        ".github/scripts/setup_git_merge_driver.sh.",
    )
    args = parser.parse_args()

    if args.merge_driver:
        base, ours, theirs = (Path(p) for p in args.merge_driver)
        try:
            merged = merge_three_way(
                _load_cache_or_empty(base),
                _load_recipe_cache(ours),
                _load_recipe_cache(theirs),
            )
        except (OSError, ValueError) as exc:
            # A non-zero exit makes git report a normal conflict.
            print(f"recipe cache merge driver failed: {exc}")
            return 1
        _save_json(ours, merged)
        return 0

    if args.merge_caches:
        if not args.output:
            parser.error("--merge-caches needs --output")
        merged = merge_cache_files([Path(p) for p in args.merge_caches])
        _save_json(Path(args.output), merged)
        print(
            f"Merged {len(args.merge_caches)} recipe caches into "
            f"{args.output}: {len(merged['packages'])} packages"
        )
        return 0
    if not args.artifacts_root or not args.repo_data_dir:
        parser.error("--artifacts-root and --repo-data-dir are required")

    artifact_root = Path(args.artifacts_root)
    repo_data_dir = Path(args.repo_data_dir)

    if not artifact_root.exists():
        print("No artifact root found; nothing to merge.")
        return 0

    rc_changed = _merge_recipe_cache_files(repo_data_dir, artifact_root)
    ex_changed = _merge_examples_files(repo_data_dir, artifact_root)
    print(
        "Merged data artifacts: "
        f"recipe_cache_changed={rc_changed}, examples_changed={ex_changed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
