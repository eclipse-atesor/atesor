#!/usr/bin/env python3
r"""Emit a GitHub Actions matrix plan for a sharded package list.

Reads a package list JSON and prints two ``KEY=VALUE`` lines suitable
for ``$GITHUB_OUTPUT``:

    total=<N>                  # number of shards (chunks of group_size)
    groups=[0, 1, ..., N-1]    # JSON array, consumable via fromJSON()

Example::

    python .github/scripts/plan_shards.py \\
        --list full --group-size 50 >> "$GITHUB_OUTPUT"

then::

    strategy:
      matrix:
        shard: ${{ fromJSON(needs.plan-full.outputs.groups) }}

An empty package list emits ``total=0`` and ``groups=[]`` — downstream
jobs are then skipped naturally because their matrix has no entries.
"""

import argparse
import json
import math
import os
import sys

_PACKAGES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "packages")
)


def _resolve_list_path(value: str) -> str:
    """Resolve a bare name under ``.github/packages/<name>.json``."""
    if os.sep in value or value.endswith(".json"):
        return value
    return os.path.join(_PACKAGES_DIR, f"{value}.json")


def _load_package_count(path: str) -> int:
    """Return the number of packages in the list at ``path``.

    Accepts both the schema used by ``.github/packages/*.json``
    (``{"packages": [...]}``) and a bare list, for robustness.

    Raises:
        ValueError: If the JSON is neither an object nor a list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        pkgs = data.get("packages", [])
    elif isinstance(data, list):
        pkgs = data
    else:
        raise ValueError(
            f"Unexpected JSON shape in {path}: {type(data).__name__}"
        )
    return len(pkgs)


def compute_plan(count: int, group_size: int) -> tuple[int, list[int]]:
    """Return ``(total_shards, group_indices)`` for ``count`` packages.

    A ``count`` of 0 or less gives ``(0, [])``.

    Raises:
        ValueError: If ``group_size`` is less than 1.
    """
    if group_size < 1:
        raise ValueError(f"group-size must be >= 1 (got {group_size})")
    if count <= 0:
        return 0, []
    total = math.ceil(count / group_size)
    return total, list(range(total))


def main(argv: list[str]) -> int:
    """Run the CLI entry point.

    Returns:
        Process exit code: always 0. argparse exits with 2 on invalid
        arguments; an unreadable list or a bad ``--group-size`` raises.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        required=True,
        help="Package list name (e.g. 'full') or path to a JSON file.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=50,
        help="Max packages per shard (default: 50).",
    )
    parser.add_argument(
        "--output-key-prefix",
        default="",
        help=(
            "Optional prefix for the emitted KEY=VALUE lines. "
            "With ``--output-key-prefix=debian`` the keys become "
            "``debian_total`` and ``debian_groups``."
        ),
    )
    args = parser.parse_args(argv)

    list_path = _resolve_list_path(args.list)
    count = _load_package_count(list_path)
    total, groups = compute_plan(count, args.group_size)

    prefix = f"{args.output_key_prefix}_" if args.output_key_prefix else ""
    print(f"{prefix}total={total}")
    print(f"{prefix}groups={json.dumps(groups)}")
    # Stderr summary for the CI log (won't pollute $GITHUB_OUTPUT).
    print(
        f"[plan_shards] list={list_path} packages={count} "
        f"group_size={args.group_size} shards={total}"
        + (f" prefix={args.output_key_prefix}" if prefix else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
