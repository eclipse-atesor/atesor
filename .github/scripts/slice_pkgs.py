#!/usr/bin/env python3
"""Slice a remaining-package JSON down to a single shard's chunk.

Used by the per-shard jobs in ``batch-port`` (``batch-port-retry``
slices with ``missing_pkgs.py`` instead) to turn a
``{"packages": [...]}`` plan (produced by
``plan_remaining.py``) into the exact positional package list that
``batch_test.py --package`` should run for shard ``--shard-index`` of
``--shard-total``.

Chunking matches ``batch_test._apply_shard`` and
``missing_pkgs._apply_shard``: contiguous slices of size
``ceil(N / total)``, every name appears in exactly one shard.

Output:

* When ``--output-key`` is set, emits ``KEY=<value>`` to stdout
  suitable for ``$GITHUB_OUTPUT``.
* Otherwise prints the slice in ``--format`` (``space`` or ``lines``).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def _load(path: str) -> list[str]:
    """Load a list of package names from a remaining-plan JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    pkgs = data.get("packages", [])
    out: list[str] = []
    for entry in pkgs:
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict) and entry.get("name"):
            out.append(entry["name"])
    return out


def apply_shard(
    names: list[str],
    shard_index: int,
    shard_total: int,
) -> list[str]:
    """Return the contiguous chunk of ``names`` for one shard.

    The chunks have ``ceil(N/total)`` names, mirroring batch_test.
    """
    if shard_total <= 1:
        return list(names)
    n = len(names)
    chunk = -(-n // shard_total)  # ceil
    start = shard_index * chunk
    end = start + chunk
    return list(names[start:end])


def main(argv: Optional[list[str]] = None) -> int:
    """Run the CLI entry point.

    Returns:
        Process exit code: 0 on success, 2 on an invalid shard index or
        shard total.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from",
        dest="src",
        required=True,
        help="Path to remaining-<platform>.json",
    )
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-total", type=int, required=True)
    p.add_argument("--format", choices=["space", "lines"], default="space")
    p.add_argument(
        "--output-key",
        default=None,
        help="When set, emit `KEY=<value>` for $GITHUB_OUTPUT.",
    )
    args = p.parse_args(argv)

    if args.shard_total < 1:
        print(
            f"[ERROR] --shard-total must be >= 1 (got {args.shard_total})",
            file=sys.stderr,
        )
        return 2
    if args.shard_index < 0 or args.shard_index >= args.shard_total:
        print(
            f"[ERROR] --shard-index={args.shard_index} out of range "
            f"[0, {args.shard_total}).",
            file=sys.stderr,
        )
        return 2

    names = _load(args.src)
    slice_ = apply_shard(names, args.shard_index, args.shard_total)

    if args.format == "space":
        value = " ".join(slice_)
    else:
        value = "\n".join(slice_)

    if args.output_key:
        print(f"{args.output_key}={value}")
    else:
        print(value)

    print(
        f"[slice_pkgs] from={args.src} "
        f"shard={args.shard_index + 1}/{args.shard_total} "
        f"total={len(names)} slice={len(slice_)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
