#!/usr/bin/env bash
#############################################################################
# Copyright (c) 2026 10xEngineers
#
# Author: Akif Ejaz <akif.ejaz@10xengineers.ai>
# This program and the accompanying materials are made available under the
# terms of the MIT License which is available at
# https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
#############################################################################
#
# Validates the built .deb in a clean ubuntu:22.04 container:
#   1. install + dpkg status
#   2. no secrets shipped (.env files / real-looking API keys)
#   3. DEBIAN/md5sums integrity verification
#   4. CLI contract: --version (matches control), --help, no-arg exit 1
#   5. --setup-only degrades cleanly without a Docker daemon (exit 1)
#   6. end-to-end recipe-cache hit via ATESOR_HOME (no docker, no keys):
#      proves seed data, env setup, and state-dir seeding all work
#   7. /opt install tree stays byte-pristine after running the CLI
#   8. bundled deps import
#   9. dpkg -r leaves no orphans in /opt or /usr/bin
#
# Heavy runtime deps (docker.io / qemu-user-static) are declared but
# skipped here via --force-depends so the validation stays fast.
#
# Usage: packaging/deb/validate_deb.sh [image]
#
# `image` defaults to ubuntu:22.04 and should match the base image the
# package was built against (build_deb.sh's first argument).

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "${here}/../.." && pwd)
out_dir="${repo_root}/dist"
image="${1:-ubuntu:22.04}"

version=$(cd "${repo_root}" && PYTHONPATH=. python3 -c \
    "from src import __version__; print(__version__)" 2>/dev/null || echo 0.1.0)

deb_path=$(ls -t "${out_dir}/atesor-ai_${version}"_*.deb 2>/dev/null | head -1)
if [ -z "${deb_path}" ]; then
    echo "ERROR: no dist/atesor-ai_${version}_*.deb found." \
         "Run build_deb.sh first." >&2
    exit 1
fi
deb=$(basename "${deb_path}")

echo ">> Validating ${deb} in a clean ${image} container"
docker run --rm -e DEB="${deb}" -e VERSION="${version}" \
    -v "${out_dir}":/out:ro "${image}" bash -c '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends python3 ca-certificates \
        >/dev/null

    echo "--- [1/9] install ---"
    # --force-depends lets the intentionally-absent heavy runtime deps
    # (docker.io / qemu-user-static) downgrade to warnings so this
    # validation stays fast. Do NOT mask the exit code: a genuine
    # unpack or postinst failure must still fail the validation.
    dpkg -i --force-depends "/out/${DEB}"
    dpkg -s atesor-ai | grep -q "^Status: install ok installed" \
        || { echo "FAIL: package not properly installed"; exit 1; }

    echo "--- [2/9] no secrets shipped ---"
    if dpkg-deb -c "/out/${DEB}" | grep -E "/\.env$"; then
        echo "FAIL: a .env file is inside the package"; exit 1
    fi
    if grep -RInE \
        "^(GOOGLE|OPENAI|OPENROUTER|GH|GIT)_?(API_)?(KEY|TOKEN)=..+" \
        /opt/atesor-ai/app --include="*" \
        --exclude=".env-example" --exclude="README.md"; then
        echo "FAIL: real-looking credential assignment in app tree"
        exit 1
    fi
    test -f /opt/atesor-ai/app/.env-example \
        || { echo "FAIL: .env-example missing"; exit 1; }

    echo "--- [3/9] md5sums integrity ---"
    test -s /var/lib/dpkg/info/atesor-ai.md5sums \
        || { echo "FAIL: md5sums missing from package"; exit 1; }
    (cd / && md5sum --quiet -c /var/lib/dpkg/info/atesor-ai.md5sums) \
        || { echo "FAIL: installed files differ from md5sums"; exit 1; }

    echo "--- [4/9] CLI contract ---"
    command -v atesor-ai
    ver_out=$(atesor-ai --version)
    echo "${ver_out}"
    echo "${ver_out}" | grep -qF "${VERSION}" \
        || { echo "FAIL: --version does not report ${VERSION}"; exit 1; }
    atesor-ai --help | head -6
    set +e
    atesor-ai </dev/null >/dev/null 2>&1
    rc=$?
    set -e
    [ "${rc}" -eq 1 ] \
        || { echo "FAIL: no-arg run must exit 1 (got ${rc})"; exit 1; }

    echo "--- [5/9] --setup-only degrades cleanly without docker ---"
    set +e
    setup_out=$(atesor-ai --setup-only </dev/null 2>&1)
    rc=$?
    set -e
    [ "${rc}" -eq 1 ] \
        || { echo "FAIL: --setup-only without docker must exit 1"; exit 1; }
    echo "${setup_out}" | grep -q "Docker is not running" \
        || { echo "FAIL: missing clear docker error message"; exit 1; }

    echo "--- [6/9] end-to-end recipe-cache hit (no docker, no keys) ---"
    export ATESOR_HOME=/root/atesor-state
    # A cache hit needs the stored repository URL: an entry of another
    # owner with the same name is a miss, and the CLI goes on to ask
    # for keys.
    seed=$(/opt/atesor-ai/venv/bin/python3 - <<PY
import json, re
data = json.load(open("/opt/atesor-ai/app/data/recipe_cache.json"))
for name, entry in data.get("packages", {}).items():
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
        continue
    recipe = entry.get("debian-riscv64") if isinstance(entry, dict) else None
    url = (recipe or {}).get("repo_url") or ""
    if url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1] == name:
        print(name, url)
        break
PY
)
    pkg=${seed%% *}
    url=${seed#* }
    [ -n "${pkg}" ] \
        || { echo "FAIL: no debian seed recipe in bundled cache"; exit 1; }
    echo "    seed package: ${pkg} (${url})"
    # Exit 0: the entry holds a verified verdict. Exit 3: the entry was
    # recorded before artifact verification existed.
    rc=0
    atesor-ai --repo "${url}" --platform debian </dev/null || rc=$?
    [ "${rc}" -eq 0 ] || [ "${rc}" -eq 3 ] \
        || { echo "FAIL: cache hit exited ${rc}"; exit 1; }
    recipe="${ATESOR_HOME}/workspace/output/${pkg}_recipe.md"
    test -s "${recipe}" \
        || { echo "FAIL: cache hit did not write ${recipe}"; exit 1; }
    echo "    recipe written: ${recipe}"
    unset ATESOR_HOME

    echo "--- [7/9] /opt stays pristine after running the CLI ---"
    dirty=$( (cd / && md5sum -c /var/lib/dpkg/info/atesor-ai.md5sums \
        2>/dev/null | grep -v ": OK$") || true)
    [ -z "${dirty}" ] \
        || { echo "FAIL: CLI runs modified /opt:"; echo "${dirty}"; exit 1; }
    pyc=$(find /opt/atesor-ai/app -name "__pycache__" -o -name "*.pyc" \
        | head -5)
    [ -z "${pyc}" ] \
        || { echo "FAIL: bytecode written into /opt: ${pyc}"; exit 1; }

    echo "--- [8/9] bundled deps import ---"
    /opt/atesor-ai/venv/bin/python3 -c \
        "import langchain, langgraph, docker, pydantic, \
google.generativeai; print(\"bundled deps import OK\")"

    echo "--- [9/9] clean removal ---"
    dpkg -r atesor-ai >/dev/null 2>&1
    [ ! -e /usr/bin/atesor-ai ] \
        || { echo "FAIL: launcher left behind after removal"; exit 1; }
    # postrm removes the whole tree (runtime bytecode caches are not
    # package-owned); the directory itself must be gone.
    [ ! -d /opt/atesor-ai ] \
        || { echo "FAIL: /opt/atesor-ai still exists after removal:"; \
             find /opt/atesor-ai | head -5; exit 1; }
    echo "removal clean"
'
echo ">> Validation OK: ${deb_path}"
