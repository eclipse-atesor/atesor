# Atesor AI — Debian package

Builds a self-contained `.deb` for **Ubuntu 22.04 / Debian-based** systems
(targets `python3.10`; architecture is derived from the build container
via `dpkg --print-architecture`, so an arm64 host produces an honest
`_arm64.deb`).

## Layout when installed

```
/opt/atesor-ai/app/     application code + Dockerfiles + seed data
/opt/atesor-ai/app/.env-example   reference for all supported env vars
/opt/atesor-ai/app/BUILD_INFO     provenance: build date, pip freeze
/opt/atesor-ai/venv/    bundled virtualenv with all Python deps
/usr/bin/atesor-ai      launcher (venv python + PYTHONPATH, refuses to
                        run against a damaged install tree)
```

Runtime state (workspace, logs, recipe cache, learned examples) is written
to `$ATESOR_HOME` if set, otherwise `~/.local/share/atesor-ai` — never into
`/opt`. Bytecode writes into `/opt` are disabled twice over: the launcher
exports `PYTHONDONTWRITEBYTECODE=1`, and a `sitecustomize.py` baked into
the venv sets `sys.dont_write_bytecode` for *any* direct invocation of
the bundled interpreter. The install tree therefore stays byte-identical
to the package (`dpkg -V atesor-ai` stays clean), and a `postrm` script
guarantees `/opt/atesor-ai` is removed wholesale on uninstall. Per-run
debug logs land under `<state>/workspace/logs/`.

## Build

```bash
packaging/deb/build_deb.sh            # -> dist/atesor-ai_<version>_<arch>.deb
```

Requires Docker on the build host. The package is built inside a clean
`ubuntu:22.04` container so the bundled virtualenv matches the target
interpreter. The full container build output is persisted next to the
artifact as `dist/atesor-ai_<version>_build.log`.

Security gates during the build:

- the developer tree's `.env` (real API keys) is never staged; the build
  **fails** if any `.env` ends up in the package root (`.env-example` is
  the only allowed environment file);
- `DEBIAN/md5sums` is generated so installed files are verifiable with
  `dpkg -V atesor-ai`.

## Validate

```bash
packaging/deb/validate_deb.sh [image]   # full contract check in a clean container
```

`image` defaults to `ubuntu:22.04` and should match the base image the
package was built against.

## CI / releasing

`.github/workflows/package-deb.yml` runs the same pipeline in CI:
test suite → lint → `build_deb.sh` → `validate_deb.sh` → artifacts.

### Cutting a release (tag-triggered)

Push a `vX.Y.Z` tag; that is the whole release procedure:

```bash
git tag v1.2.0
git push origin v1.2.0
```

The workflow derives the version from the tag (`v1.2.0` → `1.2.0`),
writes it into `src/__init__.py` **before** building so the control
file, the artifact filename and `atesor-ai --version` cannot disagree,
then publishes a GitHub release `v1.2.0` containing:

- `atesor-ai_1.2.0_amd64.deb` and its `.sha256`
- release notes with install / setup / porting / teardown commands

Pre-release suffixes work too (`v0.2.0-rc1`). A tag that is not a valid
version (or contains shell metacharacters) fails the job before
anything is built. The release is **published**, not drafted — the tag
is the explicit intent. Re-pushing the same tag updates the notes and
re-uploads the assets.

Because the tag sets the version, bump nothing by hand: `src/__init__.py`
is rewritten during the run.

### Manual runs

Actions → **Package .deb** → *Run workflow*. Inputs: `version` (empty =
use the source's version), `base_image` (default `ubuntu:22.04`, used
for both build and validation), and `create_release` (produces a
*draft* release). Every run — tagged or manual — uploads the `.deb`,
its checksum, and the full build log as a workflow artifact.

Checks, in order: install status → no shipped secrets → md5sums
integrity → CLI contract (`--version` matches control, `--help`,
no-arg exits 1) → `--setup-only` degrades cleanly without a Docker
daemon → an end-to-end recipe-cache hit through `$ATESOR_HOME` (no
docker, no API keys needed — proves seed data, env handling, and
state-dir seeding) → `/opt` still byte-pristine after running the
CLI → bundled imports → clean `dpkg -r` removal.

## Install (on a target machine)

```bash
sudo apt-get install ./atesor-ai_<version>_<arch>.deb
```

`apt` resolves the declared dependencies: `python3`, a Docker engine
(`docker.io` | `docker-ce`), `qemu-user-static`, `binfmt-support`, and
`ca-certificates` (TLS to the LLM providers). `git` is recommended.

## Prerequisites that cannot be bundled

Atesor AI orchestrates Docker; these must exist on the target host:

- a running Docker daemon (and your user in the `docker` group),
- `qemu-user-static` + `binfmt-support` for `riscv64` emulation,
- an LLM API key (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, or
  `OPENROUTER_API_KEY`) — see `/opt/atesor-ai/app/.env-example`.

## Notes

- Targeting another distro/python version requires rebuilding with the
  matching base image (e.g. `packaging/deb/build_deb.sh debian:12`).
- The runtime dependency set is pinned in `requirements-runtime.txt`
  (dev-only tools are excluded). The exact resolved versions of a given
  build are recorded in the package's `BUILD_INFO`.
- `batch_test.py` (the CI batch sweep driver) is intentionally NOT
  packaged: it assumes a source checkout and spawns `python3 main.py`
  workers. The installed CLI covers single-package porting; batch runs
  belong to the repository/CI workflow.
