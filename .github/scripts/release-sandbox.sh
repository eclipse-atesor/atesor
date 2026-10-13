#!/usr/bin/env bash
# release-sandbox.sh — publish the riscv64 sandbox image(s) to Docker Hub.
#
# Tagging scheme (single repo, variant-prefixed):
#   <variant>-latest         moving tag, always points at newest build
#   <variant>-v<MAJOR>.<MINOR>  immutable, one per release
#
# Default action per variant (alpine | debian):
#   1) Discover highest existing `<variant>-v<M>.<n>` on the registry.
#   2) Build a fresh linux/riscv64 image from the matching Dockerfile.
#   3) Push the new build as `<variant>-v<M>.<next>` (immutable) and then
#      `<variant>-latest` (pointer flip). The run stops if the pinned tag
#      already exists.
#
# Version bumping:
#   default      bump MINOR within highest MAJOR  (v1.5 -> v1.6)
#   --bump-major increment MAJOR, MINOR=0          (v1.5 -> v2.0)
#   --version X.Y  explicit override               (e.g. --version 3.2)
#   first ever release -> v1.0
#
# Legacy `:latest` (variant-ambiguous) is intentionally NOT touched. To
# preserve it as a snapshot under the new scheme, run manually, e.g.:
#   docker buildx imagetools create \
#       -t cloudv10x/atesor-sandbox:legacy-latest \
#       cloudv10x/atesor-sandbox:latest
#
# Usage:
#   .github/scripts/release-sandbox.sh                 # release both variants
#   .github/scripts/release-sandbox.sh --alpine        # alpine only
#   .github/scripts/release-sandbox.sh --debian        # debian only
#   .github/scripts/release-sandbox.sh --bump-major    # new major version
#   .github/scripts/release-sandbox.sh --version 2.3   # pin explicit version
#   .github/scripts/release-sandbox.sh --yes           # skip confirmation
#   .github/scripts/release-sandbox.sh --dry-run       # plan only, no changes
#   .github/scripts/release-sandbox.sh --repo foo/bar  # override target repo
#   .github/scripts/release-sandbox.sh --no-cache      # docker build --no-cache
#
# Requirements: bash 4+, docker (with buildx), curl, python3.
# A push needs a Docker Hub login. ATESOR_DOCKERHUB_REPO sets the default
# repo. Destructive: the run replaces `<variant>-latest` on the registry.
# It asks first unless --yes or --dry-run is set. It can also register
# riscv64 binfmt_misc on the host with a privileged container.

set -euo pipefail

# Defaults
REPO="${ATESOR_DOCKERHUB_REPO:-cloudv10x/atesor-sandbox}"
PLATFORM="linux/riscv64"
DOCKERFILE_ALPINE="Dockerfile"
DOCKERFILE_DEBIAN="Dockerfile.debian"

DO_ALPINE=1
DO_DEBIAN=1
ASSUME_YES=0
DRY_RUN=0
NO_CACHE=0
BUMP_MAJOR=0
EXPLICIT_VERSION=""   # set via --version X.Y

# Pretty logging
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
    C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'; C_CYAN=$'\033[36m'
else
    C_RESET=""; C_BOLD=""; C_DIM=""
    C_RED=""; C_GREEN=""; C_YELLOW=""; C_CYAN=""
fi

log()  { printf '%s[%s]%s %s\n'   "$C_CYAN"   "$(date +%H:%M:%S)" "$C_RESET" "$*"; }
ok()   { printf '%s[ OK ]%s %s\n' "$C_GREEN"  "$C_RESET" "$*"; }
warn() { printf '%s[WARN]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()  { printf '%s[FAIL]%s %s\n' "$C_RED"    "$C_RESET" "$*" >&2; exit 1; }
hr()   { printf '%s%s%s\n' "$C_DIM" "------------------------------------------------------------" "$C_RESET"; }

# CLI parsing
usage() { sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --alpine)      DO_ALPINE=1; DO_DEBIAN=0 ;;
        --debian)      DO_ALPINE=0; DO_DEBIAN=1 ;;
        --both)        DO_ALPINE=1; DO_DEBIAN=1 ;;
        --yes|-y)      ASSUME_YES=1 ;;
        --dry-run)     DRY_RUN=1 ;;
        --no-cache)    NO_CACHE=1 ;;
        --bump-major)  BUMP_MAJOR=1 ;;
        --version)     EXPLICIT_VERSION="${2:?--version needs X.Y}"; shift ;;
        --version=*)   EXPLICIT_VERSION="${1#*=}" ;;
        --repo)        REPO="${2:?--repo needs a value}"; shift ;;
        --repo=*)      REPO="${1#*=}" ;;
        -h|--help)     usage 0 ;;
        *)             warn "Unknown arg: $1"; usage 2 ;;
    esac
    shift
done

(( DO_ALPINE || DO_DEBIAN )) || die "Nothing to do: --alpine and --debian both disabled."

if (( BUMP_MAJOR )) && [[ -n "$EXPLICIT_VERSION" ]]; then
    die "--bump-major and --version are mutually exclusive."
fi

if [[ -n "$EXPLICIT_VERSION" ]]; then
    [[ "$EXPLICIT_VERSION" =~ ^[0-9]+\.[0-9]+$ ]] \
        || die "--version must look like 'MAJOR.MINOR' (e.g. 2.3), got: $EXPLICIT_VERSION"
fi

# Anchor to repo root (parent of .github) so Docker build context is correct.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Preflight
preflight() {
    log "Preflight checks (repo=$REPO, platform=$PLATFORM)"

    command -v docker  >/dev/null || die "docker not found in PATH."
    command -v curl    >/dev/null || die "curl not found in PATH."
    command -v python3 >/dev/null || die "python3 not found in PATH."
    docker buildx version >/dev/null 2>&1 || die "docker buildx not available."

    (( DO_ALPINE )) && [[ ! -f "$DOCKERFILE_ALPINE" ]] && die "Missing $DOCKERFILE_ALPINE in $REPO_ROOT"
    (( DO_DEBIAN )) && [[ ! -f "$DOCKERFILE_DEBIAN" ]] && die "Missing $DOCKERFILE_DEBIAN in $REPO_ROOT"

    # binfmt_misc registration for riscv64 (idempotent).
    if ! docker buildx inspect --bootstrap 2>/dev/null \
            | grep -qi "linux/riscv64"; then
        log "Registering riscv64 binfmt via tonistiigi/binfmt"
        if (( DRY_RUN )); then
            log "[dry-run] would run: docker run --privileged --rm tonistiigi/binfmt --install riscv64"
        else
            docker run --privileged --rm tonistiigi/binfmt --install riscv64 \
                >/dev/null || die "Failed to register riscv64 binfmt."
        fi
    fi

    # Docker Hub login probe: look for an entry in ~/.docker/config.json.
    local cfg="${DOCKER_CONFIG:-$HOME/.docker}/config.json"
    if [[ -f "$cfg" ]] \
        && python3 -c "
import json, sys
try:
    cfg = json.load(open('$cfg'))
except Exception:
    sys.exit(1)
auths = cfg.get('auths') or {}
keys = list(auths) + list(cfg.get('credHelpers') or {})
sys.exit(0 if any('docker.io' in k or 'index.docker.io' in k for k in keys) else 1)
" 2>/dev/null; then
        ok "Docker Hub credentials present in $cfg"
    else
        warn "No Docker Hub credentials detected. Run: docker login"
        if (( ! ASSUME_YES )); then
            read -r -p "Continue anyway? [y/N] " ans
            [[ "$ans" =~ ^[Yy]$ ]] || die "Aborted by user."
        fi
    fi

    ok "Preflight passed"
}

# Registry helpers

# Echo all tags from Docker Hub for $REPO, one per line. Paginated.
fetch_tags() {
    local url="https://hub.docker.com/v2/repositories/${REPO}/tags?page_size=100"
    local page tags
    tags=""
    while [[ -n "$url" && "$url" != "null" ]]; do
        page="$(curl -fsSL "$url" 2>/dev/null || true)"
        [[ -z "$page" ]] && break
        tags+="$(printf '%s' "$page" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for r in d.get("results", []):
    print(r["name"])
')"$'\n'
        url="$(printf '%s' "$page" | python3 -c '
import json, sys
print((json.load(sys.stdin).get("next") or "").strip())
')"
    done
    printf '%s' "$tags"
}

# True if `<tag>` exists on the remote.
remote_tag_exists() {
    local tag="$1"
    docker buildx imagetools inspect "${REPO}:${tag}" >/dev/null 2>&1
}

# Compute the next version for a variant.
#
# Reads existing `<variant>-v<MAJOR>.<MINOR>` tags from the registry and
# echoes the next version per the active bump mode. Writes the result on
# stdout as "MAJOR.MINOR".
next_version_for() {
    local variant="$1" tags
    tags="$(fetch_tags)"

    # Echo result via python3 so version arithmetic is unambiguous.
    BUMP_MAJOR="$BUMP_MAJOR" EXPLICIT_VERSION="$EXPLICIT_VERSION" \
    VARIANT="$variant" TAGS="$tags" python3 <<'PY'
import os, re, sys
variant = os.environ["VARIANT"]
explicit = os.environ.get("EXPLICIT_VERSION", "").strip()
bump_major = os.environ.get("BUMP_MAJOR") == "1"
tags = os.environ.get("TAGS", "").splitlines()

pat = re.compile(rf"^{re.escape(variant)}-v(\d+)\.(\d+)$")
versions = []
for t in tags:
    m = pat.match(t.strip())
    if m:
        versions.append((int(m.group(1)), int(m.group(2))))

if explicit:
    print(explicit)
    sys.exit(0)

if not versions:
    print("1.0")
    sys.exit(0)

max_major = max(v[0] for v in versions)
if bump_major:
    print(f"{max_major + 1}.0")
else:
    max_minor_in_major = max(v[1] for v in versions if v[0] == max_major)
    print(f"{max_major}.{max_minor_in_major + 1}")
PY
}

# Registry-side copy: src_tag -> dst_tag (no local pull/push). Idempotent.
registry_copy() {
    local src="$1" dst="$2"
    log "Archiving ${REPO}:${src} -> ${REPO}:${dst} (registry-side copy)"
    if (( DRY_RUN )); then
        log "[dry-run] docker buildx imagetools create -t ${REPO}:${dst} ${REPO}:${src}"
        return 0
    fi
    docker buildx imagetools create \
        -t "${REPO}:${dst}" "${REPO}:${src}" \
        || die "Failed to archive ${REPO}:${src} as ${REPO}:${dst}"
    ok "Archived ${REPO}:${src} as ${REPO}:${dst}"
}

# Per-variant release
release_variant() {
    local variant="$1" dockerfile="$2"
    hr
    log "${C_BOLD}Releasing variant: ${variant}${C_RESET}"
    log "Dockerfile: $dockerfile"

    local next_ver pinned_tag moving_tag
    next_ver="$(next_version_for "$variant")"
    [[ -n "$next_ver" ]] || die "Failed to compute next version for $variant"
    pinned_tag="${variant}-v${next_ver}"
    moving_tag="${variant}-latest"

    log "Next version       : v${next_ver}"
    log "Pinned tag to push : ${REPO}:${pinned_tag}"
    log "Moving tag to push : ${REPO}:${moving_tag}"

    # Safety: refuse to clobber an existing pinned tag (those are immutable).
    if remote_tag_exists "$pinned_tag"; then
        die "${REPO}:${pinned_tag} already exists. Pinned tags are immutable. \
Use --bump-major or --version X.Y to choose a different version."
    fi

    # Step A — log whether `<variant>-latest` exists. The new build
    # replaces it, and nothing archives it first. An earlier run of this
    # script pushed that image under its own pinned tag too. To keep a
    # moving tag that has no pinned tag, copy it by hand first (see the
    # header).
    if remote_tag_exists "$moving_tag"; then
        log "Found existing ${REPO}:${moving_tag} — will be replaced by new build."
    else
        warn "${REPO}:${moving_tag} does not exist on registry yet — first release for this variant."
    fi

    # Step B — build the new image locally, tagged with both names.
    local build_args=(
        buildx build
        --platform "$PLATFORM"
        --file "$dockerfile"
        --tag "${REPO}:${pinned_tag}"
        --tag "${REPO}:${moving_tag}"
        --load
    )
    (( NO_CACHE )) && build_args+=(--no-cache)
    build_args+=(.)

    log "Building: docker ${build_args[*]}"
    if (( DRY_RUN )); then
        log "[dry-run] skip build"
    else
        docker "${build_args[@]}" || die "Build failed for ${variant}"
        ok "Built ${REPO}:${pinned_tag} and ${REPO}:${moving_tag}"
    fi

    # Step C — push pinned first (immutable record), then moving (pointer flip).
    for tag in "$pinned_tag" "$moving_tag"; do
        log "Pushing ${REPO}:${tag}"
        if (( DRY_RUN )); then
            log "[dry-run] docker push ${REPO}:${tag}"
        else
            docker push "${REPO}:${tag}" \
                || die "docker push failed for ${REPO}:${tag}"
            ok "Pushed ${REPO}:${tag}"
        fi
    done

    log "${C_GREEN}${C_BOLD}Variant ${variant} released as v${next_ver}${C_RESET}"
}

# Main
main() {
    preflight

    hr
    log "${C_BOLD}Release plan${C_RESET}"
    log "  Repo        : $REPO"
    log "  Variants    : $( ((DO_ALPINE)) && echo -n "alpine " ; ((DO_DEBIAN)) && echo -n "debian" )"
    log "  Version mode: $(
        if [[ -n "$EXPLICIT_VERSION" ]]; then echo "explicit v$EXPLICIT_VERSION";
        elif (( BUMP_MAJOR )); then           echo "bump-major";
        else                                  echo "bump-minor (default)"; fi)"
    log "  No cache    : $((NO_CACHE))"
    log "  Dry run     : $((DRY_RUN))"

    if (( ! ASSUME_YES && ! DRY_RUN )); then
        read -r -p "Proceed with release? [y/N] " ans
        [[ "$ans" =~ ^[Yy]$ ]] || die "Aborted by user."
    fi

    (( DO_ALPINE )) && release_variant "alpine" "$DOCKERFILE_ALPINE"
    (( DO_DEBIAN )) && release_variant "debian" "$DOCKERFILE_DEBIAN"

    hr
    ok "All done."
    log "Pulls for CI:"
    (( DO_ALPINE )) && log "  docker pull ${REPO}:alpine-latest"
    (( DO_DEBIAN )) && log "  docker pull ${REPO}:debian-latest"
}

main "$@"
