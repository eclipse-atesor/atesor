#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${HOME}/projects/atesor-ai"
shopt -s nullglob

log() {
	echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

cleanup_env() {
	log "cleanup: stopping running containers"
	docker ps -q | xargs -r docker stop

	log "cleanup: docker system prune"
	docker system prune -a --volumes -f

	log "cleanup: removing workspace repos"
	cd "${ROOT_DIR}/workspace"
	rm -rf repos/* || true
	log "cleanup: done"
}

lists=("${ROOT_DIR}"/.github/packages/failed-*.json)
if [[ ${#lists[@]} -eq 0 ]]; then
	log "ERROR: no failed-*.json files found in ${ROOT_DIR}/.github/packages"
	exit 1
fi

for list_file in "${lists[@]}"; do
	list_name="$(basename "${list_file}" .json)"
	log "Starting ${list_name} batch test"
	cleanup_env

	cd "${ROOT_DIR}"
	log "running: python3 .github/scripts/batch_test.py --list ${list_name} --workers 16 --platform ubuntu"
	# Force unbuffered output + stream setup logs in batch preflight.
	set +e
	PYTHONUNBUFFERED=1 ATESOR_SETUP_STREAM=1 \
		python3 .github/scripts/batch_test.py \
		--list "${list_name}" --workers 16 --platform ubuntu
	status=$?
	set -e
	log "completed: ${list_name} (exit ${status})"

done

log "all failed-* batches completed"
