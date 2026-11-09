#!/usr/bin/env bash
# Add one source layer on top of the image already built from commit 4438450.
# Does not push, does not prune, and does not rebuild the dependency layers.
set -euo pipefail

if [[ "${1:-}" == "--push" || "${1:-}" == "push" ]]; then
  echo "Refusing to push. This build stays on the local machine." >&2
  exit 2
fi

cd "$(dirname "$0")/.."

if ! command -v podman >/dev/null 2>&1; then
  echo "podman is not installed." >&2
  exit 1
fi

BASE_IMAGE="${BASE_IMAGE:-localhost/mirofish:4438450ae19f1de735796cbe46fa97572945dbf7}"
if ! podman image exists "${BASE_IMAGE}"; then
  echo "Base image ${BASE_IMAGE} is not in the local store." >&2
  echo "A cold Dockerfile build is a separate decision. Do not prune the existing image." >&2
  exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SHA="$(git rev-parse --short=12 HEAD)"
else
  SHA="${MIROFISH_SOURCE_SHA:-unpinned}"
fi

echo "Overlay ${BASE_IMAGE} -> localhost/mirofish:overlay-${SHA} (no push)."
podman build \
  --platform linux/amd64 \
  -f Dockerfile.local-overlay \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "localhost/mirofish:overlay-${SHA}" \
  -t localhost/mirofish:local \
  backend/app/utils

echo "Previous image tag remains ${BASE_IMAGE} for rollback."
podman image inspect "localhost/mirofish:overlay-${SHA}" --format 'id={{.Id}}'
echo "Recreate only after stopping the container on ports 3000 and 5001."
echo "Keep .env and backend/uploads. Do not pass --build to compose:"
echo "  podman compose -f docker-compose.wsl-local.yml up -d --no-build --force-recreate"
echo "Rollback:"
echo "  podman tag ${BASE_IMAGE} localhost/mirofish:local"
echo "  podman compose -f docker-compose.wsl-local.yml up -d --no-build --force-recreate"
