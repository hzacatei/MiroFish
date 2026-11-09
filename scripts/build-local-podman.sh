#!/usr/bin/env bash
# Build a local linux/amd64 image with rootless Podman. Never pushes.
# Intended for WSL2. Run from an extracted source archive; a git checkout
# is optional and is used only to label the image with the commit.
set -euo pipefail

if [[ "${1:-}" == "--push" || "${1:-}" == "push" ]]; then
  echo "Refusing to push. This build stays on the local machine." >&2
  exit 2
fi

cd "$(dirname "$0")/.."

if ! command -v podman >/dev/null 2>&1; then
  echo "podman is not installed. Install rootless Podman in WSL2, then rerun." >&2
  exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SHA="$(git rev-parse HEAD)"
else
  SHA="${MIROFISH_SOURCE_SHA:-unpinned}"
fi

echo "Building localhost/mirofish:${SHA} for linux/amd64 (no push)."
podman build \
  --platform linux/amd64 \
  --pull=true \
  -f Dockerfile \
  -t "localhost/mirofish:${SHA}" \
  -t localhost/mirofish:local \
  .

echo "Image ID:"
podman image inspect "localhost/mirofish:${SHA}" --format 'id={{.Id}} digest={{.Digest}}'
echo "Resolved base images:"
podman image inspect python:3.11 --format 'python:3.11 {{.Digest}} {{.Id}}' || true
podman image inspect ghcr.io/astral-sh/uv:0.9.26 --format 'uv:0.9.26 {{.Digest}} {{.Id}}' || true
echo "Save for offline transfer (still local):"
echo "  podman save localhost/mirofish:${SHA} | gzip > mirofish-${SHA}-amd64.tar.gz"
echo "Load on the destination WSL2 distro:"
echo "  gunzip -c mirofish-${SHA}-amd64.tar.gz | podman load"
echo "  podman tag localhost/mirofish:${SHA} localhost/mirofish:local"
echo "Do not run podman compose with the stock docker-compose.yml."
echo "After the current loopback app on ports 3000 and 5001 is stopped:"
echo "  podman compose -f docker-compose.wsl-local.yml up -d"
