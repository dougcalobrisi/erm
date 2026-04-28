#!/bin/bash
# Build wheel + sdist into dist/.
# Usage: ./scripts/build-python.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

cd "$PROJECT_ROOT"

info "Cleaning previous build artifacts..."
rm -rf dist/ build/ src/*.egg-info/

require_command uv "https://docs.astral.sh/uv/getting-started/installation/"

info "Running uv build..."
uv build

echo ""
success "Build complete. Artifacts:"
ls -la dist/
