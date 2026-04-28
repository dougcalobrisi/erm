#!/bin/bash
# Publish built artifacts in dist/ to PyPI (or TestPyPI).
# Usage: ./scripts/publish-python.sh [--test] [--dry-run]
#   --test      Publish to TestPyPI instead of PyPI.
#   --dry-run   Validate without uploading.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

USE_TEST_PYPI=false
parse_common_args "$@"
for arg in "$@"; do
    case $arg in
        --test) USE_TEST_PYPI=true ;;
        --help|-h)
            echo "Usage: ./scripts/publish-python.sh [--test] [--dry-run]"
            echo "  --test      Publish to TestPyPI instead of PyPI."
            echo "  --dry-run   Validate without uploading."
            exit 0
            ;;
    esac
done

cd "$PROJECT_ROOT"

if [ ! -d "dist" ] || [ -z "$(ls -A dist/ 2>/dev/null)" ]; then
    error "No distribution files found in dist/"
    echo "  Run: make build"
    exit 1
fi

EXPECTED=$(get_pyproject_version)
for artifact in dist/*; do
    base="$(basename "$artifact")"
    if ! echo "$base" | grep -q "$EXPECTED"; then
        error "Stale artifact: $base (expected version $EXPECTED)"
        echo "  Run: make build"
        exit 1
    fi
done

info "Validating packages with twine..."
uv run --with twine -- twine check dist/*

if [ "$DRY_RUN" = true ]; then
    success "Dry run complete - packages validated."
    echo ""
    echo "Would upload:"
    ls -la dist/
    exit 0
fi

if [ "$USE_TEST_PYPI" = true ]; then
    info "Publishing to TestPyPI..."
    uv run --with twine -- twine upload --repository testpypi dist/*
    echo ""
    success "Published to TestPyPI: https://test.pypi.org/project/erm/"
else
    info "Publishing to PyPI..."
    uv run --with twine -- twine upload dist/*
    echo ""
    success "Published to PyPI: https://pypi.org/project/erm/"
fi
