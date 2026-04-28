#!/bin/bash
# Synchronize the project version across pyproject.toml and src/erm/__init__.py.
# Usage: ./scripts/sync-version.sh <version> [--commit] [--dry-run]
#   <version>   New version (PEP 440: X.Y.Z, X.Y.ZbN, X.Y.ZrcN, X.Y.ZaN).
#   --commit    Create a git commit with the version bump.
#   --dry-run   Show what would change without modifying files.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

NEW_VERSION=""
CREATE_COMMIT=false
parse_common_args "$@"
for arg in "$@"; do
    case $arg in
        --commit) CREATE_COMMIT=true ;;
        --help|-h)
            sed -n '2,7p' "$0"
            exit 0
            ;;
        --*) ;;
        *)
            [ -z "$NEW_VERSION" ] && NEW_VERSION="$arg"
            ;;
    esac
done

if [ -z "$NEW_VERSION" ]; then
    error "Version argument required."
    echo "Usage: ./scripts/sync-version.sh <version> [--commit] [--dry-run]"
    exit 1
fi

# PEP 440 (subset): release + optional pre-release segment.
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$'; then
    error "Invalid version: $NEW_VERSION"
    echo "Expected PEP 440: X.Y.Z or X.Y.Z{a,b,rc}N (e.g. 0.2.0, 0.2.0b1, 0.2.0rc1)"
    exit 1
fi

cd "$PROJECT_ROOT"

CURRENT_PY=$(get_pyproject_version)
CURRENT_INIT=$(get_init_version)

info "Current versions:"
echo "  pyproject.toml:        $CURRENT_PY"
echo "  src/erm/__init__.py:   $CURRENT_INIT"
echo ""
info "New version: $NEW_VERSION"
echo ""

if [ "$DRY_RUN" = true ]; then
    info "DRY RUN — no files will be modified"
    echo "Would update:"
    echo "  $PYPROJECT  -> $NEW_VERSION"
    echo "  $INIT_PY    -> $NEW_VERSION"
    [ -f "$PROJECT_ROOT/uv.lock" ] && echo "  $PROJECT_ROOT/uv.lock (re-locked)"
    [ "$CREATE_COMMIT" = true ] && echo "Would create commit: chore: bump version to $NEW_VERSION"
    exit 0
fi

info "Updating pyproject.toml..."
sed_inplace "s/^version = \"${CURRENT_PY}\"/version = \"${NEW_VERSION}\"/" "$PYPROJECT"

info "Updating src/erm/__init__.py..."
sed_inplace "s/^__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" "$INIT_PY"

# Verify
NEW_PY=$(get_pyproject_version)
NEW_INIT=$(get_init_version)
if [ "$NEW_PY" != "$NEW_VERSION" ] || [ "$NEW_INIT" != "$NEW_VERSION" ]; then
    error "Version update failed (pyproject=$NEW_PY init=$NEW_INIT)"
    exit 1
fi
success "Versions updated to $NEW_VERSION"

# Re-lock if uv.lock exists.
if [ -f "$PROJECT_ROOT/uv.lock" ]; then
    info "Updating uv.lock..."
    (cd "$PROJECT_ROOT" && uv lock 2>/dev/null || true)
fi

if [ "$CREATE_COMMIT" = true ]; then
    info "Creating git commit..."
    FILES=("$PYPROJECT" "$INIT_PY")
    [ -f "$PROJECT_ROOT/uv.lock" ] && FILES+=("$PROJECT_ROOT/uv.lock")
    if git commit --only "${FILES[@]}" -m "chore: bump version to $NEW_VERSION"; then
        success "Created commit"
    else
        error "git commit failed"
        exit 1
    fi
fi

echo ""
success "Version sync complete!"
