#!/bin/bash
# Common helpers for build/publish/version scripts.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYPROJECT="$PROJECT_ROOT/pyproject.toml"
INIT_PY="$PROJECT_ROOT/src/erm/__init__.py"

case "$(uname -s)" in
    Darwin*) PLATFORM="macos" ;;
    Linux*)  PLATFORM="linux" ;;
    *)       PLATFORM="unknown" ;;
esac

if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Cross-platform `sed -i`. macOS `sed` requires an explicit empty backup arg.
sed_inplace() {
    if [ "$PLATFORM" = "macos" ]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

require_command() {
    local cmd="$1" hint="${2:-}"
    if ! command -v "$cmd" &> /dev/null; then
        error "$cmd not found."
        [ -n "$hint" ] && echo "  Install with: $hint"
        exit 1
    fi
}

# Read the version line from pyproject.toml's [project] table.
get_pyproject_version() {
    grep -E '^version[[:space:]]*=' "$PYPROJECT" | head -1 | sed 's/.*=[[:space:]]*"\([^"]*\)".*/\1/'
}

# Read __version__ from src/erm/__init__.py.
get_init_version() {
    grep -E '^__version__[[:space:]]*=' "$INIT_PY" | head -1 | sed 's/.*=[[:space:]]*"\([^"]*\)".*/\1/'
}

# Common --dry-run flag handling. Sets DRY_RUN=true|false.
DRY_RUN=false
parse_common_args() {
    for arg in "$@"; do
        case $arg in
            --dry-run) DRY_RUN=true ;;
        esac
    done
}
