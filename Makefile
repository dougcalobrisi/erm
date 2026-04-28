# erm Makefile

.PHONY: all setup lint test test-cov clean build build-python publish publish-python publish-test publish-dry-run sync-version help

all: help

# Install dev dependencies (creates .venv via uv)
setup:
	uv sync --extra dev

# Check Python syntax
lint:
	uv run python -m py_compile src/erm/*.py
	@echo "Syntax check passed"

# Run pytest
test:
	uv run pytest tests/ -v

# Run pytest with coverage
test-cov:
	uv run pytest tests/ -v --cov=src/erm --cov-report=term-missing

# Remove build/dist artifacts
clean:
	rm -rf build/ dist/ src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Build distribution packages (wheel + sdist)
build: build-python

build-python:
	./scripts/build-python.sh

# Publish to PyPI (production)
publish: publish-python

publish-python:
	./scripts/publish-python.sh

# Publish to TestPyPI
publish-test:
	./scripts/publish-python.sh --test

# Validate dist/ without uploading
publish-dry-run:
	./scripts/publish-python.sh --dry-run

# Bump version. Usage: make sync-version VERSION=0.2.0
sync-version:
	@test -n "$(VERSION)" || (echo "Usage: make sync-version VERSION=x.y.z" && exit 1)
	./scripts/sync-version.sh $(VERSION)

help:
	@echo "erm Makefile"
	@echo ""
	@echo "Setup & checks:"
	@echo "  setup            Install dev deps (uv sync --extra dev)"
	@echo "  lint             Compile-check Python sources"
	@echo "  test             Run pytest"
	@echo "  test-cov         Run pytest with coverage"
	@echo "  clean            Remove build/dist/__pycache__ artifacts"
	@echo ""
	@echo "Build & publish:"
	@echo "  build            Build wheel + sdist into dist/"
	@echo "  publish          Upload dist/ to PyPI"
	@echo "  publish-test     Upload dist/ to TestPyPI"
	@echo "  publish-dry-run  Validate dist/ without uploading"
	@echo ""
	@echo "Version:"
	@echo "  sync-version VERSION=x.y.z   Bump version in pyproject.toml + __init__.py"
