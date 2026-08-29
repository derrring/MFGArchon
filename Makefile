.PHONY: help test test-fast lint type-check coverage clean install format

help:
	@echo "MFGarchon Development Commands"
	@echo "============================"
	@echo "test          - Run all tests"
	@echo "test-fast     - Run fast tests only"
	@echo "lint          - Run linting checks"
	@echo "type-check    - Run mypy type checking"
	@echo "coverage      - Run tests with coverage report"
	@echo "format        - Format code with ruff"
	@echo "clean         - Remove cache and build files"
	@echo "install       - Install package in development mode"

test:
	pytest

test-fast:
	pytest -m "not slow" -x

lint:
	@command -v ruff >/dev/null || { echo 'ruff is not installed: it has one owner and is read at runtime (#2172).'; echo '  uv pip install "ruff==$$(python scripts/update_ruff_version.py --print-current)"'; exit 1; }
	ruff check mfgarchon/

type-check:
	mypy mfgarchon/ --show-error-codes

coverage:
	pytest --cov=mfgarchon --cov-report=html --cov-report=term

format:
	@command -v ruff >/dev/null || { echo 'ruff is not installed: it has one owner and is read at runtime (#2172).'; echo '  uv pip install "ruff==$$(python scripts/update_ruff_version.py --print-current)"'; exit 1; }
	ruff format mfgarchon/ tests/ examples/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

install:
	pip install -e . --group dev

.DEFAULT_GOAL := help
