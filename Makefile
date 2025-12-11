.PHONY: help install install-dev format lint test test-cov clean build publish pre-commit

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

install-test: ## Install package with test dependencies only
	pip install -e ".[test]"

format: ## Format code with black and isort
	@echo "Running black..."
	black src/ tests/ examples/
	@echo "Running isort..."
	isort src/ tests/ examples/

lint: ## Lint code with ruff and run type checking with mypy
	@echo "Running ruff check..."
	ruff check src/ tests/ examples/
	@echo "Running mypy..."
	mypy src/

lint-fix: ## Lint and auto-fix issues with ruff
	@echo "Running ruff check with auto-fix..."
	ruff check --fix src/ tests/ examples/

test: ## Run tests with pytest
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=src/spark_dist_fit --cov-report=term-missing --cov-report=html -v

test-fast: ## Run tests without coverage
	pytest -v --no-cov

clean: ## Clean build artifacts, cache files, and coverage reports
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build distribution packages
	pip install --upgrade build
	python -m build

publish-test: build ## Publish to TestPyPI
	pip install --upgrade twine
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	pip install --upgrade twine
	twine upload dist/*

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

check: pre-commit lint test ## Run all checks (pre-commit, lint, test)

setup: install-dev ## Initial setup for development
	@echo "✓ Development environment setup complete!"
	@echo "  Run 'make test' to verify everything works"
