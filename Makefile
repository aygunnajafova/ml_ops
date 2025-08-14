.PHONY: help format lint test clean install

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install development dependencies
	pip install ruff isort black pre-commit

format: ## Format code with black and isort
	black src/
	isort src/

lint: ## Lint code with ruff
	ruff check src/

lint-fix: ## Lint and auto-fix code with ruff
	ruff check --fix src/

check: ## Run all code quality checks
	ruff check src/
	isort --check-only src/
	black --check src/

quality: format lint ## Format and lint code

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

test: ## Run basic tests
	python -c "import pandas, sklearn, joblib; print('Dependencies OK')"

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete

all: quality test ## Run all checks and tests
