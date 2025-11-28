ACE_DATASET ?= outputs/ace_datasets/ace_dataset_auto.json
ACE_ITERATIONS ?= 1
ACE_LIMIT ?=
ACE_SHUFFLE ?=

DATASET_FLAGS := --output $(ACE_DATASET)
DATASET_FLAGS +=$(if $(ACE_LIMIT), --limit $(ACE_LIMIT),)
DATASET_FLAGS +=$(if $(ACE_SHUFFLE), --shuffle-seed $(ACE_SHUFFLE),)

.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry src

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: test-offline
test-offline: ## Run the offline-safe subset of tests
	@echo "🚀 Testing code (offline subset)"
	@uv run python -m pytest -m offline

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: run
run: ## Run the companion in interactive chat mode
	@echo "🚀 Starting IBM Gen AI Companion..."
	@uv run genai-companion chat

.PHONY: ace-improve
ace-improve: ## Generate a dataset and run ACE self-improvement cycle
	@echo "🧱 Building ACE dataset -> $(ACE_DATASET)"
	@uv run python scripts/build_ace_dataset.py $(DATASET_FLAGS)
	@echo "🤖 Running ACE iterations (count=$(ACE_ITERATIONS))"
	@uv run python scripts/run_ace_iterations.py --dataset $(ACE_DATASET) --iterations $(ACE_ITERATIONS)

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
