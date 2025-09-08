.PHONY: help setup install run test update_prices

help: ## Displays this information.
	@printf '%s\n' "Usage: make <command>"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@printf '\n'

setup: ## Create and activate Python virtual environment
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

install: ## Install dependencies from requirements.txt
	pip install -r requirements.txt

run: ## Run the CLI tool with an optional initial message
	python3 main_aisuite.py

test: ## Run tests with pytest
	pytest

update_prices: ## Run the pricing data update script. Wobbly at best.
	python3 scripts/extract_prices.py
