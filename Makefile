SHELL := /bin/bash

init:
	@command -v poetry >/dev/null 2>&1  || echo "Poetry not installed"

	poetry install

format:
	poetry run autoflake --remove-unused-variables --remove-all-unused-imports --recursive .
	poetry run isort --atomic . $(EXTRA_FLAGS)
	poetry run black --safe . $(EXTRA_FLAGS)
	poetry run flake8 --max-line-length 100 .

typecheck:
	poetry run mypy forecasting_library
