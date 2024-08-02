.PHONY: lint lint-fix format format-fix all

lint:
	ruff check

lint-fix:
	ruff check --fix

format-check:
	ruff format --check

format:
	ruff format

all: lint format
