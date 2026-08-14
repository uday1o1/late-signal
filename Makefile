.PHONY: format lint typecheck test check

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
