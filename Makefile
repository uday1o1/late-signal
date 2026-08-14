.PHONY: audit check format format-check lint test typecheck

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

audit:
	uv run latesignal audit

check: format-check lint typecheck test audit
