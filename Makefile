.PHONY: fmt fmt-check

fmt:
	npx oxfmt .
	uv run --with ruff ruff format .

fmt-check:
	npx oxfmt --check .
	uv run --with ruff ruff format --check .
