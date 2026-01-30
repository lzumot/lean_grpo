.PHONY: help install install-dev test lint format clean docker-build docker-run

help:
	@echo "Lean GRPO - Available commands:"
	@echo ""
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make example      - Run example training"
	@echo ""

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=lean_grpo --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

example:
	python examples/train_example.py

# Docker commands
docker-build:
	docker build -t lean-grpo:latest .

docker-run:
	docker run --gpus all -it -v $(PWD):/workspace lean-grpo:latest
