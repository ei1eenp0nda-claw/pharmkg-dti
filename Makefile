.PHONY: help install test lint format docker train eval benchmark serve clean

help:
	@echo "PharmKG-DTI Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  install-dev  Install with dev dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  docker       Build Docker image"
	@echo "  train        Train model (fast config)"
	@echo "  eval         Evaluate model"
	@echo "  benchmark    Run benchmarks"
	@echo "  serve        Start API server"
	@echo "  clean        Clean generated files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest black flake8 mypy optuna

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=127
	mypy src/ --ignore-missing-imports || true

format:
	black src/ tests/

docker:
	docker build -f docker/Dockerfile -t pharmkg-dti:latest .

train:
	python -m src.cli train --fast

eval:
	python -m src.cli evaluate --checkpoint checkpoints/best_model.pt --dataset bindingdb

benchmark:
	python -m src.cli benchmark --full

serve:
	python -m src.cli serve --port 8000

clean:
	rm -rf __pycache__ .pytest_cache htmlcov
	rm -rf logs/ checkpoints/ experiments/ outputs/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete
