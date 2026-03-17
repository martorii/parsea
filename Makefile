# Include environment variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Default host and port if not specified in .env
API_HOST ?= 0.0.0.0
API_PORT ?= 8000

.PHONY: build run stop logs test clean help

help:
	@echo "Available commands:"
	@echo "  make build  - Build the Docker image"
	@echo "  make run    - Start the container in detached mode"
	@echo "  make stop   - Stop and remove the container"
	@echo "  make logs   - Show container logs"
	@echo "  make test   - Run deployment tests"
	@echo "  make clean  - Remove Docker images and containers"

build:
	docker compose build

run:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f

test:
	@echo "Running deployment tests against http://$(API_HOST):$(API_PORT)..."
	uv run python tests/test_deployment.py --check-all --token $(API_TOKEN)

clean:
	docker compose down --rmi all --volumes --remove-orphans
