.PHONY: install test test-integration test-all lint format run clean help

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help:
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  install            Install dependencies and pre-commit hooks"
	@echo "  test               Run unit and security tests"
	@echo "  test-integration   Run integration tests"
	@echo "  test-all           Run all tests (unit, security, integration)"
	@echo "  lint               Check code with ruff"
	@echo "  format             Format code with ruff and fix issues"
	@echo "  run                Run the application"
	@echo "  clean              Clean build artifacts and cache"
	@echo "  help               Show this help message"

install:
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(YELLOW)Installing pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Installation complete!$(NC)"

test:
	@echo "$(YELLOW)Running unit tests...$(NC)"
	pytest tests/ -v --tb=short
	@echo "$(YELLOW)Running security checks...$(NC)"
	pytest tests/ -v --tb=short -m security || true
	@echo "$(GREEN)Tests complete!$(NC)"

test-integration:
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/ -v --tb=short -m integration || echo "$(YELLOW)No integration tests marked with @pytest.mark.integration$(NC)"
	@echo "$(GREEN)Integration tests complete!$(NC)"

test-all: test test-integration
	@echo "$(GREEN)All tests complete!$(NC)"

RUFF_FLAGS := --select E,W,F,I,C4,B,UP --ignore E501,B008,B017,BLE001,RUF059,RUF013,DTZ,PLW0602,TRY401,G201,SIM115,PYI034,PERF102

lint:
	@echo "$(YELLOW)Linting code with ruff...$(NC)"
	ruff check $(RUFF_FLAGS) src/ tests/ app.py
	@echo "$(GREEN)Linting complete!$(NC)"

format:
	@echo "$(YELLOW)Fixing issues with ruff...$(NC)"
	ruff check $(RUFF_FLAGS) --fix src/ tests/ app.py
	@echo "$(YELLOW)Formatting code with ruff...$(NC)"
	ruff format src/ tests/ app.py
	@echo "$(GREEN)Formatting complete!$(NC)"

run:
	@echo "$(YELLOW)Starting application...$(NC)"
	python app.py

clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete 2>/dev/null || true
	find . -type f -name .coverage.* -delete 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"
