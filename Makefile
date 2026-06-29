# Replayable Mini RAG Pipeline
# Convenience targets for a clean-checkout evaluator.

PYTHON ?= python3
VENV   ?= .venv
PY      = $(VENV)/bin/python
PIP     = $(VENV)/bin/pip

LLM_MODEL   ?= llama3.2:1b
EMBED_MODEL ?= nomic-embed-text

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Replayable Mini RAG - available targets:"
	@echo "  make setup     Create venv, install deps, pull Ollama models"
	@echo "  make run       Run the full pipeline (interactive human review)"
	@echo "  make run-auto  Run the full pipeline non-interactively (no overrides)"
	@echo "  make validate  Validate generated artifacts against all requirements"
	@echo "  make test      Run unit tests (pytest)"
	@echo "  make clean     Delete generated artifacts"
	@echo "  make distclean Delete artifacts and the virtualenv"

$(VENV):
	$(PYTHON) -m venv $(VENV)

.PHONY: setup
setup: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Pulling Ollama models (requires 'ollama' installed and running)..."
	-ollama pull $(LLM_MODEL)
	-ollama pull $(EMBED_MODEL)
	@echo "Setup complete."

.PHONY: run
run:
	$(PY) run.py --model $(LLM_MODEL) --embed-model $(EMBED_MODEL)

.PHONY: run-auto
run-auto:
	$(PY) run.py --auto-continue --model $(LLM_MODEL) --embed-model $(EMBED_MODEL)

.PHONY: validate
validate:
	$(PY) validate.py

.PHONY: test
test:
	$(PY) -m pytest -q

.PHONY: clean
clean:
	rm -rf artifacts
	@echo "Removed artifacts/."

.PHONY: distclean
distclean: clean
	rm -rf $(VENV)
	@echo "Removed $(VENV)/."
