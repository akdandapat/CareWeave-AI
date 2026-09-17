# CareWeave AI
# Everything here runs offline with no API keys. `make all` is the full pipeline.

PY := PYTHONPATH=backend python3
MEMBERS ?= 2000

.PHONY: help install data train rag extract eval scenarios test api demo all clean

help:
	@echo "make install    install pinned dependencies"
	@echo "make data       generate the synthetic ecosystem (MEMBERS=$(MEMBERS))"
	@echo "make train      train and evaluate the friction risk model"
	@echo "make rag        evaluate policy retrieval"
	@echo "make extract    evaluate signal extraction"
	@echo "make eval       run all three evaluations"
	@echo "make scenarios  run the six end-to-end demo scenarios"
	@echo "make test       run the invariant test suite"
	@echo "make api        serve the API and UI on :8000"
	@echo "make all        data -> train -> eval -> scenarios -> test"

install:
	pip install -r backend/requirements.txt

data:
	$(PY) -m careweave.data.generator.run --members $(MEMBERS)

train:
	$(PY) -m careweave.ml.train

rag:
	$(PY) -m careweave.rag.evaluate

extract:
	$(PY) -m careweave.nlp.evaluate

eval: train rag extract

scenarios:
	$(PY) -m careweave.scenarios

test:
	$(PY) -m pytest tests/ -q

api:
	cd backend && PYTHONPATH=. python3 -m uvicorn careweave.api.main:app --reload --port 8000

demo: api

all: data train rag extract scenarios test
	@echo ""
	@echo "Pipeline complete. Reports are in data/artifacts/."
	@echo "Run 'make api' and open http://localhost:8000"

clean:
	rm -rf data/synthetic/* data/documents/* data/artifacts/*
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
