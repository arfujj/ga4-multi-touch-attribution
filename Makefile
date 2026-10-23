# GA4 Multi-Touch Attribution Lab: `make all` runs everything end to end.
SHELL := /bin/bash
PY    := .venv/bin/python
DBT   := ../.venv/bin/dbt
ENV   := set -a && . ./.env && set +a
PORT  ?= 8501

.PHONY: all setup profile dryrun dbt extract analysis writeup test pytest dbt-test dashboard screenshots clean

all: check-env clear-usage profile dbt extract analysis pytest writeup  ## full pipeline
	@echo "Done. Outputs in reports/, docs/ and README.md. Run 'make dashboard' to explore."

setup:  ## create venv and install pinned requirements
	python3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip -q
	$(PY) -m pip install -r requirements.txt -q
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example; set GCP_PROJECT_ID")

check-env:
	@test -f .env || (echo "Missing .env; run: cp .env.example .env and set GCP_PROJECT_ID" && exit 1)

clear-usage:
	@rm -f data/cache/bq_usage.jsonl

profile:  ## profile raw data -> docs/data_profile.md
	$(ENV) && PYTHONPATH=src $(PY) -m mta.profile

dryrun:  ## compile dbt and print bytes each model/test would scan
	$(ENV) && cd dbt && $(DBT) compile --profiles-dir . -q
	$(ENV) && PYTHONPATH=src $(PY) -m mta.dbt_dryrun

dbt:  ## seeds + models + all tests; fails the build if any test fails
	$(ENV) && cd dbt && $(DBT) build --profiles-dir .

dbt-test:
	$(ENV) && cd dbt && $(DBT) test --profiles-dir .

extract:  ## dbt marts -> data/cache/*.parquet
	$(ENV) && PYTHONPATH=src $(PY) -m mta.extract

analysis:  ## 8 models, bootstrap, funnel, budget sim -> reports/
	PYTHONPATH=src $(PY) -m mta.analysis --n-boot 1000

writeup:  ## README.md, docs/findings_memo.md, reports/report.html
	PYTHONPATH=src $(PY) -m mta.writeup

pytest:
	$(PY) -m pytest -q

test: pytest dbt-test  ## unit tests + dbt data tests

dashboard:  ## Streamlit dashboard
	$(PY) -m streamlit run dashboard/app.py --server.port $(PORT)

screenshots:  ## start the dashboard headless and capture README screenshots
	@$(PY) -m streamlit run dashboard/app.py --server.port 8599 >/dev/null 2>&1 & echo $$! > .st.pid; \
	for i in $$(seq 1 30); do curl -s localhost:8599/_stcore/health >/dev/null && break; sleep 1; done; \
	PYTHONPATH=src $(PY) -m mta.screenshots --port 8599; status=$$?; kill $$(cat .st.pid); rm -f .st.pid; exit $$status

clean:
	rm -rf data/cache reports/*.csv reports/*.json reports/*.parquet reports/figures dbt/target
