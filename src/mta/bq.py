"""BigQuery helper with a cost guard: every query is dry-run first, bytes are
printed, and anything above ``MAX_BYTES`` is refused."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

PROJECT = os.environ.get("GCP_PROJECT_ID", "")
DATASET = os.environ.get("BQ_DATASET", "ga4_mta")
LOCATION = os.environ.get("BQ_LOCATION", "US")
MAX_BYTES = int(float(os.environ.get("BQ_MAX_BYTES", 2e9)))  # 2 GB per query hard cap

USAGE_LOG = ROOT / "data" / "cache" / "bq_usage.jsonl"  # bytes processed per query (cleared by `make all`)
SOURCE = "bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*"
START_SUFFIX, END_SUFFIX = "20201101", "20210131"

_client: bigquery.Client | None = None


def client() -> bigquery.Client:
    global _client
    if _client is None:
        if not PROJECT:
            raise RuntimeError("Set GCP_PROJECT_ID in .env (see .env.example)")
        _client = bigquery.Client(project=PROJECT, location=LOCATION)
    return _client


def dry_run(sql: str) -> int:
    """Return bytes the query would process, without running it."""
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    return int(client().query(sql, job_config=cfg).total_bytes_processed or 0)


def query(sql: str, label: str = "query", max_bytes: int = MAX_BYTES) -> pd.DataFrame:
    """Dry-run, print the cost, refuse if above ``max_bytes``, then run."""
    nbytes = dry_run(sql)
    print(f"[bq] {label}: dry run = {nbytes / 1e6:,.1f} MB")
    if nbytes > max_bytes:
        raise RuntimeError(f"{label} would scan {nbytes / 1e9:.2f} GB > cap {max_bytes / 1e9:.2f} GB")
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=max(max_bytes, 10 * 1024**2))
    job = client().query(sql, job_config=cfg)
    df = job.to_dataframe()
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a") as fh:
        fh.write(json.dumps({"label": label, "bytes_processed": int(job.total_bytes_processed or 0)}) + "\n")
    return df


def table(name: str) -> str:
    """Fully-qualified name of a dbt-built table in this project."""
    return f"`{PROJECT}.{DATASET}.{name}`"
