"""Pull the dbt marts out of BigQuery into local parquet files (data/cache/).

Everything downstream (models, dashboard, report) reads these files, so the
public dataset is scanned only once per `make all`. Each query is dry-run first.
Run: python -m mta.extract
"""

from __future__ import annotations

import pandas as pd

from mta import bq

CACHE = bq.ROOT / "data" / "cache"

QUERIES: dict[str, str] = {
    "journey_touches": f"""
        SELECT journey_id, touch_number, touch_ts, channel, journey_converted AS converted,
               conversion_ts, is_conversion_touch, revenue_usd
        FROM {bq.table('fct_journey_touches')}
    """,
    "journeys": f"""
        SELECT journey_id, converted, conversion_ts, first_touch_ts, last_touch_ts, n_touches,
               n_distinct_channels, path, revenue_usd, days_to_conversion
        FROM {bq.table('fct_journeys')}
    """,
    "funnel": f"SELECT * FROM {bq.table('fct_funnel')}",
    "channel_map": f"SELECT * FROM {bq.table('int_channel_map')} ORDER BY sessions DESC",
    # Session grain is aggregated in BigQuery to keep the download small.
    "sessions_daily": f"""
        SELECT DATE(session_start_ts) AS date, channel, device_category,
               COUNT(*) AS sessions, COUNT(DISTINCT user_pseudo_id) AS users,
               COUNTIF(converted) AS converting_sessions, SUM(n_purchases) AS purchases,
               SUM(revenue_usd) AS revenue_usd, COUNTIF(source_inferred) AS inferred_sessions,
               SUM(session_duration_sec) AS duration_sec
        FROM {bq.table('fct_sessions')}
        GROUP BY 1, 2, 3
    """,
    "kpis": f"""
        SELECT COUNT(*) AS sessions, COUNT(DISTINCT user_pseudo_id) AS users,
               COUNTIF(converted) AS converting_sessions, SUM(n_purchases) AS purchases,
               SUM(revenue_usd) AS revenue_usd,
               COUNT(DISTINCT IF(converted, user_pseudo_id, NULL)) AS purchasing_users,
               COUNTIF(source_inferred) AS inferred_sessions,
               COUNTIF(channel = 'Other/Unknown') AS other_unknown_sessions
        FROM {bq.table('fct_sessions')}
    """,
}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, sql in QUERIES.items():
        df = bq.query(sql, f"extract: {name}")
        # db-dtypes extension types (dbdate) don't round-trip through parquet; store plain datetimes
        for col in df.columns:
            if str(df[col].dtype) in ("dbdate", "dbtime"):
                df[col] = pd.to_datetime(df[col].astype(str))
        df.to_parquet(CACHE / f"{name}.parquet", index=False)
        print(f"    -> data/cache/{name}.parquet ({len(df):,} rows)")


if __name__ == "__main__":
    main()
