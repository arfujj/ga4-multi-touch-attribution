"""Profile the raw GA4 export and write docs/data_profile.md.

Every number in the output document is computed by the queries below.
Run: python -m mta.profile
"""

from __future__ import annotations

import json

import pandas as pd

from mta import bq

W = f"_TABLE_SUFFIX BETWEEN '{bq.START_SUFFIX}' AND '{bq.END_SUFFIX}'"
P = lambda k, t="string": f"(SELECT value.{t}_value FROM UNNEST(event_params) WHERE key = '{k}')"  # noqa: E731

OVERVIEW_SQL = f"""
SELECT
  MIN(PARSE_DATE('%Y%m%d', event_date)) AS min_date,
  MAX(PARSE_DATE('%Y%m%d', event_date)) AS max_date,
  COUNT(DISTINCT event_date) AS n_days,
  COUNT(*) AS events,
  COUNT(DISTINCT user_pseudo_id) AS users,
  COUNT(DISTINCT CONCAT(user_pseudo_id, '-', CAST({P('ga_session_id', 'int')} AS STRING))) AS sessions,
  COUNTIF({P('ga_session_id', 'int')} IS NULL) AS events_without_session_id,
  COUNT(DISTINCT user_id) AS logged_in_user_ids,
  COUNT(DISTINCT platform) AS platforms
FROM `{bq.SOURCE}` WHERE {W}
"""

EVENTS_SQL = f"""
SELECT event_name, COUNT(*) AS events, COUNT(DISTINCT user_pseudo_id) AS users,
  ROUND(COUNTIF({P('source')} IS NOT NULL) / COUNT(*), 4) AS share_with_source_param
FROM `{bq.SOURCE}` WHERE {W}
GROUP BY 1 ORDER BY 2 DESC
"""

# Session-level traffic source = first event in the session carrying a source/medium param.
SESSION_SOURCE_SQL = f"""
WITH e AS (
  SELECT user_pseudo_id AS u, {P('ga_session_id', 'int')} AS sid, event_timestamp AS ts,
         {P('source')} AS src, {P('medium')} AS med, {P('campaign')} AS cmp
  FROM `{bq.SOURCE}` WHERE {W}
), s AS (
  SELECT u, sid,
    ARRAY_AGG(IF(src IS NULL AND med IS NULL, NULL, STRUCT(src, med, cmp)) IGNORE NULLS ORDER BY ts LIMIT 1)[SAFE_OFFSET(0)] AS f
  FROM e GROUP BY 1, 2
)
SELECT IFNULL(f.src, '(null)') AS source, IFNULL(f.med, '(null)') AS medium, COUNT(*) AS sessions
FROM s GROUP BY 1, 2 ORDER BY 3 DESC
"""

FIELD_OBFUSCATION_SQL = f"""
WITH e AS (
  SELECT {P('source')} AS ev_source, {P('medium')} AS ev_medium, {P('campaign')} AS ev_campaign,
         traffic_source.source AS user_source, traffic_source.medium AS user_medium,
         traffic_source.name AS user_campaign, geo.country AS country, device.category AS device,
         ecommerce.transaction_id AS transaction_id, event_name
  FROM `{bq.SOURCE}` WHERE {W}
)
SELECT field,
  COUNT(*) AS rows_evaluated,
  COUNTIF(v IS NULL) AS null_values,
  COUNTIF(v = '<Other>') AS other_values,
  COUNTIF(v = '(data deleted)') AS data_deleted_values,
  COUNTIF(v IN ('(not set)', '')) AS not_set_values
FROM e, UNNEST([
  STRUCT('event_param.source' AS field, ev_source AS v, TRUE AS keep),
  ('event_param.medium', ev_medium, TRUE),
  ('event_param.campaign', ev_campaign, TRUE),
  ('traffic_source.source (user first touch)', user_source, TRUE),
  ('traffic_source.medium (user first touch)', user_medium, TRUE),
  ('traffic_source.name (user first touch)', user_campaign, TRUE),
  ('geo.country', country, TRUE),
  ('device.category', device, TRUE),
  ('ecommerce.transaction_id (purchase events)', transaction_id, event_name = 'purchase')
])
WHERE keep
GROUP BY field ORDER BY field
"""

PURCHASE_SQL = f"""
WITH p AS (
  SELECT user_pseudo_id AS u, {P('ga_session_id', 'int')} AS sid, event_timestamp AS ts,
         COALESCE(ecommerce.transaction_id, {P('transaction_id')},
                  CAST({P('transaction_id', 'int')} AS STRING)) AS tid,
         ecommerce.purchase_revenue_in_usd AS rev
  FROM `{bq.SOURCE}` WHERE {W} AND event_name = 'purchase'
), flagged AS (
  SELECT *, tid IS NOT NULL AND tid != '(not set)'
            AND ROW_NUMBER() OVER (PARTITION BY u, sid, tid ORDER BY ts) > 1 AS is_duplicate
  FROM p
)
SELECT
  COUNT(*) AS purchase_events,
  COUNTIF(is_duplicate) AS duplicate_events,
  SUM(rev) AS gross_revenue_usd,
  SUM(IF(is_duplicate, 0, rev)) AS dedup_revenue_usd,
  COUNTIF(tid IS NULL) AS null_transaction_id,
  COUNTIF(tid = '(not set)') AS not_set_transaction_id,
  COUNTIF(rev = 0) AS zero_revenue_events,
  COUNT(DISTINCT CONCAT(u, '-', CAST(sid AS STRING))) AS purchase_sessions,
  COUNT(DISTINCT u) AS purchasing_users
FROM flagged
"""


def _md(df: pd.DataFrame, floatfmt: str = ",.4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt, intfmt=",")


def main() -> None:
    ov = bq.query(OVERVIEW_SQL, "profile: overview").iloc[0]
    ev = bq.query(EVENTS_SQL, "profile: event types")
    ss = bq.query(SESSION_SOURCE_SQL, "profile: session source/medium")
    ob = bq.query(FIELD_OBFUSCATION_SQL, "profile: obfuscation")
    pu = bq.query(PURCHASE_SQL, "profile: purchases").iloc[0]

    n_sess = int(ss["sessions"].sum())
    no_params = int(ss.loc[(ss.source == "(null)") & (ss.medium == "(null)"), "sessions"].sum())
    other_like = {"<Other>", "(data deleted)"}
    obf_sess = int(ss.loc[ss.source.isin(other_like) | ss.medium.isin(other_like), "sessions"].sum())
    self_ref = int(ss.loc[ss.source.str.contains("googlemerchandisestore.com", regex=False), "sessions"].sum())

    for c in ["null_values", "other_values", "data_deleted_values", "not_set_values"]:
        ob[c.replace("_values", "_pct")] = (ob[c] / ob["rows_evaluated"] * 100).round(2)
    ob_view = ob[["field", "rows_evaluated", "null_pct", "other_pct", "data_deleted_pct", "not_set_pct"]]

    top_ss = ss.head(20).assign(share_pct=lambda d: (d.sessions / n_sess * 100).round(2))
    dup_rev = float(pu.gross_revenue_usd - pu.dedup_revenue_usd)

    metrics = {
        "min_date": str(ov.min_date), "max_date": str(ov.max_date), "n_days": int(ov.n_days),
        "events": int(ov.events), "users": int(ov.users), "sessions": int(ov.sessions),
        "sessions_without_source_params": no_params, "sessions_obfuscated_source": obf_sess,
        "sessions_self_referral": self_ref, "purchase_events": int(pu.purchase_events),
        "duplicate_purchase_events": int(pu.duplicate_events),
        "gross_revenue_usd": float(pu.gross_revenue_usd), "dedup_revenue_usd": float(pu.dedup_revenue_usd),
    }
    (bq.ROOT / "reports").mkdir(exist_ok=True)
    (bq.ROOT / "reports" / "data_profile_metrics.json").write_text(json.dumps(metrics, indent=2))

    doc = f"""# Data profile: GA4 obfuscated sample e-commerce

_Generated by `src/mta/profile.py`. Every number below comes from a query against
`{bq.SOURCE}` filtered to `_TABLE_SUFFIX` {bq.START_SUFFIX} to {bq.END_SUFFIX}._

## Overview

| Metric | Value |
|---|---|
| Date range | {ov.min_date} to {ov.max_date} ({ov.n_days} days) |
| Events | {ov.events:,} |
| Users (`user_pseudo_id`) | {ov.users:,} |
| Sessions (`user_pseudo_id` + `ga_session_id`) | {ov.sessions:,} |
| Events with no `ga_session_id` | {ov.events_without_session_id:,} |
| Distinct logged-in `user_id`s | {ov.logged_in_user_ids:,} |
| Platforms | {ov.platforms} |

## Event types

{_md(ev)}

## How traffic source is populated

* `traffic_source.*` is **user-scoped first touch**: it is the same on every event a user
  ever sends, so it cannot describe later sessions and is not used for multi-touch work.
* Session-scoped source is only available through the `source` / `medium` / `campaign`
  **event parameters**, which GA4 writes on some events of a session (never on
  `session_start`). We take the first event in each session that carries them.
* **{no_params:,} of {n_sess:,} sessions ({no_params / n_sess:.1%}) carry no source/medium
  parameter at all.** GA4 reports such sessions as `(direct) / (none)`; we follow that
  convention and flag them as `source_inferred = true`.
* **{self_ref:,} sessions ({self_ref / n_sess:.1%}) are self-referrals** from
  `googlemerchandisestore.com` domains (most likely cross-domain hops between the store's own domains). GA4 would treat these
  as unwanted referrals; we map them to Direct so they don't inflate Referral.
* **{obf_sess:,} sessions ({obf_sess / n_sess:.1%}) have an obfuscated source or medium**
  (`<Other>` or `(data deleted)`). When the medium survives (e.g. `<Other> / organic`) the
  channel can still be classified; otherwise it lands in Other/Unknown.

### Top 20 session source / medium combinations

{_md(top_ss, ",.2f")}

## Obfuscation by field (event level)

Percent of evaluated rows that are null, `<Other>`, `(data deleted)` or `(not set)`.

{_md(ob_view, ",.2f")}

## Purchases and revenue quality

| Metric | Value |
|---|---|
| Purchase events | {pu.purchase_events:,} |
| Duplicate purchase events (same user + session + transaction_id) | {pu.duplicate_events:,} |
| Gross revenue (USD) | ${pu.gross_revenue_usd:,.0f} |
| Revenue after de-duplication (USD) | ${pu.dedup_revenue_usd:,.0f} |
| Revenue removed as duplicates | ${dup_rev:,.0f} ({dup_rev / pu.gross_revenue_usd:.1%}) |
| `transaction_id` null / `(not set)` | {pu.null_transaction_id:,} / {pu.not_set_transaction_id:,} |
| Zero-revenue purchase events | {pu.zero_revenue_events:,} |
| Sessions with a purchase | {pu.purchase_sessions:,} |
| Purchasing users | {pu.purchasing_users:,} |

Duplicates fire within milliseconds of each other with identical revenue, so they look like a
double-firing tag rather than repeat orders. Purchases with a null or `(not set)`
transaction id can't be de-duplicated and are kept.

## Implications for the analysis

1. Channel analysis must use event-param source/medium; the `traffic_source` struct is first touch only.
2. Treating parameter-less sessions as Direct follows GA4, but it is an assumption. Direct is the
   channel most exposed to this choice.
3. Obfuscation (`<Other>`, `(data deleted)`) is reported as its own channel rather than imputed.
4. Revenue is de-duplicated on (user, session, transaction_id); dbt tests reconcile modeled
   revenue against this rule recomputed directly from raw events.
"""
    (bq.ROOT / "docs" / "data_profile.md").write_text(doc)
    print("wrote docs/data_profile.md")


if __name__ == "__main__":
    main()
