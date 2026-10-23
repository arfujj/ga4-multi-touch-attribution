-- One row per de-duplicated purchase event.

select
    event_key as purchase_key,
    session_key,
    user_pseudo_id,
    event_ts as purchase_ts,
    transaction_id,
    purchase_revenue_usd as revenue_usd
from {{ ref('stg_ga4__events') }}
where event_name = 'purchase'
  and not is_duplicate_purchase
