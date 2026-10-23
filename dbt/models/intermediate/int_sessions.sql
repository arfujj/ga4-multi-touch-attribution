-- One row per session (user_pseudo_id + ga_session_id).
-- Session traffic source = first event in the session that carries a source/medium param.

with events as (

    select * from {{ ref('stg_ga4__events') }}

),

sessions as (

    select
        session_key,
        any_value(user_pseudo_id) as user_pseudo_id,
        any_value(ga_session_id) as ga_session_id,
        max(ga_session_number) as ga_session_number,
        min(event_ts) as session_start_ts,
        max(event_ts) as session_end_ts,
        count(*) as n_events,

        array_agg(
            if(event_name = 'page_view', page_location, null) ignore nulls
            order by event_timestamp_micros limit 1
        )[safe_offset(0)] as landing_page_location,

        array_agg(
            if(source is null and medium is null, null, struct(source, medium, campaign)) ignore nulls
            order by event_timestamp_micros limit 1
        )[safe_offset(0)] as traffic,

        array_agg(device_category ignore nulls order by event_timestamp_micros limit 1)[safe_offset(0)] as device_category,
        array_agg(country ignore nulls order by event_timestamp_micros limit 1)[safe_offset(0)] as country,

        logical_or(event_name = 'view_item') as has_view_item,
        logical_or(event_name = 'add_to_cart') as has_add_to_cart,
        logical_or(event_name = 'begin_checkout') as has_begin_checkout,
        logical_or(event_name = 'add_payment_info') as has_add_payment_info,

        countif(event_name = 'purchase' and not is_duplicate_purchase) as n_purchases,
        min(if(event_name = 'purchase', event_ts, null)) as first_purchase_ts,
        coalesce(sum(if(event_name = 'purchase' and not is_duplicate_purchase, purchase_revenue_usd, 0)), 0) as revenue_usd
    from events
    group by session_key

)

select
    session_key,
    user_pseudo_id,
    ga_session_id,
    ga_session_number,
    session_start_ts,
    session_end_ts,
    timestamp_diff(session_end_ts, session_start_ts, second) as session_duration_sec,
    n_events,
    landing_page_location,
    coalesce(
        nullif(regexp_extract(landing_page_location, r'^https?://[^/]+(/[^?#]*)'), ''),
        '(not set)'
    ) as landing_page_path,
    traffic.source as source,
    traffic.medium as medium,
    traffic.campaign as campaign,
    traffic is null as source_inferred,
    coalesce(device_category, '(not set)') as device_category,
    coalesce(country, '(not set)') as country,
    has_view_item,
    has_add_to_cart,
    has_begin_checkout,
    has_add_payment_info,
    n_purchases,
    n_purchases > 0 as converted,
    first_purchase_ts,
    revenue_usd
from sessions
