{{ config(
    materialized='table',
    cluster_by=['event_name', 'session_key']
) }}

-- Not date-partitioned on purpose: the BigQuery Sandbox applies a 60-day partition
-- expiration, which would silently drop every 2020/2021 partition on write.

-- One row per raw GA4 event, flattened. Only the columns needed downstream are read.
-- Duplicate purchase events (same user + session + transaction_id, a double-firing tag)
-- are flagged here and excluded from revenue downstream.

with source as (

    select
        event_date,
        event_timestamp,
        event_name,
        event_params,
        user_pseudo_id,
        device.category as device_category,
        geo.country as country,
        ecommerce.transaction_id as ecommerce_transaction_id,
        ecommerce.purchase_revenue_in_usd as purchase_revenue_in_usd
    from {{ source('ga4', 'events') }}
    where _TABLE_SUFFIX between '{{ var("start_suffix") }}' and '{{ var("end_suffix") }}'

),

flattened as (

    select
        parse_date('%Y%m%d', event_date) as event_date,
        timestamp_micros(event_timestamp) as event_ts,
        event_timestamp as event_timestamp_micros,
        event_name,
        user_pseudo_id,
        {{ event_param('ga_session_id', 'int') }} as ga_session_id,
        {{ event_param('ga_session_number', 'int') }} as ga_session_number,
        {{ event_param('page_location') }} as page_location,
        {{ event_param('page_referrer') }} as page_referrer,
        {{ event_param('source') }} as source,
        {{ event_param('medium') }} as medium,
        {{ event_param('campaign') }} as campaign,
        coalesce(
            ecommerce_transaction_id,
            {{ event_param('transaction_id') }},
            cast({{ event_param('transaction_id', 'int') }} as string)
        ) as transaction_id,
        if(event_name = 'purchase', coalesce(purchase_revenue_in_usd, 0), null) as purchase_revenue_usd,
        device_category,
        country
    from source

),

keyed as (

    select
        *,
        concat(user_pseudo_id, '-', cast(ga_session_id as string)) as session_key,
        row_number() over (
            partition by user_pseudo_id, event_timestamp_micros, event_name
            order by page_location, transaction_id
        ) as _dup_rank,
        event_name = 'purchase'
            and transaction_id is not null
            and transaction_id != '(not set)'
            and row_number() over (
                partition by user_pseudo_id, ga_session_id, if(event_name = 'purchase', transaction_id, null)
                order by event_timestamp_micros
            ) > 1 as is_duplicate_purchase
    from flattened

)

select
    to_hex(md5(concat(user_pseudo_id, '|', cast(event_timestamp_micros as string), '|', event_name, '|', cast(_dup_rank as string)))) as event_key,
    * except (_dup_rank)
from keyed
