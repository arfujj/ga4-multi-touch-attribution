-- Fails (returns rows) if modeled revenue differs from raw revenue by more than
-- var('revenue_tolerance_pct') percent. Raw revenue is recomputed straight from the public
-- source with the documented de-duplication rule, independently of the staging model.

with raw_purchases as (

    select
        user_pseudo_id,
        (select value.int_value from unnest(event_params) where key = 'ga_session_id') as sid,
        event_timestamp,
        coalesce(
            ecommerce.transaction_id,
            (select value.string_value from unnest(event_params) where key = 'transaction_id'),
            cast((select value.int_value from unnest(event_params) where key = 'transaction_id') as string)
        ) as tid,
        coalesce(ecommerce.purchase_revenue_in_usd, 0) as rev
    from {{ source('ga4', 'events') }}
    where _TABLE_SUFFIX between '{{ var("start_suffix") }}' and '{{ var("end_suffix") }}'
      and event_name = 'purchase'

),

deduped as (

    select *
    from raw_purchases
    where true
    qualify not (
        tid is not null and tid != '(not set)'
        and row_number() over (partition by user_pseudo_id, sid, tid order by event_timestamp) > 1
    )

),

raw as (

    select sum(rev) as revenue from deduped

),

modeled as (

    select 'fct_purchases' as model, coalesce(sum(revenue_usd), 0) as revenue from {{ ref('fct_purchases') }}
    union all
    select 'fct_sessions', coalesce(sum(revenue_usd), 0) from {{ ref('fct_sessions') }}
    union all
    select 'fct_journeys', coalesce(sum(revenue_usd), 0) from {{ ref('fct_journeys') }}

)

select
    m.model,
    m.revenue as modeled_revenue,
    r.revenue as raw_revenue,
    100 * abs(m.revenue - r.revenue) / nullif(r.revenue, 0) as pct_diff
from modeled as m
cross join raw as r
-- coalesce so an empty model or empty raw side fails instead of silently passing
where coalesce(100 * abs(m.revenue - r.revenue) / nullif(r.revenue, 0), 100) > {{ var('revenue_tolerance_pct') }}
